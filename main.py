
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import swisseph as swe
import math
from datetime import datetime, timedelta
import sqlite3
import random
import statistics


app = FastAPI(title="EPHEMERIS Predictive + Validation + Alignment + Profiles + Statistics")


# ==================================================
# DATABASE
# ==================================================

DB_PATH = "ephemeris_events.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ---- user profiles ----
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            birth_date TEXT,
            birth_time TEXT,
            latitude REAL,
            longitude REAL
        )
    """)

    # ---- life events ----
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            date TEXT,
            event_type TEXT,
            intensity INTEGER,
            notes TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ==================================================
# ENUMS
# ==================================================

class EpistemicState(str, Enum):
    VOID = "VOID"
    CONFLICT = "CONFLICT"
    WEAK = "WEAK"
    CONDITIONAL = "CONDITIONAL"
    STRONG = "STRONG"
    ORACLE = "ORACLE"


class GovernanceDecision(str, Enum):
    SILENCE = "SILENCE"
    CONDITIONAL_LANGUAGE = "CONDITIONAL_LANGUAGE"
    PREDICTION_ALLOWED = "PREDICTION_ALLOWED"
    ORACLE_PERMISSION = "ORACLE_PERMISSION"


# ==================================================
# INPUT MODELS
# ==================================================

class GateInput(BaseModel):
    birth_data_complete: bool
    structural_support: bool
    timing_active: bool
    tradition_consensus: float
    uncertainty_penalty: float
    inner_relevance: float
    outer_relevance: float


class BirthInput(BaseModel):
    user_id: str
    date: str
    time: str
    latitude: float
    longitude: float


class InterpretationInput(BaseModel):
    gate: GateInput
    birth: BirthInput


class EventInput(BaseModel):
    user_id: str
    date: str
    event_type: str
    intensity: int
    notes: str | None = None


# ==================================================
# USER PROFILE ENDPOINTS
# ==================================================

@app.post("/create_user")
def create_user(birth: BirthInput):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO users (user_id, birth_date, birth_time, latitude, longitude)
        VALUES (?, ?, ?, ?, ?)
    """, (birth.user_id, birth.date, birth.time, birth.latitude, birth.longitude))

    conn.commit()
    conn.close()

    return {"status": "user profile saved", "user_id": birth.user_id}


@app.get("/user/{user_id}")
def get_user(user_id: str):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT birth_date, birth_time, latitude, longitude
        FROM users
        WHERE user_id = ?
    """, (user_id,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return {"error": "User not found"}

    return {
        "user_id": user_id,
        "birth_date": row[0],
        "birth_time": row[1],
        "latitude": row[2],
        "longitude": row[3],
    }


# ==================================================
# ASTRONOMY HELPERS
# ==================================================

def planets_from_jd(jd):
    return {
        "Sun": swe.calc_ut(jd, swe.SUN)[0][0],
        "Moon": swe.calc_ut(jd, swe.MOON)[0][0],
        "Mercury": swe.calc_ut(jd, swe.MERCURY)[0][0],
        "Venus": swe.calc_ut(jd, swe.VENUS)[0][0],
        "Mars": swe.calc_ut(jd, swe.MARS)[0][0],
        "Jupiter": swe.calc_ut(jd, swe.JUPITER)[0][0],
        "Saturn": swe.calc_ut(jd, swe.SATURN)[0][0],
    }


def julian_from_datetime(dt):
    return swe.julday(dt.year, dt.month, dt.day, dt.hour + dt.minute / 60.0)


def natal_jd(date: str, time: str):
    y, m, d = map(int, date.split("-"))
    hh, mm = map(int, time.split(":"))
    return swe.julday(y, m, d, hh + mm / 60.0)


# ==================================================
# TRANSIT ACTIVATION
# ==================================================

def angle_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def closeness(angle):
    return 1 - min(angle / 180, 1)


def transit_activation(natal, transit):
    return round(
        0.4 * closeness(angle_diff(natal["Sun"], transit["Sun"])) +
        0.3 * closeness(angle_diff(natal["Moon"], transit["Moon"])) +
        0.3 * closeness(angle_diff(natal["Moon"], transit["Saturn"])),
        3
    )


# ==================================================
# CERTAINTY DECAY
# ==================================================

DECAY_CONSTANT_DAYS = 120


def decay_adjusted(score, days_ahead):
    decay = math.exp(-days_ahead / DECAY_CONSTANT_DAYS)
    return round(score * decay, 4)


# ==================================================
# FUTURE WINDOW SCANNER
# ==================================================

def future_windows(natal):

    now = datetime.utcnow()
    windows = []

    for days in range(0, 181, 15):
        future_time = now + timedelta(days=days)

        jd_future = julian_from_datetime(future_time)
        transit = planets_from_jd(jd_future)

        activation = transit_activation(natal, transit)
        effective = decay_adjusted(activation, days)

        windows.append({
            "date": future_time.strftime("%Y-%m-%d"),
            "activation": activation,
            "effective_certainty": effective
        })

    return sorted(windows, key=lambda x: x["effective_certainty"], reverse=True)[:3]


# ==================================================
# EPISTEMIC + GOVERNANCE
# ==================================================

def compute_base_state(gate: GateInput):

    if not gate.birth_data_complete:
        return EpistemicState.VOID, 0.0

    base = (
        (0.4 if gate.structural_support else 0.0) +
        (0.3 if gate.timing_active else 0.0) +
        (0.3 * gate.tradition_consensus) -
        (0.5 * gate.uncertainty_penalty)
    )

    base = max(0.0, min(1.0, base))

    if base < 0.2:
        return EpistemicState.WEAK, base
    elif base < 0.5:
        return EpistemicState.CONDITIONAL, base
    else:
        return EpistemicState.STRONG, base


ORACLE_FLOOR = 0.92


def oracle_score(score, gate):
    return max(
        0.0,
        min(
            1.0,
            score *
            gate.tradition_consensus *
            (1 - gate.uncertainty_penalty) *
            math.sqrt(gate.inner_relevance * gate.outer_relevance),
        ),
    )


def governance(state, oracle):

    if state in [EpistemicState.VOID, EpistemicState.WEAK]:
        return GovernanceDecision.SILENCE

    if state == EpistemicState.CONDITIONAL:
        return GovernanceDecision.CONDITIONAL_LANGUAGE

    if state == EpistemicState.STRONG and oracle < ORACLE_FLOOR:
        return GovernanceDecision.PREDICTION_ALLOWED

    if oracle >= ORACLE_FLOOR:
        return GovernanceDecision.ORACLE_PERMISSION

    return GovernanceDecision.SILENCE


# ==================================================
# INTERPRETATION
# ==================================================

def interpret(decision):

    if decision == GovernanceDecision.SILENCE:
        return {"mode": "silence", "message": None}

    if decision == GovernanceDecision.CONDITIONAL_LANGUAGE:
        return {"mode": "probabilistic", "message": "Future signals weak or uncertain."}

    if decision == GovernanceDecision.PREDICTION_ALLOWED:
        return {"mode": "structured", "message": "Limited predictive windows detected."}

    if decision == GovernanceDecision.ORACLE_PERMISSION:
        return {"mode": "oracle", "message": "Rare high-certainty future convergence."}


# ==================================================
# FULL READING ENDPOINT
# ==================================================

@app.post("/full_reading/{user_id}")
def full_reading(user_id: str, gate: GateInput):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT birth_date, birth_time FROM users WHERE user_id = ?", (user_id,))
    birth = cursor.fetchone()
    conn.close()

    if not birth:
        return {"error": "User profile missing"}

    natal = planets_from_jd(natal_jd(birth[0], birth[1]))
    peaks = future_windows(natal)

    strongest = peaks[0]["effective_certainty"]

    state, base = compute_base_state(gate)
    oracle = oracle_score(strongest * base, gate)

    decision = governance(state, oracle)
    interpretation = interpret(decision)

    return {
        "natal_planets": natal,
        "future_peaks": peaks,
        "epistemic_state": state,
        "oracle_score": round(oracle, 3),
        "governance_decision": decision,
        "interpretation": interpretation,
    }


# ==================================================
# EVENT LOGGER
# ==================================================

@app.post("/log_event")
def log_event(event: EventInput):

    if not (1 <= event.intensity <= 5):
        return {"error": "Intensity must be between 1 and 5"}

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO events (user_id, date, event_type, intensity, notes)
        VALUES (?, ?, ?, ?, ?)
    """, (event.user_id, event.date, event.event_type, event.intensity, event.notes))

    conn.commit()
    conn.close()

    return {"status": "event logged"}


# ==================================================
# ALIGNMENT VALIDATION
# ==================================================

def date_distance_days(d1: str, d2: str) -> int:
    dt1 = datetime.strptime(d1, "%Y-%m-%d")
    dt2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((dt1 - dt2).days)


def alignment_score(event_date: str, peaks: list) -> float:
    distances = [date_distance_days(event_date, p["date"]) for p in peaks]
    nearest = min(distances)
    return round(math.exp(-nearest / 14), 3)


@app.get("/validate/{user_id}")
def validate_user(user_id: str):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT birth_date, birth_time FROM users WHERE user_id = ?", (user_id,))
    birth = cursor.fetchone()

    if not birth:
        return {"error": "User profile missing"}

    natal = planets_from_jd(natal_jd(birth[0], birth[1]))
    peaks = future_windows(natal)

    cursor.execute("SELECT date FROM events WHERE user_id = ?", (user_id,))
    events = cursor.fetchall()
    conn.close()

    if not events:
        return {"message": "No events logged"}

    scores = [alignment_score(e[0], peaks) for e in events]
    avg = round(sum(scores) / len(scores), 3)

    return {
        "events": len(events),
        "average_alignment": avg,
        "scores": scores,
        "future_peaks": peaks
    }


# ==================================================
# STATISTICAL SIGNIFICANCE ENGINE
# ==================================================

def simulate_random_alignment(num_events: int, peaks: list, trials: int = 500):

    if num_events == 0:
        return 0.0, 0.0

    random_scores = []

    for _ in range(trials):
        trial_scores = []

        for _ in range(num_events):
            rand_day = random.randint(0, 180)
            rand_date = (datetime.utcnow() + timedelta(days=rand_day)).strftime("%Y-%m-%d")
            trial_scores.append(alignment_score(rand_date, peaks))

        random_scores.append(sum(trial_scores) / len(trial_scores))

    return statistics.mean(random_scores), statistics.stdev(random_scores)


def interpret_confidence(z: float) -> str:
    if z < 0.5:
        return "indistinguishable_from_chance"
    elif z < 1.5:
        return "weak_signal"
    elif z < 2.5:
        return "moderate_evidence"
    else:
        return "strong_evidence"


@app.get("/statistics/{user_id}")
def statistical_validation(user_id: str):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT birth_date, birth_time FROM users WHERE user_id = ?", (user_id,))
    birth = cursor.fetchone()

    if not birth:
        return {"error": "User profile missing"}

    natal = planets_from_jd(natal_jd(birth[0], birth[1]))
    peaks = future_windows(natal)

    cursor.execute("SELECT date FROM events WHERE user_id = ?", (user_id,))
    events = cursor.fetchall()
    conn.close()

    if not events:
        return {"message": "No events logged"}

    observed_scores = [alignment_score(e[0], peaks) for e in events]
    observed_mean = sum(observed_scores) / len(observed_scores)

    rand_mean, rand_std = simulate_random_alignment(len(events), peaks)

    if rand_std == 0:
        return {"error": "Insufficient data for statistics"}

    z_score = (observed_mean - rand_mean) / rand_std

    return {
        "events": len(events),
        "observed_alignment": round(observed_mean, 3),
        "random_alignment_mean": round(rand_mean, 3),
        "random_alignment_std": round(rand_std, 3),
        "confidence_z": round(z_score, 3),
        "interpretation": interpret_confidence(z_score),
    }

# ==================================================
# PRIVACY POLICY ENDPOINT (REQUIRED FOR GPT ACTIONS)
# ==================================================

@app.get("/privacy")
def privacy():
    return {
        "service": "EPHEMERIS-01",
        "policy": (
            "EPHEMERIS-01 does not store, sell, or share personal data. "
            "Birth details and life events are processed only to generate "
            "symbolic astrological interpretations requested directly by the user. "
            "No tracking, advertising, or third-party data sharing is performed."
        ),
        "data_storage": "Minimal temporary processing only",
        "contact": "Vansh Bhatnagar"
    }
