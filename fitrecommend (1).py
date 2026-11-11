# BIG PICTURE 
# --------------------------------------
#   1. KNOWLEDGE BASE  - load a CSV of exercises -> clean table `exercise_kb`.
#   2. ENGINE          - safety_filter -> rank_exercises -> assemble_weekly_plan
#   3. USER INTERFACE  - Streamlit pages + a tiny router at the bottom.
#
# REMINDER: Streamlit re-runs this whole file on every click, so anything that
# must be remembered (current page, plan, feedback, progression...) lives in
# `st.session_state`.
# =============================================================================

import hashlib
import json
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# TUNABLE CONSTANTS - change these numbers to tweak the algorithm
# =============================================================================
SECONDS_PER_REP = 4                 # estimated time to perform one repetition
SETUP_SECONDS_PER_EXERCISE = 30     # walking to the rack, loading weights, etc. (0 = pure formula)
WARMUP_SECONDS = 300                # 5-minute dynamic warm-up
COOLDOWN_SECONDS = 300              # 5-minute static cooldown
COUNT_WARMUP_COOLDOWN_IN_DURATION = True   # True: warm-up + cooldown come out of the chosen session length
MAX_RESISTANCE_EXERCISES = 8        # safety cap so very long sessions don't explode
MIN_UNILATERAL_PER_SESSION = 1      # at least this many one-limb exercises per session

# Progressive-overload settings (see evaluate_week below).
# Sets never change. Each week the reps go up; when they reach the top of the rep
# target, the user adds weight and the reps start again ("double progression").
PERFECT_REP_STEP = 1                # a "Perfect" week adds 1 rep per set
EASY_REP_STEP = 2                   # a "Too Easy" week adds 2 reps per set
MAX_BODYWEIGHT_EXTRA_REPS = 10      # exercises without weights can grow this far above the start
MAX_DIFFICULTY_BUMP = 1             # allow at most ONE tier above the self-rated level
# Equipment whose load the user chooses in kg (everything else progresses by reps only).
WEIGHTED_EQUIPMENT = {"dumbbells", "barbell", "e-z curl bar", "kettlebells", "machine", "cable", "medicine ball"}

# Feedback tag names
TAG_TOO_HEAVY = "Too Heavy"
TAG_EXHAUSTING = "Cardiovascularly Exhausting"
TAG_JOINT_PAIN = "Joint Pain"

# Age and beginner safety rules (design decisions, not taken from a guideline)
MIN_AGE = 18                        # the app is for adults
SENIOR_AGE = 50                     # from this age: no jump / plyometric exercises, slightly more rest
SENIOR_EXTRA_REST = 15              # seconds added to rest on strength exercises

LEVEL_RANK = {"beginner": 1, "intermediate": 2, "advanced": 3}
LEVEL_NAMES = {1: "beginner", 2: "intermediate", 3: "advanced"}
SAME_LEVEL_BONUS = 0.15            # ranking bonus for exercises at the user's own level
SECTION_ORDER = ["Warm-up", "Main", "Finisher", "Cooldown"]

# --- Configuration & Initialization ---
st.set_page_config(page_title="FitRecommend", layout="centered")
def apply_background():
    """Injects only the FitPro matte dark ambient gradient background."""
    custom_css = """
    <style>
    .stApp {
        background: radial-gradient(circle at 50% 0%, #112217 0%, #0B0E0C 55%, #070908 100%) !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Run the background styling
apply_background()

st.markdown("""
<style>

/* Button animation */
.stButton > button {
    transition: all 0.25s ease-in-out;
    border-radius: 10px;
    font-weight: 600;
}

/* Hover effect */
.stButton > button:hover {
    transform: translateY(-3px) scale(1.03);
    box-shadow: 0 5px 15px rgba(0, 255, 170, 0.3);
    border-color: #00FFAA;
    color: #00FFAA;
}

/* Click effect */
.stButton > button:active {
    transform: scale(0.96);
}

</style>
""", unsafe_allow_html=True)

def new_progression():
    """Starting state of the progressive overload engine."""
    return {
        "reps_bump": 0,          # reps added for weighted exercises (goes back to 0 when weight goes up)
        "bodyweight_reps": 0,    # reps added for exercises without weights (never resets)
        "weight_stage": 0,       # how many times the plan has told the user to add weight
        "weight_up_week": 0,     # week number in which a weight increase applies (0 = none)
        "difficulty_bump": 0,    # how many tiers above your level we may recommend
        "history": [],           # human-readable log shown on the Progress page
    }


# Default values for everything we want to remember between re-runs.
# "if ... not in" means: only set it the FIRST time the app opens.
_defaults = {
    "page": "home",
    "feedback": {},                 # format: {exercise_name: {tag: count}}
    "weekly_plan": pd.DataFrame(),
    "selected_exercise": None,
    "completed_workouts": 0,
    "progression": new_progression(),   # overload state
    "weights": {},                      # {exercise_name: kg} counted weights (days with feedback submitted)
    "weight_drafts": {},                # {day: {exercise_name: kg}} typed but NOT counted until that day's feedback is submitted
    "weight_log": [],                   # [{"week", "exercise", "kg"}] one entry per exercise per week
    "week_number": 1,
    "week_feedback": {},                # {day: "Too Easy"/"Perfect"/"Too Hard"} for this week
    "session_log": [],                  # one entry per rated session
    "blacklisted_exercises": set(),     # names removed forever after "Joint Pain"
    "blacklisted_patterns": set(),      # (joint_action, muscle_group) pairs removed forever
    "feedback_day": None,               # which day the feedback form is about
    "user_id": None,                    # logged-in user (None = guest)
    "nickname": None,
    "flash": None,                      # one-time message shown after a rerun
}

#safety for app memory. When user press any button, code run again from top to bottom,
#preserve the user data
for _key, _value in _defaults.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value

default_profile = {
    "age": 21, "level": "Beginner", "goal": "Muscle Gain", "location": "Gym",
    "equipment": None, "days": 4, "duration": 60,     # None = not chosen yet (Gym -> everything ticked)
}
if 'profile' not in st.session_state:
    st.session_state.profile = default_profile


def navigate_to(page_name):
    """Switch page by changing the remembered page name (call st.rerun() after)."""
    st.session_state.page = page_name


def show_flash():
    """Show a one-time message (e.g. 'Feedback saved') then forget it."""
    flash = st.session_state.get("flash")
    if flash:
        kind, text = flash
        {"success": st.success, "warning": st.warning}.get(kind, st.info)(text)
        st.session_state.flash = None


# =============================================================================
# 1. DATA & KNOWLEDGE BASE BUILDER
# =============================================================================

KB_FILE = "exercise_kb_final.csv"   # produced by the preprocessing notebook (dp_final.ipynb)


@st.cache_data
def load_and_build_kb():
    """Load the knowledge base prepared OFFLINE in dp_final.ipynb.
    All preprocessing (Free Exercise DB cleaning, megaGym rating filter, hand-written
    warm-ups / cool-downs, compound / unilateral / joint-action labels) happens there.
    This function only reads the result and restores the column types."""
    path = Path(__file__).with_name(KB_FILE)
    try:
        kb = pd.read_csv(path, keep_default_na=False)
    except FileNotFoundError:
        st.error(f"{KB_FILE} was not found next to the app. Run dp_final.ipynb and upload the file.")
        st.stop()

    kb["location_suitability"] = kb["location_suitability"].apply(json.loads)
    kb["is_compound"] = kb["is_compound"].astype(str).str.lower().eq("true")
    for col in ["exercise_category", "major_muscle_group", "equipment_needed", "difficulty_level",
                "movement_type", "joint_action"]:
        kb[col] = kb[col].astype(str).str.strip().str.lower()
    kb["instructions"] = kb["instructions"].astype(str)
    for col in ["image_1", "image_2"]:
        if col not in kb.columns:
            kb[col] = ""
    return kb.drop_duplicates("exercise_name").reset_index(drop=True)


exercise_kb = load_and_build_kb()


EQUIPMENT_ORDER = ["dumbbells", "barbell", "e-z curl bar", "kettlebells", "bench", "pull-up bar", "dip bars", "resistance band",
                   "machine", "cable", "exercise ball", "foam roll", "lacrosse ball", "jump rope"]


def equipment_choices(kb):
    """Equipment labels for the profile form, built from exercise_kb_final.csv (bodyweight is always allowed)."""
    values = {str(v).strip().lower() for v in kb["equipment_needed"]} - {"bodyweight", ""}
    ordered = [v for v in EQUIPMENT_ORDER if v in values] + sorted(values - set(EQUIPMENT_ORDER))
    return [v.title() for v in ordered]      # the safety filter lowercases labels, so they match the KB values
# =============================================================================
# 2. RECOMMENDATION ENGINE PIPELINE
# =============================================================================

def blacklist_mask(frame, names, patterns):
    """True for every row that is blacklisted, either because its NAME is
    banned or because it matches a banned (joint_action, muscle_group) PATTERN."""
    if frame.empty:
        return pd.Series(False, index=frame.index, dtype=bool)
    by_name = frame["exercise_name"].astype(str).str.lower().isin(names)
    by_pattern = pd.Series(
        [(a, g) in patterns for a, g in zip(frame["joint_action"], frame["major_muscle_group"])],
        index=frame.index, dtype=bool,
    )
    return by_name | by_pattern


def effective_level_rank(profile):
    """self-rated level (1-3) PLUS any 'harder variant' unlocked by progression."""
    base = LEVEL_RANK.get(str(profile["self_rated_fitness_level"]).lower(), 1)
    bump = (profile.get("progression") or {}).get("difficulty_bump", 0)
    return min(3, base + bump)


def safety_filter(exercises, profile, feedback):
    """STEP A: remove exercises that are unsafe or impossible for this user."""
    user_level = effective_level_rank(profile)
    allowed_equipment = {str(x).strip().lower() for x in profile["available_equipment"]}
    location = profile["workout_location"].strip().lower()
    # permanent Joint-Pain blacklist
    banned_names = {str(x).lower() for x in profile.get("blacklisted_exercises", [])}
    banned_patterns = set(profile.get("blacklisted_patterns", []))

    rows = exercises.copy()
    rows = rows[rows["difficulty_level"].map(LEVEL_RANK).fillna(3) <= user_level]
    rows = rows[rows["equipment_needed"].astype(str).str.lower().isin(allowed_equipment) | (rows["equipment_needed"] == "bodyweight")]
    rows = rows[rows["location_suitability"].apply(lambda places: location in [str(p).lower() for p in places])]
    # conservative rule for users aged 50+ -> no jump or plyometric exercises
    if int(profile.get("age") or 0) >= SENIOR_AGE:
        rows = rows[rows["exercise_category"] != "plyometrics"]
        rows = rows[~rows["exercise_name"].astype(str).str.lower().str.contains("jump|plyometric", regex=True)]
    rows = rows[~blacklist_mask(rows, banned_names, banned_patterns)]
    # NOTE: "Too Hard"-style feedback no longer REMOVES exercises. "Too Heavy" now only
    # lowers the ranking score (see rank_exercises); only Joint Pain removes exercises.
    return rows.reset_index(drop=True)


def training_days(profile, n_days):
    """Weekly split (what each training day focuses on) based on fitness level."""
    level = str(profile.get("self_rated_fitness_level", "beginner")).lower()
    if level == "beginner":
        base = [
            {"label": "Full Body", "groups": ["quads", "chest", "back", "shoulders", "glutes", "core"], "finisher": None},
            {"label": "Full Body + Cardio", "groups": ["glutes", "back", "chest", "quads", "core", "shoulders"], "finisher": "cardio"},
            {"label": "Full Body + Flexibility", "groups": ["chest", "quads", "back", "glutes", "shoulders", "core"], "finisher": "flexibility"},
        ]
    elif level == "intermediate":
        base = [
            {"label": "Upper Body", "groups": ["chest", "back", "shoulders", "biceps", "triceps"], "finisher": None},
            {"label": "Lower Body", "groups": ["quads", "hamstrings", "glutes", "calves", "core"], "finisher": None},
            {"label": "Upper Body + Cardio", "groups": ["back", "chest", "shoulders", "biceps", "triceps"], "finisher": "cardio"},
            {"label": "Lower Body + Flexibility", "groups": ["hamstrings", "quads", "glutes", "calves", "core"], "finisher": "flexibility"},
        ]
    else:
        base = [
            {"label": "Upper Body (Strength)", "groups": ["chest", "back", "shoulders", "biceps", "triceps"], "finisher": None},
            {"label": "Lower Body (Strength)", "groups": ["quads", "hamstrings", "glutes", "calves", "core"], "finisher": None},
            {"label": "Push", "groups": ["chest", "shoulders", "triceps"], "finisher": None},
            {"label": "Pull + Cardio", "groups": ["back", "biceps", "shoulders"], "finisher": "cardio"},
            {"label": "Legs + Flexibility", "groups": ["quads", "hamstrings", "glutes", "calves", "core"], "finisher": "flexibility"},
        ]
    days_out = [dict(base[index % len(base)]) for index in range(n_days)]
    if str(profile.get("fitness_goal", "")).lower().replace(" ", "_") == "weight_loss":
        for d in days_out:               # weight loss: every day = resistance block + cardio block
            d["finisher"] = "cardio"
            d["label"] = d["label"].split(" + ")[0] + " + Cardio"
    return days_out


def preferred_type(goal):
    """Compound lifts are preferred for every goal except Muscle Gain, where isolation
    work is equally useful for growing individual muscles (no preference)."""
    return None if str(goal).lower().replace(" ", "_") == "muscle_gain" else "compound"


def rating_score(value):
    """megaGym rating (8.0-10.0 after the notebook filter) scaled to 0.33-1.0; unrated = 0.5."""
    try:
        return round(min(max((float(value) - 7.0) / 3.0, 0.0), 1.0), 3)
    except (TypeError, ValueError):
        return 0.5


def exercise_features(row, profile):
    """What each exercise offers. Only features that DIFFER between exercises are used
    (goal, location and category are already fixed by the safety filter and the pools)."""
    owned = {str(x).strip().lower() for x in profile["available_equipment"]}
    equipment = str(row["equipment_needed"]).lower()
    features = {
        "difficulty": str(row["difficulty_level"]).lower(),
        "own_equipment": 1.0 if equipment != "bodyweight" and equipment in owned else 0.0,
        "rating": rating_score(row.get("rating")),
    }
    if str(row["exercise_category"]).lower() == "resistance":
        features["type"] = "compound" if bool(row["is_compound"]) else "isolation"
    return features


def user_features(profile):
    """The user's 'ideal exercise': at their level, using the equipment they own, highly
    rated, and of the type that suits their goal."""
    ideal = {
        # after progression unlocks a harder tier, we aim at that tier.
        "difficulty": LEVEL_NAMES[effective_level_rank(profile)],
        "own_equipment": 1.0,
        "rating": 1.0,
    }
    if preferred_type(profile["fitness_goal"]):
        ideal["type"] = preferred_type(profile["fitness_goal"])
    return ideal


def feedback_penalty(tags):
    """score reduction for one exercise from its negative feedback tags.
    `tags` looks like {"Too Heavy": 2}. Each report counts, capped at 3 reports."""
    heavy = min(tags.get(TAG_TOO_HEAVY, 0), 3)
    exhausted = min(tags.get(TAG_EXHAUSTING, 0), 3)
    return 0.3 * heavy + 0.15 * exhausted


def rank_exercises(safe_pool, profile, feedback):
    """STEP B: score every safe exercise (cosine similarity to the user's ideal)."""
    if safe_pool.empty:
        return safe_pool.assign(recommendation_score=pd.Series(dtype=float))

    item_dicts = [exercise_features(row, profile) for _, row in safe_pool.iterrows()]
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(item_dicts + [user_features(profile)])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()

    ranked = safe_pool.copy()
    ranked["recommendation_score"] = scores

    # exercises at the user's own level are preferred; easier ones stay allowed as a fallback.
    own_level = LEVEL_NAMES[effective_level_rank(profile)]
    ranked.loc[ranked["difficulty_level"].astype(str).str.lower() == own_level, "recommendation_score"] += SAME_LEVEL_BONUS

    for idx, row in ranked.iterrows():
        # "Too Heavy" / "Cardiovascularly Exhausting" lower the score.
        ranked.loc[idx, "recommendation_score"] -= feedback_penalty(feedback.get(row["exercise_name"], {}))

    return ranked.sort_values("recommendation_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# prescription (sets / reps / rest) and the time-cost formula
# ---------------------------------------------------------------------------
def estimate_exercise_seconds(sets, reps, rest_seconds, sides=1):
    """(feature 1) (Sets x Reps x 4 s) + ((Sets - 1) x Rest).
    `sides` is 2 for one-limb exercises, because you do every rep on BOTH sides."""
    work = sets * reps * SECONDS_PER_REP * sides
    rest = max(sets - 1, 0) * rest_seconds
    return work + rest


# ---------------------------------------------------------------------------
# Starting-point recommendations by level + goal (proposed ranges for this system,
# not scientifically proven optimal values). "Strength" has no row of its own, so it
# uses the Hypertrophy row.
# ---------------------------------------------------------------------------
EXPERIENCE_LABELS = {
    "Beginner": "Beginner (0–6 months of training)",
    "Intermediate": "Intermediate (6–24 months of training)",
    "Advanced": "Advanced (2+ years of training)",
}

def rec_group(goal):
    g = str(goal).lower().replace(" ", "_")
    return {"weight_loss": "weight_loss", "general_fitness": "general_fitness"}.get(g, "hypertrophy")

DAYS_REC = {   # (level, group) -> (min days, max days)
    ("beginner", "hypertrophy"): (3, 3), ("intermediate", "hypertrophy"): (4, 4), ("advanced", "hypertrophy"): (4, 5),
    ("beginner", "weight_loss"): (3, 3), ("intermediate", "weight_loss"): (3, 4), ("advanced", "weight_loss"): (4, 4),
    ("beginner", "general_fitness"): (3, 3), ("intermediate", "general_fitness"): (3, 4), ("advanced", "general_fitness"): (4, 4),
}
DURATION_REC = {   # (level, group) -> (min minutes, max minutes)
    ("beginner", "hypertrophy"): (30, 45), ("intermediate", "hypertrophy"): (45, 60), ("advanced", "hypertrophy"): (60, 75),
    ("beginner", "weight_loss"): (30, 45), ("intermediate", "weight_loss"): (45, 60), ("advanced", "weight_loss"): (60, 75),
    ("beginner", "general_fitness"): (30, 45), ("intermediate", "general_fitness"): (40, 60), ("advanced", "general_fitness"): (45, 75),
}
# Weight loss: resistance and cardio are planned separately (minutes, by level).
WL_RESISTANCE_MIN = {"beginner": (30, 35), "intermediate": (35, 45), "advanced": (40, 50)}
WL_CARDIO_MIN = {"beginner": (15, 20), "intermediate": (20, 30), "advanced": (25, 30)}
WL_CARDIO_MAX_SHARE = 0.4          # cardio never takes more than 40% of a short session


def _fmt_range(lo, hi, unit):
    return f"{lo} {unit}" if lo == hi else f"{lo}–{hi} {unit}"


def days_recommendation(level, goal):
    lo, hi = DAYS_REC[(str(level).lower(), rec_group(goal))]
    return lo, hi


def duration_recommendation(level, goal):
    return DURATION_REC[(str(level).lower(), rec_group(goal))]


def cardio_minutes(session_minutes, user_goal="general_fitness", level_rank=2):
    """Cardio length. Weight loss: a separate cardio block sized by level (upper end of the
    range, capped at 40% of the session). Other goals: about 20% of the session."""
    if user_goal == "weight_loss":
        level = LEVEL_NAMES.get(level_rank, "intermediate")
        target = WL_CARDIO_MIN[level][1]
        cap = int(session_minutes * WL_CARDIO_MAX_SHARE // 5 * 5)
        return int(max(5, min(target, cap)))
    return int(np.clip(round(session_minutes * 0.2 / 5) * 5, 5, 20))


# Reference ranges by goal (reps, sets, rest in seconds). The starting values below sit
# inside these ranges; the top of the rep range is the limit for exercises without weights.
#   Goal in the app   -> row of the reference table
#   Strength          -> Maximal strength   (1-6 reps, 2-4 sets, 3-5 min rest)
#   Muscle Gain       -> Muscle hypertrophy (5-30 reps, 2-4 sets, 1-3 min rest)
#   General Fitness   -> General fitness    (8-15 reps, 2-3 sets, 1-2 min rest)
#   Weight Loss       -> Weight loss        (8-15 reps, 2-4 sets, 1-2 min rest)
GOAL_RANGES = {
    "strength":        {"reps": (1, 6),  "sets": (2, 4), "rest": (180, 300)},
    "muscle_gain":     {"reps": (5, 30), "sets": (2, 4), "rest": (60, 180)},
    "general_fitness": {"reps": (8, 15), "sets": (2, 3), "rest": (60, 120)},
    "weight_loss":     {"reps": (8, 15), "sets": (2, 4), "rest": (60, 120)},
}

# Starting point (sets, min reps, max reps, rest seconds) inside those ranges.
# Beginners start with fewer sets / moderate reps; intermediate and advanced start higher.
BEGINNER_START = {
    "strength":        (3, 4, 6, 180),
    "muscle_gain":     (3, 8, 12, 90),
    "general_fitness": (2, 10, 12, 60),
    "weight_loss":     (3, 10, 12, 60),
}
EXPERIENCED_START = {
    "strength":        (4, 3, 5, 240),
    "muscle_gain":     (4, 6, 12, 120),
    "general_fitness": (3, 8, 12, 90),
    "weight_loss":     (4, 10, 15, 60),
}


def goal_key(user_goal):
    return user_goal if user_goal in GOAL_RANGES else "general_fitness"


def base_prescription(user_goal, level_rank=2):
    """(sets, min reps, max reps, rest seconds) for strength exercises, by goal AND level."""
    key = goal_key(user_goal)
    return (BEGINNER_START if level_rank <= 1 else EXPERIENCED_START)[key]


def is_weighted(equipment):
    """True when the user picks the load in kg (dumbbells, barbell, machine ...)."""
    return str(equipment).strip().lower() in WEIGHTED_EQUIPMENT


def rep_limits(user_goal, level_rank=2):
    """(start reps, top reps for weighted exercises, top reps for exercises without weights)."""
    _, low, high, _ = base_prescription(user_goal, level_rank)
    goal_top = GOAL_RANGES[goal_key(user_goal)]["reps"][1]
    return low, high, min(goal_top, low + MAX_BODYWEIGHT_EXTRA_REPS)


def prescribe_exercise(row, user_goal, session_minutes, progression, feedback, section, level_rank=2, age=0):
    """work out sets, reps, rest AND the estimated duration of one exercise.
    Sets and rest are fixed by goal and level. Reps are ONE number that grows with the
    overload state. The 'Cardiovascularly Exhausting' tag adds rest / shortens cardio."""
    category = str(row["exercise_category"]).lower()
    tags = feedback.get(row["exercise_name"], {})
    exhausted = min(tags.get(TAG_EXHAUSTING, 0), 2)

    if category == "cardio":
        minutes = max(5, cardio_minutes(session_minutes, user_goal, level_rank) - 5 * exhausted)
        return {"sets": 1, "reps": f"{minutes} minutes", "rest": "No rest",
                "seconds": minutes * 60, "setup": 0}

    if category in ("mobility", "flexibility"):
        sets, hold = 2, 30
        rest = 10 if section in ("Warm-up", "Cooldown") else 30
        label = "30 seconds" if category == "mobility" else "30 seconds per stretch"
        return {"sets": sets, "reps": label, "rest": f"{rest} seconds",
                "seconds": sets * hold + (sets - 1) * rest, "setup": 0}

    sides = 2 if (category == "resistance" and str(row["movement_type"]) == "unilateral") else 1
    suffix = " per side" if sides == 2 else ""

    if category == "plyometrics":
        sets, reps, rest = 3, 10, 60
    else:
        sets, _, _, rest = base_prescription(user_goal, level_rank)
        low, top_weighted, top_bodyweight = rep_limits(user_goal, level_rank)
        if is_weighted(row["equipment_needed"]):
            reps = min(low + progression["reps_bump"], top_weighted)        # then the weight goes up
        else:
            reps = min(low + progression["bodyweight_reps"], top_bodyweight)
    rest += 15 * exhausted
    if age >= SENIOR_AGE:
        rest += SENIOR_EXTRA_REST     # a little more recovery for users aged 50+

    seconds = estimate_exercise_seconds(sets, reps, rest, sides)
    return {"sets": sets, "reps": f"{reps} reps{suffix}", "rest": f"{rest} seconds",
            "seconds": seconds, "setup": SETUP_SECONDS_PER_EXERCISE}


def build_plan_row(row, rx, day, focus, section):
    """turn one exercise + its prescription into one row of the plan table."""
    category = str(row["exercise_category"]).lower()
    is_strength = category == "resistance"
    return {
        "Day": day, "Focus": focus, "Section": section,
        "Exercise": row["exercise_name"], "Sets": str(rx["sets"]), "Reps": rx["reps"], "Rest": rx["rest"],
        "Target": row["target_muscle"].title(),
        "Type": ("Compound" if bool(row["is_compound"]) else "Isolation") if is_strength else category.title(),
        "Movement": str(row["movement_type"]).title() if is_strength else "—",
        "Difficulty": row["difficulty_level"].title(),
        "Equipment": row["equipment_needed"].title(),
        "Instructions": row["instructions"],
        "Seconds": rx["seconds"] + rx["setup"],      # total time cost incl. setup
        "Score": round(float(row.get("recommendation_score", 0) or 0), 3),
    }


def pick_support_exercises(pool, groups, target_seconds, section, exclude_names, rx_for):
    """(feature 7) choose warm-up or cooldown exercises worth ~target_seconds.
    Order of preference: one per trained muscle group -> more for those groups ->
    full-body movements -> anything left. A `while` loop adds items until the
    next one would push us past the target."""
    if pool.empty:
        return []
    ordered, seen = [], set(exclude_names)

    def queue(frame):
        for _, candidate in frame.iterrows():
            if candidate["exercise_name"] not in seen:
                seen.add(candidate["exercise_name"])
                ordered.append(candidate)

    for group in groups:
        queue(pool[pool["major_muscle_group"] == group].head(1))
    queue(pool[pool["major_muscle_group"].isin(groups)])
    queue(pool[pool["major_muscle_group"] == "other"])
    queue(pool)

    picked, total, idx = [], 0, 0
    while idx < len(ordered):
        rx = rx_for(ordered[idx], section)
        if total + rx["seconds"] <= target_seconds:
            picked.append((ordered[idx], rx))
            total += rx["seconds"]
        idx += 1
    return picked


def assemble_weekly_plan(ranked, profile, feedback):
    """STEP C: build the weekly schedule.
    Each day = Warm-up -> Main (compound first) -> Finisher -> Cooldown, sized to
    fill the user's chosen session length using the time formula."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    n_days = int(np.clip(profile["preferred_days_per_week"], 2, 6))
    session_minutes = int(np.clip(profile.get("session_duration_minutes", 60), 30, 90))
    day_indices = {
        2: [0, 3], 3: [0, 2, 4], 4: [0, 1, 3, 5],
        5: [0, 1, 3, 4, 6], 6: [0, 1, 2, 4, 5, 6]
    }[n_days]

    if ranked.empty:
        return pd.DataFrame()

    progression = profile.get("progression") or new_progression()
    user_goal = str(profile.get("fitness_goal", "General Fitness")).lower().replace(" ", "_")
    split_days = training_days(profile, n_days)
    level_rank = LEVEL_RANK.get(str(profile.get("self_rated_fitness_level", "beginner")).lower(), 1)
    age = int(profile.get("age") or 0)

    # total time we may spend, in seconds.
    total_budget = session_minutes * 60
    if COUNT_WARMUP_COOLDOWN_IN_DURATION:
        total_budget -= WARMUP_SECONDS + COOLDOWN_SECONDS

    def rx_for(row, section):
        return prescribe_exercise(row, user_goal, session_minutes, progression, feedback, section, level_rank, age)

    used, previous_muscles, muscle_usage, rows_out = set(), set(), {}, []

    for i, day_index in enumerate(day_indices):
        day = days[day_index]
        session = split_days[i]
        finisher = session["finisher"]

        # --- candidate ordering ---
        unused_candidates = ranked[~ranked["exercise_name"].isin(used)].copy()
        recycled_candidates = ranked[ranked["exercise_name"].isin(used)].copy()
        unused_candidates["previously_used"] = 0
        recycled_candidates["previously_used"] = 1
        candidates = pd.concat([unused_candidates, recycled_candidates], ignore_index=True)
        candidates["is_compound_int"] = candidates["is_compound"].astype(int)
        # Flag exercises with negative feedback so they are picked LAST. (The score
        # penalty alone would only matter as the final tie-breaker below.)
        candidates["penalised"] = candidates["exercise_name"].map(
            lambda n: int(feedback_penalty(feedback.get(n, {})) > 0))
        candidates["muscle_usage"] = candidates["target_muscle"].astype(str).str.lower().map(
            lambda muscle: muscle_usage.get(muscle, 0))
        # (feature 2) is_compound sits right after "previously_used", so compound
        # lifts are always preferred over isolation work.
        candidates = candidates.sort_values(
            ["previously_used", "is_compound_int", "penalised", "muscle_usage", "recommendation_score"],
            ascending=[True, False, True, True, False])

        resistance_pool = candidates[candidates["exercise_category"] == "resistance"]

        # --- reserve time for the finisher (cardio / flexibility), if any ---
        finisher_row, finisher_rx = None, None
        if finisher is not None:
            finisher_pool = candidates[candidates["exercise_category"] == finisher]
            if finisher == "cardio":
                # plyometrics also count as a cardio-type finisher (users aged 50+ never get them:
                # the safety filter removed them). Weight loss keeps real cardio first.
                plyo = candidates[candidates["exercise_category"] == "plyometrics"]
                if user_goal == "weight_loss":
                    finisher_pool = pd.concat([finisher_pool, plyo])
                else:
                    finisher_pool = candidates[candidates["exercise_category"].isin(["cardio", "plyometrics"])]
            if not finisher_pool.empty:
                finisher_row = finisher_pool.iloc[0]
                finisher_rx = rx_for(finisher_row, "Finisher")
        main_budget = max(0, total_budget - (finisher_rx["seconds"] if finisher_rx else 0))

        # --- pick the main exercises ---
        chosen, session_muscles, state = [], set(), {"time": 0}

        def try_add(row, enforce_variety=True):
            """Add `row` if it is new, passes the variety rule AND fits in the time left."""
            name = row["exercise_name"]
            if len(chosen) >= MAX_RESISTANCE_EXERCISES:
                return False
            if any(item["row"]["exercise_name"] == name for item in chosen):
                return False
            category = str(row["exercise_category"]).lower()
            muscle = str(row["target_muscle"]).lower()
            if enforce_variety and category == "resistance" and (muscle in previous_muscles or muscle in session_muscles):
                return False
            rx = rx_for(row, "Main")
            cost = rx["seconds"] + rx["setup"]
            if state["time"] + cost > main_budget:
                return False
            chosen.append({"row": row, "rx": rx})
            state["time"] += cost
            if category == "resistance":
                session_muscles.add(muscle)
            return True

        def unilateral_count():
            return sum(1 for item in chosen if item["row"]["movement_type"] == "unilateral")

        # Only exercises for today's muscle groups may enter the main block, so an
        # Upper Body day never gets leg exercises (and the other way round).
        today_pool = resistance_pool[resistance_pool["major_muscle_group"].isin(session["groups"])]

        # Phase 0 (feature 6): guarantee the minimum number of one-limb exercises
        # for today's muscle groups (skipped if there is none).
        unilateral_today = today_pool[today_pool["movement_type"] == "unilateral"]
        for enforce in (True, False):
            for _, row in unilateral_today.iterrows():
                if unilateral_count() >= MIN_UNILATERAL_PER_SESSION:
                    break
                try_add(row, enforce)

        # Phase 1: cover each muscle group in today's template (one exercise each).
        for muscle_group in session["groups"]:
            if state["time"] >= main_budget:
                break
            if any(item["row"]["major_muscle_group"] == muscle_group for item in chosen):
                continue
            for _, row in resistance_pool[resistance_pool["major_muscle_group"] == muscle_group].iterrows():
                if try_add(row, True):
                    break

        # Phase 2 (feature 1): keep adding the next best exercise for today's muscle groups
        # WHILE time remains. First respecting muscle variety, then relaxing it if the window
        # is still unfilled. If today's groups run out, the session is simply shorter.
        for enforce in (True, False):
            queue = [row for _, row in today_pool.iterrows()]
            idx = 0
            while idx < len(queue) and len(chosen) < MAX_RESISTANCE_EXERCISES and state["time"] < main_budget:
                try_add(queue[idx], enforce)
                idx += 1

        # Fallback: if there is no strength option at all, allow any non-warm-up exercise.
        if not chosen:
            fallback = candidates[~candidates["exercise_category"].isin(["mobility", "flexibility"])]
            for _, row in fallback.iterrows():
                try_add(row, False)

        # (feature 2) compound lifts first. sorted() is stable, so the order
        # within compound / isolation groups is kept.
        ordered_main = sorted(chosen, key=lambda item: 0 if bool(item["row"]["is_compound"]) else 1)

        # --- (feature 7) warm-up + cooldown targeting today's muscle groups ---
        trained_groups = []
        for item in ordered_main:
            if str(item["row"]["exercise_category"]).lower() == "resistance":
                group = item["row"]["major_muscle_group"]
                if group not in trained_groups:
                    trained_groups.append(group)
        if not trained_groups:
            trained_groups = list(session["groups"])

        in_session = {item["row"]["exercise_name"] for item in ordered_main}
        if finisher_row is not None:
            in_session.add(finisher_row["exercise_name"])
        warmups = pick_support_exercises(
            candidates[candidates["exercise_category"] == "mobility"],
            trained_groups, WARMUP_SECONDS, "Warm-up", in_session, rx_for)
        cooldowns = pick_support_exercises(
            candidates[candidates["exercise_category"] == "flexibility"],
            trained_groups, COOLDOWN_SECONDS, "Cooldown", in_session, rx_for)

        # --- write the day's rows: Warm-up -> Main -> Finisher -> Cooldown ---
        session_focus = session["label"]
        current_muscles = set()
        for row, rx in warmups:
            rows_out.append(build_plan_row(row, rx, day, session_focus, "Warm-up"))
            used.add(row["exercise_name"])
        for item in ordered_main:
            row, rx = item["row"], item["rx"]
            rows_out.append(build_plan_row(row, rx, day, session_focus, "Main"))
            used.add(row["exercise_name"])
            if str(row["exercise_category"]).lower() == "resistance":
                muscle = str(row["target_muscle"]).lower()
                current_muscles.add(muscle)
                muscle_usage[muscle] = muscle_usage.get(muscle, 0) + 1
        if finisher_row is not None:
            rows_out.append(build_plan_row(finisher_row, finisher_rx, day, session_focus, "Finisher"))
            used.add(finisher_row["exercise_name"])
        for row, rx in cooldowns:
            rows_out.append(build_plan_row(row, rx, day, session_focus, "Cooldown"))
            used.add(row["exercise_name"])
        previous_muscles = current_muscles

    return pd.DataFrame(rows_out)


def generate_plan():
    """Glue: translate the UI profile for the engine and run the whole pipeline."""
    p = st.session_state.profile
    f = st.session_state.feedback

    backend_profile = {
        "age": p.get("age"),
        "fitness_goal": p.get("goal"),
        "workout_location": p.get("location"),
        "available_equipment": p.get("equipment") or [],
        "preferred_days_per_week": p.get("days"),
        "session_duration_minutes": p.get("duration"),
        "self_rated_fitness_level": p.get("level"),
        # progression state and the permanent Joint-Pain blacklist
        "progression": st.session_state.progression,
        "blacklisted_exercises": st.session_state.blacklisted_exercises,
        "blacklisted_patterns": st.session_state.blacklisted_patterns,
    }

    safe_exercises = safety_filter(exercise_kb, backend_profile, f)
    ranked_exercises = rank_exercises(safe_exercises, backend_profile, f)
    weekly_plan = assemble_weekly_plan(ranked_exercises, backend_profile, f)

    return weekly_plan


# =============================================================================
# (feature 4) JOINT-PAIN BLACKLIST
# =============================================================================
def blacklist_exercise(name):
    """Permanently ban an exercise AND similar ones (same joint action + same muscle
    group). Returns how many OTHER exercises were banned along with it.
    For signed-in users the ban is saved with the rest of their data (see autosave)."""
    st.session_state.blacklisted_exercises.add(name.lower())
    match = exercise_kb[exercise_kb["exercise_name"].str.lower() == name.lower()]
    if match.empty:
        return 0
    action, group = match.iloc[0]["joint_action"], match.iloc[0]["major_muscle_group"]
    if action == "other":
        return 0
    st.session_state.blacklisted_patterns.add((action, group))
    similar = exercise_kb[(exercise_kb["joint_action"] == action) & (exercise_kb["major_muscle_group"] == group)]
    return max(0, len(similar) - 1)


def purge_blacklisted(plan_df, rated_days):
    """After a Joint Pain report, rebuild the days that have NOT been rated yet so they
    no longer contain the banned exercises (or similar ones). Days already rated are
    kept exactly as they were. Returns (new_plan, messages)."""
    fresh = generate_plan()    # safety_filter already excludes the blacklist
    if fresh.empty:
        return plan_df, ["Could not rebuild the remaining days without the banned exercises."]
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    kept = plan_df[plan_df["Day"].isin(rated_days)]
    rebuilt = fresh[~fresh["Day"].isin(rated_days)]
    combined = pd.concat([kept, rebuilt], ignore_index=True)
    combined["_order"] = combined["Day"].map(day_order.index)
    combined = combined.sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)
    return combined, ["Your remaining days were rebuilt without those exercises."] if not rebuilt.empty else []


# =============================================================================
# (feature 3) PROGRESSIVE OVERLOAD ENGINE
# =============================================================================
def apply_progress(prog, step, label, rep_span, bodyweight_span, can_unlock=True):
    """A 'Perfect' or 'Too Easy' week. Reps go up by `step`. If the reps were already at
    the top of the target, the user adds weight instead and the reps start again."""
    prog["bodyweight_reps"] = min(bodyweight_span, prog["bodyweight_reps"] + step)
    if prog["reps_bump"] >= rep_span:
        prog["reps_bump"] = 0
        prog["weight_stage"] += 1
        message = (f"{label} -> you reached the top of the rep target. Add a little weight "
                   "(the next size up) and start again from the lower reps.")
        if can_unlock and prog["difficulty_bump"] < MAX_DIFFICULTY_BUMP:
            prog["difficulty_bump"] += 1
            message += " Harder exercises are now unlocked."
        return message, True
    added = min(step, rep_span - prog["reps_bump"])
    prog["reps_bump"] += added
    return f"{label} -> +{added} rep{'s' if added > 1 else ''} per set, same weight.", False


def apply_too_hard(prog):
    """A 'Too Hard' week steps ONE change back so you can recover."""
    if prog["reps_bump"] > 0 or prog["bodyweight_reps"] > 0:
        prog["reps_bump"] = max(0, prog["reps_bump"] - 1)
        prog["bodyweight_reps"] = max(0, prog["bodyweight_reps"] - 1)
        return "Too Hard -> 1 rep less per set."
    if prog["difficulty_bump"] > 0:
        prog["difficulty_bump"] -= 1
        return "Too Hard -> went back to your original exercise difficulty."
    return "Too Hard -> you are at the starting reps; use a lighter weight this week."


def evaluate_week(prog, ratings, rep_span, bodyweight_span, can_unlock=True):
    """Look at this week's session ratings and update the progression state.
    Priority: any 'Too Hard' (safety) > any 'Too Easy' > all 'Perfect'.
    Returns (messages, weight_goes_up)."""
    if not ratings:
        return ["No sessions were rated this week, so nothing changed."], False
    if "Too Hard" in ratings:
        return [apply_too_hard(prog)], False
    if "Too Easy" in ratings:
        message, weight_up = apply_progress(prog, EASY_REP_STEP, "Too Easy", rep_span, bodyweight_span, can_unlock)
    else:
        message, weight_up = apply_progress(prog, PERFECT_REP_STEP, "Perfect", rep_span, bodyweight_span, can_unlock)
    return [message], weight_up


def finish_week():
    """Apply progression, move to the next week and build its plan."""
    week = st.session_state.week_number
    profile = st.session_state.profile
    prog = st.session_state.progression
    level_rank = LEVEL_RANK.get(str(profile.get("level", "Beginner")).lower(), 1)
    user_goal = goal_key(str(profile.get("goal", "General Fitness")).lower().replace(" ", "_"))
    low, top_weighted, top_bodyweight = rep_limits(user_goal, level_rank)
    can_unlock = level_rank + prog["difficulty_bump"] < 3      # is there a harder level left?
    messages, weight_up = evaluate_week(prog, list(st.session_state.week_feedback.values()),
                                        top_weighted - low, top_bodyweight - low, can_unlock)
    if weight_up:
        prog["weight_up_week"] = week + 1
    for message in messages:
        prog["history"].append(f"Week {week}: {message}")
    st.session_state.week_number += 1
    st.session_state.week_feedback = {}
    st.session_state.weight_drafts = {}      # weights typed for skipped (unrated) days are not counted
    st.session_state.weekly_plan = generate_plan()
    return messages


def save_weight(exercise, kg):
    """Remember the weight the user typed in, and log it once per week for the Progress page."""
    week = st.session_state.week_number
    st.session_state.weights[exercise] = kg
    log = [e for e in st.session_state.weight_log if not (e["week"] == week and e["exercise"] == exercise)]
    log.append({"week": week, "exercise": exercise, "kg": kg})
    st.session_state.weight_log = log


def record_tag(name, tag):
    """Count one negative-feedback tag for an exercise."""
    counts = st.session_state.feedback.setdefault(name, {})
    counts[tag] = counts.get(tag, 0) + 1


# =============================================================================
# PERSISTENT STORAGE (nickname + PIN login, Supabase)
# -----------------------------------------------------------------------------
# Needs two secrets (Streamlit Cloud -> app Settings -> Secrets, or locally in
# .streamlit/secrets.toml, which must NOT be pushed to GitHub):
#     SUPABASE_URL = "https://xxxx.supabase.co"
#     SUPABASE_KEY = "your secret / service_role key"
# Without them the app still works, in guest mode only.
# Tables: users, user_state, session_ratings, workout_feedback (SQL in supabase_setup.sql).
# =============================================================================
PIN_ITERATIONS = 200_000

# session_state keys that are saved for a logged-in user
PERSISTED_KEYS = ["profile", "feedback", "weekly_plan", "completed_workouts", "progression",
                  "week_number", "week_feedback", "session_log",
                  "blacklisted_exercises", "blacklisted_patterns", "weights", "weight_drafts", "weight_log"]


@st.cache_resource
def _supabase_client():
    from supabase import create_client          # raises if the package is missing
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])  # raises if secrets are missing


def get_supabase():
    """One shared Supabase client, or None if it is not configured / not installed.
    (Failures are not cached, so adding the secrets later works without a code change.)"""
    try:
        return _supabase_client()
    except Exception:
        return None


def storage_ready():
    return get_supabase() is not None


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_pin(pin, salt):
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), bytes.fromhex(salt), PIN_ITERATIONS).hex()


def db_create_user(nickname, pin):
    """Returns (user_id, error_message)."""
    sb = get_supabase()
    taken = sb.table("users").select("id").eq("nickname", nickname).execute().data
    if taken:
        return None, "That nickname is already taken. Please choose another one."
    salt = secrets.token_hex(16)
    try:
        rows = sb.table("users").insert({
            "nickname": nickname, "pin_salt": salt, "pin_hash": _hash_pin(pin, salt),
            "consent_at": _now(), "created_at": _now(),
        }).execute().data
    except Exception as error:
        if "23505" in str(error) or "duplicate" in str(error).lower():
            return None, "That nickname is already taken. Please choose another one."
        raise
    return rows[0]["id"], None


def db_check_login(nickname, pin):
    rows = (get_supabase().table("users").select("id, pin_salt, pin_hash")
            .eq("nickname", nickname).execute().data)
    if rows and secrets.compare_digest(_hash_pin(pin, rows[0]["pin_salt"]), rows[0]["pin_hash"]):
        return rows[0]["id"]
    return None


def db_save_state(user_id, state_json):
    sb = get_supabase()
    sb.table("user_state").upsert({
        "user_id": user_id, "state_json": json.loads(state_json), "updated_at": _now(),
    }, on_conflict="user_id").execute()
    ratings = [{"user_id": user_id, "week": e.get("week"), "day": e.get("day"),
                "difficulty": e.get("difficulty"), "rating": e.get("rating"),
                "notes": str(e.get("notes", "")), "saved_at": _now()}
               for e in st.session_state.session_log]
    if ratings:
        sb.table("session_ratings").upsert(ratings, on_conflict="user_id,week,day").execute()


def db_load_state(user_id):
    rows = get_supabase().table("user_state").select("state_json").eq("user_id", user_id).execute().data
    if not rows:
        return None
    data = rows[0]["state_json"]
    return json.loads(data) if isinstance(data, str) else data


def db_log_feedback(user_id, nickname, entry):
    """Append-only log of every feedback submission (one row per click), for analysis."""
    get_supabase().table("workout_feedback").insert({
        "user_id": user_id, "username": nickname, "week": entry["week"], "day": entry["day"],
        "rating": entry["rating"], "difficulty": entry["difficulty"], "notes": entry["notes"],
    }).execute()


def serialise_state():
    data = {}
    for key in PERSISTED_KEYS:
        value = st.session_state[key]
        if key == "weekly_plan":
            value = value.to_dict(orient="records")
        elif key == "blacklisted_exercises":
            value = sorted(value)
        elif key == "blacklisted_patterns":
            value = sorted([list(p) for p in value])
        data[key] = value
    return json.dumps(data, sort_keys=True, default=str)


def restore_state(data):
    """Put a saved blob back into session_state (missing keys keep their defaults)."""
    for key in PERSISTED_KEYS:
        if key not in data:
            continue
        value = data[key]
        if key == "weekly_plan":
            value = pd.DataFrame(value)
        elif key == "blacklisted_exercises":
            value = set(value)
        elif key == "blacklisted_patterns":
            value = {tuple(p) for p in value}
        elif key == "progression":
            # accounts saved by an older version: keep only the fields that still exist
            fresh = new_progression()
            value = {**fresh, **{k: v for k, v in value.items() if k in fresh}}
        st.session_state[key] = value


def reset_to_defaults():
    for key, value in _defaults.items():
        st.session_state[key] = value() if callable(value) else (value.copy() if hasattr(value, "copy") else value)
    st.session_state.progression = new_progression()
    st.session_state.profile = dict(default_profile)


def autosave():
    """Called at the end of every run: saves only when something changed."""
    user_id = st.session_state.get("user_id")
    if not user_id:
        return
    blob = serialise_state()
    digest = hashlib.sha256(blob.encode()).hexdigest()
    if digest != st.session_state.get("_saved_digest"):
        try:
            db_save_state(user_id, blob)
            st.session_state["_saved_digest"] = digest
        except Exception as error:
            st.warning(f"Could not save your progress right now ({type(error).__name__}). It will retry on your next click.")


def account_box():
    """Login / register / guest panel on the home page."""
    if st.session_state.get("user_id"):
        c1, c2 = st.columns([3, 1])
        c1.success(f"Signed in as **{st.session_state.nickname}**. Your progress is saved automatically.")
        if c2.button("Log out", use_container_width=True):
            autosave()
            reset_to_defaults()
            st.session_state.user_id = None
            st.session_state.nickname = None
            st.session_state["_saved_digest"] = None
            st.rerun()
        return

    if not storage_ready():
        st.caption("Accounts are not available right now (database not configured), so you are using the app as a guest.")
        return

    with st.expander("🔐 Sign in to save your progress (optional)", expanded=False):
        st.caption("Use a nickname and a 4–6 digit PIN, not your real name. Without signing in you can still use "
                   "the app as a guest, but your progress is lost when the page is refreshed.")
        tab_login, tab_register = st.tabs(["Sign in", "Create account"])
        with tab_login:
            nick = st.text_input("Nickname", key="login_nick").strip().lower()
            pin = st.text_input("PIN", type="password", key="login_pin")
            if st.button("Sign in", key="btn_login"):
                try:
                    user_id = db_check_login(nick, pin) if nick and pin else None
                except Exception as error:
                    st.error(f"Could not reach the database ({type(error).__name__}). Please try again.")
                    return
                if user_id is None:
                    st.error("Nickname or PIN is incorrect.")
                else:
                    reset_to_defaults()
                    try:
                        saved = db_load_state(user_id)
                    except Exception:
                        saved = None
                    if saved:
                        restore_state(saved)
                    st.session_state.user_id, st.session_state.nickname = user_id, nick
                    st.session_state["_saved_digest"] = None
                    st.session_state.flash = ("success", f"Welcome back, {nick}!" + (" Your saved plan has been loaded." if saved else ""))
                    st.rerun()
        with tab_register:
            nick = st.text_input("Choose a nickname (3–20 letters or numbers)", key="reg_nick").strip().lower()
            pin = st.text_input("Choose a PIN (4–6 digits)", type="password", key="reg_pin")
            consent = st.checkbox(
                "I agree that my nickname, profile answers, workout plan and session ratings are stored by this app "
                "for the purpose of this final year project. No real name or contact details are collected.",
                key="reg_consent")
            if st.button("Create account", key="btn_register"):
                if not re.fullmatch(r"[a-z0-9_]{3,20}", nick):
                    st.error("Nickname must be 3–20 letters, numbers or underscores.")
                elif not re.fullmatch(r"\d{4,6}", pin):
                    st.error("PIN must be 4–6 digits.")
                elif not consent:
                    st.error("Please tick the consent box to create an account.")
                else:
                    try:
                        user_id, error = db_create_user(nick, pin)
                    except Exception as exc:
                        user_id, error = None, f"Could not reach the database ({type(exc).__name__}). Please try again."
                    if error:
                        st.error(error)
                    else:
                        # keep whatever the guest already did, and save it under the new account
                        st.session_state.user_id, st.session_state.nickname = user_id, nick
                        st.session_state["_saved_digest"] = None
                        st.session_state.flash = ("success", f"Account created. Signed in as {nick}.")
                        st.rerun()


# =============================================================================
# 3. UI PAGES
# =============================================================================
def page_home():
    st.markdown("<h1 style='text-align: center;'>FitRecommend</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Personalized Workout Recommendation System</h3>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<p style='text-align: center;'>Get a workout plan designed for your goals, fitness level and equipment.</p>", unsafe_allow_html=True)

    if not st.session_state.get("user_id"):   # guests: sign-in panel at the top
        account_box()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Create / Edit My Plan", use_container_width=True):
            navigate_to('profile')
            st.rerun()
        if not st.session_state.weekly_plan.empty:
            if st.button("View Current Plan", use_container_width=True):
                navigate_to('plan')
                st.rerun()
        if st.button("My Progress", use_container_width=True):
            navigate_to('progress')
            st.rerun()
        if st.button("Training Guide", use_container_width=True):
            navigate_to('learn')
            st.rerun()

    if st.session_state.get("user_id"):       # signed in: "Signed in as ..." + Log out at the bottom
        st.write("")
        st.divider()
        show_flash()                           # "Account created..." / "Welcome back..." message
        account_box()


def page_profile():
    p = st.session_state.profile
    st.title("Create Your Profile")
    st.write("These inputs feed directly into the recommendation algorithm.")

    age = st.number_input("Age", min_value=MIN_AGE, max_value=99, value=max(MIN_AGE, int(p['age'])))
    st.caption(f"You must be {MIN_AGE} or older. For users aged {SENIOR_AGE}+ the plan avoids jump and "
               "plyometric exercises and adds a little rest.")

    levels = ["Beginner", "Intermediate", "Advanced"]
    level = st.radio("Training Experience / Fitness Level", levels, index=levels.index(p['level']),
                     format_func=lambda x: EXPERIENCE_LABELS[x],
                     help="Beginner: 0–6 months. Intermediate: 6–24 months. Advanced: 2+ years.")

    goals = ["Muscle Gain", "Weight Loss", "General Fitness", "Strength"]
    goal = st.selectbox("Main Goal", goals, index=goals.index(p['goal']))

    locations = ["Gym", "Home", "Outdoor"]
    location = st.radio("Workout Location", locations, index=locations.index(p['location']))

    # options come straight from the knowledge base, so every choice matches real exercises
    equipment_options = equipment_choices(exercise_kb)
    saved = p.get('equipment')
    if "equip_ms" not in st.session_state:          # first time the form is shown
        st.session_state.equip_ms = (list(equipment_options) if saved is None and location == "Gym"
                                     else [e for e in (saved or []) if e in equipment_options])
    elif st.session_state.get("_loc_sig") not in (None, location) and location == "Gym":
        st.session_state.equip_ms = list(equipment_options)   # switched to Gym -> tick everything
    st.session_state["_loc_sig"] = location
    equipment = st.multiselect(
        "Available Equipment",
        equipment_options,
        key="equip_ms",
        help="Bodyweight exercises are always included. Only equipment used by at least one exercise is listed."
    )
    if location == "Gym":
        st.caption("At the gym all equipment is ticked. Untick anything your gym does not have.")

    # --- days per week: recommendation pops up once level + goal are chosen ---
    d_lo, d_hi = days_recommendation(level, goal)
    days_text = _fmt_range(d_lo, d_hi, "days per week") + (" of resistance training" if rec_group(goal) == "weight_loss" else "")
    if st.session_state.get("_days_sig") != (level, goal):
        st.session_state["_days_sig"] = (level, goal)
        st.toast(f"Recommended for {level} + {goal}: {days_text}", icon="📅")
    days = st.slider("Days per Week", min_value=2, max_value=6, value=int(np.clip(p['days'], 2, 6)))
    st.info(f"📅 **Recommended for {level} + {goal}:** {days_text}.")
    if not d_lo <= days <= d_hi:
        st.caption(f"You chose {days} days, which is outside the recommended range. The plan will still be built for {days} days.")

    # --- session duration: recommendation pops up once level, goal and days are chosen ---
    m_lo, m_hi = duration_recommendation(level, goal)
    if st.session_state.get("_dur_sig") != (level, goal, days):
        st.session_state["_dur_sig"] = (level, goal, days)
        st.toast(f"Recommended session length for {level} + {goal}: {m_lo}–{m_hi} min", icon="⏱️")
    duration_options = [30, 45, 60, 75, 90]
    dur_idx = duration_options.index(p['duration']) if p['duration'] in duration_options else 2
    duration = st.select_slider("Session Duration (minutes)", options=duration_options, value=duration_options[dur_idx])
    st.info(f"⏱️ **Recommended session length for {level} + {goal}:** {m_lo}–{m_hi} minutes "
            f"({days} days per week).")
    if rec_group(goal) == "weight_loss":
        lv = level.lower()
        r_lo, r_hi = WL_RESISTANCE_MIN[lv]
        c_lo, c_hi = WL_CARDIO_MIN[lv]
        st.caption(f"Weight loss is planned as two blocks: resistance training {r_lo}–{r_hi} min plus cardio "
                   f"{c_lo}–{c_hi} min (plus a 5 min warm-up and cooldown). Short sessions get a shorter cardio block.")
    if not m_lo <= duration <= m_hi:
        st.caption("These are proposed starting ranges, not proven optimal values. The actual time depends on the "
                   "number of exercises, sets, rest periods and your own capacity.")

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("⬅ Back"):
            navigate_to('home')
            st.rerun()
    with col2:
        if st.button("Generate My Plan ⚡"):
            st.session_state.profile.update({
                "age": age, "level": level, "goal": goal, "location": location,
                "equipment": equipment, "days": days, "duration": duration
            })
            navigate_to('loading')
            st.rerun()


def page_loading():
    st.title("Creating Your Personalized Plan...")
    progress_text = st.empty()
    progress_bar = st.progress(0)

    steps = [
        "Checking your fitness level and progression...",
        "Filtering unsuitable or blacklisted exercises via safety rules...",
        "Matching exercises to your goal using content-scoring...",
        "Filling your time window with compound-first sessions, warm-ups and cooldowns...",
    ]

    for i, step in enumerate(steps):
        progress_text.write(f"✓ {step}")
        progress_bar.progress((i + 1) * 25)
        time.sleep(0.6)

    st.session_state.weekly_plan = generate_plan()
    time.sleep(0.5)
    navigate_to('plan')
    st.rerun()


def page_plan():
    st.title("Your Personalized Plan")
    p = st.session_state.profile
    prog = st.session_state.progression
    st.caption(f"**Week {st.session_state.week_number}** | **Goal:** {p.get('goal')} | **Level:** {p.get('level')} | **{p.get('days')} Days / Week**")
    show_flash()
    if prog["weight_up_week"] == st.session_state.week_number:
        st.success("🏋️ **Time to add weight.** You reached the top of your rep target last week. "
                   "For exercises with weights, pick the next size up (about 1-2.5 kg more) "
                   "and type the new weight in the table. Your reps start again from the lower number.")
    st.info("**Choosing your weight:** pick a weight you can lift comfortably for every rep with good form. "
            "The last 2 reps should feel hard but still controlled. If you cannot finish the reps, go lighter. "
            "Type the weight you use in the **Weight (kg)** column. It only counts towards your progress "
            "once you submit the feedback for that day. "
            "Sets stay the same every week: you progress by adding reps, then weight.")

    if st.button("⌂ Back to Home"):
        navigate_to('home')
        st.rerun()

    plan_df = st.session_state.weekly_plan

    if plan_df.empty:
        st.error("No exercises matched your criteria. Please adjust your equipment, location, or fitness level.")
        return

    session_minutes = int(p.get("duration", 60))

    # A week normally ends when every day is rated. If the user skipped a workout,
    # they can end the week here so progression does not get stuck.
    n_rated, n_days = len(st.session_state.week_feedback), plan_df["Day"].nunique()
    if 0 < n_rated < n_days:
        with st.container(border=True):
            st.write(f"You have rated {n_rated} of {n_days} workouts this week.")
            if st.button("Finish this week now (skip the unrated workouts)"):
                week = st.session_state.week_number
                messages = finish_week()
                st.session_state.flash = ("success", f"Week {week} complete. " + " ".join(messages)
                                          + f" Week {week + 1}'s plan is ready.")
                navigate_to('progress')
                st.rerun()

    for day in plan_df['Day'].unique():
        day_data = plan_df[plan_df['Day'] == day]
        rated = day in st.session_state.week_feedback
        st.subheader(f"{day} - {day_data['Focus'].iloc[0]}")
        total_minutes = day_data["Seconds"].sum() / 60
        st.caption(f"Estimated time: ~{total_minutes:.0f} min of your {session_minutes} min window"
                   + (" ✅ feedback submitted" if rated else ""))
        if total_minutes < 0.8 * session_minutes:
            st.caption(f"⚠️ This day is shorter than your chosen time: there are not enough "
                       f"{day_data['Focus'].iloc[0].split(' + ')[0]} exercises for your equipment and location. "
                       "Adding equipment in your profile gives more options.")

        # One table per day, ordered Warm-up -> Main -> Finisher -> Cooldown.
        ordered = day_data.assign(_s=day_data["Section"].map(SECTION_ORDER.index)).sort_values("_s", kind="stable")
        table = ordered[["Section", "Exercise", "Target", "Type", "Movement", "Equipment", "Sets", "Reps", "Rest"]].reset_index(drop=True)
        # Only main exercises that use weights get a weight. Others show "—".
        uses_weight = (table["Section"] == "Main") & table["Equipment"].map(is_weighted)
        # Weighted exercises get an empty box to type in (or the saved weight); the others show "—".
        # Before the day's feedback is submitted, typed weights are only drafts for THAT day.
        drafts = st.session_state.weight_drafts.setdefault(day, {})
        def weight_text(name, ok):
            if not ok:
                return "—"
            kg = drafts.get(name, st.session_state.weights.get(name)) if not rated else st.session_state.weights.get(name)
            return "" if kg is None else f"{kg:g}"
        table["Weight (kg)"] = [weight_text(name, ok) for name, ok in zip(table["Exercise"], uses_weight)]
        # the weight sits next to the sets and reps so it is visible without scrolling sideways
        table = table[["Section", "Exercise", "Sets", "Reps", "Weight (kg)", "Rest",
                       "Equipment", "Target", "Type", "Movement"]]
        edited = st.data_editor(
            table, hide_index=True, use_container_width=True,
            height=35 * (len(table) + 1) + 3,          # show every row, no scrolling inside the table
            key=f"plan_{st.session_state.week_number}_{day}_{st.session_state.get('_editor_reset', 0)}",
            disabled=[c for c in table.columns if c != "Weight (kg)"],
            column_config={"Section": st.column_config.TextColumn(width="small"),
                           "Sets": st.column_config.TextColumn(width="small"),
                           "Weight (kg)": st.column_config.TextColumn(
                "Weight (kg)", max_chars=6, validate=r"^(\d{1,3}([.,]\d{1,2})?)?$",
                help="Type the weight you use, for example 12.5. Only for exercises with weights; "
                     "rows marked — (bodyweight, band and timed exercises) do not need one.")},
        )
        for name, ok, text in zip(edited["Exercise"], uses_weight, edited["Weight (kg)"]):
            text = str(text or "").replace(",", ".").strip()
            if not ok:
                continue
            valid = bool(re.fullmatch(r"\d{1,3}(\.\d{1,2})?", text)) and float(text or 0) > 0
            if rated:       # feedback already submitted: a correction counts straight away
                if valid and st.session_state.weights.get(name) != float(text):
                    save_weight(name, float(text))
            elif valid and drafts.get(name, st.session_state.weights.get(name)) != float(text):
                drafts[name] = float(text)       # draft only; counted when this day's feedback is submitted
            elif text == "" and name in drafts:
                del drafts[name]
        if not rated:
            st.caption("Weights you type here count towards your progress once you submit this day's feedback.")
        # typed into a row that has no weight (shown as —): put the — back
        if any((not ok) and str(text) != "—" for ok, text in zip(uses_weight, edited["Weight (kg)"])):
            st.session_state["_editor_reset"] = st.session_state.get("_editor_reset", 0) + 1
            st.toast("That exercise does not use weights, so there is nothing to type there.")
            st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            selected = st.selectbox("View instructions for:", list(ordered["Exercise"]), key=f"sel_{day}")
            if st.button("View Exercise Details", key=f"btn_{day}"):
                st.session_state.selected_exercise = ordered[ordered["Exercise"] == selected].iloc[0]
                navigate_to('exercise_detail')
                st.rerun()
        with col2:
            st.write("")
            st.write("")
            # Greyed out (not clickable) once this day's feedback has been submitted.
            if st.button("Complete Workout & Give Feedback", key=f"comp_{day}", disabled=rated):
                st.session_state.feedback_day = day
                navigate_to('feedback')
                st.rerun()
        st.divider()


PHOTO_FRAME_CSS = """
<style>
.ex-photos {
    display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
    padding: 14px; margin: 6px 0 4px 0;
    background: linear-gradient(180deg, rgba(0,255,170,0.07), rgba(0,255,170,0.02));
    border: 1px solid rgba(0,255,170,0.35); border-radius: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}
.ex-photos figure { margin: 0; display: flex; flex-direction: column; gap: 8px; }
.ex-photos .ex-photo {
    background: #0b0e0c; border-radius: 12px; overflow: hidden;
    border: 1px solid rgba(255,255,255,0.15); aspect-ratio: 3 / 2;
    display: flex; align-items: center; justify-content: center;
}
.ex-photos img { width: 100%; height: 100%; object-fit: cover; display: block; }
.ex-photos figcaption {
    text-align: center; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.04em;
    color: #00FFAA; text-transform: uppercase;
}
.ex-photos figcaption span { color: #9aa5a0; font-weight: 500; margin-right: 6px; }
@media (max-width: 480px) { .ex-photos { gap: 8px; padding: 8px; } .ex-photos figcaption { font-size: 0.72rem; } }
</style>
"""


def show_exercise_photos(image_1, image_2):
    """Two photos side by side (start and end position) inside a rounded frame."""
    from html import escape
    st.markdown(PHOTO_FRAME_CSS + f"""
<div class="ex-photos">
  <figure><div class="ex-photo"><img src="{escape(image_1)}" alt="Start position" loading="lazy"></div>
          <figcaption><span>1</span>Start position</figcaption></figure>
  <figure><div class="ex-photo"><img src="{escape(image_2)}" alt="End position" loading="lazy"></div>
          <figcaption><span>2</span>End position</figcaption></figure>
</div>""", unsafe_allow_html=True)


def page_exercise_detail():
    ex = st.session_state.selected_exercise
    if st.button("⬅ Back to Plan"):
        navigate_to('plan')
        st.rerun()

    st.title(ex["Exercise"])

    col1, col2 = st.columns(2)
    col1.metric("Target Muscle", ex["Target"])
    col1.metric("Equipment", ex["Equipment"])
    col1.metric("Type", ex["Type"])
    col2.metric("Difficulty", ex["Difficulty"])
    col2.metric("Prescription", f"{ex['Sets']} Sets × {ex['Reps']}")
    col2.metric("Rest", ex.get("Rest", "Not specified"))

    # two photos (start / end position) from the knowledge base, shown in a frame
    key = str(ex["Exercise"]).strip().lower()
    match = exercise_kb[exercise_kb["exercise_name"].astype(str).str.strip().str.lower() == key]
    kb_row = match.iloc[0] if not match.empty else None
    if kb_row is not None and kb_row["image_1"] and kb_row["image_2"]:
        show_exercise_photos(kb_row["image_1"], kb_row["image_2"])
        source_name = str(kb_row.get("photo_source", "") or "")
        credit = "Photos: Free Exercise DB (public domain)"
        if source_name and source_name.lower() != key:
            credit += f", entry “{source_name}”"
        st.caption(credit + ".")
    else:
        st.info("Photos are not available for this exercise. If you created your plan before the "
                "latest update, go back and generate a new plan.")

    st.subheader("How to perform this exercise:")
    st.info(ex["Instructions"])

    st.subheader("Why we recommended this:")
    reasons = f"""
    ✓ Matches your **{st.session_state.profile.get('goal')}** goal  
    ✓ Available at your selected location (**{st.session_state.profile.get('location')}**)  
    ✓ Suitable for your **{st.session_state.profile.get('level')}** fitness level  
    ✓ Uses equipment you have (**{ex['Equipment']}**)
    """
    if ex["Type"] == "Compound":
        reasons += "\n    ✓ Compound lift, placed early in your session while you are fresh"
    if ex["Movement"] == "Unilateral":
        reasons += "\n    ✓ One-sided movement, which helps balance strength between your left and right"
    if kb_row is not None and str(kb_row.get("rating", "")).strip() not in ("", "nan"):
        reasons += f"\n    ✓ Highly rated by other users ({float(kb_row['rating']):.1f} / 10 on megaGym)"
    st.success(reasons)
    if "Score" in ex and str(ex["Score"]) not in ("", "nan"):
        st.caption(f"Match score: {float(ex['Score']):.2f}. It is higher when the exercise is at your level, "
                   "uses equipment you own, is highly rated and suits your goal.")


def page_feedback():
    if st.button("⬅ Back to Plan"):
        navigate_to('plan')
        st.rerun()

    day = st.session_state.feedback_day
    plan_df = st.session_state.weekly_plan
    if day is None or plan_df.empty or day not in plan_df["Day"].values:
        st.info("Pick a workout day on the plan page to leave feedback.")
        return

    st.title("Workout Complete! 🎉")
    st.write(f"Feedback for **{day}**. It improves future recommendations.")

    rating = st.slider("Overall Rating (1 = Poor, 5 = Excellent)", 1, 5, 4)
    # three options map directly onto the overload engine.
    difficulty = st.select_slider("How difficult was today's workout?",
                                  options=["Too Easy", "Perfect", "Too Hard"], value="Perfect")

    # (feature 4) granular, per-exercise tags.
    exercises_today = list(dict.fromkeys(plan_df[plan_df["Day"] == day]["Exercise"]))
    st.markdown("**Anything specific? (optional)**")
    too_heavy = st.multiselect(f"{TAG_TOO_HEAVY} (we'll rank these lower)", exercises_today)
    exhausting = st.multiselect(f"{TAG_EXHAUSTING} (more rest / shorter cardio next time)", exercises_today)
    joint_pain = st.multiselect(f"{TAG_JOINT_PAIN} (removed for good, plus similar exercises)", exercises_today)

    notes = st.text_area("Anything else?")

    if st.button("Submit Feedback"):
        week = st.session_state.week_number
        first_time = day not in st.session_state.week_feedback
        n_training_days = plan_df["Day"].nunique()
        st.session_state.week_feedback[day] = difficulty
        notes_out = []

        if first_time:   # don't double-count if the same day is submitted twice
            for name in too_heavy:
                record_tag(name, TAG_TOO_HEAVY)
            for name in exhausting:
                record_tag(name, TAG_EXHAUSTING)
            st.session_state.completed_workouts += 1
        st.session_state.session_log = [
            e for e in st.session_state.session_log if not (e["week"] == week and e["day"] == day)
        ] + [{"week": week, "day": day, "difficulty": difficulty, "rating": rating, "notes": notes}]
        if st.session_state.get("user_id"):          # only signed-in (consenting) users are logged
            try:
                db_log_feedback(st.session_state.user_id, st.session_state.nickname, st.session_state.session_log[-1])
            except Exception:
                pass                                 # the full state is still saved by autosave()

        # Count this day's weights now that its feedback is submitted: the typed weight,
        # or the last saved weight if the user left the box as it was.
        typed = st.session_state.weight_drafts.pop(day, {})
        main_today = plan_df[(plan_df["Day"] == day) & (plan_df["Section"] == "Main")]
        for name, equipment in zip(main_today["Exercise"], main_today["Equipment"]):
            kg = typed.get(name, st.session_state.weights.get(name))
            if kg and is_weighted(equipment):
                save_weight(name, kg)

        # Joint pain: ban now, and clean up the rest of this week's plan.
        for name in joint_pain:
            n_similar = blacklist_exercise(name)
            notes_out.append(f"{name} was blacklisted" + (f" along with {n_similar} similar exercise(s)." if n_similar else "."))
        if joint_pain:
            new_plan, messages = purge_blacklisted(plan_df, set(st.session_state.week_feedback))
            st.session_state.weekly_plan = new_plan
            notes_out.extend(messages)

        # Every training day rated -> the week is over: apply progression automatically.
        if len(st.session_state.week_feedback) >= n_training_days:
            messages = finish_week()
            notes_out.append(f"Week {week} complete. " + " ".join(messages) + f" Week {week + 1}'s plan is ready.")

        st.session_state.flash = ("success", "Feedback saved. " + " ".join(notes_out))
        navigate_to('progress')
        st.rerun()


def page_progress():
    if st.button("⬅ Back to Home"):
        navigate_to('home')
        st.rerun()

    st.title("My Progress")
    show_flash()

    # a real average instead of a hard-coded number.
    scale = {"Too Easy": 1, "Perfect": 3, "Too Hard": 5}
    log = st.session_state.session_log
    average = f"{np.mean([scale[e['difficulty']] for e in log]):.1f} / 5" if log else "—"

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Workouts Completed", st.session_state.completed_workouts)
        col2.metric("Average Difficulty", average)
        col3.metric("Current Week", st.session_state.week_number)

    prog = st.session_state.progression
    profile = st.session_state.profile
    level_rank = LEVEL_RANK.get(str(profile.get("level", "Beginner")).lower(), 1)
    user_goal = goal_key(str(profile.get("goal", "General Fitness")).lower().replace(" ", "_"))
    sets, _, _, _ = base_prescription(user_goal, level_rank)
    low, top_weighted, top_bodyweight = rep_limits(user_goal, level_rank)
    reps_now = min(low + prog["reps_bump"], top_weighted)
    bodyweight_now = min(low + prog["bodyweight_reps"], top_bodyweight)

    st.subheader("How your plan progresses")
    st.caption("Your sets stay the same every week. You get stronger by adding reps first, then weight.")
    st.markdown(f"""
    * **Sets:** {sets} per exercise (fixed)
    * **Reps with weights:** {reps_now} per set now (starts at {low}, goes up to {top_weighted}, then you add weight and start again at {low})
    * **Reps without weights:** {bodyweight_now} per set now (starts at {low}, can go up to {top_bodyweight})
    * **Times you were asked to add weight:** {prog['weight_stage']}
    * **Harder exercises:** {"unlocked - you may now see exercises one level up" if prog['difficulty_bump'] else "not yet - you only get exercises at your level"}
    """)
    with st.expander("How does this work?"):
        st.markdown(f"""
        * After each workout you rate how hard it was. At the end of the week your plan changes:
          **Perfect** adds {PERFECT_REP_STEP} rep per set, **Too Easy** adds {EASY_REP_STEP} reps per set,
          **Too Hard** takes 1 rep away.
        * When you reach the top of the rep target ({top_weighted} reps) and the week still feels fine,
          it is time to **add weight**: pick the next size up and go back to {low} reps.
        * Exercises without weights (push-ups, planks, bands) cannot get heavier, so they simply keep adding reps.
        * **Harder exercises** are tougher versions, one level above the level you picked.
          They unlock the first time you are asked to add weight.
        * Always choose a weight you can lift comfortably with good form for every rep.
        """)

    st.subheader("My Weights")
    log = pd.DataFrame(st.session_state.weight_log)
    if log.empty:
        st.write("Type the weights you use in the **Weight (kg)** column of your plan to follow them here.")
    else:
        log = log.sort_values("week")
        summary = log.groupby("exercise").agg(first_week=("week", "first"), start=("kg", "first"),
                                              now=("kg", "last"), best=("kg", "max")).reset_index()
        summary["change"] = summary["now"] - summary["start"]
        improved = int((summary["change"] > 0).sum())
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Exercises tracked", len(summary))
            c2.metric("Exercises with more weight", improved)
            c3.metric("Total weight added", f"{summary['change'].clip(lower=0).sum():.1f} kg")
        st.dataframe(
            summary.rename(columns={"exercise": "Exercise", "first_week": "Since week", "start": "Start (kg)",
                                    "now": "Now (kg)", "best": "Best (kg)", "change": "Change (kg)"}),
            hide_index=True, use_container_width=True,
            column_config={c: st.column_config.NumberColumn(format="%.1f")
                           for c in ["Start (kg)", "Now (kg)", "Best (kg)", "Change (kg)"]})
        choice = st.selectbox("Show weight by week for:", sorted(summary["exercise"]))
        history = log[log["exercise"] == choice].set_index("week")["kg"]
        if len(history) > 1:
            st.line_chart(history, y_label="kg", x_label="Week")
        else:
            st.caption(f"{choice}: {history.iloc[0]:.1f} kg in week {history.index[0]}. "
                       "The chart appears when you have typed a weight in two different weeks.")

    st.subheader("Progression History")
    if prog["history"]:
        for entry in reversed(prog["history"]):
            st.markdown(f"* {entry}")
    else:
        st.write("Complete a week of sessions to see your plan adapt.")

    if st.session_state.blacklisted_exercises:
        st.subheader("Excluded for Joint Pain")
        st.write(", ".join(sorted(x.title() for x in st.session_state.blacklisted_exercises)))

def page_learn():
    if st.button("⌂ Back to Home"):
        navigate_to('home')
        st.rerun()

    st.title("Training Guide")
    st.write("Understand the terms used in your workout plan and how they support safe, effective training.")

    glossary = [
        ("Workout Basics", "Sets", "A set is a group of repetitions performed before resting.", "3 sets of 10 means you perform 10 repetitions, rest, and repeat twice."),
        ("Workout Basics", "Reps", "Repetitions are the number of times you complete a movement in one set.", "8–12 reps means aim for at least 8 controlled repetitions and no more than 12."),
        ("Workout Basics", "Rest", "Rest is the recovery time between sets so you can maintain technique and effort.", "Heavy strength sets usually need more rest than lighter accessory exercises."),
        ("Movement Terms", "Compound Exercise", "A compound exercise uses more than one joint and trains several muscle groups together.", "A squat trains the legs and glutes while your core stabilizes your body."),
        ("Movement Terms", "Isolation Exercise", "An isolation exercise focuses more directly on one muscle group or joint action.", "A bicep curl mainly trains elbow flexion and the biceps."),
        ("Movement Terms", "Unilateral", "A unilateral exercise works one side of the body at a time.", "A reverse lunge or single-arm dumbbell row can help reveal left-to-right differences."),
        ("Movement Terms", "Bilateral", "A bilateral exercise uses both sides of the body together.", "A barbell squat, bench press, or standard push-up is bilateral."),
        ("Plan Structure", "Warm-up", "A warm-up raises body temperature and prepares the joints and muscles for the main workout.", "Use light movement and rehearsal sets before demanding lifts."),
        ("Plan Structure", "Finisher", "A finisher is a short final block, often cardio or flexibility work, completed after resistance training.", "Brisk walking or a chest stretch can appear at the end of a session."),
        ("Progression", "Progressive Overload", "Progressive overload is a gradual increase in training challenge as your body adapts.", "In this app your sets stay the same. You add a repetition each week, and when you reach the top of the rep target you add a little weight and start again from the lower reps."),
        ("Progression", "Choosing Your Weight", "Pick a weight you can lift comfortably for every repetition with good form. The last two repetitions should feel hard but still controlled.", "If you cannot finish the reps, go lighter. If all reps feel very easy, rate the workout Too Easy. Type the weight you use in the Weight (kg) column of your plan."),
        ("Progression", "Recovery", "Recovery is the time your body needs to repair and adapt between training sessions.", "Sleep, nutrition, rest days, and manageable training volume all support recovery."),
    ]

    workout_tips = [
        ("Build consistency before complexity", "The most productive program is one you can repeat week after week. Start with a schedule that fits your life, learn the core movements, and avoid changing exercises or methods every session. Complex methods are optional; regular training and steady effort matter more for most healthy adults.", "ACSM resistance-training guidelines", "https://acsm.org/resistance-training-guidelines-update-2026/"),
        ("Train every major muscle group", "A balanced program should include the legs, hips, back, chest, abdomen, shoulders, and arms. Covering these areas supports general strength and reduces the chance that your plan is dominated by only the muscles you enjoy training. The app uses this principle when building its weekly split.", "U.S. Physical Activity Guidelines", "https://www.health.gov/PAguidelines/pdf/paguide.pdf"),
        ("Strength train at least twice each week", "For general health, muscle-strengthening work for all major muscle groups should be included on at least two days each week. More days can work well when volume is distributed sensibly, but beginning with a repeatable two-day foundation is better than an unsustainable ambitious plan.", "U.S. Physical Activity Guidelines", "https://www.health.gov/PAguidelines/pdf/paguide.pdf"),
        ("Progress gradually", "Increase only one or two training demands at a time: for example, one extra repetition or a small amount of weight. This app keeps your sets the same and raises the reps first, then the weight. Starting too hard and raising volume, intensity, and frequency together makes it harder to recover and makes it difficult to identify what caused pain or excessive fatigue.", "U.S. Physical Activity Guidelines", "https://www.health.gov/PAguidelines/pdf/paguide.pdf"),
        ("Match training to the goal", "Strength development generally benefits from heavier loading and lower repetition ranges, while muscle growth is supported by sufficient weekly training volume. Use the plan as a starting point, then select loads that let you complete the prescribed repetitions with controlled technique.", "ACSM resistance-training guidelines", "https://acsm.org/resistance-training-guidelines-update-2026/"),
        ("Accumulate useful weekly volume", "Muscle growth is influenced by the amount of challenging work a muscle receives over the week, not by one exhausting workout alone. Spread your work across sessions so that quality repetitions remain high and each muscle can recover before it is trained hard again.", "ACSM resistance-training guidelines", "https://acsm.org/resistance-training-guidelines-update-2026/"),
        ("Use controlled repetitions", "Do not rush the lowering phase or let momentum replace muscular control. A controlled repetition makes it easier to maintain positions, notice form breakdown, and use a load that matches your current ability. If control disappears, reduce the load, range, or repetition target.", "ACSM injury-prevention guidance", "https://www.acsm.org/docs/default-source/files-for-resource-library/smb-resistance-training-and-injury-prevention.pdf"),
        ("Warm up for the session you are about to do", "Use a short period of easy movement and lighter rehearsal sets before your demanding work. This gives you time to check how your body feels, practise technique with low risk, and prepare for the positions and range of motion required by the session.", "CDC guidance on exercise pain", "https://archive.cdc.gov/www_cdc_gov/arthritis/basics/physical-activity/pain.html"),
        ("Treat joint pain differently from normal effort", "Muscular effort and temporary fatigue can be expected, but joint pain, increasing swelling, altered movement, or a worsening symptom is a reason to stop or modify the activity. Do not repeatedly train through pain; choose a pain-free alternative and seek a qualified health professional when symptoms persist or are severe.", "CDC joint-friendly activity guidance", "https://www.cdc.gov/arthritis/prevention/"),
        ("Use a pain-free resistance level", "Choose a weight, band, or bodyweight variation that allows stable, controlled repetitions without joint pain. A lighter variation is not a failure: it is an appropriate starting point that can be progressed as confidence, technique, and tolerance improve.", "CDC joint-friendly activity guidance", "https://www.cdc.gov/arthritis/prevention/"),
        ("Keep cardio and flexibility as finishers", "When strength or muscle gain is the main goal, perform the technical resistance exercises while you are freshest. Cardio or flexibility can still support health and recovery, but placing them after the main work helps preserve attention and energy for the lifts that require the most control.", "ACSM resistance-training guidelines", "https://acsm.org/resistance-training-guidelines-update-2026/"),
        ("Use compound movements as the foundation", "Squats, presses, rows, hinges, lunges, and pull variations involve several joints and muscle groups. They provide an efficient base for a time-limited workout; smaller isolation exercises can then address muscles that need extra attention.", "ACSM resistance-training guidelines", "https://acsm.org/resistance-training-guidelines-update-2026/"),
        ("Include both sides of the body", "Bilateral lifts can build overall strength efficiently, while unilateral exercises can help you compare how each side controls the movement. Keep the exercise choice appropriate to your level, and use the same controlled volume on both sides unless a qualified clinician gives different instructions.", "ACSM injury-prevention guidance", "https://www.acsm.org/docs/default-source/files-for-resource-library/smb-resistance-training-and-injury-prevention.pdf"),
        ("Give recovery the same priority as training", "Adaptation happens between workouts. Poor sleep, persistent fatigue, performance decline, or increasing soreness are signals to review training stress and recovery habits. A lighter session, an extra rest day, or a reduced load can be a sensible training decision rather than lost progress.", "CDC injury-prevention recommendations", "https://www.cdc.gov/Mmwr/preview/mmwrhtml/rr4902a3.htm"),
        ("Choose equipment you can use safely", "Barbells, machines, bands, dumbbells, and bodyweight can all be effective. Select the option that fits your location, confidence, and available instruction, then progress the challenge over time instead of assuming expensive or advanced equipment is required.", "ACSM resistance-training guidelines", "https://acsm.org/resistance-training-guidelines-update-2026/"),
    ]

    search = st.text_input("Search a term", placeholder="Try unilateral, sets, or progressive overload").strip().lower()
    categories = ["Workout Basics", "Movement Terms", "Plan Structure", "Progression", "Workout Tips"]
    tabs = st.tabs(categories)

    matches = 0
    for category, tab in zip(categories, tabs):
        with tab:
            if category == "Workout Tips":
                entries = [tip for tip in workout_tips if not search or search in " ".join(tip[:-1]).lower()]
                if not entries:
                    st.caption("No matching tips in this section.")
                    continue
                for title, explanation, source_name, source_url in entries:
                    matches += 1
                    with st.expander(title):
                        st.write(explanation)
                        st.markdown(f"Source: [{source_name}]({source_url})")
                continue
            entries = [item for item in glossary if item[0] == category and (not search or search in " ".join(item[1:]).lower())]
            if not entries:
                st.caption("No matching terms in this section.")
                continue
            for _, term, definition, example in entries:
                matches += 1
                with st.expander(term):
                    st.write(definition)
                    st.caption(f"Example: {example}")

    if search and not matches:
        st.info("No matching term found. Try a broader word such as 'movement', 'rest', or 'recovery'.")

    st.warning("This guide is educational and does not replace advice from a qualified health or fitness professional. Stop an exercise if you feel pain.")


# --- Router ---
pages = {
    'home': page_home,
    'profile': page_profile,
    'loading': page_loading,
    'plan': page_plan,
    'exercise_detail': page_exercise_detail,
    'feedback': page_feedback,
    'progress': page_progress,
    'learn': page_learn
}

pages.get(st.session_state.page, page_home)()
autosave()
