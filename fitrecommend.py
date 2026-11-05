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

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# TUNABLE CONSTANTS (NEW) - change these numbers to tweak the algorithm
# =============================================================================
SECONDS_PER_REP = 4                 # estimated time to perform one repetition
SETUP_SECONDS_PER_EXERCISE = 30     # walking to the rack, loading weights, etc. (0 = pure formula)
WARMUP_SECONDS = 300                # 5-minute dynamic warm-up
COOLDOWN_SECONDS = 300              # 5-minute static cooldown
COUNT_WARMUP_COOLDOWN_IN_DURATION = True   # True: warm-up + cooldown come out of the chosen session length
MAX_RESISTANCE_EXERCISES = 8        # safety cap so very long sessions don't explode
MIN_UNILATERAL_PER_SESSION = 1      # at least this many one-limb exercises per session

# Progressive-overload limits (see evaluate_week below)
MAX_EXTRA_SETS = 2                  # never add more than +2 sets
MAX_REST_REDUCTION = 30             # never cut rest by more than 30 s in total
MIN_REST_SECONDS = 30               # rest never goes below this
REPS_BUMP_STEP = 1                  # e.g. 10 -> 11 reps
MAX_REPS_BUMP = 2                   # at most +4 reps before a harder variant is unlocked
MAX_DIFFICULTY_BUMP = 1             # allow at most ONE tier above the self-rated level
STREAK_FOR_REP_BUMP = 1             # "Perfect" weeks in a row needed to bump reps
OVERLOAD_LEVERS = ["sets", "rest", "difficulty"]   # rotated when a week is "Too Easy"

# Feedback tag names (NEW)
TAG_TOO_HEAVY = "Too Heavy"
TAG_EXHAUSTING = "Cardiovascularly Exhausting"
TAG_JOINT_PAIN = "Joint Pain"
FEEDBACK_TAGS = [TAG_TOO_HEAVY, TAG_EXHAUSTING, TAG_JOINT_PAIN]

# Age and beginner safety rules (design decisions, not taken from a guideline)
MIN_AGE = 18                        # the app is for adults
SENIOR_AGE = 50                     # from this age: no jump / plyometric exercises, slightly more rest
SENIOR_EXTRA_REST = 15              # seconds added to rest on strength exercises
BEGINNER_MAX_EXTRA_SETS = 1         # beginners never get more than +1 set from progression

LEVEL_RANK = {"beginner": 1, "intermediate": 2, "advanced": 3}
LEVEL_NAMES = {1: "beginner", 2: "intermediate", 3: "advanced"}
SECTION_ORDER = ["Warm-up", "Main", "Finisher", "Cooldown"]

# --- Configuration & Initialization ---
st.set_page_config(page_title="FitRecommend", layout="centered")


def new_progression():
    """Baseline state for progressive overload engine. Memory of how much difficulty increased"""
    return {
        "extra_sets": 0,        # sets added on top of the normal prescription
        "rest_delta": 0,        # seconds added to rest (negative = shorter rest)
        "reps_bump": 0,         # reps added to the rep target
        "difficulty_bump": 0,   # how many tiers above your level we may recommend
        "perfect_streak": 0,    # consecutive weeks rated "Perfect"
        "overload_cursor": 0,   # which lever to pull next on a "Too Easy" week
        "history": [],          # human-readable log shown on the Progress page
    }


# Default values for everything we want to remember between re-runs.
# "if ... not in" means: only set it the FIRST time the app opens.
_defaults = {
    "page": "home",
    "feedback": {},                 # NEW format: {exercise_name: {tag: count}}
    "weekly_plan": pd.DataFrame(),
    "selected_exercise": None,
    "completed_workouts": 0,
    "progression": new_progression(),   # NEW: overload state
    "week_number": 1,                   # NEW
    "week_feedback": {},                # NEW: {day: "Too Easy"/"Perfect"/"Too Hard"} for this week
    "session_log": [],                  # NEW: one entry per rated session
    "blacklisted_exercises": set(),     # NEW: names removed forever after "Joint Pain"
    "blacklisted_patterns": set(),      # NEW: (joint_action, muscle_group) pairs removed forever
    "feedback_day": None,               # NEW: which day the feedback form is about
    "flash": None,                      # NEW: one-time message shown after a rerun
}

#safety for app memory. When user press any button, code run again from top to bottom,
#preserve the user data
for _key, _value in _defaults.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value

default_profile = {
    "age": 21, "level": "Beginner", "goal": "Muscle Gain", "location": "Gym",
    "equipment": ["Dumbbells", "Barbell", "Bench"], "days": 4, "duration": 60,
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


def major_muscle_group(value):
    """Convert a messy muscle name into a simple group ('Lats' -> 'back')."""
    text = str(value).lower()
    if any(term in text for term in ["chest", "pectoral"]):
        return "chest"
    if any(term in text for term in ["lat", "back", "rhomboid", "trap"]):
        return "back"
    if any(term in text for term in ["shoulder", "deltoid"]):
        return "shoulders"
    if any(term in text for term in ["quad", "leg"]):
        return "quads"
    if "hamstring" in text:
        return "hamstrings"
    if "glute" in text:
        return "glutes"
    if any(term in text for term in ["abdominal", "core", "oblique"]):
        return "core"
    if "bicep" in text:
        return "biceps"
    if "tricep" in text:
        return "triceps"
    if "calf" in text:
        return "calves"
    return "other"


# =============================================================================
# 1. DATA & KNOWLEDGE BASE BUILDER
# =============================================================================

# --- NEW: helpers that label each exercise by reading its title -------------
def _contains_any(text, keywords):
    """True if any keyword appears at the START of a word in `text`.
    (The \\b means 'word boundary', so 'row' matches 'rows' but not 'arrow'.)"""
    return any(re.search(r"\b" + re.escape(k), text) for k in keywords)


# Multi-joint lifts that should ALWAYS count as compound.
_STRONG_COMPOUND = ["squat", "deadlift", "lunge", "bench press", "push-up", "pushup", "push up",
                    "pull-up", "pullup", "pull up", "chin-up", "chin up", "dip"]
# Single-joint movements.
_ISOLATION = ["curl", "extension", "raise", "fly", "flye", "kickback", "crunch", "sit-up", "situp",
              "shrug", "pushdown", "push-down", "skull", "pullover", "plank", "wrist", "adduct",
              "abduct", "tricep", "bicep", "calf"]
# Words that suggest compound if nothing above already decided.
_WEAK_COMPOUND = ["press", "row", "pulldown", "pull-down", "thrust", "bridge", "clean", "snatch",
                  "step-up", "step up", "good morning", "swing", "burpee"]
_UNILATERAL = ["lunge", "single", "one-arm", "one arm", "one-leg", "one leg", "unilateral",
               "concentration", "split squat", "step-up", "step up", "bulgarian", "alternating",
               "pistol", "kickback"]
# (keyword, joint action) - FIRST match wins, so specific words come first.
_JOINT_ACTIONS = [
    ("leg curl", "knee_flexion"), ("hamstring curl", "knee_flexion"),
    ("leg extension", "knee_extension"), ("calf", "ankle_plantarflexion"),
    ("deadlift", "hip_hinge"), ("good morning", "hip_hinge"), ("hip thrust", "hip_hinge"),
    ("thrust", "hip_hinge"), ("bridge", "hip_hinge"), ("back extension", "hip_hinge"), ("swing", "hip_hinge"),
    ("squat", "squat_pattern"), ("lunge", "squat_pattern"), ("leg press", "squat_pattern"),
    ("step-up", "squat_pattern"), ("step up", "squat_pattern"),
    ("pulldown", "vertical_pull"), ("pull-down", "vertical_pull"), ("pull-up", "vertical_pull"),
    ("pullup", "vertical_pull"), ("pull up", "vertical_pull"), ("chin-up", "vertical_pull"), ("chin up", "vertical_pull"),
    ("row", "horizontal_pull"), ("face pull", "horizontal_pull"),
    ("overhead press", "vertical_push"), ("shoulder press", "vertical_push"),
    ("military press", "vertical_push"), ("arnold", "vertical_push"),
    ("lateral raise", "shoulder_raise"), ("front raise", "shoulder_raise"),
    ("leg raise", "trunk_flexion"),
    ("bench press", "horizontal_push"), ("chest press", "horizontal_push"), ("push-up", "horizontal_push"),
    ("pushup", "horizontal_push"), ("push up", "horizontal_push"), ("dip", "horizontal_push"),
    ("fly", "chest_fly"), ("flye", "chest_fly"), ("press", "horizontal_push"),
    ("pushdown", "elbow_extension"), ("push-down", "elbow_extension"), ("skull", "elbow_extension"),
    ("tricep", "elbow_extension"), ("curl", "elbow_flexion"),
    ("crunch", "trunk_flexion"), ("sit-up", "trunk_flexion"), ("situp", "trunk_flexion"),
    ("plank", "trunk_stability"), ("twist", "trunk_rotation"),
]


def infer_compound(name, category):
    """NEW: True for multi-joint lifts (squat, bench press), False for single-joint
    ones (curl, calf raise). Only strength exercises can be 'compound'."""
    if category != "resistance":
        return False
    text = str(name).lower()
    if _contains_any(text, _STRONG_COMPOUND):
        return True
    if _contains_any(text, _ISOLATION):
        return False
    return _contains_any(text, _WEAK_COMPOUND)


def infer_movement_type(name, category):
    """NEW: 'unilateral' = one limb at a time (lunge), otherwise 'bilateral'."""
    if category == "resistance" and _contains_any(str(name).lower(), _UNILATERAL):
        return "unilateral"
    return "bilateral"


def infer_joint_action(name, category):
    """which joint movement an exercise trains (e.g. 'horizontal_push').
    Used to find 'similar' exercises when the user reports Joint Pain.
    Warm-ups, stretches and cardio return 'other' so they are never caught."""
    if category != "resistance":
        return "other"
    text = str(name).lower()
    for keyword, action in _JOINT_ACTIONS:
        if re.search(r"\b" + re.escape(keyword), text):
            return action
    return "other"


@st.cache_data
def load_and_build_kb():
    try:
        exercise = pd.read_csv(Path(__file__).with_name("megaGymDataset.csv"))
    except FileNotFoundError:
        exercise = pd.DataFrame()

    def first_col(frame, candidates):
        return next((c for c in candidates if c in frame.columns), None)

    title_col = first_col(exercise, ["Title", "Exercise", "Name"])
    body_col = first_col(exercise, ["BodyPart", "Target Muscle", "Muscle"])
    equipment_col = first_col(exercise, ["Equipment", "equipment_needed"])
    level_col = first_col(exercise, ["Level", "Difficulty", "difficulty_level"])
    type_col = first_col(exercise, ["Type", "exercise_category"])
    desc_col = first_col(exercise, ["Desc", "Description", "desc"])

    if not exercise.empty:
        if desc_col is not None:
            exercise = exercise.dropna(subset=[desc_col])
            exercise = exercise[exercise[desc_col].astype(str).str.strip().ne("")]
        if "Rating" in exercise.columns:
            ratings = pd.to_numeric(exercise["Rating"], errors="coerce")
            exercise = exercise[ratings > 8.5]
        if "RatingDesc" in exercise.columns:
            exercise = exercise.drop(columns=["RatingDesc"])
        exercise = exercise.dropna()
        if type_col is not None:
            excluded_types = ["Olympic Weightlifting", "Strongman", "Powerlifting"]
            exercise = exercise[~exercise[type_col].isin(excluded_types)]
        if equipment_col is not None:
            equipment_values = exercise[equipment_col].astype(str).str.strip().str.lower()
            exercise = exercise[equipment_values != "other"]

    if title_col is None:
        source = pd.DataFrame()
    else:
        source = exercise.dropna(subset=[title_col]).drop_duplicates(subset=[title_col]).copy()

    def text_value(row, column, default=""):
        if column is None or pd.isna(row.get(column, np.nan)):
            return default
        value = str(row[column]).strip()
        return value if value and value.lower() != "nan" else default

    def normalise_level(value):
        value = str(value).lower()
        if "beginner" in value or "novice" in value: return "beginner"
        if "intermediate" in value: return "intermediate"
        if "expert" in value or "advanced" in value: return "advanced"
        return "beginner"

    # The CSV says "Dumbbell"/"Bands" but the profile form says "Dumbbells"/
    # "Resistance Band". Without this mapping the equipment filter never matched.
    equipment_aliases = {"dumbbell": "dumbbells", "bands": "resistance band", "band": "resistance band"}

    def normalise_equipment(value):
        value = str(value).strip().lower()
        if value in {"", "none", "nan", "body only", "bodyweight"}: return "bodyweight"
        return equipment_aliases.get(value, value)

    def classify_category(row):
        raw_type = text_value(row, type_col).lower()
        body = text_value(row, body_col).lower()
        title = text_value(row, title_col).lower()
        combined = f"{raw_type} {body} {title}"
        if any(w in combined for w in ["cardio", "running", "cycling", "jump rope", "rowing machine"]):
            return "cardio"
        if "plyometric" in raw_type:
            return "plyometrics"
        #dynamic warm-up movements get their own category.
        if any(w in combined for w in ["mobility", "dynamic", "warm-up", "warm up", "warmup"]):
            return "mobility"
        if any(w in combined for w in ["stretch", "flexibility", "yoga"]):
            return "flexibility"
        return "resistance"

    records = []
    for _, row in source.iterrows():
        equipment = normalise_equipment(text_value(row, equipment_col, "bodyweight"))

        actual_desc = text_value(row, desc_col, "")
        if not actual_desc:
            continue

        records.append({
            "exercise_name": text_value(row, title_col, "Unnamed exercise"),
            "exercise_category": classify_category(row),
            "target_muscle": text_value(row, body_col, "full body").title(),
            "major_muscle_group": major_muscle_group(text_value(row, body_col, "full body")),
            "equipment_needed": equipment,
            "location_suitability": ["gym"] if equipment != "bodyweight" else ["home", "gym", "outdoor"],
            "difficulty_level": normalise_level(text_value(row, level_col, "beginner")),
            "instructions": actual_desc
        })

    # ---- Supplemental exercises (hand-written so the app always has the basics) ----
    anywhere = ["home", "outdoor", "gym"]
    supplemental = [
        # --- original basics ---
        ("Brisk Walking", "cardio", "full body", "bodyweight", anywhere, "beginner", None),
        ("Stationary Cycling", "cardio", "legs", "stationary bike", ["gym"], "beginner", None),
        ("Jump Rope", "cardio", "full body", "jump rope", anywhere, "intermediate", None),
        ("Standing Chest Stretch", "flexibility", "chest", "bodyweight", anywhere, "beginner",
         "1. Clasp your hands behind your back.\n2. Straighten your arms and lift them slightly.\n3. Open your chest and hold the stretch.\n4. Breathe slowly."),
        ("Bodyweight Squat", "resistance", "legs", "bodyweight", anywhere, "beginner", None),
        ("Wall Push-up", "resistance", "chest", "bodyweight", anywhere, "beginner", None),
        ("Glute Bridge", "resistance", "glutes", "bodyweight", anywhere, "beginner", None),
        ("Bench Press", "resistance", "chest", "barbell", ["gym"], "intermediate", None),
        ("Lat Pulldown", "resistance", "back", "machine", ["gym"], "beginner", None),
        
        # --- unilateral + isolation strength options ---
        ("Reverse Lunge", "resistance", "quads", "bodyweight", anywhere, "beginner",
         "1. Stand tall with hands on hips.\n2. Step one foot back and lower until both knees are about 90 degrees.\n3. Push through the front heel to stand.\n4. Complete all reps on one side, then switch."),
        ("Single-Leg Glute Bridge", "resistance", "glutes", "bodyweight", anywhere, "beginner",
         "1. Lie on your back, knees bent, one leg extended.\n2. Press through the planted heel to lift your hips.\n3. Squeeze your glute at the top.\n4. Lower slowly; complete reps, then switch legs."),
        ("Single-Arm Dumbbell Row", "resistance", "back", "dumbbells", ["home", "gym"], "beginner",
         "1. Place one hand and knee on a bench, dumbbell in the other hand.\n2. Keep your back flat.\n3. Pull the dumbbell toward your hip.\n4. Lower with control; complete reps, then switch arms."),
        ("Calf Raise", "resistance", "calves", "bodyweight", anywhere, "beginner",
         "1. Stand with feet hip-width apart.\n2. Rise onto your toes as high as you can.\n3. Pause for a second.\n4. Lower slowly."),
        ("Dead Bug", "resistance", "core", "bodyweight", anywhere, "beginner",
         "1. Lie on your back, arms up, knees bent over hips.\n2. Lower the opposite arm and leg toward the floor.\n3. Keep your lower back pressed down.\n4. Return and alternate sides."),
        
        # =====================================================================
        # --- NEW: dynamic warm-up (category 'mobility') - FROM MUSCLE & STRENGTH
        # =====================================================================
        ("Fast-Paced Walk", "mobility", "full body", "bodyweight", anywhere, "beginner", 
         "Walk at a brisk pace to elevate your core body temperature and blood flow."),
        ("Plantar Fascia w/ Lacrosse Ball", "mobility", "calves", "lacrosse ball", anywhere, "beginner", 
         "While standing or seated, roll a lacrosse ball under the arch of each foot to release tension."),
        ("Donkey Calf Raise", "mobility", "calves", "bodyweight", anywhere, "beginner", 
         "Hinge at the hips (placing hands on a bench or wall) and perform controlled calf raises."),
        ("Dumbbell Goblet Squat", "mobility", "quads", "dumbbells", ["home", "gym"], "beginner", 
         "Hold a dumbbell vertically against your chest and perform a controlled squat to open the hips."),
        ("Bodyweight Walking Lunge", "mobility", "quads", "bodyweight", anywhere, "beginner", 
         "Step forward into a lunge, alternating legs as you walk forward."),
        ("Side Plank", "mobility", "core", "bodyweight", anywhere, "beginner", 
         "Prop yourself up on one forearm and the side of your foot, keeping your body perfectly straight."),
        ("Plyometric Push Up", "mobility", "chest", "bodyweight", anywhere, "intermediate", 
         "Perform a push-up with enough explosive force to briefly lift your hands off the ground."),
        ("Rack Lat Stretch", "mobility", "back", "machine", ["gym"], "beginner", 
         "Hold onto a squat rack or sturdy pole, hinge at the hips, and pull back gently to stretch your lats."),
        ("Bodyweight Standing Calf Raise", "mobility", "calves", "bodyweight", anywhere, "beginner", 
         "Stand tall and push up onto your toes, then lower with control to warm up the ankles."),
        ("IT Band Foam Roller", "mobility", "quads", "foam roll", ["home", "gym"], "beginner", 
         "Lie on your side and slowly roll the foam roller along your outer thigh."),
        ("Bodyweight Single-Leg Deadlift", "mobility", "hamstrings", "bodyweight", anywhere, "beginner", 
         "Hinge at the hips on one leg, keeping your back straight and the other leg extended behind you."),
        ("Prisoner Squat", "mobility", "quads", "bodyweight", anywhere, "beginner", 
         "Place your hands behind your head and perform a bodyweight squat."),
        ("Prisoner Squat (Hold at the Bottom)", "mobility", "quads", "bodyweight", anywhere, "beginner", 
         "Perform a prisoner squat and hold the bottom position isometrically to improve hip mobility."),
        ("Superman", "mobility", "back", "bodyweight", anywhere, "beginner", 
         "Lie face down and lift your arms and legs off the floor simultaneously, squeezing the lower back."),
        ("Plank", "mobility", "core", "bodyweight", anywhere, "beginner", 
         "Hold a push-up position on your forearms with a straight back and tight core."),
        ("Band Pull-Apart", "mobility", "shoulders", "resistance band", anywhere, "beginner", 
         "Hold a resistance band with straight arms and pull it apart until it touches your chest."),
        ("Warm-up Push Up", "mobility", "chest", "bodyweight", anywhere, "beginner", 
         "Lower your body until your chest is just above the floor, then push back up."),
        ("Warm-up Bench Dips", "mobility", "triceps", "bench", ["gym", "home"], "beginner", 
         "Place your hands on a bench behind you, lower your hips toward the floor, then press back up."),
        ("Arm Circles (Clockwise)", "mobility", "shoulders", "bodyweight", anywhere, "beginner", 
         "Extend your arms straight out to the sides and make controlled circles clockwise."),
        ("Arm Circles (Counter Clockwise)", "mobility", "shoulders", "bodyweight", anywhere, "beginner", 
         "Extend your arms straight out to the sides and make controlled circles counter-clockwise."),
        ("Eccentric Only Pull Up", "mobility", "back", "bodyweight", ["gym", "outdoor"], "intermediate", 
         "Jump to the top of a pull-up position, then lower yourself down as slowly as possible."),
        ("Seated Zottman Curl", "mobility", "biceps", "dumbbells", ["gym", "home"], "beginner", 
         "Curl dumbbells up with palms facing up, rotate wrists at the top, and lower with palms facing down."),
        ("90/90 Hip Crossover", "mobility", "glutes", "bodyweight", anywhere, "beginner", 
         "Sit with both legs bent at 90 degrees and rotate your hips/knees from side to side on the floor."),
        ("Warm-up Sit Up", "mobility", "core", "bodyweight", anywhere, "beginner", 
         "Lie on your back, knees bent, and sit all the way up to warm up the abdominals."),
        ("Jump Squat", "mobility", "quads", "bodyweight", anywhere, "beginner", 
         "Perform a bodyweight squat and explode upward into a jump to activate the nervous system."),

        # --- static cooldown stretches (category 'flexibility') ---
        ("Cross-Body Shoulder Stretch", "flexibility", "shoulders", "bodyweight", anywhere, "beginner",
         "1. Bring one arm across your chest.\n2. Hold it with the other arm.\n3. Hold the stretch, then switch."),
        ("Child's Pose", "flexibility", "back", "bodyweight", anywhere, "beginner",
         "1. Kneel and sit back on your heels.\n2. Reach your arms forward along the floor.\n3. Relax and breathe deeply."),
        ("Overhead Side Reach Lat Stretch", "flexibility", "back", "bodyweight", anywhere, "beginner",
         "1. Raise one arm overhead.\n2. Lean gently to the opposite side.\n3. Hold, then switch."),
        ("Standing Quad Stretch", "flexibility", "quads", "bodyweight", anywhere, "beginner",
         "1. Stand on one leg (hold a wall).\n2. Pull the other heel toward your glute.\n3. Keep knees together and hold, then switch."),
        ("Seated Hamstring Stretch", "flexibility", "hamstrings", "bodyweight", anywhere, "beginner",
         "1. Sit with one leg straight, the other bent.\n2. Hinge forward from your hips.\n3. Hold, then switch."),
        ("Figure-Four Glute Stretch", "flexibility", "glutes", "bodyweight", anywhere, "beginner",
         "1. Lie on your back, cross one ankle over the opposite knee.\n2. Pull the thigh toward your chest.\n3. Hold, then switch."),
        ("Cobra Stretch", "flexibility", "core", "bodyweight", anywhere, "beginner",
         "1. Lie face down, hands under shoulders.\n2. Gently press up, keeping hips on the floor.\n3. Hold and breathe."),
        ("Doorway Biceps Stretch", "flexibility", "biceps", "bodyweight", anywhere, "beginner",
         "1. Place your palm on a wall, arm straight, thumb down.\n2. Slowly turn your body away.\n3. Hold, then switch."),
        ("Overhead Triceps Stretch", "flexibility", "triceps", "bodyweight", anywhere, "beginner",
         "1. Raise one arm and bend the elbow behind your head.\n2. Gently press the elbow with your other hand.\n3. Hold, then switch."),
        ("Wall Calf Stretch", "flexibility", "calves", "bodyweight", anywhere, "beginner",
         "1. Stand facing a wall, one foot back with the heel down.\n2. Lean forward until you feel the calf stretch.\n3. Hold, then switch."),
        ("Standing Forward Fold", "flexibility", "full body", "bodyweight", anywhere, "beginner",
         "1. Stand with feet hip-width apart.\n2. Fold forward from the hips, knees slightly bent.\n3. Let your head and arms hang and breathe."),
    ]

    existing = {r["exercise_name"].lower() for r in records}
    for name, category, muscle, equipment, locations, level, instructions in supplemental:
        if name.lower() not in existing:
            records.append({
                "exercise_name": name, "exercise_category": category, "target_muscle": muscle.title(),
                "major_muscle_group": major_muscle_group(muscle),
                "equipment_needed": equipment, "location_suitability": locations,
                "difficulty_level": level,
                "instructions": instructions or f"1. Setup your {equipment} securely.\n2. Brace your core.\n3. Contract your {muscle} to initiate the movement.\n4. Complete the reps with a controlled tempo."
            })

    kb = pd.DataFrame(records).drop_duplicates("exercise_name").reset_index(drop=True)
    # NEW columns, computed once for every exercise (CSV rows and supplemental rows alike).
    kb["is_compound"] = [infer_compound(n, c) for n, c in zip(kb["exercise_name"], kb["exercise_category"])]
    kb["movement_type"] = [infer_movement_type(n, c) for n, c in zip(kb["exercise_name"], kb["exercise_category"])]
    kb["joint_action"] = [infer_joint_action(n, c) for n, c in zip(kb["exercise_name"], kb["exercise_category"])]
    return kb


exercise_kb = load_and_build_kb()
# =============================================================================
# 2. RECOMMENDATION ENGINE PIPELINE
# =============================================================================

def blacklist_mask(frame, names, patterns):
    """NEW: True for every row that is blacklisted, either because its NAME is
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
    """NEW: self-rated level (1-3) PLUS any 'harder variant' unlocked by progression."""
    base = LEVEL_RANK.get(str(profile["self_rated_fitness_level"]).lower(), 1)
    bump = (profile.get("progression") or {}).get("difficulty_bump", 0)
    return min(3, base + bump)


def safety_filter(exercises, profile, feedback):
    """STEP A: remove exercises that are unsafe or impossible for this user."""
    user_level = effective_level_rank(profile)
    allowed_equipment = {str(x).strip().lower() for x in profile["available_equipment"]}
    location = profile["workout_location"].strip().lower()
    disliked = {x.strip().lower() for x in profile.get("disliked_exercises", [])}
    # NEW: permanent Joint-Pain blacklist
    banned_names = {str(x).lower() for x in profile.get("blacklisted_exercises", [])}
    banned_patterns = set(profile.get("blacklisted_patterns", []))

    rows = exercises.copy()
    rows = rows[rows["difficulty_level"].map(LEVEL_RANK).fillna(3) <= user_level]
    rows = rows[rows["equipment_needed"].astype(str).str.lower().isin(allowed_equipment) | (rows["equipment_needed"] == "bodyweight")]
    rows = rows[rows["location_suitability"].apply(lambda places: location in [str(p).lower() for p in places])]
    rows = rows[~rows["exercise_name"].astype(str).str.lower().isin(disliked)]
    # NEW: conservative rule for users aged 50+ -> no jump or plyometric exercises
    if int(profile.get("age") or 0) >= SENIOR_AGE:
        rows = rows[rows["exercise_category"] != "plyometrics"]
        rows = rows[~rows["exercise_name"].astype(str).str.lower().str.contains("jump|plyometric", regex=True)]
    rows = rows[~blacklist_mask(rows, banned_names, banned_patterns)]
    # NOTE: "Too Hard"-style feedback no longer REMOVES exercises. "Too Heavy" now only
    # lowers the ranking score (see rank_exercises); only Joint Pain removes exercises.
    return rows.reset_index(drop=True)


def goal_category(goal):
    goal = str(goal).lower().replace(" ", "_")
    if goal in {"muscle_gain", "strength"}: return "resistance"
    if goal == "weight_loss": return "cardio"
    return "mixed"


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


def exercise_features(row, profile):
    return {
        "goal": goal_category(profile["fitness_goal"]),
        "exercise_category": str(row["exercise_category"]).lower(),
        "target_muscle": str(row["target_muscle"]).lower(),
        "equipment": str(row["equipment_needed"]).lower(),
        "difficulty": str(row["difficulty_level"]).lower(),
        "location": profile["workout_location"].lower(),
        "location_match": profile["workout_location"].lower() in [str(x).lower() for x in row["location_suitability"]]
    }


def user_features(profile):
    return {
        "goal": goal_category(profile["fitness_goal"]),
        "exercise_category": goal_category(profile["fitness_goal"]),
        "target_muscle": "any", "equipment": "any",
        # NEW: after progression unlocks a harder tier, we aim at that tier.
        "difficulty": LEVEL_NAMES[effective_level_rank(profile)],
        "location": profile["workout_location"].lower(),
        "location_match": True
    }


def feedback_penalty(tags):
    """NEW: score reduction for one exercise from its negative feedback tags.
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

    liked = {x.strip().lower() for x in profile.get("liked_exercises", [])}
    for idx, row in ranked.iterrows():
        if row["exercise_name"].lower() in liked or row["target_muscle"].lower() in liked:
            ranked.loc[idx, "recommendation_score"] += 0.5
        # NEW: "Too Heavy" / "Cardiovascularly Exhausting" lower the score.
        ranked.loc[idx, "recommendation_score"] -= feedback_penalty(feedback.get(row["exercise_name"], {}))

    return ranked.sort_values("recommendation_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# NEW: prescription (sets / reps / rest) and the time-cost formula
# ---------------------------------------------------------------------------
def estimate_exercise_seconds(sets, reps, rest_seconds, sides=1):
    """NEW (feature 1): (Sets x Reps x 4 s) + ((Sets - 1) x Rest).
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


# Prescription ranges by goal (reps, sets, rest in seconds). Progression can move
# the numbers around, but they are always clamped back into these ranges.
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


def prescribe_exercise(row, user_goal, session_minutes, progression, feedback, section, level_rank=2, age=0):
    """NEW: work out sets, reps, rest AND the estimated duration of one exercise.
    This applies the overload state (extra sets, shorter rest, more reps) and the
    'Cardiovascularly Exhausting' tag (more rest / shorter cardio)."""
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
        sets, low, high, rest = 3, 10, 15, 60
    else:
        sets, low, high, rest = base_prescription(user_goal, level_rank)
        sets += (min(progression["extra_sets"], BEGINNER_MAX_EXTRA_SETS) if level_rank <= 1 else progression["extra_sets"])                          # overload: more sets
        low += progression["reps_bump"]                            # overload: more reps
        high += progression["reps_bump"]
        rest = max(MIN_REST_SECONDS, rest + progression["rest_delta"])   # overload: less rest
        # keep everything inside the reference ranges for this goal
        limits = GOAL_RANGES[goal_key(user_goal)]
        sets = int(np.clip(sets, *limits["sets"]))
        high = min(high, limits["reps"][1])
        low = min(low, high)
        rest = int(np.clip(rest, *limits["rest"]))
    rest += 15 * exhausted
    if age >= SENIOR_AGE:
        rest += SENIOR_EXTRA_REST     # NEW: a little more recovery for users aged 50+

    reps_for_timing = round((low + high) / 2)    # use the middle of the rep range
    seconds = estimate_exercise_seconds(sets, reps_for_timing, rest, sides)
    return {"sets": sets, "reps": f"{low}–{high} reps{suffix}", "rest": f"{rest} seconds",
            "seconds": seconds, "setup": SETUP_SECONDS_PER_EXERCISE}


def build_plan_row(row, rx, day, focus, section):
    """NEW: turn one exercise + its prescription into one row of the plan table."""
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
    }


def pick_support_exercises(pool, groups, target_seconds, section, exclude_names, rx_for):
    """NEW (feature 7): choose warm-up or cooldown exercises worth ~target_seconds.
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
        1: [0], 2: [0, 3], 3: [0, 2, 4], 4: [0, 1, 3, 5],
        5: [0, 1, 3, 4, 6], 6: [0, 1, 2, 4, 5, 6], 7: list(range(7))
    }[n_days]

    if ranked.empty:
        return pd.DataFrame()

    progression = profile.get("progression") or new_progression()
    user_goal = str(profile.get("fitness_goal", "General Fitness")).lower().replace(" ", "_")
    split_days = training_days(profile, n_days)
    level_rank = LEVEL_RANK.get(str(profile.get("self_rated_fitness_level", "beginner")).lower(), 1)
    age = int(profile.get("age") or 0)

    # NEW: total time we may spend, in seconds.
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
        # NEW (feature 2): is_compound sits right after "previously_used", so compound
        # lifts are always preferred over isolation work.
        candidates = candidates.sort_values(
            ["previously_used", "is_compound_int", "penalised", "muscle_usage", "recommendation_score"],
            ascending=[True, False, True, True, False])

        resistance_pool = candidates[candidates["exercise_category"] == "resistance"]

        # --- reserve time for the finisher (cardio / flexibility), if any ---
        finisher_row, finisher_rx = None, None
        if finisher is not None:
            finisher_pool = candidates[candidates["exercise_category"] == finisher]
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

        # Phase 0 (feature 6): guarantee the minimum number of one-limb exercises,
        # preferring ones that train today's muscle groups.
        unilateral_pool = resistance_pool[resistance_pool["movement_type"] == "unilateral"]
        unilateral_today = unilateral_pool[unilateral_pool["major_muscle_group"].isin(session["groups"])]
        for enforce in (True, False):
            for pool in (unilateral_today, unilateral_pool):
                for _, row in pool.iterrows():
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

        # Phase 2 (feature 1): keep adding the next best exercise WHILE time remains.
        # First respecting muscle variety, then relaxing it if the window is still unfilled.
        for enforce in (True, False):
            queue = [row for _, row in resistance_pool.iterrows()]
            idx = 0
            while idx < len(queue) and len(chosen) < MAX_RESISTANCE_EXERCISES and state["time"] < main_budget:
                try_add(queue[idx], enforce)
                idx += 1

        # Fallback: if there is no strength option at all, allow any non-warm-up exercise.
        if not chosen:
            fallback = candidates[~candidates["exercise_category"].isin(["mobility", "flexibility"])]
            for _, row in fallback.iterrows():
                try_add(row, False)

        # NEW (feature 2): compound lifts first. sorted() is stable, so the order
        # within compound / isolation groups is kept.
        ordered_main = sorted(chosen, key=lambda item: 0 if bool(item["row"]["is_compound"]) else 1)

        # --- NEW (feature 7): warm-up + cooldown targeting today's muscle groups ---
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
        "available_equipment": p.get("equipment"),
        "preferred_days_per_week": p.get("days"),
        "session_duration_minutes": p.get("duration"),
        "self_rated_fitness_level": p.get("level"),
        "liked_exercises": set(),
        "disliked_exercises": set(),
        # NEW: progression state and the permanent Joint-Pain blacklist
        "progression": st.session_state.progression,
        "blacklisted_exercises": st.session_state.blacklisted_exercises,
        "blacklisted_patterns": st.session_state.blacklisted_patterns,
    }

    safe_exercises = safety_filter(exercise_kb, backend_profile, f)
    ranked_exercises = rank_exercises(safe_exercises, backend_profile, f)
    weekly_plan = assemble_weekly_plan(ranked_exercises, backend_profile, f)

    return weekly_plan


# =============================================================================
# NEW (feature 4): JOINT-PAIN BLACKLIST
# =============================================================================
def blacklist_exercise(name):
    """Permanently ban an exercise AND similar ones (same joint action + same muscle
    group). Returns how many OTHER exercises were banned along with it.
    Note: 'permanent' means for this session; refreshing the browser resets it
    unless you later save st.session_state to a file or database."""
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
# NEW (feature 3): PROGRESSIVE OVERLOAD ENGINE
# =============================================================================
def apply_too_easy(prog, max_extra_sets=MAX_EXTRA_SETS):
    """A 'Too Easy' week pulls ONE lever (sets -> rest -> harder variants, rotating)
    so we never change everything at once."""
    for _ in range(len(OVERLOAD_LEVERS)):
        lever = OVERLOAD_LEVERS[prog["overload_cursor"] % len(OVERLOAD_LEVERS)]
        prog["overload_cursor"] += 1
        if lever == "sets" and prog["extra_sets"] < max_extra_sets:
            prog["extra_sets"] += 1
            return "Too Easy -> added 1 set to strength exercises."
        if lever == "rest" and prog["rest_delta"] > -MAX_REST_REDUCTION:
            prog["rest_delta"] -= 15
            return "Too Easy -> reduced rest by 15 seconds."
        if lever == "difficulty" and prog["difficulty_bump"] < MAX_DIFFICULTY_BUMP:
            prog["difficulty_bump"] += 1
            return "Too Easy -> unlocked harder exercise variants."
    return "Too Easy -> already at the app's progression limits; try using heavier weights."


def apply_perfect_streak(prog):
    """After enough 'Perfect' weeks: bump reps first, then unlock harder variants."""
    if prog["reps_bump"] < MAX_REPS_BUMP:
        prog["reps_bump"] += REPS_BUMP_STEP
        return f"{STREAK_FOR_REP_BUMP} Perfect weeks in a row -> rep targets +{REPS_BUMP_STEP}."
    if prog["difficulty_bump"] < MAX_DIFFICULTY_BUMP:
        prog["difficulty_bump"] += 1
        prog["reps_bump"] = 0
        return "Rep targets maxed out -> unlocked harder exercise variants and reset reps."
    return "Perfect streak -> already at the progression limits; try using heavier weights."


def apply_too_hard(prog):
    """A 'Too Hard' week steps ONE change back so you can recover."""
    if prog["extra_sets"] > 0:
        prog["extra_sets"] -= 1
        return "Too Hard -> removed 1 extra set."
    if prog["rest_delta"] < 0:
        prog["rest_delta"] += 15
        return "Too Hard -> restored 15 seconds of rest."
    if prog["reps_bump"] > 0:
        prog["reps_bump"] -= REPS_BUMP_STEP
        return f"Too Hard -> lowered rep targets by {REPS_BUMP_STEP}."
    if prog["difficulty_bump"] > 0:
        prog["difficulty_bump"] -= 1
        return "Too Hard -> went back to your original exercise difficulty."
    return "Too Hard -> already at baseline; consider lighter weights."


def evaluate_week(prog, ratings, max_extra_sets=MAX_EXTRA_SETS):
    """Look at this week's session ratings and update the progression state.
    Priority: any 'Too Hard' (safety) > any 'Too Easy' > all 'Perfect'."""
    if not ratings:
        return ["No sessions were rated this week, so nothing changed."]
    if "Too Hard" in ratings:
        prog["perfect_streak"] = 0
        return [apply_too_hard(prog)]
    if "Too Easy" in ratings:
        prog["perfect_streak"] = 0
        return [apply_too_easy(prog, max_extra_sets)]
    prog["perfect_streak"] += 1
    if prog["perfect_streak"] >= STREAK_FOR_REP_BUMP:
        prog["perfect_streak"] = 0
        return [apply_perfect_streak(prog)]
    return [f"Perfect week {prog['perfect_streak']} of {STREAK_FOR_REP_BUMP} - one more to progress."]


def finish_week():
    """Apply progression, move to the next week and build its plan."""
    week = st.session_state.week_number
    level = str(st.session_state.profile.get("level", "Beginner")).lower()
    cap = BEGINNER_MAX_EXTRA_SETS if LEVEL_RANK.get(level, 1) == 1 else MAX_EXTRA_SETS
    messages = evaluate_week(st.session_state.progression,
                             list(st.session_state.week_feedback.values()), cap)
    for message in messages:
        st.session_state.progression["history"].append(f"Week {week}: {message}")
    st.session_state.week_number += 1
    st.session_state.week_feedback = {}
    st.session_state.weekly_plan = generate_plan()
    return messages


def record_tag(name, tag):
    """Count one negative-feedback tag for an exercise."""
    counts = st.session_state.feedback.setdefault(name, {})
    counts[tag] = counts.get(tag, 0) + 1


# =============================================================================
# 3. UI PAGES
# =============================================================================

def page_home():
    st.markdown("<h1 style='text-align: center;'>FitRecommend</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Personalized Workout Recommendation System</h3>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<p style='text-align: center;'>Get a workout plan designed for your goals, fitness level and equipment.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Create / Edit My Plan", use_container_width=True):
            navigate_to('profile')
            st.rerun()
        if not st.session_state.weekly_plan.empty:
            if st.button("View Current Plan", use_container_width=True):
                navigate_to('plan')
                st.rerun()
        if st.button("My Progress & Download Data", use_container_width=True):
            navigate_to('progress')
            st.rerun()
        # --- NEW TRAINING GUIDE BUTTON ---
        if st.button("Training Guide", use_container_width=True):
            navigate_to('learn')
            st.rerun()


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

    equipment = st.multiselect(
        "Available Equipment",
        ["Dumbbells", "Barbell", "Bench", "Resistance Band", "Machine", "Cable", "Stationary Bike", "Jump Rope"],
        default=p['equipment']
    )

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
    # NEW: show the overload adjustments currently applied
    st.caption(
        f"Overload: +{prog['extra_sets']} sets, rest {prog['rest_delta']:+d}s, "
        f"+{prog['reps_bump']} reps, harder variants: {'on' if prog['difficulty_bump'] else 'off'}")
    show_flash()

    if st.button("⌂ Back to Home"):
        navigate_to('home')
        st.rerun()

    plan_df = st.session_state.weekly_plan

    if plan_df.empty:
        st.error("No exercises matched your criteria. Please adjust your equipment, location, or fitness level.")
        return

    session_minutes = int(p.get("duration", 60))

    for day in plan_df['Day'].unique():
        day_data = plan_df[plan_df['Day'] == day]
        rated = day in st.session_state.week_feedback
        st.subheader(f"{day} - {day_data['Focus'].iloc[0]}")
        total_minutes = day_data["Seconds"].sum() / 60
        st.caption(f"Estimated time: ~{total_minutes:.0f} min of your {session_minutes} min window"
                   + (" ✅ feedback submitted" if rated else ""))

        # One table per day, ordered Warm-up -> Main -> Finisher -> Cooldown.
        ordered = day_data.assign(_s=day_data["Section"].map(SECTION_ORDER.index)).sort_values("_s", kind="stable")
        table = ordered[["Section", "Exercise", "Target", "Type", "Movement", "Sets", "Reps", "Rest"]]
        st.dataframe(table, hide_index=True, use_container_width=True)

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
    st.success(reasons)


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
    # NEW: three options map directly onto the overload engine.
    difficulty = st.select_slider("How difficult was today's workout?",
                                  options=["Too Easy", "Perfect", "Too Hard"], value="Perfect")

    # NEW (feature 4): granular, per-exercise tags.
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


def build_export():
    """Everything from this session as JSON text (no name or ID is collected)."""
    data = {
        "profile": st.session_state.profile,
        "week_number": st.session_state.week_number,
        "completed_workouts": st.session_state.completed_workouts,
        "session_log": st.session_state.session_log,
        "progression": st.session_state.progression,
        "excluded_exercises": sorted(st.session_state.blacklisted_exercises),
        "weekly_plan": st.session_state.weekly_plan.to_dict(orient="records"),
    }
    return json.dumps(data, indent=2, default=str)


def page_progress():
    if st.button("⬅ Back to Home"):
        navigate_to('home')
        st.rerun()

    st.title("My Progress")
    show_flash()

    # NEW: a real average instead of a hard-coded number.
    scale = {"Too Easy": 1, "Perfect": 3, "Too Hard": 5}
    log = st.session_state.session_log
    average = f"{np.mean([scale[e['difficulty']] for e in log]):.1f} / 5" if log else "—"

    col1, col2 = st.columns(2)
    col1.metric("Workouts Completed", st.session_state.completed_workouts)
    col2.metric("Average Difficulty", average)
    st.metric("Current Week", st.session_state.week_number)

    prog = st.session_state.progression
    st.subheader("Current Overload Settings")
    st.markdown(f"""
    * **Extra sets:** +{prog['extra_sets']}
    * **Rest change:** {prog['rest_delta']:+d} seconds
    * **Rep target bump:** +{prog['reps_bump']}
    * **Harder variants unlocked:** {'Yes' if prog['difficulty_bump'] else 'No'}
    * **Consecutive 'Perfect' weeks:** {prog['perfect_streak']} / {STREAK_FOR_REP_BUMP}
    """)

    st.subheader("Progression History")
    if prog["history"]:
        for entry in reversed(prog["history"]):
            st.markdown(f"* {entry}")
    else:
        st.write("Complete a week of sessions to see your plan adapt.")

    st.subheader("Download My Session Data")
    st.caption("Nothing is saved on the server. Download your plan and feedback as a file "
               "(it contains no name or ID) if you want to keep it or send it to the researcher.")
    st.download_button("⬇ Download session data (JSON)", data=build_export(),
                       file_name="my_workout_session.json", mime="application/json")

    if st.session_state.blacklisted_exercises:
        st.subheader("Excluded for Joint Pain")
        st.write(", ".join(sorted(x.title() for x in st.session_state.blacklisted_exercises)))

# --- NEW TRAINING GUIDE INTEGRATION ---
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
        ("Progression", "Progressive Overload", "Progressive overload is a gradual increase in training challenge as your body adapts.", "You may add a repetition, a set, a little weight, or improve technique over time."),
        ("Progression", "Recovery", "Recovery is the time your body needs to repair and adapt between training sessions.", "Sleep, nutrition, rest days, and manageable training volume all support recovery."),
    ]

    workout_tips = [
        ("Build consistency before complexity", "The most productive program is one you can repeat week after week. Start with a schedule that fits your life, learn the core movements, and avoid changing exercises or methods every session. Complex methods are optional; regular training and steady effort matter more for most healthy adults.", "ACSM resistance-training guidelines", "https://acsm.org/resistance-training-guidelines-update-2026/"),
        ("Train every major muscle group", "A balanced program should include the legs, hips, back, chest, abdomen, shoulders, and arms. Covering these areas supports general strength and reduces the chance that your plan is dominated by only the muscles you enjoy training. The app uses this principle when building its weekly split.", "U.S. Physical Activity Guidelines", "https://www.health.gov/PAguidelines/pdf/paguide.pdf"),
        ("Strength train at least twice each week", "For general health, muscle-strengthening work for all major muscle groups should be included on at least two days each week. More days can work well when volume is distributed sensibly, but beginning with a repeatable two-day foundation is better than an unsustainable ambitious plan.", "U.S. Physical Activity Guidelines", "https://www.health.gov/PAguidelines/pdf/paguide.pdf"),
        ("Progress gradually", "Increase only one or two training demands at a time: for example, a small amount of weight, one extra repetition, or one additional set. Starting too hard and raising volume, intensity, and frequency together makes it harder to recover and makes it difficult to identify what caused pain or excessive fatigue.", "U.S. Physical Activity Guidelines", "https://www.health.gov/PAguidelines/pdf/paguide.pdf"),
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
    'learn': page_learn # <--- INTEGRATED ROUTE HERE
}

pages.get(st.session_state.page, page_home)()