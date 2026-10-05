"""
Project: Building an Intelligent System for Automated Text-to-Sign
         Language Translation (recorded retrieval + LSTM recognition)
Target language: Nigerian Sign Language (NSL)
Architecture:    Retrieval-based (MediaPipe Holistic keypoints + custom LSTM)
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# PROJECT ROOT & FOLDER LAYOUT
# ---------------------------------------------------------------------------
# All paths are resolved relative to this file so the project can be moved
# anywhere on disk (or into Google Drive for Colab) without edits elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".runtime/matplotlib"))

DATASET_DIR    = PROJECT_ROOT / "dataset"
CLIPS_DIR      = DATASET_DIR / "clips"        # Phase 1 output: per-sign .mp4 clips
KEYPOINTS_DIR  = DATASET_DIR / "keypoints"    # Phase 2 output: (30, 1662) .npy arrays
LANDMARKS_DIR  = DATASET_DIR / "landmarks"    # Phase 2 output: raw landmark .pkl files
ALPHABET_DIR   = LANDMARKS_DIR / "alphabet"   # A-Z fingerspelling landmark sequences

MODELS_DIR     = PROJECT_ROOT / "models"
RESULTS_DIR    = PROJECT_ROOT / "results"

BEST_MODEL_PATH  = MODELS_DIR / "baseline" / "best.keras"
FINAL_MODEL_PATH = MODELS_DIR / "baseline" / "final.keras"
LABEL_MAP_PATH   = MODELS_DIR / "baseline" / "label_map.json"
HISTORY_PATH     = MODELS_DIR / "baseline" / "training_history.json"


# CLIP EXTRACTION SETTINGS

MIN_CLIP_DURATION_MS = 300      # Skip annotations shorter than 0.3 seconds
FFMPEG_BINARY        = "ffmpeg" # Change to an absolute path if not on PATH
MAX_GLOSS_WORDS = 4   # annotations with more words are sentence-level → skip
TIERS_TO_PROCESS = ["words", "words_1", "words_2"]


SEQUENCE_LENGTH = 30            # Every sign is padded/trimmed to 30 frames

# Keypoint dimensionality per frame (MediaPipe Holistic):
#   pose:       33 landmarks x 4 (x, y, z, visibility) = 132
#   face:      468 landmarks x 3 (x, y, z)             = 1404
#   left hand:  21 landmarks x 3 (x, y, z)             = 63
#   right hand: 21 landmarks x 3 (x, y, z)             = 63
POSE_DIM       = 33 * 4
FACE_DIM       = 468 * 3
LEFT_HAND_DIM  = 21 * 3
RIGHT_HAND_DIM = 21 * 3
KEYPOINT_DIM   = POSE_DIM + FACE_DIM + LEFT_HAND_DIM + RIGHT_HAND_DIM  # = 1662


# LSTM TRAINING HYPERPARAMETERS

TRAIN_SPLIT      = 0.8
BATCH_SIZE       = 16
MAX_EPOCHS       = 100
EARLY_STOP_PATIENCE = 15
LR_PLATEAU_PATIENCE = 5
LR_PLATEAU_FACTOR   = 0.5
RANDOM_SEED      = 42

# Data augmentation (applied to training split only)
AUG_NOISE_SIGMA  = 0.01   # Gaussian noise std-dev added to keypoints
AUG_TIMEWARP_FACTOR = 0.1 # +/- temporal stretch factor

# ---------------------------------------------------------------------------
# PHASE 4 — STICKMAN RENDERER SETTINGS
# ---------------------------------------------------------------------------
CANVAS_WIDTH  = 640
CANVAS_HEIGHT = 480
RENDER_FPS    = 30
CANVAS_BG_COLOR = (30, 30, 30)  # Dark grey background (BGR) for visibility

# ---------------------------------------------------------------------------
# PHASE 5 — PREPROCESSOR SETTINGS
# ---------------------------------------------------------------------------
NSL_FILLER_WORDS = []  # Preserve meaning; no unvalidated grammar deletion.

MANIFEST_PATH = DATASET_DIR / "manifest.json"
CURATION_PATH = DATASET_DIR / "curation.json"
ALIAS_PATH = PROJECT_ROOT / "label_aliases.json"
MIN_HAND_COVERAGE = 0.5
MIN_REAL_FRAMES = 6
SPLIT_PATH = RESULTS_DIR / "split_manifest.json"

# English processing generates lookup candidates, not validated NSL grammar.
SPACY_MODEL = 'en_core_web_sm'
NLP_ENABLED = True
NLP_LEMMATIZE_POS = ('NOUN', 'VERB', 'ADJ', 'ADV')
# No filtering by default: preserve negation, auxiliaries, time and function words.
NLP_DROP_WORDS = ()
NLP_DROP_POS = ()
FUZZY_MATCH_ENABLED = True
FUZZY_MATCH_THRESHOLD = 85.0
# DEBUG includes user input; INFO keeps normal operation quiet.
TRANSLATION_LOG_LEVEL = 'INFO'
