"""
Feature Extraction: MediaPipe Holistic Keypoints

Processes every per-sign clip produced by extract_clips through MediaPipe
Holistic and saves the result in TWO formats per clip:

  1. dataset/keypoints/{GLOSS}/{clip_name}.npy
        numpy array, shape (30, 1662) — training input for the LSTM.
  2. dataset/landmarks/{GLOSS}/{clip_name}.pkl
        raw MediaPipe landmark objects per frame — used by the Phase 4
        stickman renderer (which needs landmark structure, not flat
        vectors, to draw body/hand connections).

Keypoint layout per frame (total 1662 values):
    pose:       33 landmarks x (x, y, z, visibility) = 132
    face:      468 landmarks x (x, y, z)             = 1404
    left hand:  21 landmarks x (x, y, z)             = 63
    right hand: 21 landmarks x (x, y, z)             = 63

Usage:
    python extract_keypoints.py            # process all clips
    python extract_keypoints.py --viz PATH # heatmap a saved .npy

"""

import pickle
import json
import sys
from pathlib import Path
from config import PROJECT_ROOT  # Sets a writable Matplotlib cache before MediaPipe imports.

import cv2
import numpy as np
from tqdm import tqdm

try:
    import mediapipe as mp
except ImportError:
    sys.exit("[ERROR] mediapipe is not installed. Run: pip install mediapipe")

from config import (CLIPS_DIR, KEYPOINTS_DIR, LANDMARKS_DIR,
                    SEQUENCE_LENGTH, KEYPOINT_DIM,
                    POSE_DIM, FACE_DIM, LEFT_HAND_DIM, RIGHT_HAND_DIM)

# CONSTANTS

MP_HOLISTIC = mp.solutions.holistic

# model_complexity=1 balances accuracy vs speed; set to 2 for max accuracy
# (much slower on CPU) or 0 for a quick smoke-test pass.
HOLISTIC_KWARGS = dict(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# If True, clips whose outputs already exist are skipped, so an
# interrupted run can be resumed without redoing hours of work.
SKIP_ALREADY_PROCESSED = True



# KEYPOINT FLATTENING

def extract_keypoints(results) -> np.ndarray:
    """
    Flatten one MediaPipe Holistic result into a single 1662-dim vector.

    Any body part MediaPipe failed to detect in this frame (e.g. a hand
    out of view) is replaced with zeros of the correct size, so the
    output shape is ALWAYS (1662,) regardless of detection quality.

    Args:
        results: object returned by holistic.process(frame).

    Returns:
        np.ndarray of shape (KEYPOINT_DIM,) = (1662,), dtype float32.
    """
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                         for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(POSE_DIM)

    if results.face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(FACE_DIM)

    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(LEFT_HAND_DIM)

    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(RIGHT_HAND_DIM)

    return np.concatenate([pose, face, lh, rh]).astype(np.float32)



# SEQUENCE LENGTH NORMALISATION

def normalise_sequence_length(keypoint_frames: list, landmark_frames: list):
    """
    Force a clip to exactly SEQUENCE_LENGTH (30) frames.

    - Shorter clips: zero-pad the keypoint array at the end; pad the
      landmark list with None (renderer treats None as an empty frame).
    - Longer clips: uniformly SAMPLE 30 frame indices across the whole
      clip instead of chopping at frame 30. Uniform sampling preserves
      the complete motion arc of the sign — a hard cut would throw away
      the entire second half of slower signs.

    The SAME indices are applied to both lists so the .npy used for
    training and the .pkl used for rendering stay frame-aligned.

    Args:
        keypoint_frames: list of (1662,) np.ndarrays, one per video frame.
        landmark_frames: list of per-frame landmark dicts (same length).

    Returns:
        (np.ndarray of shape (30, 1662), list of 30 landmark dicts/None)
    """
    n = len(keypoint_frames)

    if n == 0:
        return None, None  # unreadable clip; caller records the failure

    if n >= SEQUENCE_LENGTH:
        # Evenly spaced indices from first to last frame, inclusive.
        idx = np.linspace(0, n - 1, SEQUENCE_LENGTH).astype(int)
        kp = np.stack([keypoint_frames[i] for i in idx])
        lm = [landmark_frames[i] for i in idx]
    else:
        pad = SEQUENCE_LENGTH - n
        kp = np.concatenate([
            np.stack(keypoint_frames),
            np.zeros((pad, KEYPOINT_DIM), dtype=np.float32),
        ])
        lm = landmark_frames + [None] * pad

    return kp, lm



# PER-CLIP PROCESSING

def process_clip(clip_path: Path, holistic) -> tuple:
    """
    Run one video clip through MediaPipe Holistic frame by frame.

    Args:
        clip_path: path to a .mp4 clip from Phase 1.
        holistic:  an initialised MP_HOLISTIC.Holistic instance
                   (reused across clips for speed).

    Returns:
        (keypoint_array (30, 1662), versioned full-frame landmark payload)
        or (None, None) if the clip could not be read.
    """
    holistic.reset()  # Reset tracking between unrelated clips.
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if not cap.isOpened():
        cap.release()
        return None, None

    keypoint_frames, landmark_frames = [], []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # MediaPipe expects RGB; OpenCV delivers BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False          # minor perf win
        results = holistic.process(rgb)

        keypoint_frames.append(extract_keypoints(results))
        # Store the raw landmark objects the stickman renderer needs.
        landmark_frames.append({
            "pose": results.pose_landmarks,
            "face": results.face_landmarks,
            "left_hand": results.left_hand_landmarks,
            "right_hand": results.right_hand_landmarks,
        })

    cap.release()
    keypoints, _ = normalise_sequence_length(keypoint_frames, landmark_frames)
    if keypoints is None:
        return None, None
    return keypoints, {"version": 2, "fps": fps, "width": width,
                       "height": height, "frames": landmark_frames}


def process_all_clips() -> None:
    """
    Walk dataset/clips/{GLOSS}/*.mp4, extract keypoints for every clip,
    and save the .npy / .pkl pair. Prints progress and a per-class
    summary at the end.
    """
    if not CLIPS_DIR.exists():
        sys.exit(f"[ERROR] Clips folder not found: {CLIPS_DIR}\n"
                 "Run extract_clips.py first.")

    gloss_dirs = sorted(d for d in CLIPS_DIR.iterdir() if d.is_dir())
    if not gloss_dirs:
        sys.exit(f"[ERROR] No gloss folders inside {CLIPS_DIR}.")

    all_clips = [(g.name, c) for g in gloss_dirs
                 for c in sorted(g.glob("*.mp4"))]
    print(f"[PHASE 2] {len(all_clips)} clips across "
          f"{len(gloss_dirs)} gloss classes.")

    per_class_counts = {}
    failed, skipped = [], 0

    # ONE Holistic instance reused for every clip — re-initialising the
    # model per clip would multiply runtime several times over.
    with MP_HOLISTIC.Holistic(**HOLISTIC_KWARGS) as holistic:
        for gloss, clip_path in tqdm(all_clips, desc="Extracting keypoints",
                                     unit="clip"):
            npy_path = KEYPOINTS_DIR / gloss / f"{clip_path.stem}.npy"
            pkl_path = LANDMARKS_DIR / gloss / f"{clip_path.stem}.pkl"

            metadata_path = pkl_path.with_suffix(".json")
            if (SKIP_ALREADY_PROCESSED and npy_path.exists() and pkl_path.exists()
                    and metadata_path.exists()
                    and json.loads(metadata_path.read_text()).get("version") == 2):
                skipped += 1
                per_class_counts[gloss] = per_class_counts.get(gloss, 0) + 1
                continue

            try:
                keypoints, landmarks = process_clip(clip_path, holistic)
            except Exception as err:  # keep the batch alive on odd clips
                print(f"\n[CLIP ERROR] {clip_path.name}: {err}")
                failed.append(clip_path.name)
                continue

            if keypoints is None:
                failed.append(clip_path.name)
                continue

            npy_path.parent.mkdir(parents=True, exist_ok=True)
            pkl_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, keypoints)
            with open(pkl_path, "wb") as f:
                pickle.dump(landmarks, f)
            metadata_path.write_text(json.dumps({"version": 2, "fps": landmarks["fps"],
                                                "frame_count": len(landmarks["frames"])}))

            per_class_counts[gloss] = per_class_counts.get(gloss, 0) + 1

    # ----- Summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE — KEYPOINT EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"  Classes processed : {len(per_class_counts)}")
    print(f"  Clips processed   : {sum(per_class_counts.values())}"
          f"  (of which resumed/skipped: {skipped})")
    print(f"  Failed clips      : {len(failed)}")
    if failed:
        for name in failed[:10]:
            print(f"      - {name}")
        if len(failed) > 10:
            print(f"      ... and {len(failed) - 10} more")
    print(f"  .npy output -> {KEYPOINTS_DIR}")
    print(f"  .pkl output -> {LANDMARKS_DIR}")
    print("-" * 60)
    print("  Samples per class (recognition requires independent source recordings):")
    for gloss in sorted(per_class_counts):
        flag = "  <-- record independent examples" if per_class_counts[gloss] < 2 else ""
        print(f"      {gloss:<30} {per_class_counts[gloss]:>4}{flag}")
    print("=" * 60)
    print("Next step -> train_lstm.py (Google Colab)")



# VERIFICATION HELPER

def visualise_keypoints_sample(npy_path) -> None:
    """
    Plot a saved keypoint sequence as a heatmap (frames x features) so
    the extraction can be visually verified.

    What to look for:
      - Horizontal banding that CHANGES over time = real motion captured.
      - Large all-black regions in the hand columns = hands not detected
        (check clip framing / lighting).
      - All-black rows at the bottom = zero-padding on a short clip
        (expected and harmless).

    Args:
        npy_path: path to a .npy file saved by this script.
    """
    import matplotlib.pyplot as plt

    npy_path = Path(npy_path)
    if not npy_path.exists():
        print(f"[ERROR] File not found: {npy_path}")
        return

    seq = np.load(npy_path)
    print(f"Loaded {npy_path.name}: shape={seq.shape}, "
          f"min={seq.min():.3f}, max={seq.max():.3f}")

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(seq, aspect="auto", cmap="viridis",
                   interpolation="nearest")
    # Mark the boundaries between body-part blocks for readability.
    for x, label in [(POSE_DIM, "pose|face"),
                     (POSE_DIM + FACE_DIM, "face|LH"),
                     (POSE_DIM + FACE_DIM + LEFT_HAND_DIM, "LH|RH")]:
        ax.axvline(x, color="white", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Feature index (pose | face | left hand | right hand)")
    ax.set_ylabel("Frame (0-29)")
    ax.set_title(f"Keypoint sequence heatmap — {npy_path.stem}")
    fig.colorbar(im, ax=ax, label="normalised coordinate value")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    print("NSL Feature Extraction — MediaPipe Holistic keypoints")

    if len(sys.argv) == 3 and sys.argv[1] == "--viz":
        visualise_keypoints_sample(sys.argv[2])
    else:
        process_all_clips()
