"""

PHASE 1 — Data Preparation: ELAN Parsing + Video Clip Extraction

Converts ELAN `.eaf` annotation files + their source videos into labelled,
per-sign video clips ready for keypoint extraction in Phase 2.

Pipeline:
    .eaf file  --XML-->  (start_ms, end_ms, gloss) annotations
    source.mp4 --ffmpeg--> clips with recording identity and exact time boundaries

Usage (single recording):
    python extract_clips.py path/to/session.eaf path/to/session.mp4

Usage (batch, edit EAF_VIDEO_PAIRS below):
    python extract_clips.py

"""

import re
import json
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

from dataset_utils import read_annotations
from config import PROJECT_ROOT, CLIPS_DIR, MAX_GLOSS_WORDS, MIN_CLIP_DURATION_MS, FFMPEG_BINARY, TIERS_TO_PROCESS 

# list of (eaf_path, video_path) tuples for every teacher

EAF_VIDEO_PAIRS = [
    ("annotations/Media_2---VID_20250119_164025_001.eaf", "videos/Media_2---VID_20250119_164025_001.mp4"),
    ("annotations/Media_3---VID_20250119_170440_001.eaf", "videos/Media_3---VID_20250119_170440_001.mp4"),
    ("annotations/Media_5---VID_20250123_075346-1_001.eaf", "videos/Media_5---VID_20250123_075346-1_001.mp4"),
    ("annotations/Media_6---VID_20250123_080426-1.eaf", "videos/Media_6---VID_20250123_080426-1.mp4"),
    ("annotations/Media_7---VID_20250128_164242.eaf", "videos/Media_7---VID_20250128_164242.mp4"),
    ("annotations/Media_9---VIDEO 1_1.eaf", "videos/Media_9---VIDEO 1_1.mp4"),
    ("annotations/Media_11---VIDEO 2_1.eaf", "videos/Media_11---VIDEO 2_1.mp4"),
    ("annotations/Media_13---VIDEO 3_1.eaf", "videos/Media_13---VIDEO 3_1.mp4"),
    ("annotations/Media_15---VIDEO 4_1.eaf", "videos/Media_15---VIDEO 4_1.mp4"),

]


# HELPERS
def sanitise_gloss(gloss: str) -> str:
    """
    Convert a raw ELAN gloss label into a filesystem-safe folder/file name.

    Args:
        gloss: Raw annotation value from the .eaf tier, e.g. "How are you?"

    Returns:
        Uppercase label with all non-alphanumeric runs collapsed to a single
        underscore, e.g. "HOW_ARE_YOU". Returns an empty string if nothing
        usable remains.
    """
    gloss = gloss.strip().upper()
    gloss = re.sub(r"[^A-Z0-9]+", "_", gloss)   # replace unsafe chars
    gloss = gloss.strip("_")                     # no leading/trailing _
    return gloss


def parse_eaf_annotations(eaf_path: Path) -> list:
    """
    Read an ELAN .eaf file and extract every time-aligned annotation
    from configured word tiers.

    Args:
        eaf_path: Path to the .eaf annotation file.

    Returns:
        List of dicts: {"start_ms": int, "end_ms": int,
                        "gloss": str, "tier": str}
        Annotations with empty gloss values are excluded.

    Raises:
        FileNotFoundError: if the .eaf file does not exist.
    """
    if not eaf_path.exists():
        raise FileNotFoundError(f".eaf file not found: {eaf_path}")

    annotations = []
    seen = set()
    for ann in read_annotations(eaf_path):
        if ann['tier'] not in TIERS_TO_PROCESS:
            continue
        gloss = ann['label']
        if not gloss or len(gloss.split('_')) > MAX_GLOSS_WORDS:
            continue
        identity = (gloss, ann['start_ms'], ann['end_ms'])
        if identity in seen:
            continue
        seen.add(identity)
        annotations.append(dict(ann, gloss=gloss))
    return annotations



def extract_clip(video_path: Path, start_ms: int, end_ms: int,
                 output_path: Path) -> bool:
    """
    Cut a time segment out of the source video using ffmpeg.

    Re-encodes (libx264) rather than stream-copying so cuts are
    frame-accurate — stream copy can only cut on keyframes, which would
    shift sign boundaries by up to a second.

    Args:
        video_path:  Path to the full teacher recording.
        start_ms:    Segment start in milliseconds.
        end_ms:      Segment end in milliseconds.
        output_path: Destination .mp4 path (parents created by caller).

    Returns:
        True if ffmpeg succeeded and the output file exists, else False.
    """
    start_s = start_ms / 1000.0
    duration_s = (end_ms - start_ms) / 1000.0

    cmd = [
        FFMPEG_BINARY,
        "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_s:.3f}",          # accurate seek (re-encode mode)
        "-i", str(video_path),
        "-t", f"{duration_s:.3f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-an",                             # audio is irrelevant for signing
        "-y",                              # overwrite if re-running
        str(output_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[FFMPEG ERROR] {output_path.name}: "
                  f"{result.stderr.strip()[:200]}")
            return False
        return output_path.exists()
    except FileNotFoundError:
        # ffmpeg binary itself is missing
        sys.exit("[ERROR] ffmpeg not found. Install it and ensure it is "
                 "on your PATH, or set FFMPEG_BINARY in config.py.")



# MAIN PROCESSING

def process_eaf_video_pair(eaf_path, video_path) -> dict:
    """
    Extract every annotated sign clip from one (eaf, video) recording pair.

    Args:
        eaf_path:   Path (or str) to the ELAN annotation file.
        video_path: Path (or str) to the matching source video.

    Returns:
        Stats dict: {"extracted": int, "skipped_short": int,
                     "failed": int, "glosses": set}
    """
    eaf_path, video_path = Path(eaf_path), Path(video_path)
    if not eaf_path.is_absolute():
        eaf_path = PROJECT_ROOT / eaf_path
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path

    if not video_path.exists():
        raise FileNotFoundError(f"Source video not found: {video_path}")

    print(f"\n Processing: {eaf_path.name}  +  {video_path.name}")
    annotations = parse_eaf_annotations(eaf_path)
    print(f"Found {len(annotations)} annotations "
          f"across selected word tiers.")

    stats = {"extracted": 0, "skipped_short": 0, "failed": 0, "glosses": set()}

    for ann in tqdm(annotations, desc="  Cutting clips", unit="clip"):
        duration_ms = ann["end_ms"] - ann["start_ms"]

        # Edge case: annotations shorter than 0.3 s are too brief to
        # contain a full sign and would yield <10 usable frames.
        if duration_ms < MIN_CLIP_DURATION_MS:
            stats["skipped_short"] += 1
            continue

        gloss = ann["gloss"]
        out_dir = CLIPS_DIR / gloss
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{gloss}__{eaf_path.stem}__{ann['start_ms']}_{ann['end_ms']}.mp4"
        if out_path.exists() and out_path.with_suffix(".json").exists():
            continue

        if extract_clip(video_path, ann["start_ms"], ann["end_ms"], out_path):
            out_path.with_suffix(".json").write_text(json.dumps(ann, indent=2))
            stats["extracted"] += 1
            stats["glosses"].add(gloss)
        else:
            stats["failed"] += 1

    return stats


def batch_process_multiple_videos(eaf_video_pairs) -> None:
    """
    Process multiple teacher recording sessions in one run.

    Args:
        eaf_video_pairs: list of (eaf_path, video_path) tuples.
    """
    if not eaf_video_pairs:
        print("No (eaf, video) pairs supplied. Edit "
              "EAF_VIDEO_PAIRS at the top of this file or pass paths "
              "on the command line.")
        return

    totals = {"extracted": 0, "skipped_short": 0, "failed": 0,
              "glosses": set()}

    for eaf_path, video_path in eaf_video_pairs:
        try:
            stats = process_eaf_video_pair(eaf_path, video_path)
        except FileNotFoundError as err:
            print(f"[SKIPPED PAIR] {err}")
            continue
        for key in ("extracted", "skipped_short", "failed"):
            totals[key] += stats[key]
        totals["glosses"] |= stats["glosses"]

    # Final summary 
    print("\n" + "=" * 60)
    print("CLIP EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"  Total clips extracted : {totals['extracted']}")
    print(f"  Unique gloss labels   : {len(totals['glosses'])}")
    print(f"  Skipped (< {MIN_CLIP_DURATION_MS} ms)    : "
          f"{totals['skipped_short']}")
    print(f"  Failed extractions    : {totals['failed']}")
    print(f"  Output folder         : {CLIPS_DIR}")
    if totals["glosses"]:
        print(f"  Labels: {', '.join(sorted(totals['glosses']))}")
    print("=" * 60)
    print("Next step -> extract_keypoints.py")

# ENTRY POINT
if __name__ == "__main__":
    print("NSL Data Preparation ELAN parsing + clip extraction")
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) == 3:

        batch_process_multiple_videos([(sys.argv[1], sys.argv[2])])
    else:
        # Batch mode: uses the EAF_VIDEO_PAIRS constant defined above
        batch_process_multiple_videos(EAF_VIDEO_PAIRS)