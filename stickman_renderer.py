"""
stickman_renderer.py
===========================
PHASE 4 — Stickman Renderer

Loads a `.pkl` landmark sequence saved by Phase 2 and draws an animated
stickman using MediaPipe's drawing utilities. The animation approximates recorded motion; keypoint tracking and sampling
can lose details. Prefer the original recording when assessing intelligibility.

Colour coding (BGR):
    pose skeleton -> white
    left hand     -> green
    right hand    -> yellow

Outputs supported:
    - live playback window (cv2.imshow)
    - animated .gif  (for the Streamlit web UI in Phase 5b)
    - .mp4 video     (for the project report / demo)

Usage:
    python stickman_renderer.py dataset/landmarks/HELLO/HELLO_1200.pkl

Requirements:
    pip install "mediapipe==0.10.21" opencv-python numpy Pillow
    (Pinned mediapipe: versions >= 0.10.30 removed mp.solutions.)
"""

import copy
import pickle
import sys
from pathlib import Path
from config import PROJECT_ROOT  # Configure writable caches before MediaPipe imports.

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    sys.exit("[ERROR] mediapipe is not installed. "
             'Run: pip install "mediapipe==0.10.21"')

from config import CANVAS_WIDTH, CANVAS_HEIGHT, RENDER_FPS, CANVAS_BG_COLOR, LANDMARKS_DIR, CLIPS_DIR

MP_DRAWING  = mp.solutions.drawing_utils
MP_HOLISTIC = mp.solutions.holistic


class StickmanRenderer:
    """
    Draws animated stickman figures from pickled MediaPipe Holistic
    landmark sequences (the .pkl files produced by Phase 2).

    Each .pkl file is a list of 30 frames; each frame is either
        {"pose": ..., "face": ..., "left_hand": ..., "right_hand": ...}
    (values are NormalizedLandmarkList protos, or None if that body part
    was not detected) or None for zero-padding frames on short clips.
    """

    def __init__(self, canvas_height: int = CANVAS_HEIGHT,
                 canvas_width: int = CANVAS_WIDTH,
                 fps: int = RENDER_FPS, close_up: bool = True):
        """
        Args:
            canvas_height: output frame height in pixels.
            canvas_width:  output frame width in pixels.
            fps:           playback / export frame rate.
        """
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.fps = fps
        self.close_up = close_up

        # Drawing specs — thickness/radius tuned for a 640x480 canvas.
        # NOTE: colours are BGR (OpenCV convention), not RGB.
        self._pose_spec = MP_DRAWING.DrawingSpec(
            color=(255, 255, 255), thickness=2, circle_radius=2)   # white
        self._left_hand_spec = MP_DRAWING.DrawingSpec(
            color=(0, 255, 0), thickness=3, circle_radius=2)       # green
        self._right_hand_spec = MP_DRAWING.DrawingSpec(
            color=(0, 230, 255), thickness=3, circle_radius=2)     # yellow

    # ------------------------------------------------------------------
    # CORE DRAWING
    # ------------------------------------------------------------------
    def _blank_canvas(self) -> np.ndarray:
        """Return a fresh dark-grey canvas (better sign visibility than
        pure black)."""
        canvas = np.empty((self.canvas_height, self.canvas_width, 3),
                          dtype=np.uint8)
        canvas[:] = CANVAS_BG_COLOR
        return canvas

    def draw_frame(self, landmark_pkl_frame, source_aspect=None) -> np.ndarray:
        """
        Draw a single stickman frame onto a blank canvas.

        Args:
            landmark_pkl_frame: one frame from a Phase 2 .pkl — a dict of
                landmark objects, or None (zero-padding frame).

        Returns:
            np.ndarray (H, W, 3) BGR image. Padding frames / undetected
            body parts simply render as the empty background.
        """
        canvas = self._blank_canvas()

        if landmark_pkl_frame is None:          # zero-padding frame
            return canvas

        # Normalized landmarks inherit the recording's aspect ratio. Drawing
        # portrait footage across a landscape canvas stretches hands and faces.
        if source_aspect and source_aspect > 0:
            width = min(self.canvas_width, round(self.canvas_height * source_aspect))
            height = min(self.canvas_height, round(self.canvas_width / source_aspect))
            canvas = np.full((max(1, height), max(1, width), 3), CANVAS_BG_COLOR, dtype=np.uint8)

        pose = landmark_pkl_frame.get("pose")
        if pose is not None:
            MP_DRAWING.draw_landmarks(
                canvas, pose, {(a, b) for a, b in MP_HOLISTIC.POSE_CONNECTIONS if a <= 16 and b <= 16},
                landmark_drawing_spec=None,
                connection_drawing_spec=self._pose_spec)

        # MediaPipe's default dots have white borders which merge over small
        # fingers. Draw compact, antialiased joints without those borders.
        for name, spec in [('left_hand', self._left_hand_spec),
                           ('right_hand', self._right_hand_spec)]:
            hand = landmark_pkl_frame.get(name)
            if hand is None:
                continue
            height, width = canvas.shape[:2]
            points = {i: (round(p.x * (width-1)), round(p.y * (height-1)))
                      for i, p in enumerate(hand.landmark)
                      if np.isfinite(p.x) and np.isfinite(p.y)
                      and 0 <= p.x <= 1 and 0 <= p.y <= 1}
            for a, b in MP_HOLISTIC.HAND_CONNECTIONS:
                if a in points and b in points:
                    cv2.line(canvas, points[a], points[b], spec.color, 1, cv2.LINE_AA)
            for point in points.values():
                cv2.circle(canvas, point, 2, spec.color, -1, cv2.LINE_AA)

        face = landmark_pkl_frame.get('face')
        if face is not None:
            spec = MP_DRAWING.DrawingSpec(color=(160, 190, 210), thickness=1, circle_radius=1)
            MP_DRAWING.draw_landmarks(canvas, face, mp.solutions.face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None, connection_drawing_spec=spec)
        output = self._blank_canvas()
        height, width = canvas.shape[:2]
        x, y = (self.canvas_width-width)//2, (self.canvas_height-height)//2
        output[y:y+height, x:x+width] = canvas
        return output

    # ------------------------------------------------------------------
    # SEQUENCE HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _load_pkl(pkl_path) -> list:
        """
        Load a .pkl landmark sequence and drop trailing padding frames
        (None entries), so exports don't end with dead frames.

        Raises:
            FileNotFoundError: if pkl_path does not exist.
        """
        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            raise FileNotFoundError(f"Landmark file not found: {pkl_path}")
        with open(pkl_path, "rb") as f:
            frames = pickle.load(f)
        if isinstance(frames, dict):
            frames = frames["frames"]
        return [fr for fr in frames if fr is not None]

    @staticmethod
    def _close_up_frames(frames, aspect):
        """Use one camera window for the whole clip, preserving motion and aspect."""
        points = []
        for frame in frames:
            for name, landmarks in (frame or {}).items():
                if landmarks is None:
                    continue
                for i, point in enumerate(landmarks.landmark):
                    # Lower-body landmarks are unnecessary for the signing view.
                    if name == 'pose' and (i > 24 or point.visibility < .5):
                        continue
                    if np.isfinite(point.x) and np.isfinite(point.y) and 0 <= point.x <= 1 and 0 <= point.y <= 1:
                        points.append((point.x, point.y))
        if not points:
            return frames, aspect
        points = np.asarray(points)
        low, high = points.min(axis=0), points.max(axis=0)
        size = np.maximum(high - low, .05) * 1.18
        low = (high + low - size) / 2
        transformed = copy.deepcopy(frames)
        for frame in transformed:
            for landmarks in (frame or {}).values():
                if landmarks is not None:
                    for point in landmarks.landmark:
                        point.x = (point.x - low[0]) / size[0]
                        point.y = (point.y - low[1]) / size[1]
        return transformed, (aspect or 1) * size[0] / size[1]

    def render_sequence(self, pkl_path) -> list:
        """Resample playback to the capture duration (legacy clips use paired video)."""
        with Path(pkl_path).open('rb') as handle:
            payload = pickle.load(handle)
        aspect = None
        if isinstance(payload, dict):
            frames = payload['frames']
            duration = len(frames) / payload['fps']
            if payload.get('width') and payload.get('height'):
                aspect = payload['width'] / payload['height']
        else:
            frames = [fr for fr in payload if fr is not None]
            duration = len(frames) / self.fps
            try:
                clip = CLIPS_DIR / Path(pkl_path).relative_to(LANDMARKS_DIR).with_suffix('.mp4')
                if clip.exists():
                    cap = cv2.VideoCapture(str(clip))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    if width > 0 and height > 0:
                        aspect = width / height
                    if fps > 0 and count > 0:
                        duration = count / fps
                    cap.release()
            except ValueError:
                pass
        if not frames:
            return []
        if self.close_up and aspect:
            frames, aspect = self._close_up_frames(frames, aspect)
        indices = np.linspace(0, len(frames)-1, max(1, round(duration * self.fps))).astype(int)
        return [self.draw_frame(frames[i], aspect) for i in indices]

    # ------------------------------------------------------------------
    # PLAYBACK & EXPORT
    # ------------------------------------------------------------------
    def play_sequence(self, pkl_path, window_name: str = "NSL Sign") -> None:
        """
        Play a landmark sequence in an OpenCV window at self.fps.
        Press Q to stop playback early.
        """
        delay_ms = max(1, int(1000 / self.fps))
        for frame in self.render_sequence(pkl_path):
            cv2.imshow(window_name, frame)
            if cv2.waitKey(delay_ms) & 0xFF in (ord("q"), ord("Q")):
                break
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)   # macOS quirk: lets the window actually close

    def sequence_to_gif(self, pkl_path, output_path) -> Path:
        """
        Save the stickman animation as an animated .gif (used by the
        Streamlit interface, which displays GIFs natively).

        Returns:
            Path to the written .gif file.
        """
        from PIL import Image   # Pillow

        frames = self.render_sequence(pkl_path)
        if not frames:
            raise ValueError(f"No drawable frames in {pkl_path}")

        # OpenCV frames are BGR; PIL expects RGB.
        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                      for f in frames]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / self.fps),   # ms per frame
            loop=0,                          # loop forever
        )
        return output_path

    def sequence_to_video(self, pkl_path, output_path) -> Path:
        """
        Save the stickman animation as an .mp4 video.

        Returns:
            Path to the written .mp4 file.
        """
        frames = self.render_sequence(pkl_path)
        if not frames:
            raise ValueError(f"No drawable frames in {pkl_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.canvas_width, self.canvas_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for "
                               f"{output_path}")
        for frame in frames:
            writer.write(frame)
        writer.release()
        return output_path


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------
def test_renderer(pkl_path) -> None:
    """
    Quick sanity check: play one sign live, then export it as .gif and
    .mp4 next to the project's results folder.

    Args:
        pkl_path: path to any .pkl produced by Phase 2, e.g.
                  dataset/landmarks/HELLO/HELLO_1200.pkl
    """
    renderer = StickmanRenderer()
    pkl_path = Path(pkl_path)

    print(f"[TEST] Playing {pkl_path.name} "
          f"({renderer.canvas_width}x{renderer.canvas_height} @ "
          f"{renderer.fps} fps). Press Q to skip.")
    try:
        renderer.play_sequence(pkl_path)
    except cv2.error:
        print("[TEST] No display available — skipping live playback "
              "(exports below still work).")

    gif = renderer.sequence_to_gif(pkl_path, f"results/{pkl_path.stem}.gif")
    vid = renderer.sequence_to_video(pkl_path, f"results/{pkl_path.stem}.mp4")
    print(f"[TEST] GIF -> {gif}")
    print(f"[TEST] MP4 -> {vid}")
    print("[TEST] Renderer OK.")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[PHASE 4] NSL Stickman Renderer")
    if len(sys.argv) != 2:
        sys.exit("Usage: python stickman_renderer.py "
                 "path/to/sign.pkl\n"
                 "e.g.   python stickman_renderer.py "
                 "dataset/landmarks/HELLO/HELLO_1200.pkl")
    test_renderer(sys.argv[1])
