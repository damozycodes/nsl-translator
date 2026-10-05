"""Export one ordered, browser-compatible sentence video; preserve source timing."""
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from config import FFMPEG_BINARY, PROJECT_ROOT, CANVAS_WIDTH, CANVAS_HEIGHT, RENDER_FPS


def _sources(plan, mode):
    for entry in plan:
        if entry['type'] == 'missing':
            raise ValueError(f"Missing sign/fingerspelling: {entry['gloss']}")
        if entry['type'] == 'sign':
            path = entry.get('clip') if mode == 'source' else entry['pkl']
            if not path:
                raise ValueError(f"No source recording for {entry['gloss']}; select stickman output.")
            yield Path(path), mode
        else:
            for char, path in entry['letters']:
                if path is None:
                    raise ValueError(f'Missing letter: {char}')
                path = Path(path)
                if mode == 'stickman' and path.suffix.lower() != '.pkl':
                    skeleton = path.with_suffix('.pkl')
                    if not skeleton.is_file():
                        raise ValueError(f'No finger-joint recording for letter {char}. '
                                         'Use Original recording, or add a paired landmark .pkl file for this letter.')
                    path = skeleton
                yield path, ('stickman' if path.suffix.lower() == '.pkl' else
                             'image' if path.suffix.lower() in {'.jpg', '.jpeg', '.png'} else 'source')



def cache_key(plan, mode, speed=1.0, close_up=True):
    fingerprints = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns, kind)
                    for p, kind in _sources(plan, mode)]
    # Include renderer code so updates cannot reuse obsolete animations.
    for name in ['stickman_renderer.py', 'playback.py', 'config.py']:
        fingerprints.append(hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest())
    return hashlib.sha256(json.dumps([mode, speed, close_up, fingerprints]).encode()).hexdigest()[:24]


def run_ffmpeg(args):
    result = subprocess.run([FFMPEG_BINARY, '-hide_banner', '-loglevel', 'error', '-y', *args],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])


def _letter_header(source, destination, letter):
    """Keep the current letter above recorded or skeleton footage."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'mp4v'),
                             cap.get(cv2.CAP_PROP_FPS) or RENDER_FPS, (CANVAS_WIDTH, CANVAS_HEIGHT))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError('Could not create letter video')
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 30, dtype=np.uint8)
            h, w = frame.shape[:2]
            scale = min(CANVAS_WIDTH/w, (CANVAS_HEIGHT-48)/h)
            frame = cv2.resize(frame, (max(1, round(w*scale)), max(1, round(h*scale))))
            y = 48 + (CANVAS_HEIGHT-48-frame.shape[0])//2
            x = (CANVAS_WIDTH-frame.shape[1])//2
            canvas[y:y+frame.shape[0], x:x+frame.shape[1]] = frame
            size = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.putText(canvas, letter, ((CANVAS_WIDTH-size[0])//2, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            writer.write(canvas)
            count += 1
    finally:
        cap.release()
        writer.release()
    if not count:
        raise ValueError(f'No frames for letter {letter}')
    return destination


def export_sentence(plan, output_path, mode='source', speed=1.0, close_up=True):
    if not .25 <= speed <= 2:
        raise ValueError('Playback speed must be between 0.25 and 2')
    if mode not in {'source', 'stickman'}:
        raise ValueError('Unknown rendering mode')
    sources = list(_sources(plan, mode))
    if not sources:
        raise ValueError('No signs to render')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = None
    with tempfile.TemporaryDirectory(prefix='nsl-render-') as temporary:
        temp = Path(temporary)
        for i, (source, kind) in enumerate(sources):
            letter = source.stem if len(source.stem) == 1 and source.stem in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' else None
            if kind == 'stickman':
                if renderer is None:
                    from stickman_renderer import StickmanRenderer
                    renderer = StickmanRenderer(close_up=close_up)
                source = renderer.sequence_to_video(source, temp / f'raw-{i}.mp4')
            if letter and kind != 'image':
                source = _letter_header(source, temp / f'label-{i}.mp4', letter)
            input_args = ['-loop', '1', '-framerate', str(RENDER_FPS)] if kind == 'image' else []
            duration_args = ['-t', str(1.2 / speed)] if kind == 'image' else []
            run_ffmpeg([*input_args, '-i', str(source), *duration_args, '-an', '-vf',
                        f'setpts=(PTS-STARTPTS)/{speed},scale={CANVAS_WIDTH}:{CANVAS_HEIGHT}:force_original_aspect_ratio=decrease,'
                        f'pad={CANVAS_WIDTH}:{CANVAS_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={RENDER_FPS}',
                        '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p', str(temp / f'{i}.mp4')])
        (temp / 'concat.txt').write_text(''.join(f"file '{i}.mp4'\n" for i in range(len(sources))))
        joined = temp / 'joined.mp4'
        run_ffmpeg(['-f', 'concat', '-safe', '0', '-i', str(temp / 'concat.txt'),
                    '-c', 'copy', '-movflags', '+faststart', str(joined)])
        # Atomic destination replacement, including when rendering across filesystems.
        import os
        import shutil
        with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix='.mp4', delete=False) as handle:
            staging = Path(handle.name)
        try:
            shutil.copyfile(joined, staging)
            os.replace(staging, output_path)
        finally:
            staging.unlink(missing_ok=True)
    return output_path
