"""Extract one alphabet video for review, without changing the training dataset."""
import argparse
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('output', type=Path, help='New .pkl file for review')
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error('Video does not exist')
    if args.output.suffix != '.pkl' or args.output.exists():
        parser.error('Choose a new .pkl output path')
    from extract_keypoints import process_clip, MP_HOLISTIC, HOLISTIC_KWARGS
    from stickman_renderer import StickmanRenderer
    with MP_HOLISTIC.Holistic(**HOLISTIC_KWARGS) as holistic:
        _, payload = process_clip(args.video, holistic)
    if not payload or not any(f.get('left_hand') or f.get('right_hand') for f in payload['frames']):
        parser.error('No hands detected; check framing and lighting')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        pickle.dump(payload, handle)
    preview = args.output.with_suffix('.preview.mp4')
    StickmanRenderer().sequence_to_video(args.output, preview)
    print(f'Extracted: {args.output}\nReview original video and preview: {preview}')
    print('Register only after a fluent NSL signer verifies the letter and rendered motion.')


if __name__ == '__main__':
    main()
