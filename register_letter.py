"""Register an NSL fingerspelling recording AFTER a fluent signer reviews it.
Example: python register_letter.py A path/to/letter.pkl --reviewer reviewer-id
Do not treat a word gloss named A as evidence of the fingerspelled letter A.
"""
import argparse
import json
import shutil
import string
from pathlib import Path
from config import ALPHABET_DIR
from dataset_utils import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('letter',choices=list(string.ascii_uppercase))
    parser.add_argument('recording',type=Path)
    parser.add_argument('--reviewer',required=True,help='ID of the fluent NSL reviewer who verified this letter')
    args=parser.parse_args()
    if not args.recording.is_file() or args.recording.suffix.lower() not in {'.pkl', '.png', '.jpg', '.jpeg', '.mp4'}:
        parser.error('recording must be .pkl, .png, .jpg, .jpeg or .mp4')
    if not args.reviewer.strip():
        parser.error('reviewer must not be blank')
    if args.recording.suffix.lower() == '.pkl':
        from stickman_renderer import StickmanRenderer
        frames = StickmanRenderer._load_pkl(args.recording)
        if not frames or not any(f.get('left_hand') or f.get('right_hand') for f in frames):
            parser.error('recording has no detected hand landmarks')
    elif args.recording.suffix.lower() == '.mp4':
        import cv2
        cap = cv2.VideoCapture(str(args.recording))
        ok, _ = cap.read()
        cap.release()
        if not ok:
            parser.error('Video cannot be decoded')
    else:
        if args.letter in {'J', 'Z'}:
            parser.error('J and Z require a motion recording, not a static image')
        from PIL import Image
        with Image.open(args.recording) as image:
            image.verify()
    ALPHABET_DIR.mkdir(parents=True,exist_ok=True)
    destination=ALPHABET_DIR/f'{args.letter}{args.recording.suffix.lower()}'
    if any(ALPHABET_DIR.glob(f'{args.letter}.*')):
        parser.error(f'{destination} already exists; preserve earlier reviewed recordings')
    shutil.copyfile(args.recording,destination)
    destination.with_suffix('.json').write_text(json.dumps({'letter':args.letter,
        'reviewer':args.reviewer,'source':str(args.recording.resolve()),'sha256':file_hash(destination)},indent=2))
    print('Registered',destination)


if __name__=='__main__':
    main()
