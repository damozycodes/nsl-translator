"""Create a small, reproducible runtime dataset without changing the full dataset."""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'demo-data'
WORDS = '''HELLO WATER GOOD MORNING AFTERNOON EVENING THANK_YOU PLEASE YES NO I YOU WE SHE THEY MY YOUR NAME WHAT WHERE WHEN WHO WHY HOW SCHOOL TEACHER STUDENT BOOK LEARN READ WRITE HOUSE HOME FAMILY MOTHER FATHER BROTHER SISTER FRIEND FOOD EAT DRINK WANT NEED GO COME HELP LOVE HAPPY TODAY TOMORROW YESTERDAY WORK DOCTOR CHILDREN NIGERIA DEAF SIGN LANGUAGE UNDERSTAND'''.split()


def main():
    manifest = json.loads((ROOT / 'dataset/manifest.json').read_text())
    selected = {}
    for row in sorted(manifest['records'], key=lambda r: (-r['hand_coverage'], -r['real_frames'], r['id'])):
        if row['eligible'] and row['label'] in WORDS and row['label'] not in selected:
            selected[row['label']] = row
    missing = set(WORDS) - selected.keys()
    if missing:
        raise SystemExit(f'Missing selected labels: {sorted(missing)}')
    for row in selected.values():
        for key in ('clip', 'landmarks'):
            relative = Path(row[key]).relative_to('dataset')
            destination = DEST / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / row[key], destination)
    alphabet = DEST / 'landmarks/alphabet'
    alphabet.mkdir(parents=True, exist_ok=True)
    for path in (ROOT / 'dataset/landmarks/alphabet').iterdir():
        if len(path.stem) == 1 and path.stem.isupper() and path.suffix in ('.mp4', '.pkl', '.json'):
            shutil.copy2(path, alphabet / path.name)
    # Preserve source provenance and review status; do not claim linguistic validation.
    output = {'version': manifest['version'], 'thresholds': manifest['thresholds'],
              'summary': {'eligible_clips': len(selected), 'canonical_labels': len(selected),
                          'note': 'Demonstration subset; one recording per selected label.'},
              'records': list(selected.values())}
    (DEST / 'manifest.json').write_text(json.dumps(output, indent=2) + '\n')
    shutil.copy2(ROOT / 'dataset/curation.json', DEST / 'curation.json')
    (DEST / 'README.md').write_text('# Demonstration dataset\n\n'
        'Small selection from the local NSL project dataset, intended for the lecturer demonstration.\n'
        'Generated with `python3 scripts/prepare_demo.py`. Original recordings remain unchanged.\n'
        'Manifest paths resolve after Docker copies this folder to /app/dataset.\n'
        'Training keypoint arrays are not included. Alphabet illustrations remain under assets/.\n\n'
        'Included labels: ' + ', '.join(sorted(selected)) + '\n')
    print(f'Prepared {len(selected)} labels; {sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file()) / 1024**2:.1f} MiB')


if __name__ == '__main__':
    main()
