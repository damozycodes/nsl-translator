"""Render available study sentences and record unavailable outputs without hiding failures."""
import argparse
import csv
from pathlib import Path
from config import RESULTS_DIR
from evaluation import COVERAGE_SENTENCES
from translation import TextPreprocessor, SignVocabulary
from playback import export_sentence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['source', 'stickman'], default='source')
    parser.add_argument('--output-dir', type=Path, default=RESULTS_DIR / 'review_study')
    args = parser.parse_args()
    sheet = args.output_dir / f'{args.mode}_review.csv'
    if sheet.exists():
        parser.error('Review sheet already exists. Use a different output directory to preserve ratings.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pp, vocab = TextPreprocessor(), SignVocabulary()
    fields = ['sentence_id', 'english_text', 'output_video', 'playback_mode', 'output_complete',
              'understood_text', 'reviewer_id', 'fluent_nsl_user', 'intelligibility_1_to_5',
              'meaning_preservation_1_to_5', 'grammar_1_to_5', 'notes']
    with sheet.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, sentence in enumerate(COVERAGE_SENTENCES, 1):
            plan = vocab.resolve_tokens(pp.preprocess(sentence))
            missing = [e['gloss'] for e in plan if e['type'] == 'missing']
            row = {'sentence_id': i, 'english_text': sentence, 'playback_mode': args.mode,
                   'output_complete': 'no' if missing else 'yes',
                   'notes': 'Missing: ' + ', '.join(missing) if missing else ''}
            if not missing:
                output = args.output_dir / f'{i:02d}_{args.mode}.mp4'
                export_sentence(plan, output, args.mode)
                row['output_video'] = str(output.resolve())
            writer.writerow(row)
            handle.flush()
            print(f'{i:02d}: {row["output_complete"]} — {sentence}', flush=True)
    print('Review sheet:', sheet)


if __name__ == '__main__':
    main()
