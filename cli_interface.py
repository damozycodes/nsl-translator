"""CLI translator. Use --text and --output for noninteractive video export."""
import argparse
import logging
from pathlib import Path
from translation import TextPreprocessor, SignVocabulary, plan_coverage


def translate(text, output, mode):
    from playback import export_sentence
    vocab = SignVocabulary()
    plan = vocab.resolve_tokens(TextPreprocessor().preprocess(text))
    print('Plan:', ' → '.join(e['gloss'] for e in plan))
    print('Coverage:', plan_coverage(plan))
    missing = [e['gloss'] for e in plan if e['type'] == 'missing']
    if missing:
        print('Cannot produce a complete translation. Missing signs/alphabet:', ', '.join(missing))
        return False
    if not plan:
        print('Enter some text to translate.')
        return False
    print('Saved:', export_sentence(plan, Path(output), mode=mode))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text')
    parser.add_argument('--output', default='results/translation.mp4')
    parser.add_argument('--mode', choices=['source', 'stickman'], default='source')
    parser.add_argument('--debug', action='store_true', help='Log NLP and string-matching decisions')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.debug:
        logging.getLogger('nsl.translation').setLevel(logging.DEBUG)
    if args.text is not None:
        raise SystemExit(0 if translate(args.text, args.output, args.mode) else 2)
    while True:
        try:
            text = input('English text (Q to quit): ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() == 'q':
            break
        if text:
            translate(text, args.output, args.mode)


if __name__ == '__main__':
    main()
