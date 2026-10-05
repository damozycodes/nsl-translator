"""Freeze disjoint source-video partitions with every evaluated class in all three."""
import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from config import MANIFEST_PATH, SPLIT_PATH
from dataset_utils import read_manifest, file_hash


def make_split(records):
    usable = [r for r in records if r['eligible'] and r.get('source')]
    by_label = defaultdict(set)
    for row in usable:
        by_label[row['label']].add(row['source'])
    sources = sorted({r['source'] for r in usable})
    if len(sources) < 3:
        raise ValueError('Need at least three source recordings for a disjoint train/validation/test split.')
    n_holdout = max(1, round(len(sources)*0.2))
    # Optimize label presence only, never feature values or model results.
    # For larger corpora bound search while retaining deterministic candidates.
    best = None
    candidates = itertools.combinations(sources, n_holdout)
    for test in itertools.islice(candidates, 2000):
        remaining = sorted(set(sources)-set(test))
        for val in itertools.islice(itertools.combinations(remaining, n_holdout), 2000):
            train = set(remaining)-set(val)
            if not train:
                continue
            shared = {label for label, groups in by_label.items()
                      if groups & train and groups & set(val) and groups & set(test)}
            score = (len(shared), sum(len(by_label[k]) for k in shared))
            if best is None or score > best[0]:
                best = (score, train, set(val), set(test), shared)
    _, train, val, test, shared = best
    if not shared:
        raise ValueError('No labels have examples across three separate recordings. Collect more independent examples.')
    label_map = {label: i for i, label in enumerate(sorted(shared))}
    partitions = {}
    groups = {'train': train, 'validation': val, 'test': test}
    for name, group in groups.items():
        partitions[name] = [r for r in usable if r['source'] in group and r['label'] in shared]
    return {'version': 1, 'strategy': 'disjoint_source_video',
            'limitation': 'Recording-disjoint, not proven signer-disjoint: signer identities are unavailable.',
            'label_map': label_map, 'groups': {k: sorted(v) for k,v in groups.items()},
            'excluded_labels': sorted(set(by_label)-shared), 'partitions': partitions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=SPLIT_PATH)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'{args.output} already exists. Use a new --output path for a new experiment.')
    split = make_split(read_manifest()['records'])
    split['dataset_manifest_sha256'] = file_hash(MANIFEST_PATH)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2))
    print(json.dumps({'classes': len(split['label_map']),
                      'samples': {k: len(v) for k,v in split['partitions'].items()},
                      'groups': split['groups'], 'excluded_labels': len(split['excluded_labels'])}, indent=2))


if __name__ == '__main__':
    main()
