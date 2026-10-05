"""Build a non-destructive, provenance-aware inventory and quality review queue."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from config import (PROJECT_ROOT, KEYPOINTS_DIR, MANIFEST_PATH, CURATION_PATH,
                    MIN_HAND_COVERAGE, MIN_REAL_FRAMES, TIERS_TO_PROCESS,
                    MIN_CLIP_DURATION_MS, RESULTS_DIR)
from dataset_utils import read_annotations, canonical_label, file_hash


def audit():
    annotation_index = defaultdict(list)
    for eaf in sorted((PROJECT_ROOT / 'annotations').glob('*.eaf')):
        for ann in read_annotations(eaf):
            annotation_index[(ann['raw_label'], ann['start_ms'])].append(ann)
    overrides = json.loads(CURATION_PATH.read_text()) if CURATION_PATH.exists() else {'excluded_clips': {}}
    records = []
    for npy in sorted(KEYPOINTS_DIR.glob('*/*.npy')):
        relative = str(npy.relative_to(KEYPOINTS_DIR).with_suffix(''))
        clip = PROJECT_ROOT / 'dataset/clips' / (relative + '.mp4')
        pkl = PROJECT_ROOT / 'dataset/landmarks' / (relative + '.pkl')
        reasons = []
        # New extractions carry an exact sidecar; legacy files are reconciled with ELAN.
        sidecar = clip.with_suffix('.json')
        if sidecar.exists():
            candidates = [json.loads(sidecar.read_text())]
        else:
            try:
                start = int(npy.stem.rsplit('_', 1)[1])
                candidates = annotation_index.get((npy.parent.name, start), [])
            except ValueError:
                candidates = []
        word_candidates = [a for a in candidates if a['tier'] in TIERS_TO_PROCESS]
        identities = {(a['source'], a['start_ms'], a['end_ms']) for a in word_candidates}
        all_intervals = {(a['source'], a['start_ms'], a['end_ms']) for a in candidates}
        if not word_candidates:
            reasons.append('no_word_tier_provenance')
        if len(identities) > 1 or len(all_intervals) > 1:
            reasons.append('ambiguous_annotation_interval')
        ann = word_candidates[0] if word_candidates else (candidates[0] if candidates else {})
        if ann and ann['end_ms'] - ann['start_ms'] < MIN_CLIP_DURATION_MS:
            reasons.append('annotation_too_short')
        if not clip.exists() or not pkl.exists():
            reasons.append('missing_paired_file')
        try:
            a = np.load(npy, allow_pickle=False)
            if a.shape != (30, 1662) or not np.isfinite(a).all():
                raise ValueError('invalid shape or nonfinite values')
            real = np.any(a != 0, axis=1)
            real_frames = int(real.sum())
            coverage = float(np.any(a[real, 1536:] != 0, axis=1).mean()) if real_frames else 0.0
            if real_frames < MIN_REAL_FRAMES:
                reasons.append('too_few_detected_frames')
            if coverage < MIN_HAND_COVERAGE:
                reasons.append('low_hand_detection')
        except Exception as error:
            reasons.append('invalid_array: ' + str(error))
            real_frames, coverage = 0, 0.0
        if relative in overrides.get('excluded_clips', {}):
            reasons.append('manual_exclusion: ' + overrides['excluded_clips'][relative])
        records.append({'id': relative, 'label': canonical_label(npy.parent.name),
                        'raw_label': npy.parent.name, 'source': ann.get('source'),
                        'start_ms': ann.get('start_ms'), 'end_ms': ann.get('end_ms'),
                        'keypoints': str(npy.relative_to(PROJECT_ROOT)),
                        'landmarks': str(pkl.relative_to(PROJECT_ROOT)),
                        'clip': str(clip.relative_to(PROJECT_ROOT)),
                        'sha256': file_hash(npy), 'real_frames': real_frames,
                        'hand_coverage': round(coverage, 4), 'eligible': not reasons,
                        'reasons': reasons, 'linguistically_validated': False})
    # Exact duplicate arrays stay in one sample only; no replication to satisfy splits.
    seen = set()
    for row in records:
        if row['eligible']:
            if row['sha256'] in seen:
                row['eligible'] = False
                row['reasons'].append('duplicate_array')
            else:
                seen.add(row['sha256'])
    usable = [r for r in records if r['eligible']]
    summary = {'total_clips': len(records), 'raw_labels': len({r['raw_label'] for r in records}),
               'eligible_clips': len(usable), 'canonical_labels': len({r['label'] for r in usable}),
               'excluded_clips': len(records) - len(usable),
               'exclusion_reasons': dict(Counter(reason for r in records for reason in r['reasons'])),
               'note': 'Automatic technical screening only; NSL correctness requires human review.'}
    MANIFEST_PATH.write_text(json.dumps({'version': 1, 'thresholds': {
        'min_hand_coverage': MIN_HAND_COVERAGE, 'min_real_frames': MIN_REAL_FRAMES},
        'summary': summary, 'records': records}, indent=2))
    RESULTS_DIR.mkdir(exist_ok=True)
    with (RESULTS_DIR / 'dataset_review.csv').open('w', newline='') as handle:
        fields = ['id', 'label', 'source', 'hand_coverage', 'real_frames', 'eligible', 'reasons', 'clip']
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)
    by_label = defaultdict(list)
    for row in records:
        by_label[row['label']].append(row)
    with (RESULTS_DIR / 'label_review.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'label', 'raw_labels', 'total_clips', 'eligible_clips',
            'eligible_source_recordings', 'needs_more_recordings', 'review_notes'])
        writer.writeheader()
        for label, rows in sorted(by_label.items()):
            selected = [r for r in rows if r['eligible']]
            sources = {r['source'] for r in selected if r['source']}
            writer.writerow({'label': label, 'raw_labels': ', '.join(sorted({r['raw_label'] for r in rows})),
                             'total_clips': len(rows), 'eligible_clips': len(selected),
                             'eligible_source_recordings': len(sources),
                             'needs_more_recordings': len(sources) < 3,
                             'review_notes': 'Verify lexical label and sign boundaries with an NSL reviewer.'})
    (RESULTS_DIR / 'dataset_audit.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    audit()
