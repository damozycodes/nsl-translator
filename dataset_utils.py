"""Shared label normalization, provenance, and reproducible dataset loading."""
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from config import ALIAS_PATH, MANIFEST_PATH, PROJECT_ROOT


def normalise_label(value):
    return re.sub(r"[^A-Z0-9]+", "_", value.strip().upper()).strip("_")


def load_aliases():
    return json.loads(ALIAS_PATH.read_text()) if ALIAS_PATH.exists() else {}


def canonical_label(value):
    label = normalise_label(value)
    return load_aliases().get(label, label)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_annotations(path):
    """Resolve both aligned and reference annotations without changing source files."""
    root = ET.parse(path).getroot()
    times = {x.attrib['TIME_SLOT_ID']: int(x.attrib['TIME_VALUE'])
             for x in root.findall('./TIME_ORDER/TIME_SLOT') if 'TIME_VALUE' in x.attrib}
    nodes = {}
    for tier in root.findall('TIER'):
        for wrapper in tier.findall('ANNOTATION'):
            for node in wrapper:
                nodes[node.attrib['ANNOTATION_ID']] = (tier.attrib['TIER_ID'], node)

    def interval(node, seen=None):
        seen = set() if seen is None else seen
        key = node.attrib['ANNOTATION_ID']
        if key in seen:
            raise ValueError(f'Cyclic annotation reference: {key}')
        seen.add(key)
        if 'ANNOTATION_REF' in node.attrib:
            return interval(nodes[node.attrib['ANNOTATION_REF']][1], seen)
        return times[node.attrib['TIME_SLOT_REF1']], times[node.attrib['TIME_SLOT_REF2']]

    for key, (tier, node) in nodes.items():
        start, end = interval(node)
        value = node.findtext('ANNOTATION_VALUE') or ''
        yield {'source': Path(path).stem, 'tier': tier, 'annotation_id': key,
               'start_ms': start, 'end_ms': end, 'raw_label': normalise_label(value),
               'label': canonical_label(value)}


def read_manifest(path=MANIFEST_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError('Dataset manifest missing. Run: python audit_dataset.py')
    return json.loads(path.read_text())


def load_split(split, partition, root=PROJECT_ROOT):
    """Verify exact saved inputs; never silently recreate a split after data changes."""
    import numpy as np
    rows = split['partitions'][partition]
    if not rows:
        raise ValueError(f'Empty {partition} partition')
    arrays, labels = [], []
    for row in rows:
        path = root / row['keypoints']
        if file_hash(path) != row['sha256']:
            raise ValueError(f'Dataset changed since split creation: {path}')
        a = np.load(path, allow_pickle=False)
        if a.shape != (30, 1662) or not np.isfinite(a).all():
            raise ValueError(f'Invalid keypoints: {path}')
        arrays.append(a)
        labels.append(split['label_map'][row['label']])
    return np.stack(arrays).astype(np.float32), np.array(labels)
