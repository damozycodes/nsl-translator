# Dataset setup

The repository contains application code and illustrated alphabet assets, but does
not distribute the research recordings, annotations, generated manifest, or
extracted motion data. Ignoring these files does not remove local copies.

## Run with an existing prepared dataset

Restore these items from the project's separate dataset backup, keeping their
paths relative to the project root:

- `dataset/clips/`: extracted sign videos.
- `dataset/landmarks/`: matching landmark files, including `alphabet/` recordings
  and their metadata.
- `dataset/manifest.json`: the manifest corresponding to those files.
- `dataset/curation.json`: tracked review exclusions; preserve any newer dataset
  backup version when restoring.

Both playback modes need the prepared vocabulary data. The GitHub repository
alone does not contain the recorded sign vocabulary. Keep the illustrated cards
and their attribution in `assets/asl_demo/` in the repository.

## Rebuild or train

Restore `videos/` and matching ELAN files under `annotations/` to re-extract the
dataset. Training and a fresh dataset audit also need `dataset/keypoints/`.
Follow the preparation commands in the main README. Do not run a fresh audit
against an incomplete runtime-only dataset; restore its matching manifest instead.

Keep model checkpoints, frozen training splits, evaluation results, original
recordings, and annotation files in a separate backed-up research archive.
They are excluded from Git, not disposable.

## Hosting

Transfer `clips/`, `landmarks/`, `manifest.json`, and `curation.json` separately
to the hosting service. Preserve the same directory structure. If a persistent
disk is mounted over the dataset directory, copy all required files onto that
disk, including the manifest and curation configuration.
