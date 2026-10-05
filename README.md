# Nigerian Sign Language text-to-sign project

This is a **retrieval-based prototype**: English text selects recorded sign clips.
A separate **three-layer LSTM** recognizes MediaPipe keypoints. It is not a
trained text-to-sign neural translator, and it does not learn NSL grammar.
MediaPipe supplies pretrained visual feature extraction; this project does not
currently train its own CNN. Confirm that this matches your approved project scope.

## Start the application

Python 3.12 and ffmpeg are required. From this project folder:

```bash
brew install python@3.12 ffmpeg
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
python -m pip check
streamlit run streamlit_app.py
```

Before starting a fresh clone, restore the prepared dataset as described in
[dataset setup](dataset/README.md). The recordings and generated manifest are
not included in Git. Create a new virtual environment on each machine.
No `main.py` is needed.
The manifest filters recordings before use; it does not delete source files.
The app shows one ordered, downloadable video. Original recording mode preserves
facial expression and source timing; stickman mode draws pose, hands, and face
contours as an approximation. Missing words are clearly reported; incomplete
fingerspelling is never represented as a complete translation.

CLI export (no desktop window needed):

```bash
python cli_interface.py --text "Hello water" --output results/demo.mp4
python cli_interface.py --text "Hello water" --mode stickman --output results/stickman.mp4
```

`Hello water` is a playback smoke test, not an example of validated NSL grammar.
English word order is retained. No articles, negation, or conjunctions are silently
deleted. Idiom substitutions are a small English paraphrase dictionary, not validated
NSL grammar rules. Phrase matching happens before individual-word fallback.

## Files and workflow

| File | Purpose |
|---|---|
| `config.py` | Paths, extraction tiers, quality thresholds |
| `extract_clips.py` | ELAN word tiers → clips with provenance sidecars |
| `extract_keypoints.py` | Clips → fixed-length training arrays and full-motion landmarks |
| `audit_dataset.py` | Technical screening, label normalization, review queue |
| `dataset_utils.py` | Shared provenance, normalization and fingerprint checks |
| `prepare_splits.py` | Frozen train/validation/test partitions by source video |
| `train_lstm.py` | Recognition training; importing it does not start training |
| `translation.py` | Preprocessing, deterministic retrieval, honest coverage |
| `stickman_renderer.py` | Pose/hand/face contour rendering with duration preservation |
| `playback.py` | Sequential browser-compatible H.264 video export |
| `cli_interface.py` / `streamlit_app.py` | User interfaces |
| `register_letter.py` | Register a reviewed fingerspelling recording |
| `prepare_review.py` | Render a study set and record unavailable sentences |
| `evaluation.py` | Recognition metrics, vocabulary coverage, human review template |
| `tests/test_pipeline.py` | Regression tests for translation and training data integrity |

## Dataset preparation and curation

The original dataset has 7,888 clip/keypoint/landmark triplets and 1,436 raw labels.
Counts after screening are written to `results/dataset_audit.json`.

```bash
python extract_clips.py                         # all configured recording pairs
python extract_clips.py annotations/session.eaf videos/session.mp4
python extract_keypoints.py
python audit_dataset.py
```

Extraction uses only `words`, `words_1`, and `words_2` tiers. Exact duplicate
annotations are collapsed. New filenames contain the recording name, start and
end times, so recordings cannot overwrite one another. Legacy filenames remain
usable through annotation reconciliation. New files may coexist with legacy files;
run the audit again before training or using updated data.

Keypoint extraction resets temporal tracking between clips. New landmark payloads
retain every captured frame and source FPS; only LSTM inputs are normalized to
30 frames. Legacy landmark files remain supported, but missing sampled motion
cannot be reconstructed. Re-extraction upgrades legacy files and can take hours;
it is not necessary to play the original videos.

`dataset/manifest.json` screens shapes, nonfinite values, matching files, provenance,
short sequences, exact duplicate arrays, and hand detection. The default minimum
is 6 non-padding frames with either hand detected in at least 50% of those frames.
These thresholds are technical heuristics, **not certification of a correct sign**.
Inspect `results/dataset_review.csv`, comparing each clip against its annotation.
`results/label_review.csv` lists independent recording counts and labels needing
additional examples.
Exclude a reviewed clip by adding its ID and reason to
`dataset/curation.json` → `excluded_clips`, then rerun the audit.

`label_aliases.json` corrects two apparent spelling errors: `ALWYAS` → `ALWAYS` and
`ANNIMAL` → `ANIMAL`. It does not rename or delete original files. Other suspected
label errors and synonym choices require annotation review; never assume signs
are interchangeable simply because their English labels look similar.

## Training and independent evaluation

```bash
python prepare_splits.py
python train_lstm.py --run-dir models/baseline
python evaluation.py --run-dir models/baseline
```

If `results/split_manifest.json` already exists, reuse it and skip the preparation
command. Use a new output path only when intentionally creating a new experiment.

The split is **source-recording-disjoint**. Every included class occurs in train,
validation, and test. Labels without sufficient independent recordings are excluded
from recognition experiments but may remain available for retrieval. Never duplicate
clips to satisfy class counts. Signer identity is not recorded, so this does not
establish signer-independent performance.

Partitions, exact paths, and file hashes are frozen in `results/split_manifest.json`.
Training copies the manifest into its run folder and rejects changed inputs.
Existing splits and nonempty experiment folders are not silently overwritten.
For new data, create a new experiment:

```bash
python prepare_splits.py --output results/split_v2.json
python train_lstm.py --split results/split_v2.json --run-dir models/experiment2
```

Only the training partition is augmented. Missing body parts, visibility, and zero
padding are preserved; the network masks padded frames. Both early stopping and
checkpoint selection use validation loss. Training never loads the test partition.
Evaluate the final selected model once on the test partition; use validation for
model development. Do not tune against test results.

The model is saved as `best.keras`; its label map, split, training history, epoch
log, and run metadata live beside it. Evaluation writes class-wise precision,
recall, F1, accuracy, confusion matrices, and training curves. Coverage reports
count source tokens correctly for multiword signs and distinguish missing output
from playable fingerspelling. Coverage is not translation accuracy.

For Colab, upload the project scripts and curated keypoints, install training
requirements, change into the project folder, and run the same commands. Keep the
saved split and run directory with the model when downloading results.

## Fingerspelling and linguistic validation still require recordings/people

The local implementation includes 20 recorded letters and 26 illustrated letter
cards. Recordings for H, K, Q, W, X, and Z are still unavailable. Recorded letters
are part of the separately transferred dataset; illustrated cards and their
source attribution are included in the repository.
After a fluent NSL signer verifies an extracted letter recording:

```bash
python register_letter.py A path/to/reviewed_A.pkl --reviewer reviewer-01
```

Repeat for A–Z. A word label named `A` is not evidence of a fingerspelled letter.
Digits currently need direct recorded vocabulary entries; unavailable digits are
reported rather than dropped.

Run `python prepare_review.py --mode source` to export available study videos
and a review sheet that also records unavailable sentences. Use `--mode stickman`
for a separate skeleton study. Prepared source videos are in `results/review_study/`.

Run `python evaluation.py` to create `results/human_review.csv`. Follow
`HUMAN_VALIDATION.md` for an intelligibility study. NSL grammar changes should be
based on that review, not improvised English word reordering.

## Verification

```bash
python -m unittest discover -s tests -v
python -m compileall -q -x 'venv|dataset|.runtime' .
python -m pip check
```

Tests cover phrase preservation, negation, idiom boundaries, missing letters,
coverage accounting, aliases, group isolation, changed-file rejection,
augmentation, and reference annotation timing.

## English NLP and gloss matching

The shared translation flow is now:

```text
English input
  → existing contraction/idiom handling
  → spaCy English tokenization, POS tagging and lemmatization
  → configurable candidate rules (no word deletion or reordering by default)
  → shared canonical_label normalization
  → longest exact surface phrase, then exact lemma phrase
  → RapidFuzz string matching for the next unmatched candidate
  → existing unknown/fingerspelling fallback
  → the same EAF-derived clip/landmark paths and source timestamps
  → existing video assembly
```

The original `TextPreprocessor.preprocess()` interface still returns a list of
string-compatible values. These carry surface/POS information so a known exact
recording such as HOW_ARE_YOU or CHILDREN wins over a transformed alternative.
Plain string lists passed to `SignVocabulary.resolve_tokens()` remain supported.
Fuzzy matching handles one remaining candidate at a time; it does not greedily
merge neighbouring words into an approximate phrase. Exact phrases retain their
existing longest-match priority. Duplicate recordings remain in the vocabulary;
the existing quality-ranked first recording is used.

### Installation

```sh
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

spaCy 3.7.5 is pinned to retain compatibility with this project's NumPy 1.26 stack.
The English model is cached in memory after first use; there are no runtime
network requests. A missing model produces an actionable setup error rather than
silently pretending that lemmatization ran. `NLP_ENABLED = False` explicitly
selects the lightweight legacy-style tokenizer.

### Configuration

All defaults are in `config.py`:

- `SPACY_MODEL = 'en_core_web_sm'`
- `NLP_ENABLED = True`
- `NLP_LEMMATIZE_POS = ('NOUN', 'VERB', 'ADJ', 'ADV')`
- `NLP_DROP_WORDS = ()`, `NLP_DROP_POS = ()`: preserve function words by default.
- `FUZZY_MATCH_ENABLED = True`, `FUZZY_MATCH_THRESHOLD = 85.0`
- `TRANSLATION_LOG_LEVEL = 'INFO'`

`CandidateRules` in `english_nlp.py` is the extension point for future reviewed
NSL rules. For an explicit experimental filter:

```python
from translation import TextPreprocessor
from english_nlp import CandidateRules
processor = TextPreprocessor(rules=CandidateRules(
    drop_words=('the', 'are', 'to', 'their')))
print(processor.preprocess('The children are running to their houses.'))
# ['CHILD', 'RUN', 'HOUSE']
```

Without that filter the candidates are THE, CHILD, ARE, RUN, TO, THEIR, HOUSE.
Filtering is a user-configured experiment, not a validated NSL grammar rule.
Lemmatization does not encode tense, number or facial grammar into the output;
linguistic validation remains necessary. The explicit “Fingerspell every word”
mode bypasses NLP/matching and keeps the user's original spelling.

### Match metadata and logging

Sign plan entries retain `type`, `gloss`, `pkl`, `clip` and `token_count`, adding
`requested_gloss`, `matched_gloss`, `match_type`, `confidence`, `source`,
`start_ms` and `end_ms`. Missing entries keep their existing letters/missing list
and also include the rejected best candidate, its score and the threshold.
For example, HOUSES → HOUSE scores approximately 90.91 with `fuzz.ratio` when
HOUSES has no exact recording. Exact matches score 100. Scores are string
similarity, **not probabilities, semantic understanding or linguistic accuracy**.
HOUSE → HOME is not introduced as a semantic synonym.

The app displays accepted fuzzy substitutions in a short caption. For detailed
NLP and retrieval diagnostics:

```sh
venv/bin/python cli_interface.py --text 'The children are running to their houses.' --debug
```

Or configure Python logging with `logging.basicConfig(level=logging.DEBUG)` and
set the `nsl.translation` logger to DEBUG. Debug output includes entered text;
normal operation remains quiet. Rejected fuzzy results log the best candidate,
score, threshold and fallback. Threshold 85 is a starting point for evaluation,
not an empirically established optimum.

EAF parsing, interval resolution, clip extraction, training data and video assembly
are unchanged. No semantic/embedding matching or new sentence-library UI is added.

References: [spaCy linguistic features](https://spacy.io/usage/linguistic-features),
[spaCy models](https://spacy.io/usage/models/),
[RapidFuzz process API](https://rapidfuzz.github.io/RapidFuzz/Usage/process.html).

## GitHub repository contents

Track the Python application and preparation scripts, tests, dependency pins,
configuration, documentation, label aliases, curation rules, and illustrated
alphabet assets with their attribution. `.streamlit/config.toml` is shareable;
`.streamlit/secrets.toml` and local `.env` files are ignored.

The ignore rules exclude virtual environments, original recordings and annotations,
extracted datasets and their generated manifest, model checkpoints, experiment
results, generated documents, caches, and temporary files. Keep separate backups
of research data and experiment outputs. See [dataset setup](dataset/README.md)
for restoring the data needed by a fresh clone or deployment.

Before committing, review `git status --short` and `git diff --cached --stat`.
Never force-add ignored secrets or large dataset folders. GitHub preparation does
not itself deploy the app; Render configuration is included; the dataset still needs a separate
transfer. See [deployment instructions](DEPLOYMENT.md).

## Docker and Render

Deployment files are included: `Dockerfile`, `.dockerignore`, `compose.yaml`, and
`render.yaml`. Follow [DEPLOYMENT.md](DEPLOYMENT.md) to push to GitHub, connect a
Render Blueprint, and transfer the prepared dataset.
