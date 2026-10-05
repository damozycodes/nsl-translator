# NSL validation and completion checklist

The application is a recorded-sign lookup prototype. Technical checks and an LSTM
recognition score do not prove that an output sentence is grammatical or understood.

1. Ask fluent Nigerian Sign Language users/teachers to inspect the vocabulary and
   sign boundaries. Record a reviewer ID and source clip for corrections. Confirm
   spelling aliases, multiword labels, dialect/context, and fingerspelling recordings.
2. Review the low-hand-detection clips in `results/dataset_review.csv`. Compare the
   original video and skeleton; exclude badly segmented/mislabeled recordings using
   `dataset/curation.json`. Re-record or re-extract low-quality examples as needed.
3. Capture independently repeated signs across sessions and signers, especially
   excluded recognition classes. Record signer IDs explicitly before claiming
   signer-independent evaluation. Do not pad the dataset with duplicate clips.
4. Prepare a fixed set of sentences and output videos. Include names, questions,
   negation, unfamiliar words, and multiword signs. Record failures as failures,
   rather than excluding them from the study.
5. Ask preferably at least two fluent reviewers to view each video and first write
   what they understood without seeing the English prompt. Afterwards show the
   intended meaning and collect intelligibility, meaning preservation, and grammar
   ratings from 1 (poor/incorrect) to 5 (clear/correct). Compare original and stickman
   playback separately and vary presentation order.
6. Enter real ratings in `results/human_review.csv`, including reviewer IDs, fluency,
   the exact output-video path, and notes. Keep separate rows per reviewer and
   playback mode. Run `python evaluation.py --human-review results/human_review.csv`.
   Report participant counts, completed/failed outputs, disagreements and limitations;
   the generated average alone is not a complete study.
7. Implement only NSL grammar mappings verified by reviewers. Freeze and version
   any resulting rules and evaluate them on sentences not used to develop them.

Unfinished external work: verified A–Z assets, linguistic/grammar review, and any
additional recordings needed for a broader classifier. No human ratings are
fabricated or prefilled by the software.

## Adding A–Z fingerspelling

Record a fluent NSL signer demonstrating each letter in a separate video, A.mp4
through Z.mp4. Obtain permission to use the recordings. Use good lighting, a plain
background and a steady camera, keeping the fingers, hands, face and upper body
visible. Preserve the complete movement for any letter that requires motion.
Have the signer verify the alphabet appropriate to the intended NSL community;
do not relabel another sign language's alphabet as NSL.

For each letter, extract a separate review asset (this does not rewrite training data):

```sh
venv/bin/python prepare_letter.py recordings/alphabet/A.mp4 results/alphabet_review/A.pkl
```

Compare the original and generated A.preview.mp4 with a fluent NSL reviewer.
Missing or incorrect hand tracking means the clip needs better capture or extraction;
a brighter skeleton cannot reconstruct missing fingers. Once approved, register it:

```sh
venv/bin/python register_letter.py A results/alphabet_review/A.pkl --reviewer actual-reviewer-id
```

Repeat with B through Z, using real reviewer identifiers. The application discovers
registered letters on its next rerun and spells unknown words when every required
letter is available. A vocabulary word labelled A is not automatically a verified
fingerspelling example. Keep original videos as the reference for future review.

For clearer playback, select Stickman animation, enable Enlarge signer, and choose
0.5× speed before pressing Translate. The crop is fixed across each clip so camera
movement does not disguise hand movement. Slow playback repeats available frames;
it does not recover motion lost in older 30-frame extractions. Use Original recording
when judging fine finger positions or facial expression.
