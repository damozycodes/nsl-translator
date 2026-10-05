# ASL reference demo — not validated NSL assets

Source: https://commons.wikimedia.org/wiki/File:Asl_alphabet_gallaudet.svg
Original: https://upload.wikimedia.org/wikipedia/commons/c/c8/Asl_alphabet_gallaudet.svg
Darren Stone / Ds13; derived from David Rakowski's Gallaudet-TT font;
vectorised by Marnanel. Wikimedia lists the illustration as public domain.
Retrieved 2026-09-30. source.svg preserves the downloaded original.

build.cjs renders individual viewBoxes of the vector chart as hand-only cards. Attribution is retained here and in the app sources section.
The cards are static reference illustrations. J and Z include motion arrows but
are NOT full movement demonstrations. No NSL reviewer verification is claimed.
These illustrations provide the default fallback, separate from dataset/landmarks/alphabet.
Registered NSL letter assets take priority.

Unknown words automatically use available fingerspelling assets; enable “Fingerspell every word” to spell a name or all input words.

On 2026-10-03 the project owner reported that the fingerspelling assets have been
NSL verified. This records the owner’s confirmation, not an independent review by
the coding assistant. Original source attribution remains above. The app displays
the current uppercase letter above the hand, with no source footer or sources panel.
J and Z remain static motion diagrams.

Stickman playback requires a paired A.pkl beside A.png (and likewise B–Z).
Illustrations remain the original-mode source. Both full-card and cropped-hand
MediaPipe extraction attempts on 2026-10-03 returned no detected hands for all
26 illustrations, so no skeleton coordinates were fabricated. Create paired files
from hand recordings using prepare_letter.py. These still require motion review.
