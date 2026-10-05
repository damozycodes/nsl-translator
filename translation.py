"""Text normalization and deterministic retrieval; independent of rendering/ML."""
import json
import re
import string
from pathlib import Path
import contractions
from config import LANDMARKS_DIR, ALPHABET_DIR, MANIFEST_PATH, PROJECT_ROOT
from dataset_utils import canonical_label
from english_nlp import EnglishPreprocessor, CandidateRules, logger
from gloss_matching import GlossMatcher
from config import FUZZY_MATCH_THRESHOLD, FUZZY_MATCH_ENABLED

IDIOM_DICTIONARY = {
    "piece of cake":            "EASY",
    "break a leg":              "GOOD LUCK",
    "under the weather":        "SICK",
    "once in a blue moon":      "RARELY",
    "cost an arm and a leg":    "VERY EXPENSIVE",
    "hit the sack":             "SLEEP",
    "hit the books":            "STUDY",
    "spill the beans":          "TELL SECRET",
    "raining cats and dogs":    "RAINING HEAVY",
    "on cloud nine":            "VERY HAPPY",
    "call it a day":            "STOP WORK",
    "in hot water":             "IN TROUBLE",
    "let the cat out of the bag": "TELL SECRET",
    "a blessing in disguise":   "HIDDEN GOOD",
    "beat around the bush":     "AVOID TOPIC",
    "see eye to eye":           "AGREE",
    "give me a hand":           "HELP ME",
    "long time no see":         "LONG TIME NOT MEET",
}

class TextPreprocessor:
    def __init__(self, idiom_dictionary=None, filler_words=None, *, nlp=None, rules=None):
        self.idioms = IDIOM_DICTIONARY if idiom_dictionary is None else idiom_dictionary
        self.nlp = nlp if nlp is not None else EnglishPreprocessor()
        self.rules = rules if rules is not None else (CandidateRules(drop_words=filler_words)
                                                     if filler_words is not None else CandidateRules())

    def preprocess(self, sentence):
        logger.debug('Input sentence: %s', sentence)
        text = contractions.fix(sentence.replace("’", "'"))
        for idiom in sorted(self.idioms, key=len, reverse=True):
            text = re.sub(r"(?<!\w)" + re.escape(idiom) + r"(?!\w)",
                          lambda _: self.idioms[idiom].lower(), text, flags=re.IGNORECASE)
        return self.rules.generate(self.nlp.analyze(text))


class SignVocabulary:
    def __init__(self, landmarks_dir=LANDMARKS_DIR, alphabet_dir=ALPHABET_DIR,
                 manifest_path=MANIFEST_PATH, *, fuzzy_threshold=FUZZY_MATCH_THRESHOLD,
                 fuzzy_enabled=FUZZY_MATCH_ENABLED):
        self.index, self.alphabet, self.metadata = {}, {}, {}
        landmarks_dir, alphabet_dir = Path(landmarks_dir), Path(alphabet_dir)
        if not landmarks_dir.exists():
            raise FileNotFoundError(f'Landmarks missing: {landmarks_dir}')
        self.curated = manifest_path is not None and Path(manifest_path).exists()
        if self.curated:
            rows = json.loads(Path(manifest_path).read_text())['records']
            for row in sorted(rows, key=lambda r: (-r['hand_coverage'], -r['real_frames'], r['id'])):
                path = PROJECT_ROOT / row['landmarks']
                if row['eligible'] and path.exists():
                    self.index.setdefault(canonical_label(row['label']), []).append(path)
                    self.metadata[str(path)] = row
        else:
            for folder in sorted(landmarks_dir.iterdir()):
                if folder.is_dir() and folder.name != 'alphabet':
                    paths = sorted(folder.glob('*.pkl'))
                    if paths:
                        self.index.setdefault(canonical_label(folder.name), []).extend(paths)
        for p in sorted(alphabet_dir.glob('*')):
            if p.is_file() and p.suffix.lower() in {'.pkl', '.png', '.jpg', '.jpeg', '.mp4'} and p.stem.upper() in string.ascii_uppercase and len(p.stem) == 1:
                letter = p.stem.upper()
                # Prefer original media; playback selects the paired skeleton.
                if letter not in self.alphabet or p.suffix.lower() != '.pkl':
                    self.alphabet[letter] = p
        self.matcher = GlossMatcher(self.index, fuzzy_threshold, fuzzy_enabled)
        self.max_words = max((len(k.split('_')) for k in self.index), default=1)

    def fingerspell(self, tokens):
        plan = []
        for word in tokens:
            letters = [(ch, self.alphabet.get(ch)) for ch in word]
            missing = [ch for ch, path in letters if path is None]
            plan.append({'type': 'missing' if missing else 'fingerspell',
                         'gloss': word, 'letters': letters, 'missing': missing, 'token_count': 1})
        return plan

    def _sign_entry(self, match, token_count):
        key = match['matched_gloss']
        path = self.index[key][0]
        row = self.metadata.get(str(path), {})
        return {'type': 'sign', 'gloss': key, 'pkl': path, 'token_count': token_count,
                'clip': PROJECT_ROOT / row['clip'] if row else None,
                'source': row.get('source'), 'start_ms': row.get('start_ms'),
                'end_ms': row.get('end_ms'), **match}

    def resolve_tokens(self, tokens):
        plan, i = [], 0
        while i < len(tokens):
            selected = None
            # Exact surface phrases take precedence, then exact lemma phrases.
            # Fuzzy lookup never consumes neighbouring words into invented phrases.
            for use_surface in (True, False):
                for n in range(min(self.max_words, len(tokens)-i), 0, -1):
                    span = tokens[i:i+n]
                    key = '_'.join(getattr(t, 'surface', str(t)) if use_surface else str(t) for t in span)
                    match = self.matcher.exact(key)
                    if match:
                        selected = (match, n)
                        break
                if selected:
                    break
            if selected:
                match, n = selected
                plan.append(self._sign_entry(match, n))
                i += n
                continue
            match = self.matcher.match(str(tokens[i]))
            if match['matched_gloss']:
                plan.append(self._sign_entry(match, 1))
            else:
                # Keep original spelling for names and unknown inflected words.
                word = canonical_label(getattr(tokens[i], 'surface', str(tokens[i])))
                fallback = self.fingerspell([word])[0]
                fallback.update(match)
                plan.append(fallback)
            i += 1
        return plan


def plan_coverage(plan):
    total = sum(e['token_count'] for e in plan)
    direct = sum(e['token_count'] for e in plan if e['type'] == 'sign')
    spelled = sum(e['token_count'] for e in plan if e['type'] == 'fingerspell')
    return {'tokens': total, 'direct_tokens': direct, 'fingerspelled_tokens': spelled,
            'missing_tokens': total-direct-spelled,
            'direct_coverage': direct / total if total else 0,
            'playable_coverage': (direct+spelled) / total if total else 0}
