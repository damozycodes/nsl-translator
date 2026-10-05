"""English analysis and configurable candidate rules, independent of retrieval."""
import logging
from dataclasses import dataclass
from functools import lru_cache
from config import (SPACY_MODEL, NLP_ENABLED, NLP_LEMMATIZE_POS,
                    NLP_DROP_WORDS, NLP_DROP_POS, TRANSLATION_LOG_LEVEL)
from dataset_utils import canonical_label

logger = logging.getLogger('nsl.translation')
logger.setLevel(TRANSLATION_LOG_LEVEL)


@dataclass(frozen=True)
class EnglishToken:
    text: str
    lemma: str
    pos: str


class GlossCandidate(str):
    """String-compatible candidate retaining the surface form for exact lookup."""
    def __new__(cls, value, surface=None, pos=''):
        instance = super().__new__(cls, value)
        instance.surface = surface or value
        instance.pos = pos
        return instance


@lru_cache(maxsize=2)
def load_english_model(name):
    import spacy
    try:
        return spacy.load(name, exclude=['ner'])
    except OSError as exc:
        raise RuntimeError(f'spaCy model {name!r} is missing. Run: '
                           f'venv/bin/python -m spacy download {name}') from exc


class EnglishPreprocessor:
    def __init__(self, enabled=NLP_ENABLED, model=SPACY_MODEL):
        self.enabled, self.model = enabled, model

    def analyze(self, text):
        if not text.strip():
            return []
        if not self.enabled:
            import re
            return [EnglishToken(t, t, '') for t in re.findall(r"[a-zA-Z0-9']+", text)]
        doc = load_english_model(self.model)(text)
        return [EnglishToken(t.text, t.lemma_ or t.text, t.pos_)
                for t in doc if not t.is_space and not t.is_punct and any(c.isalnum() for c in t.text)]


class CandidateRules:
    """Extension point for reviewed NSL rules. Default retains English order."""
    def __init__(self, drop_words=NLP_DROP_WORDS, drop_pos=NLP_DROP_POS,
                 lemmatize_pos=NLP_LEMMATIZE_POS):
        self.drop_words = {canonical_label(w) for w in drop_words}
        self.drop_pos = set(drop_pos)
        self.lemmatize_pos = set(lemmatize_pos)

    def generate(self, tokens):
        candidates = []
        for token in tokens:
            surface = canonical_label(token.text.replace("'", ''))
            lemma = canonical_label(token.lemma.replace("'", ''))
            if surface in self.drop_words or token.pos in self.drop_pos:
                logger.debug('Filtered: %s (%s)', token.text, token.pos)
                continue
            value = lemma if token.pos in self.lemmatize_pos and lemma else surface
            if value:
                candidates.append(GlossCandidate(value, surface, token.pos))
                logger.debug('NLP: %s -> %s (POS=%s)', token.text, value, token.pos)
        logger.debug('Candidate glosses: %s', ', '.join(candidates))
        return candidates
