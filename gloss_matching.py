"""Exact then fuzzy STRING similarity over available glosses; no semantics."""
from rapidfuzz import process, fuzz
from config import FUZZY_MATCH_ENABLED, FUZZY_MATCH_THRESHOLD
from dataset_utils import canonical_label
from english_nlp import logger


class GlossMatcher:
    def __init__(self, glosses, threshold=FUZZY_MATCH_THRESHOLD, fuzzy_enabled=FUZZY_MATCH_ENABLED):
        if not 0 <= threshold <= 100:
            raise ValueError('Fuzzy threshold must be between 0 and 100')
        self.glosses = tuple(sorted({canonical_label(g) for g in glosses}))
        self.exact_glosses = set(self.glosses)
        self.threshold, self.fuzzy_enabled = threshold, fuzzy_enabled

    def exact(self, requested):
        key = canonical_label(requested)
        if key in self.exact_glosses:
            logger.debug('%s -> exact match', key)
            return dict(requested_gloss=key, matched_gloss=key, match_type='exact', confidence=100.0)
        return None

    def match(self, requested):
        key = canonical_label(requested)
        exact = self.exact(key)
        if exact:
            return exact
        best = process.extractOne(key, self.glosses, scorer=fuzz.ratio) if self.fuzzy_enabled and key else None
        if best and best[1] >= self.threshold:
            logger.debug('%s -> %s; match=fuzzy; score=%.2f; threshold=%.2f', key, best[0], best[1], self.threshold)
            return dict(requested_gloss=key, matched_gloss=best[0], match_type='fuzzy', confidence=float(best[1]))
        logger.debug('%s -> no acceptable match; best=%s; score=%.2f; threshold=%.2f; unknown/fingerspelling fallback',
                     key, best[0] if best else None, best[1] if best else 0, self.threshold)
        return dict(requested_gloss=key, matched_gloss=None, match_type='unknown', confidence=0.0,
                    best_fuzzy_candidate=best[0] if best else None, best_fuzzy_score=float(best[1]) if best else None,
                    threshold=self.threshold)
