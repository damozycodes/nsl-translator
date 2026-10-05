import json
import tempfile
import unittest
from pathlib import Path
from translation import TextPreprocessor, SignVocabulary
from english_nlp import CandidateRules, EnglishPreprocessor
from gloss_matching import GlossMatcher


class NLPMatchingTests(unittest.TestCase):
    def test_exact_wins_and_normalizes(self):
        match = GlossMatcher(['HOUSE', 'HOUSES']).match('houses')
        self.assertEqual((match['matched_gloss'], match['match_type'], match['confidence']), ('HOUSES', 'exact', 100))
        self.assertEqual(GlossMatcher(['HOW_ARE_YOU']).match('how are you?')['match_type'], 'exact')

    def test_fuzzy_threshold_inclusive_and_disabled(self):
        result = GlossMatcher(['HOUSE']).match('HOUSES')
        self.assertEqual(result['match_type'], 'fuzzy')
        score = result['confidence']
        self.assertEqual(GlossMatcher(['HOUSE'], threshold=score).match('HOUSES')['match_type'], 'fuzzy')
        self.assertIsNone(GlossMatcher(['HOUSE'], threshold=score+.01).match('HOUSES')['matched_gloss'])
        self.assertIsNone(GlossMatcher(['HOUSE'], fuzzy_enabled=False).match('HOUSES')['matched_gloss'])
        with self.assertRaises(ValueError):
            GlossMatcher([], threshold=101)

    def test_rejection_and_no_semantics(self):
        result = GlossMatcher(['PLANE']).match('AIRPLANE')
        self.assertIsNone(result['matched_gloss'])
        self.assertEqual(result['best_fuzzy_candidate'], 'PLANE')
        self.assertLess(result['best_fuzzy_score'], 85)
        self.assertIsNone(GlossMatcher(['HOME']).match('HOUSE')['matched_gloss'])
        self.assertIsNone(GlossMatcher([]).match('HOUSE')['matched_gloss'])

    def test_lemmas_pos_and_configurable_filtering(self):
        text = 'The children are running to their houses.'
        candidates = TextPreprocessor({}).preprocess(text)
        self.assertEqual(candidates, ['THE', 'CHILD', 'ARE', 'RUN', 'TO', 'THEIR', 'HOUSE'])
        self.assertEqual(candidates[1].surface, 'CHILDREN')
        self.assertEqual(candidates[1].pos, 'NOUN')
        filtered = TextPreprocessor({}, rules=CandidateRules(drop_words=['the','are','to','their'])).preprocess(text)
        self.assertEqual(filtered, ['CHILD', 'RUN', 'HOUSE'])
        self.assertIn('WALK', TextPreprocessor({}).preprocess('They walked past cars.'))
        self.assertIn('CAR', TextPreprocessor({}).preprocess('They walked past cars.'))

    def test_empty_punctuation_case_and_disabled_nlp(self):
        pp = TextPreprocessor({})
        self.assertEqual(pp.preprocess(''), [])
        self.assertEqual(pp.preprocess('...!?'), [])
        self.assertEqual(pp.preprocess('hello!'), pp.preprocess('HELLO!'))
        legacy = TextPreprocessor({}, nlp=EnglishPreprocessor(enabled=False))
        self.assertEqual(legacy.preprocess('children running houses'), ['CHILDREN','RUNNING','HOUSES'])

    def test_existing_phrase_before_lemma_or_fuzzy_and_unknown_spelling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ['HOW_ARE_YOU', 'HOUSE', 'CHILD', 'RUN']:
                (root/name).mkdir(); (root/name/'a.pkl').touch()
            vocab = SignVocabulary(root, root/'alphabet', None)
            plan = vocab.resolve_tokens(TextPreprocessor().preprocess('How are you?'))
            self.assertEqual([e['gloss'] for e in plan], ['HOW_ARE_YOU'])
            self.assertEqual(plan[0]['token_count'], 3)
            plan = vocab.resolve_tokens(TextPreprocessor({}).preprocess('The children are running to their houses.'))
            self.assertTrue({'CHILD','RUN','HOUSE'} <= {e['gloss'] for e in plan})
            unknown = vocab.resolve_tokens(['ZZZXQV'])[0]
            self.assertEqual(unknown['type'], 'missing')
            self.assertEqual(unknown['match_type'], 'unknown')
            (root/'alphabet').mkdir()
            for c in 'ZQXV': (root/'alphabet'/f'{c}.png').touch()
            vocab = SignVocabulary(root, root/'alphabet', None)
            self.assertEqual(vocab.resolve_tokens(['ZZZXQV'])[0]['type'], 'fingerspell')
            self.assertEqual(vocab.resolve_tokens(['HOUSES'])[0]['match_type'], 'fuzzy')

    def test_duplicate_annotations_keep_provenance_and_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for i, coverage in enumerate([.6, 1.]):
                p = root/f'{i}.pkl'; p.touch()
                rows.append(dict(id=str(i), label='HOUSE', landmarks=str(p), clip=str(root/f'{i}.mp4'),
                                 eligible=True, hand_coverage=coverage, real_frames=20,
                                 source=f'video{i}', start_ms=100+i, end_ms=500+i))
            manifest = root/'manifest.json'; manifest.write_text(json.dumps({'records':rows}))
            vocab = SignVocabulary(root, root/'alphabet', manifest)
            self.assertEqual(len(vocab.index['HOUSE']), 2)
            matched = vocab.resolve_tokens(['houses'])[0]
            self.assertEqual(matched['source'], 'video1')
            self.assertEqual(matched['start_ms'], 101)
            self.assertEqual(matched['end_ms'], 501)
            self.assertEqual(matched['pkl'], root/'1.pkl')
            self.assertEqual(matched['requested_gloss'], 'HOUSES')
            self.assertEqual(matched['matched_gloss'], 'HOUSE')

    def test_debug_logging(self):
        with self.assertLogs('nsl.translation', level='DEBUG') as logs:
            GlossMatcher(['HOUSE']).match('HOUSES')
            GlossMatcher(['PLANE']).match('AIRPLANE')
        self.assertTrue(any('score=' in line for line in logs.output))
        self.assertTrue(any('fallback' in line for line in logs.output))
