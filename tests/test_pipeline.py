import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from translation import TextPreprocessor, SignVocabulary, plan_coverage
from train_lstm import add_gaussian_noise, time_warp
from prepare_splits import make_split
from dataset_utils import load_split, file_hash, read_annotations


class TranslationTests(unittest.TestCase):
    def make_vocab(self, root, signs, letters=()):
        landmarks=root/'landmarks'; landmarks.mkdir()
        alphabet=landmarks/'alphabet'; alphabet.mkdir()
        for label in signs:
            folder=landmarks/label; folder.mkdir(); (folder/'example.pkl').write_bytes(b'fixture')
        for letter in letters:
            (alphabet/f'{letter}.pkl').write_bytes(b'fixture')
        return SignVocabulary(landmarks,alphabet,manifest_path=None)

    def test_phrase_preserves_function_words(self):
        with tempfile.TemporaryDirectory() as temp:
            vocab=self.make_vocab(Path(temp),['HOW_ARE_YOU','HOW','YOU'])
            plan=vocab.resolve_tokens(TextPreprocessor().preprocess('How are you?'))
            self.assertEqual([p['gloss'] for p in plan],['HOW_ARE_YOU'])
            self.assertEqual(plan_coverage(plan)['direct_tokens'],3)

    def test_contractions_negation_and_conjunction(self):
        tokens=TextPreprocessor().preprocess("I don't want tea or coffee")
        self.assertIn('NOT',tokens); self.assertIn('OR',tokens)

    def test_idiom_word_boundaries_and_empty_override(self):
        pp=TextPreprocessor({'cat':'DOG'})
        self.assertEqual(pp.preprocess('cat catalog'),['DOG','CATALOG'])
        self.assertEqual(TextPreprocessor({}).preprocess('piece of cake'),['PIECE','OF','CAKE'])

    def test_missing_letter_never_counts_as_playable(self):
        with tempfile.TemporaryDirectory() as temp:
            vocab=self.make_vocab(Path(temp),[],['A','B'])
            plan=vocab.resolve_tokens(['AB','AC','12'])
            self.assertEqual([e['type'] for e in plan],['fingerspell','missing','missing'])
            self.assertAlmostEqual(plan_coverage(plan)['playable_coverage'],1/3)

    def test_typo_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            vocab=self.make_vocab(Path(temp),['ANNIMAL'])
            self.assertEqual(vocab.resolve_tokens(['ANIMAL'])[0]['type'],'sign')


class TrainingTests(unittest.TestCase):
    def test_augmentation_preserves_missing_parts_and_padding(self):
        data=np.zeros((30,1662),dtype=np.float32); data[:8,:132]=.5
        for result in [add_gaussian_noise(data,rng=np.random.default_rng(1)),
                       time_warp(data,rng=np.random.default_rng(1))]:
            self.assertFalse(result[8:].any()); self.assertFalse(result[:,132:].any())
            np.testing.assert_equal(result[:8,3:132:4],data[:8,3:132:4])

    def test_groups_are_disjoint_and_all_labels_represented(self):
        records=[{'eligible':True,'source':source,'label':label}
                 for source in ['a','b','c','d','e'] for label in ['HELLO','WATER']]
        records.append({'eligible':True,'source':'a','label':'RARE'})
        split=make_split(records)
        groups=list(map(set,split['groups'].values()))
        for i in range(3):
            for j in range(i): self.assertFalse(groups[i]&groups[j])
        for rows in split['partitions'].values():
            self.assertEqual({r['label'] for r in rows},set(split['label_map']))
        self.assertIn('RARE',split['excluded_labels'])

    def test_changed_array_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); path=root/'sample.npy'; np.save(path,np.ones((30,1662)))
            split={'label_map':{'A':0},'partitions':{'test':[{'keypoints':'sample.npy','label':'A','sha256':file_hash(path)}]}}
            X,y=load_split(split,'test',root); self.assertEqual(X.shape,(1,30,1662))
            np.save(path,np.zeros((30,1662)))
            with self.assertRaisesRegex(ValueError,'changed'): load_split(split,'test',root)

    def test_training_module_has_no_side_effects(self):
        import train_lstm
        self.assertTrue(callable(train_lstm.train))


class ExtractionTests(unittest.TestCase):
    def test_reference_annotations_resolve_source_interval(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'session.eaf'
            path.write_text('''<ANNOTATION_DOCUMENT><TIME_ORDER><TIME_SLOT TIME_SLOT_ID="s1" TIME_VALUE="10"/><TIME_SLOT TIME_SLOT_ID="s2" TIME_VALUE="500"/></TIME_ORDER><TIER TIER_ID="words"><ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a" TIME_SLOT_REF1="s1" TIME_SLOT_REF2="s2"><ANNOTATION_VALUE>Hello</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION></TIER><TIER TIER_ID="words_1"><ANNOTATION><REF_ANNOTATION ANNOTATION_ID="b" ANNOTATION_REF="a"><ANNOTATION_VALUE>Hello</ANNOTATION_VALUE></REF_ANNOTATION></ANNOTATION></TIER></ANNOTATION_DOCUMENT>''')
            rows=list(read_annotations(path)); self.assertEqual(len(rows),2)
            self.assertEqual(rows[1]['end_ms'],500); self.assertEqual(rows[1]['source'],'session')


if __name__=='__main__':
    unittest.main()
