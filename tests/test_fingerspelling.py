import tempfile
import unittest
from pathlib import Path
import cv2
from config import PROJECT_ROOT
from translation import SignVocabulary
from playback import export_sentence, _sources


class FingerspellingTests(unittest.TestCase):
    def test_demo_is_separate_and_complete(self):
        demo = SignVocabulary(alphabet_dir=PROJECT_ROOT/'assets/asl_demo')
        self.assertEqual(len(demo.alphabet), 26)
        self.assertTrue(all(p.parent.name == 'asl_demo' for p in demo.alphabet.values()))
        self.assertFalse(any(p.parent.name == 'asl_demo' for p in SignVocabulary().alphabet.values()))
        self.assertEqual([c for c, _ in demo.fingerspell(['HELLO'])[0]['letters']], list('HELLO'))
        self.assertEqual(demo.fingerspell(['123'])[0]['type'], 'missing')

    def test_image_letters_export_in_order(self):
        demo = SignVocabulary(alphabet_dir=PROJECT_ROOT/'assets/asl_demo')
        with tempfile.TemporaryDirectory() as temp:
            out = export_sentence(demo.fingerspell(['AZ']), Path(temp)/'demo.mp4')
            cap = cv2.VideoCapture(str(out))
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS),2.4,delta=.1)
            ok, a = cap.read()
            self.assertTrue(ok)
            cap.set(cv2.CAP_PROP_POS_MSEC, 1500)
            ok, z = cap.read()
            self.assertTrue(ok)
            cap.release()
            self.assertGreater(cv2.absdiff(a,z).mean(), 2)

    def test_stickman_requires_paired_landmarks_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp)/'A.png'
            image.touch()
            plan = [{'type': 'fingerspell', 'letters': [('A', image)]}]
            self.assertEqual(list(_sources(plan, 'source')), [(image, 'image')])
            with self.assertRaisesRegex(ValueError, 'letter A'):
                list(_sources(plan, 'stickman'))
            landmarks = image.with_suffix('.pkl')
            landmarks.touch()
            self.assertEqual(list(_sources(plan, 'stickman')), [(landmarks, 'stickman')])
            self.assertEqual(list(_sources(plan, 'source')), [(image, 'image')])


    def test_registered_recordings_have_paired_skeletons(self):
        vocab = SignVocabulary()
        self.assertEqual(set(vocab.alphabet), set('ABCDEFGIJLMNOPRSTUVY'))
        plan = vocab.fingerspell(['ABFI'])
        self.assertTrue(all(kind == 'source' and path.suffix == '.mp4' for path, kind in _sources(plan, 'source')))
        self.assertTrue(all(kind == 'stickman' and path.is_file() for path, kind in _sources(plan, 'stickman')))
