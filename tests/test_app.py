"""Exercise real app submission and errors without launching a desktop window."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    def test_translate_then_missing_word_clears_previous_output(self):
        app = AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=60).run()
        self.assertEqual(len(app.exception),0)
        app.text_area[0].set_value('Hello water')
        app.button[0].click().run()
        self.assertEqual(len(app.exception),0)
        self.assertTrue(Path(app.session_state['translation']['path']).exists())
        app.text_area[0].set_value('123')
        app.button[0].click().run()
        self.assertEqual(len(app.exception),0)
        self.assertGreater(len(app.error),0)
        self.assertNotIn('translation',app.session_state)

    def test_unknown_word_spelled_without_alphabet_gallery(self):
        app = AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=60).run()
        self.assertFalse(any(c.label == 'Use ASL fingerspelling demo' for c in app.checkbox))
        self.assertEqual(len(app.get('imgs')), 0)
        app.text_area[0].set_value('zzzxqv')
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertTrue(Path(app.session_state['translation']['path']).exists())
        self.assertTrue(app.session_state['translation']['reference_letters'])

    def test_checked_fingerspelling_uses_cards(self):
        from translation import SignVocabulary
        from playback import cache_key
        app = AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=60).run()
        app.text_area[0].set_value('DAVID')
        next(c for c in app.checkbox if c.label == 'Fingerspell every word').check()
        app.radio[0].set_value('Stickman animation')
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        expected = cache_key(SignVocabulary(alphabet_dir=ROOT/'assets/asl_demo').fingerspell(['DAVID']), 'source')
        self.assertEqual(Path(app.session_state['translation']['path']).stem, expected)
        next(c for c in app.checkbox if c.label == 'Fingerspell every word').uncheck()
        app.radio[0].set_value('Original recording')
        app.text_area[0].set_value('Hello')
        app.button[0].click().run()
        expected = cache_key(SignVocabulary().resolve_tokens(['HELLO']), 'source')
        self.assertEqual(Path(app.session_state['translation']['path']).stem, expected)

    def test_empty_submission(self):
        app = AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=60).run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception),0)
        self.assertGreater(len(app.warning),0)


if __name__=='__main__':
    unittest.main()
