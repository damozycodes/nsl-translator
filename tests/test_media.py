import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from playback import export_sentence, cache_key


class MediaTests(unittest.TestCase):
    def test_source_video_order_and_duration(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); plan=[]
            for i,colour in enumerate([(0,0,255),(0,255,0)]):
                path=root/f'{i}.mp4'
                writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),25,(64,64))
                self.assertTrue(writer.isOpened())
                for _ in range(25): writer.write(np.full((64,64,3),colour,dtype=np.uint8))
                writer.release()
                plan.append({'type':'sign','gloss':str(i),'clip':path})
            output=export_sentence(plan,root/'ordered.mp4')
            cap=cv2.VideoCapture(str(output)); fps=cap.get(cv2.CAP_PROP_FPS)
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps,2,delta=.1)
            cap.set(cv2.CAP_PROP_POS_MSEC,250); ok,first=cap.read(); self.assertTrue(ok)
            cap.set(cv2.CAP_PROP_POS_MSEC,1250); ok,second=cap.read(); self.assertTrue(ok)
            cap.release()
            self.assertNotEqual(cache_key(plan, 'source'), cache_key(plan, 'source', speed=.5))
            self.assertNotEqual(cache_key(plan, 'source'), cache_key(plan, 'source', close_up=False))
            slow = export_sentence(plan, root/'slow.mp4', speed=.5)
            cap = cv2.VideoCapture(str(slow))
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS), 4, delta=.15)
            cap.release()
            self.assertGreater(first[240,320,2],200)
            self.assertGreater(second[240,320,1],200)

    def test_incomplete_plan_produces_no_video(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'no.mp4'
            with self.assertRaisesRegex(ValueError,'Missing'):
                export_sentence([{'type':'missing','gloss':'NO'}],path)
            self.assertFalse(path.exists())


if __name__=='__main__':
    unittest.main()
