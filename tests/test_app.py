import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import create_app
from MachineLearning.machinelearning import load_training_manifest
from WebApp.main_page import allowed_file


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.app = create_app({'TESTING': True, 'SECRET_KEY': 'test-only',
                               'UPLOAD_FOLDER': str(self.root / 'uploads')})
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp.cleanup()

    def test_pages_do_not_download_model(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.get('/machinelearning/').status_code, 200)
        self.assertEqual(self.client.post('/machinelearning/machinelearning').status_code, 400)

    def test_upload_validation(self):
        self.assertFalse(allowed_file('payload.txt'))
        self.assertTrue(allowed_file('sample.WAV'))
        self.assertEqual(self.client.post('/', data={}).status_code, 302)
        self.client.post('/', data={'file': (io.BytesIO(b'bad'), 'payload.txt')})
        self.assertEqual(list((self.root / 'uploads').iterdir()), [])

    def test_wav_upload_uses_isolated_job(self):
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as audio:
            audio.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
            audio.writeframes(b'\0\0' * 8000)
        buffer.seek(0)
        response = self.client.post('/', data={'file': (buffer, '../../sample.wav')})
        self.assertEqual(response.status_code, 302)
        with self.client.session_transaction() as session:
            job = self.root / 'uploads' / session['job_id']
        self.assertTrue((job / 'sample.wav').is_file())
        self.assertEqual(len(list((job / 'wav_plots').glob('*.png'))), 2)
        self.assertFalse((self.root / 'sample.wav').exists())

    def test_explicit_labels_and_traversal(self):
        manifest = self.root / 'labels.csv'
        manifest.write_text('path,label\n../outside.png,x\n')
        with self.assertRaises(ValueError):
            load_training_manifest(manifest)
        manifest.write_text('path,label\n')
        with self.assertRaises(ValueError):
            load_training_manifest(manifest)
        for i in range(4):
            (self.root / f'{i}.png').write_bytes(b'fixture')
        manifest.write_text('path,label\n0.png,a\n1.png,a\n2.png,b\n3.png,b\n')
        self.assertEqual(len(load_training_manifest(manifest)), 4)
        self.app.config['TRAINING_MANIFEST'] = str(manifest)
        with patch('MachineLearning.machinelearning.get_features', side_effect=lambda p: [0.0 if p.stem in {'0', '1'} else 1.0]):
            response = self.client.post('/machinelearning/machinelearning')
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()
