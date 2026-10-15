# Development and validation

This guide supplements the historical README; it describes the modernization branch. Use Python 3.11 or 3.12 and an isolated environment:

```sh
python -m pip install -r requirements.txt
python app.py
python -m unittest discover -s tests -v
```

Open the loopback address printed by Flask. MP3 decoding additionally needs FFmpeg on PATH; WAV uploads work without FFmpeg. Uploads are limited to 20 MiB and 60 seconds, with at most 12 waveform windows. Each upload has its own job folder under `instance/uploads/`. This is a local development app: there is no login, CSRF protection, quota management, or automatic retention policy. Delete obsolete local job folders yourself. Set `FLASK_SECRET_KEY` for persistent sessions across restarts; otherwise a temporary random key is generated.

Model weights load lazily only when training is requested. TensorFlow 2/Keras replaces the old global TF1 graph. Training requires `TRAINING_MANIFEST` pointing to a CSV with exactly `path,label` as its header, relative image paths below that CSV's directory, and at least two labelled examples per class across at least two classes. For example, supply your own `class_a/example1.png,class_a` rows. No labels are inferred from file order. No employer data is included.

The VGG19-to-linear-SVM experiment can then evaluate a stratified image split and classify the uploaded waveform windows. ImageNet weights require a first-use download. Splitting waveform windows from the same recording across train/test can inflate accuracy; independent recording-level validation and real labels are still required. The app is not a validated medical classifier.

Tests use a synthetic silent WAV and mocked features: they cover upload isolation, traversal rejection, missing labels, route loading, and classifier plumbing. Full VGG19 extraction/training and MP3/FFmpeg integration are not claimed as tested.
