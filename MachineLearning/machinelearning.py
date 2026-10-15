"""VGG19 features + linear SVM, using explicitly supplied training labels."""
import csv
from functools import lru_cache
from pathlib import Path
import re

from flask import Blueprint, current_app, flash, render_template, session
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

machinelearning = Blueprint('machinelearning', __name__)


@lru_cache(maxsize=1)
def feature_model():
    from tensorflow.keras.applications.vgg19 import VGG19
    from tensorflow.keras.models import Model
    base = VGG19(weights='imagenet')
    return Model(inputs=base.input, outputs=base.get_layer('flatten').output)


def get_features(path):
    from tensorflow.keras.utils import load_img, img_to_array
    from tensorflow.keras.applications.vgg19 import preprocess_input
    image = load_img(path, target_size=(224, 224))
    batch = np.expand_dims(img_to_array(image), axis=0)
    return feature_model().predict(preprocess_input(batch), verbose=0)[0]


def load_training_manifest(path):
    manifest = Path(path).resolve()
    root = manifest.parent
    rows = []
    with manifest.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['path', 'label']:
            raise ValueError('Training manifest must have the header path,label.')
        for row in reader:
            if None in row or not row.get('path') or not (row.get('label') or '').strip():
                raise ValueError('Each training image needs an explicit label.')
            image = (root / row['path']).resolve()
            if not image.is_relative_to(root) or image.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
                raise ValueError('Training image paths must stay within the manifest directory.')
            if not image.is_file():
                raise ValueError('A training image listed in the manifest is missing.')
            rows.append((image, row['label'].strip()))
    labels, counts = np.unique([label for _, label in rows], return_counts=True)
    if len(labels) < 2 or min(counts) < 2:
        raise ValueError('Provide at least two classes with at least two labelled images per class.')
    return rows


@machinelearning.get('/')
def upload_machinelearning_form():
    return render_template('machinelearning.html')


@machinelearning.post('/machinelearning')
def run_machinelearning():
    manifest = current_app.config['TRAINING_MANIFEST']
    if not manifest:
        flash('Set TRAINING_MANIFEST to a CSV of labelled waveform images before training.')
        return render_template('machinelearning.html'), 400
    try:
        rows = load_training_manifest(manifest)
    except (OSError, ValueError) as error:
        # Do not expose local filesystem paths in the UI.
        flash(str(error) if isinstance(error, ValueError) else 'Could not read the training manifest.')
        return render_template('machinelearning.html'), 400
    features = [get_features(path) for path, _ in rows]
    labels = [label for _, label in rows]
    classes = len(set(labels))
    test_count = max(classes, int(np.ceil(len(labels) * 0.30)))
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=test_count, random_state=42, stratify=labels)
    classifier = LinearSVC(random_state=42, tol=1e-5, max_iter=10000)
    classifier.fit(train_x, train_y)
    flash(f'Held-out accuracy on the supplied images: {accuracy_score(test_y, classifier.predict(test_x)):.3f}')
    job_id = session.get('job_id', '')
    if re.fullmatch(r'[0-9a-f]{32}', job_id):
        plots = Path(current_app.config['UPLOAD_FOLDER']) / job_id / 'wav_plots'
        for image in sorted(plots.glob('*.png')):
            prediction = classifier.predict([get_features(image)])[0]
            flash(f'{image.stem}: {prediction}')
    return render_template('machinelearning.html')
