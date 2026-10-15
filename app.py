"""Local audio experiment. Heavy ML dependencies load only when requested."""
import os
from pathlib import Path
import secrets

from flask import Flask
from WebApp.main_page import main_page
from MachineLearning.machinelearning import machinelearning


def create_app(config=None):
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.getenv('FLASK_SECRET_KEY') or secrets.token_hex(32),
        UPLOAD_FOLDER=str(Path(app.instance_path) / 'uploads'),
        TRAINING_MANIFEST=os.getenv('TRAINING_MANIFEST', ''),
        MAX_CONTENT_LENGTH=20 * 1024 * 1024,
        MAX_AUDIO_MS=60_000,
    )
    if config:
        app.config.update(config)
    Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    app.register_blueprint(main_page)
    app.register_blueprint(machinelearning, url_prefix='/machinelearning')
    return app


app = create_app()

if __name__ == '__main__':
    app.run(debug=False)
