# Audio Classification Prototype

**Python · Flask · pydub · Matplotlib · VGG19 features · Linear SVM**

A local prototype that turns short audio uploads into waveform images and explores classification with pretrained image features. Training uses an explicitly supplied labelled-image manifest; VGG19 supplies features and a linear SVM learns the class boundaries.

## What this project demonstrates

- Separating upload handling, feature extraction, and classification.
- Validating and isolating local uploaded artifacts.
- Making labels explicit and keeping heavyweight model loading lazy.

## Start here

| Read | Use it for |
| --- | --- |
| [Project walkthrough](docs/PROJECT_GUIDE.md) | Purpose, workflow, repository map, and a concrete example |
| [Code and artifact reference](docs/CODE_REFERENCE.md) | Important files, functions, classes, and their responsibilities |
| [Development guide](docs/DEVELOPMENT.md) | Setup, configuration, commands, tests, and remaining limitations |

## Explore locally

Use Python 3.11 or 3.12 in a virtual environment.

```sh
python -m pip install -r requirements.txt
python app.py
python -m unittest discover -s tests -v
```

MP3 decoding requires FFmpeg. Training additionally needs a labelled-image manifest and the first-use ImageNet weight download; see the development guide.

## Validation and scope

Four tests pass using a synthetic WAV and mocked image features. Full VGG19 extraction, labelled-data quality, and MP3 integration remain unverified. This is a local learning prototype and is not a validated medical classifier or a production upload service.

[Original project demonstration](https://vimeo.com/378198308) shows the historical implementation.

---

[GitHub profile](https://github.com/tariksalay) · [Portfolio](https://tariksalay.github.io/) · [Earlier README](README_HISTORY.md)

The earlier README is preserved as historical context; the guides above describe this branch's implementation.
