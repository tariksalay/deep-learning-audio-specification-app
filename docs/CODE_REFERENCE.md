# Code and artifact reference

[Overview](../README.md) · [Walkthrough](PROJECT_GUIDE.md) · [Development guide](DEVELOPMENT.md)

This reference describes the files on this branch. Function names are navigation aids; the walkthrough explains how their inputs, state, and outputs fit together.

## Python modules

### [machinelearning.py](../MachineLearning/machinelearning.py)

Training-manifest validation, VGG19 feature extraction, Linear SVM fitting, evaluation, and job predictions.

- `feature_model()`
- `get_features(path)`
- `load_training_manifest(path)`
- `upload_machinelearning_form()`
- `run_machinelearning()`

### [main_page.py](../WebApp/main_page.py)

Upload validation, isolated job folders, audio decoding, and waveform image extraction.

- `allowed_file(filename)`
- `extract_audio(audio, job)`
- `upload_form()`
- `upload_file()`

### [app.py](../app.py)

Application factory, blueprints, upload limits, and configuration.

- `create_app(config=None)`

## Objects passed between modules

| Object | Meaning |
| --- | --- |
| Upload job | UUID-named folder containing one submitted audio file and generated clips/images. |
| Training manifest | CSV with exactly `path,label`, pointing to local labeled images beneath the manifest directory. |
| Feature vector | Flattened VGG19 output for a resized/preprocessed image. |
| Classifier | Request-trained Linear SVM, evaluated on a stratified holdout; not persisted as a deployed model. |
| Session job ID | Identifies the upload to predict after training; it is not user authentication. |

## Checks and setup

[Tests](../tests/) contains the focused checks described in the [development guide](DEVELOPMENT.md). Consult that guide for dependencies, commands, and the boundary between automated validation and untested integrations.
