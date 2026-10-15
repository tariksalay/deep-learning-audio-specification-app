"""Validate uploads and generate waveform images in a separate job directory."""
from pathlib import Path
from uuid import uuid4
import shutil

from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

main_page = Blueprint('main_page', __name__)
ALLOWED_EXTENSIONS = {'wav', 'mp3'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_audio(audio, job):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    import numpy as np
    parts = job / 'audio_parts'
    plots = job / 'wav_plots'
    parts.mkdir()
    plots.mkdir()
    # Keep the prototype bounded: at most 12 short, overlapping waveform windows.
    for number, start in enumerate(range(0, min(len(audio), 6000), 500), 1):
        clip = audio[start:min(start + 1500, len(audio))].set_channels(1)
        with (parts / f'audio_parts{number}.wav').open('wb') as output:
            clip.export(output, format='wav')
        samples = np.array(clip.get_array_of_samples())
        figure = Figure(figsize=(6.4, 4.8))
        axis = figure.subplots()
        axis.plot(np.arange(len(samples)) / clip.frame_rate, samples)
        axis.set(xlabel='Time (seconds)', ylabel='Amplitude')
        figure.savefig(plots / f'audio_parts{number}.png')
        figure.clear()


@main_page.get('/')
def upload_form():
    return render_template('upload.html')


@main_page.post('/')
def upload_file():
    file = request.files.get('file')
    if file is None or not file.filename:
        flash('Choose an audio file.')
        return redirect(url_for('main_page.upload_form'))
    filename = secure_filename(file.filename)
    if not filename or not allowed_file(filename):
        flash('Allowed file types are mp3 and wav.')
        return redirect(url_for('main_page.upload_form'))
    job_id = uuid4().hex
    job = Path(current_app.config['UPLOAD_FOLDER']) / job_id
    job.mkdir()
    try:
        from pydub import AudioSegment
        from pydub.exceptions import CouldntDecodeError
        source = job / filename
        file.save(source)
        try:
            with source.open('rb') as stream:
                audio = AudioSegment.from_file(stream, format=source.suffix.lstrip('.'))
        except (CouldntDecodeError, OSError):
            flash('Could not decode this audio. MP3 files require FFmpeg on PATH.')
            shutil.rmtree(job)
            return redirect(url_for('main_page.upload_form'))
        if not 0 < len(audio) <= current_app.config['MAX_AUDIO_MS']:
            flash('Use an audio clip between 1 millisecond and 60 seconds.')
            shutil.rmtree(job)
            return redirect(url_for('main_page.upload_form'))
        extract_audio(audio, job)
    except Exception:
        shutil.rmtree(job, ignore_errors=True)
        raise
    session['job_id'] = job_id
    flash('Audio uploaded and waveform windows generated.')
    return redirect(url_for('machinelearning.upload_machinelearning_form'))
