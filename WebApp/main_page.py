from os import walk
import os
from flask import Flask, flash, request, redirect, render_template, Blueprint, session
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from pathlib import Path
import ffmpeg

# Variables

app = Flask(__name__)
main_page = Blueprint('main_page',__name__, template_folder='/templates')
UPLOAD_FOLDER = 'C:/Users/tondi/OneDrive/Documents/GitHub/CSEE5590-IOT-Robotics/ICP 9/downloads'
ALLOWED_EXTENSIONS = set(['txt', 'wav', 'mp3'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'secret key'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def plot_waves():
    if not os.path.exists("downloads/wav_plots"):
        os.makedirs("downloads/wav_plots")
    wavs = []
    print("line 29")
    for (_, _, filenames) in walk('downloads/audio_parts'):
        wavs.extend(filenames)
        print("line 32")
        break
    for wav in wavs:
        # read audio samples
        input_data = read("downloads/audio_parts/" + wav)
        audio = input_data[1]
        # plot the first 1024 samples
        plt.plot(audio)
        # label the axes
        plt.ylabel("Amplitude")
        plt.xlabel("Time")
        # set the title
        # plt.title("Sample Wav")
        # display the plot
        plt.savefig("downloads/wav_plots/" + wav.split('.')[0] + '.png')
        # plt.show()
        plt.close('all')
        print("line 36")
    print("line 37")
    return redirect('/machinelearning')


def extract_audio(filename):
    if not os.path.exists("downloads/audio_parts"):
        os.makedirs("downloads/audio_parts")
    count = 1
    for i in range(1, 60, 5):
        t1 = i * 100
        t2 = (i+15) * 100
        newAudio = filename
        newAudio = newAudio[t1:t2]
        newAudio.export(UPLOAD_FOLDER + "/audio_parts/audio_parts" + str(count) + '.wav', format="wav")  # Exports to a wav file in the current path.
        count += 1
    return plot_waves()
# Flask


@main_page.route('/')
def upload_form():
    return render_template('upload.html')


@main_page.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], UPLOAD_FOLDER + '/' + filename))
            print(filename)
            ##AudioSegment.converter = "C:/Users/tondi/OneDrive/Documents/Packages/ffmpeg-20191031-7c872df-win64-static/bin/ffmpeg.exe"
            filename = AudioSegment.from_file(UPLOAD_FOLDER + '/' + filename, format="mp3")
            print(filename)
            filename.export(UPLOAD_FOLDER + "/wavfile.wav", format="wav")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], "wavfile.wav"))
            ##session['filename'] = filename
            flash('File successfully uploaded')
            print(filename)
            return extract_audio(filename)
        else:
            flash('Allowed file types are mp3, wav')
            return redirect(request.url)



