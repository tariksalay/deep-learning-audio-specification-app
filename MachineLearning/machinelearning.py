from flask import Flask, flash, render_template, Blueprint, session
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import os
import tensorflow as tf

# Variables
UPLOAD_FOLDER = 'C:/Users/tondi/OneDrive/Documents/GitHub/CSEE5590-IOT-Robotics/ICP 9/downloads'
machinelearning = Blueprint('machinelearning',__name__, template_folder='/templates')


# Model information
base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)
global graph
graph = tf.get_default_graph()


def get_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    with graph.as_default():
        flatten = model.predict(x)
    return list(flatten[0])


# plot waveforms


def plot_waveforms(file_name):
    audio = file_name[1]
    plt.plot(audio)
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.savefig("downloads/wav_plots/" + "test_sample.png")
    plt.close("all")
    with open(UPLOAD_FOLDER + "/wav_plots/test_sample.png") as f:
        wave_form = f.read()
    return wave_form

# Flask


@machinelearning.route('/')
def upload_machinelearning_form():
    flash('File successfully uploaded and ready for machine learning')
    return render_template('machinelearning.html')


@machinelearning.route('/machinelearning', methods=['POST'])
def run_machinelearning():
    print("line 56")
    X = []
    y = []
    audio_plots = []
    audio_parts_plots = []
    for (_, _, filenames) in os.walk('downloads/wav_plots/'):
        audio_plots.extend(filenames)
        break
    for (_, _, filenames) in os.walk('downloads/audio_parts_classes/'):
        audio_parts_plots.extend(filenames)
        break
    print(audio_plots)
    count = 1
    for aplot in audio_plots:
        if count < len(audio_plots)/2:
            X.append(get_features(UPLOAD_FOLDER + '/wav_plots/' + aplot))
            y.append(0)
            count += 1
        else:
            X.append(get_features(UPLOAD_FOLDER + '/wav_plots/' + aplot))
            y.append(1)
            count += 1
    count = 1
    for aplot in audio_parts_plots:
            X.append(get_features(UPLOAD_FOLDER + '/audio_parts_classes/' + aplot))
            y.append(2)
            count += 1
            # X.append(get_features(UPLOAD_FOLDER + '/audio_parts_classes/' + aplot))
            # y.append(3)
            # count += 1

    print(X, "line 71")
    print(y, "line 72")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    clf = LinearSVC(random_state=0, tol=1e-5)
    labels = np.unique(X_train)
    print(labels, "line 80")
    labels = np.unique(y_train)
    print(labels, "line 82")
    print(clf, "line clf")
    clf.fit(X_train, y_train)
    print("line 83")
    predicted = clf.predict(X_test)

    # get the accuracy
    flash(accuracy_score(y_test, predicted))
    print(accuracy_score(y_test, predicted))
    return render_template('machinelearning.html')


#    with open(UPLOAD_FOLDER + "/wav_plots/") as f:
#        print("line 57")
#        file_content = f.read()
#    X = []
#    Y = []
#    X.append(get_features(plot_waveforms(file_content)))
#    Y.append(0)
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
#    clf = LinearSVC(random_state=0, tol=1e-5)
#    clf.fit(X_train, y_train)
#    predicted = clf.predict(X_test)
#    # get the accuracy
#    flash(accuracy_score(y_test, predicted))
#    return render_template('machinelearning.html')
