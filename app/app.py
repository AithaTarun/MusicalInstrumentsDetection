import math
from flask import Flask, request, render_template
import pickle
import librosa
from tensorflow.keras.models import load_model
import numpy as np

# Declaring constants
ALLOWED_EXTENSIONS = {'wav'}
SAMPLE_RATE = 22050
N_SEGMENTS = 10
DURATION = 2
SAMPLE_PER_TRACK = SAMPLE_RATE * DURATION
HOP_LENGTH = 512
N_FFT = 2048
N_MFCC = 64
INSTRUMENTS = [
    "cel",
    "cla",
    "flu",
    "gac",
    "gel",
    "mridangam",
    "org",
    "pia",
    "sax",
    "sitar",
    "tabla",
    "tru",
    "veena",
    "vio",
    "voi"
]

INSTRUMENT_NAMES = [
    "Cello",
    "Clarinet",
    "Flute",
    "Acoustic Guitar",
    "Electric Guitar",
    "Mridangam",
    "Organ",
    "Piano",
    "Saxophone",
    "Sitar",
    "Tabla",
    "Trumpet",
    "Veena",
    "Violin",
    "Voice"
]

n_samples_per_segment = int(SAMPLE_PER_TRACK / N_SEGMENTS)
expected_n_mfcc_vectors_per_segment = math.ceil(n_samples_per_segment / HOP_LENGTH)

# Configuring flask server
app = Flask(__name__)
app._static_folder = "static"

# Loading model
MODEL_PATH = "model/model.hdf5"


def load_model_data(model_path):
    model = load_model(model_path)
    return model


model = load_model_data(MODEL_PATH)


def predictProbabilities(model, X):
    prediction = model.predict(X)

    prediction = prediction[0]

    np.set_printoptions(suppress=True)
    print("Predicted probabilities : ", prediction)

    # Extract index with max value

    # predicted_index = np.argmax(prediction, axis=1)
    idx = np.argpartition(prediction, -2)[-2:]
    indices = idx[np.argsort((-prediction)[idx])]

    prediction.sort()

    return [indices, [prediction[-1], prediction[-2]]]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html', predictions = [])


@app.route('/predict', methods=['POST'])
def find():
    print("Here")
    if request.method == 'GET':
        return


@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['audioFileInput']

        if file and allowed_file(file.filename):
            signal, sr = librosa.load(file, sr=SAMPLE_RATE)

            data = []

            for s in range(1):
                start_sample = n_samples_per_segment * s
                finish_sample = start_sample + n_samples_per_segment

                mfcc = librosa.feature.mfcc(signal[start_sample: finish_sample], sr=sr, n_fft=N_FFT, n_mfcc=N_MFCC,
                                            hop_length=HOP_LENGTH)
                mfcc = mfcc.T

                if len(mfcc) == expected_n_mfcc_vectors_per_segment:
                    data.append(mfcc.tolist())


            data = np.array(data)
            data = data[..., np.newaxis]

            predictions = predictProbabilities(model, data)

            print("Top 2 Predictions : ", predictions)

    return render_template('index.html', predictions = [[INSTRUMENTS[predictions[0][0]], INSTRUMENTS[predictions[0][1]]], predictions[1], [INSTRUMENT_NAMES[predictions[0][0]], INSTRUMENT_NAMES[predictions[0][1]]]])


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
