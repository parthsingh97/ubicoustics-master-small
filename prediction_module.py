from keras.models import load_model
import tensorflow as tf
import numpy as np
from vggish_input import waveform_to_examples
import ubicoustics
import pyaudio
from pathlib import Path
import time
import wget

# Variables
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = RATE
MICROPHONES_DESCRIPTION = []
FPS = 60.0

###########################
# Download model, if it doesn't exist
###########################
MODEL_URL = "https://www.dropbox.com/s/cq1d7uqg0l28211/example_model.hdf5?dl=1"
MODEL_PATH = "models/example_model.hdf5"
# print("=====")
# print("2 / 2: Checking model... ")
# print("=====")
model_filename = "models/example_model.hdf5"
ubicoustics_model = Path(model_filename)
if (not ubicoustics_model.is_file()):
#     print("Downloading example_model.hdf5 [867MB]: ")
    wget.download(MODEL_URL,MODEL_PATH)

##############################
# Load Deep Learning Model
##############################
# print("Using deep learning model: %s" % (model_filename))
model = load_model(model_filename)
graph = tf.get_default_graph()
context = ubicoustics.everything

label = dict()
for k in range(len(context)):
    label[k] = context[k]


##############################
# Prediction Function
##############################
def predict_wav(in_data, sample_rate=RATE):
    global graph
    
#     np_wav = np.fromstring(in_data, dtype=np.int16) / 32768.0 # Convert to [-1.0, +1.0]
#     x = waveform_to_examples(np_wav, sample_rate)
    
    assert in_data.dtype == np.int16, 'Bad sample type: %r' % in_data.dtype
    np_wav = in_data / 32768.0  # Convert to [-1.0, +1.0]
    x = waveform_to_examples(np_wav, sample_rate)
    
    predictions = []
    output_list = []
    with graph.as_default():

        x = x.reshape(len(x), 96, 64, 1)
        predictions = model.predict(x)

        for k in range(len(predictions)):
            prediction = predictions[k]
            m = np.argmax(prediction)
#             print("Prediction: %s (%0.2f)" % (ubicoustics.to_human_labels[label[m]], prediction[m]))
            output_list.append([ubicoustics.to_human_labels[label[m]], prediction[m]])

    return output_list