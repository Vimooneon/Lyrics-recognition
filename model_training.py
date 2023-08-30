import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from keras import layers
from keras import models
from IPython import display

# much of the following code is adapted from an example from the official Tensorflow website
# https://www.tensorflow.org/tutorials/audio/simple_audio

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram


# this function is unused, but is (supposedly) necessary for a couple of intermediate steps
# commenting it resulted in an error, hence why it remains here
def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)


def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

# exports the model to save it for later use without having to train the model again
# the exported model are then used by the main program
class ExportModel(tf.Module):
  def __init__(self, model):
    self.model = model

    self.__call__.get_concrete_function(
        x=tf.TensorSpec(shape=(), dtype=tf.string))
    self.__call__.get_concrete_function(
       x=tf.TensorSpec(shape=[None, 64000], dtype=tf.float32))


  @tf.function
  def __call__(self, x):
    # If they pass a string, load the file and decode it. 
    if x.dtype == tf.string:
      x = tf.io.read_file(x)
      x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=64000,)
      x = tf.squeeze(x, axis=-1)
      x = x[tf.newaxis, :]

    x = get_spectrogram(x)  
    result = self.model(x, training=False)

    class_ids = tf.argmax(result, axis=-1)
    class_names = tf.gather(label_names, class_ids)
    return {'predictions':result,
            'class_ids': class_ids,
            'class_names': class_names}

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# the path to the split up audio files that have already been separated into folders based on their contents
# every folder is named after the words that it contains 
DATASET_PATH = "C:\\lyrics_recognition\\Lyrics-recognition\\audio_data"

print("##########################################################")
data_dir = pathlib.Path(DATASET_PATH)

# some preliminary steps before training the model:
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
print('Commands:', commands)
# load the audio files
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory( 
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,  
    seed=0,
    output_sequence_length=64000, 
    subset='both'
    )
# preparing the data for use in training the model:
label_names = np.array(train_ds.class_names)
print()
print("label names:", label_names)

print(train_ds.element_spec)

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

for example_audio, example_labels in train_ds.take(1):  
  print(example_audio.shape)
  print(example_labels.shape)

print(label_names[[1,1,3,0]])


train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    rows = 3
    cols = 3
    n = rows*cols


input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)


norm_layer = layers.Normalization()
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

# create a sequential model; sequential models are more effective for ASR systems
model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    layers.Resizing(32, 32),
    # Normalize.
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# training the model:
EPOCHS = 10 # the amount of epochs can be increased to make the training process longer (it will still stop because of the EarlyStopping line below), 
# though the actual benefits of that are a bit dubious
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2), # the patience parameter can also be increased to make the training process
) # less receptive to lack of growth which in combination with an increased amount of epochs will result in a longer training time

# the audio file used for visual showcase of the model's precision
x = "C:\\lyrics_recognition\\Lyrics-recognition\\audio_data\\father\\r_father3678.wav"
x = tf.io.read_file(str(x))
x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=64000,)
x = tf.squeeze(x, axis=-1) 
print("here? 0.5")
waveform = x
x = get_spectrogram(x)
x = x[tf.newaxis,...]
# check the input data's parameters
prediction = model(x)

# creates a wordlist of all words used while training, is later used in printing the answer and for the bar graph
wordlist = []
for directory_name in os.listdir("C:\\lyrics_recognition\\Lyrics-recognition\\audio_data"):
  wordlist.append(directory_name)


# used in order to get the chance of each word being the answer
preddd = model.predict(x)
classes = np.argmax(preddd, axis = 1)
print(classes)

# prints out the most likely answer
print("####################################################")
print("The likeliest:")
print(wordlist[classes[0]])

res = np.argsort(preddd)
actual_results = res[0]
res = np.sort(preddd)
probabilities = res[0]

# prints out the top 10 most likely words
print("The top 10:")
i = 0
while i < 10:
  print(str(wordlist[actual_results[684-i]]) + ": " + str(probabilities[684-i]))
  i += 1

#creates a bar graph with the most likely word being the highest
x_labels = wordlist
bar = plt.bar(x_labels, tf.nn.softmax(prediction[0]))
plt.title('Who cares the The')
plt.show()
display.display(display.Audio(waveform, rate=64000))

#exports the trained model
export_path = "C:\\lyrics_recognition\\Lyrics-recognition\\newsaved9" #8  9  10
model.save(export_path)

