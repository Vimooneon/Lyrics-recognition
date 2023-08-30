#Files
import os
import pathlib
import csv

#Math
import numpy as np

#AI
import tensorflow as tf
from keras import layers
from keras import models

#Visualisation
import matplotlib.pyplot as plt
from IPython import display
from tkinter import *
from tkinter import filedialog
import time

#Audio
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence
import soundfile as sf
import playsound
from winsound import *


# much of the following code is adapted from an example from the official Tensorflow website
# https://www.tensorflow.org/tutorials/audio/simple_audio

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# exports the model to save it for later use without having to train the model again
# the exported model are then used by this program
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
      x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
      x = tf.squeeze(x, axis=-1)
      x = x[tf.newaxis, :]

    x = get_spectrogram(x)  
    result = self.model(x, training=False)

    class_ids = tf.argmax(result, axis=-1)
    class_names = tf.gather(label_names, class_ids)
    return {'predictions':result,
            'class_ids': class_ids,
            'class_names': class_names}
  

def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

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

# this function trims an audio file (i.e. removes silence at the beginning and end of audio file)
def trim_func(file_path):
    # Lines like the ones directly below are for updating the progress bar in the UI 
    #0%-10%
    label_loading.configure(text="Starting trim " + "[▯▯▯▯▯▯▯▯▯▯]")
    window.update()
    print("Starting trim...")
    y, sr = librosa.load(file_path)
    label_loading.configure(text="File loaded " + "[▮▯▯▯▯▯▯▯▯▯]")
    window.update()
    #10%-30%
    print("File Acquired")
    y_trimmed, index = librosa.effects.trim(y, top_db=16)
    label_loading.configure(text="Trim finished " + "[▮▮▮▯▯▯▯▯▯▯]")
    window.update()
    print("File trimmed")
    return y_trimmed, sr

# this function is vital for recognizing the lyrics of a song
# it separates the vocal from the instrumentals at the cost of audio quality
def vocal_sep_func(y_trimmed, sr):
    label_loading.configure(text="Vocal separation started " + "[▮▮▮▮▯▯▯▯▯▯]")
    window.update()
    lower_limit = 0
    duration = int(librosa.get_duration(y=y_trimmed, sr=sr))
    S_full, phase = librosa.magphase(librosa.stft(y_trimmed))
    S_filter = librosa.decompose.nn_filter(S_full,
                                        aggregate=np.median,
                                        metric='cosine',
                                        width=int(librosa.time_to_frames(2, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)
    margin_i, margin_v = 2, 10
    power = 2
    mask_i = librosa.util.softmask(S_filter,
                                margin_i * (S_full - S_filter),
                                power=power)
    mask_v = librosa.util.softmask(S_full - S_filter,
                                margin_v * S_filter,
                                power=power)
    S_foreground = mask_v * S_full
    y_foreground = librosa.istft(S_foreground * phase)
    label_loading.configure(text="Vocal separation finished " + "[▮▮▮▮▮▮▮▯▯▯]")
    window.update()
    print("Vocal separation ended")
    return y_foreground[lower_limit*sr:duration*sr]

# this function normalizes an audio file, making its volume more consistent
# it also makes the audio 3x louder to counteract the loss of volume from the vocal separation step
def normalization_function(data):
    data = 3 * librosa.util.normalize(data)
    label_loading.configure(text="Normalization finished " + "[▮▮▮▮▯▯▯▯▯▯]")
    window.update()
    return data

# this function splits an audio file into fragments
# and applies the aforementioned audio preprocessing steps to the audio file
def Splitter(path):
    print("Splitting Started")
    amount_of_exported_chunks = 0
    try:
        #0-30%
        trimmed_file, sr = trim_func(path)
        #30%-40%
        normalized_file = normalization_function(trimmed_file)
        #40%-70%
        sep_file = vocal_sep_func(normalized_file, sr)
        sf.write('temp_file.wav', sep_file, sr, subtype='PCM_24')
        print("Temp file created")
        sound_file = AudioSegment.from_wav('temp_file.wav')
        # surprisingly enough this is one of the most important steps of this whole project - separating audio files 
        # this is perhaps the simplest way of separating an audio file into chunks though it has its flaws
        # it sometimes results in a chunk with multiple words or a chunk with half a word
        # it can also interpret noises (such as music) as words though vocal separation largely negates that
        # the variables below are set so that the chunks would contain single words as often as possible
        audio_chunks = split_on_silence(sound_file, 
            min_silence_len=75,
            silence_thresh=-41
        )
        print("Splitting worked")
        label_loading.configure(text="Creating temporary files " + "[▮▮▮▮▮▮▮▯▯▯]")
        window.update()
        for i, chunk in enumerate(audio_chunks):                             
            out_file = ".//temp_folder//" + "temp_file_" + str(i) + ".wav"
            chunk.export(out_file, format="wav")
            amount_of_exported_chunks += 1 
        label_loading.configure(text="Splitting finished " + "[▮▮▮▮▮▮▮▮▮▯]")
        window.update()
        return True 
    except:
        print("Splitting failed")
        return False

def empty_temp_folder():
   i = 0
   for audio_file in os.listdir("C:\\lyrics_recognition\\Lyrics-recognition\\temp_folder"):
      i += 1
      os.remove("C:\\lyrics_recognition\\Lyrics-recognition\\temp_folder\\" + audio_file)
      if i > 250:
        break
    
NEW_SAMPLE_RATE = 48000

# this function resamples an audio file into the sample above and saves the resampled files into a temporary folder 
# that is emptied out at the start and end of the program
def resample():
    for audio_file in os.listdir("C:\\lyrics_recognition\\Lyrics-recognition\\temp_folder\\"): 
        if audio_file[0:2]=='r_': 
            continue
        location = "C:\\lyrics_recognition\\Lyrics-recognition\\temp_folder\\" + audio_file
        y, sr_orig = librosa.load(location, sr=None) 
        # Resample the audio to the new sample rate
        y_resampled = librosa.resample(y, orig_sr=sr_orig, target_sr=NEW_SAMPLE_RATE)
        # Save the resampled audio to a new file
        resampled_audio_file = "C:\\lyrics_recognition\\Lyrics-recognition\\temp_folder\\" + "r_" + audio_file
        sf.write(resampled_audio_file, y_resampled, NEW_SAMPLE_RATE)
        os.remove(location)

# the path to the split up audio files that have already been separated into folders based on their contents
# every folder is named after the words that it contains        
DATASET_PATH = "C:\\lyrics_recognition\\Lyrics-recognition\\audio_data"

print("##########################################################")
data_dir = pathlib.Path(DATASET_PATH)

# some preliminary steps before training the model:
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
print('Commands:', commands)

# preparing the data for use in training the model:
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=64000,
    subset='both')

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

# Imports created model
print("Importing model")
export_path = "C:\\lyrics_recognition\\Lyrics-recognition\\newsaved7"
loaded = tf.saved_model.load(export_path)
model = loaded



# Function for opening the file explorer window
def browseFiles():
    # empty previous files if any
    empty_temp_folder()
    global filename
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("Sound files",
                                                        "*.wav*"),
                                                        ("Sound files",
                                                        "*.mp3*"),
                                                       ("all files",
                                                        "*.*")))
      
    # Change label contents
    label_file_explorer.configure(text="File Opened: "+filename)
    #print(filename)
    # filename = chosen file (full path)

def getAnswer():
    global filename
    # label that shows the completion of the process
    label_loading.grid(column=1, row=11)
    window.update()
    
    path_to_split = filename
    # 0%-90%
    if not Splitter(path_to_split):
        print("Did not split")
    # change the sample rate of the audio
    resample()
    
    wordlist = []
    answer = []
    for directory_name in os.listdir("C:\\lyrics_recognition\\Lyrics-recognition\\audio_data"):
        wordlist.append(directory_name)
    # 90%-100%
    for file in os.listdir("C:\\lyrics_recognition\\Lyrics-recognition\\temp_folder"):
        x = "C:\\lyrics_recognition\\Lyrics-recognition\\temp_folder" + "\\" + file
        x = tf.io.read_file(str(x))
        x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=64000,)
        x = tf.squeeze(x, axis=-1)
        x = get_spectrogram(x)
        x = x[tf.newaxis,...]
        preddd = model(x)
        classes = np.argmax(preddd, axis = 1)
        answer.append(wordlist[classes[0]])
    label_loading.configure(text="File processed " + "[▮▮▮▮▮▮▮▮▮▮]")
    window.update()
    time.sleep(0.25)
    # label_empty.grid_forget()
    label_loading.grid_forget()
    window.update()
    return answer

def write():
    # get a list of most likely words
    answer = getAnswer()
    # writes the answer to file
    file = open("demofile2.txt", "w")
    for word in answer:
       file.write(word + " ")
    file.close()
    

    
def karaoke():
    # get a list of most likely words
    answer = getAnswer()
    i = 0
    times = []
    # reads temporary folder to determine lenght of each audio file (later used for waiting)
    for audio_file in os.listdir("C:\\lyrics_recognition\\Lyrics-recognition\\temp_folder"):
        i += 1
        y, sr = librosa.load("C:\\lyrics_recognition\\Lyrics-recognition\\temp_folder\\"+ audio_file)
        duration = librosa.get_duration(y=y, sr=sr)
        times.append(duration)
        if i > 2500:
            break
    button_explore.grid_forget()
    button_write.grid_forget()
    button_karaoke.grid_forget()
    button_exit.grid_forget()
    karaoke_text2.grid(column = 1, row = 2)
    karaoke_text1.grid(column = 1, row = 3)
    button_back.grid(column = 1, row = 5)

    # starts playing the chosen audiofile and continues the work of program
    PlaySound(filename, SND_ALIAS|SND_ASYNC)

    i = 0
    for audio_file_length in times:   
        # adds a word to the bottom row
        karaoke_text1.config(text=karaoke_text1.cget("text") + " " + answer[i])
        window.update()
        # when 6 words are written they are trasnferred to the top row, so new words can be added to the bottom row
        if i%5==0:
            karaoke_text2.config(text=karaoke_text1.cget("text"))
            karaoke_text1.config(text="")
            window.update() 
        karaoke_wait(audio_file_length)
        i+=1
    
# wait for the lenght of audio file + pause(estimated to be 0.375 on average)
def karaoke_wait(waiting_time):
    start = time.time ()
    while time.time () < start + waiting_time + 0.375:
        pass

# add created above buttons to the screen
def create_main_buttons():
    label_file_explorer.grid(column = 1, row = 1)
    button_explore.grid(column = 1, row = 2)
    button_write.grid(column = 1, row = 4)
    button_karaoke.grid(column = 1, row = 5)
    button_exit.grid(column = 1,row = 6)
    button_back.grid_forget()
    karaoke_text1.grid_forget()
    karaoke_text2.grid_forget()
    #button_back.grid(column = 100, row = 100)
    #karaoke_text1.grid(column = 100, row = 100)
    #karaoke_text2.grid(column = 100, row = 100)



# Create the root window
window = Tk()
  
# Set window title
window.title('ASRinterface')
  
# Set window size
window.geometry("1000x500")
  
#Set window background color
window.config(background = "white")
  
# Create a File Explorer label
label_file_explorer = Label(window,
                            text = "Select a file",
                            width = 100, height = 4,
                            fg = "blue")
  
      
button_explore = Button(window,
                        text = "Browse Files",
                        command = browseFiles)
  
button_exit = Button(window,
                     text = "Exit",
                     command = exit)

button_write = Button(window,
                     text = "Write",
                     command = write)

button_karaoke = Button(window,
                     text = "karaoke",
                     command = karaoke)
button_back = Button(window,
                    text = "back",
                    command = create_main_buttons)

karaoke_text1 = Label(window,
                    text = "",)

karaoke_text2 = Label(window,
                    text = "",)

label_loading = Label(window,
                      text="Loading...")

#add created above buttons to the screen
create_main_buttons()

# Let the window wait for any events
window.mainloop()