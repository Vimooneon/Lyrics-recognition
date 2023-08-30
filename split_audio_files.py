import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
from IPython.display import Audio
import librosa
from pydub import AudioSegment
from pydub.playback import play
import soundfile as sf
from pydub.silence import split_on_silence
import pydub
import numpy as np
import csv

# NewSplitter is the first step in training a model - it splits the dataset into word-sized audio chunks (that will be used for training)

# this function trims an audio file (i.e. removes silence at the beginning and end of audio file)
def trim_func(file_path):
    y, sr = sf.read(file_path)
    y_trimmed, index = librosa.effects.trim(y=y, top_db=16)
    return y_trimmed, sr

# this function is vital for recognizing the lyrics of a song
# it separates the vocal from the instrumentals at the cost of audio quality
def vocal_sep_func(y_trimmed, sr):
    lower_limit = 0
    duration = int(librosa.get_duration(y=y_trimmed, sr=sr))
    S_full, phase = librosa.magphase(librosa.stft(y_trimmed))
    idx = slice(*librosa.time_to_frames([lower_limit, duration], sr=sr))
    fig, ax = plt.subplots()
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
    S_background = mask_i * S_full
    y_foreground = librosa.istft(S_foreground * phase)
    return y_foreground[lower_limit*sr:duration*sr]

# this function normalizes an audio file, making its volume more consistent
# it also makes the audio 3x louder to counteract the loss of volume from the vocal separation step
def normalization_function(data):
    data = 3 * librosa.util.normalize(data)
    return data

skip_first = True
# CONST is the amount of words that are actually considered
CONST = 5
wordlist = {
    
} 
# this was tuned to the Common Voice dataset, one of the biggest datasets of its kind
# it comes with a list of all validated files (validated.tsv) that also has additional information on each file
# such as the sentence spoken, the speaker's nationality, gender etc.
amount_of_exported_chunks = 0
with open(r"C:\zpd_data\validated.tsv", encoding="utf8") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        if amount_of_exported_chunks % 100 == 0 and amount_of_exported_chunks>0:
            print(amount_of_exported_chunks)
        if amount_of_exported_chunks>20000:
            break
        try:
            if skip_first:
                skip_first = False
                continue           
            # line[7] of the validated.tsv file contains the nationality of the speaker
            if not ("united states" in line[7].lower() or "us" in line[7].lower() or "uk" in line[7].lower() or "united kingdom" in line[7].lower() or "great britain" in line[7].lower() or "england" in line[7].lower()):
                continue 
            # line[1] of the validated.tsv file contains the path to the file
            pathing = line[1]  
            # line[2] of the validated.tsv file contains the sentence spoken
            # Divide the sentence into words
            sentence = line[2].split(" ")         
            file_path = r"C:\zpd_data\clips" + "\\" + pathing
            # Trim the beginning and ending of the audio file
            trimmed_file, sr = trim_func(file_path)
            # normalize the file also increasing it's volume
            normal_file = normalization_function(trimmed_file)
            sf.write('temp_file.wav', normal_file, sr, subtype='PCM_24')
            sound_file = AudioSegment.from_wav('temp_file.wav')
            
            # The splitting process
            audio_chunks = split_on_silence(sound_file, 
                min_silence_len=75,
                # consider it silent if quieter than -20 dBFS
                silence_thresh=-20
            )

            # determines if incorrect amount of audio chunks was produced 
            if len(audio_chunks) != len(sentence):
                continue
            filename = pathing[-12:-4]
            
            for i, chunk in enumerate(audio_chunks):
                # since the quality of such splitting might decrease over the amount of chunks per sentecnde
                # a constant that stops the process at certain point was introduced
                if (i>CONST): 
                    break
                word = sentence[i]
                # These "if" statements exist for cases when word length is unnatural for the audio chunk length
                if len(word) <= 7 and chunk.duration_seconds > 0.550:
                    break
                if len(word) >= 5 and chunk.duration_seconds < 0.200:
                    break                           
                if wordlist.get(word) is None:
                    wordlist.update({word:1})
                else:
                    wordlist.update({word:wordlist.get(word)+1})
                amount_of_exported_chunks += 1
        except:
            continue
file.close()

# writes all words encountered in splitting process to a seperate file
file = open("C:\\lyrics_recognition\\Lyrics-recognition\\wordlistnew.txt", "w") 
for i, v in enumerate(wordlist):
    file.write(v + " " + str(wordlist.get(v)) + "\n")
file.close() 