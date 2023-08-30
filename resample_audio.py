import librosa
import soundfile as sf
import os

NEW_SAMPLE_RATE = 48000
for directory_name in os.listdir("C:\\lyrics_recognition\\Lyrics-recognition\\audio_data"):
    for audio_file in os.listdir("C:\\lyrics_recognition\\Lyrics-recognition\\audio_data\\" + directory_name): 
        if audio_file[0:2]=='r_': 
            continue
        location = "C:\\lyrics_recognition\\Lyrics-recognition\\audio_data\\" + directory_name + "\\" + audio_file
        
        y, sr_orig = librosa.load(location, sr=None)  # sr=None to preserve the original sample rate
        # Resample the audio to the new sample rate
        y_resampled = librosa.resample(y, orig_sr=sr_orig, target_sr=NEW_SAMPLE_RATE)
        # Save the resampled audio to a new file
        resampled_audio_file = "C:\\lyrics_recognition\\Lyrics-recognition\\audio_data\\" + directory_name + "\\" + "r_" + audio_file
        sf.write(resampled_audio_file, y_resampled, NEW_SAMPLE_RATE)
        os.remove(location)
