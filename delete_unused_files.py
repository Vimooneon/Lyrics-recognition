import os

i = 0
for directory_name in os.listdir("C:\\lyrics_recognition\\Lyrics-recognition\\audio_data"):
  try: 
    # rmdir removes empty directories, if they are not empty, it creates an error and enters the except clause
    os.rmdir("C:\\lyrics_recognition\\Lyrics-recognition\\audio_data\\" + directory_name)
  except:
    for audio_file in os.listdir("C:\\lyrics_recognition\\Lyrics-recognition\\audio_data\\" + directory_name):
      #deletes non .wav files, is used after convert_mp3_to_wav.py to get rid of .mp3 files
      if not audio_file[len(audio_file)-4:] == ".wav":
         os.remove("C:\\lyrics_recognition\\Lyrics-recognition\\audio_data\\" + directory_name + "\\" + audio_file)
