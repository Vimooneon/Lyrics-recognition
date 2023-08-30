import os
from pydub import AudioSegment

f = open("cut_down_list_new.txt")
wordlist = []
for line in f:
    wordlist.append(line[:-1])


i=0
# folder names are words from wordlist
for word in wordlist:
    try:
        for filename in os.listdir("C:\\lyrics_recognition\\Lyrics-recognition\\audio_data\\"+word):
            try:
                # converts mp3 to wav
                dst = "C:\\lyrics_recognition\\Lyrics-recognition\\audio_data\\"+word+"\\"+filename[:-4]+".wav"
                # checks if the wav file already exist
                if not os.path.exists(dst):
                    src = "C:\\lyrics_recognition\\Lyrics-recognition\\audio_data\\"+word+"\\"+filename
                    sound = AudioSegment.from_mp3(src)
                    sound.export(dst, format="wav")
            except:
                continue
    except:
        continue