import os
import shutil

file = open("cut_down_list_new.txt")
wordlist = []
for line in file:
    wordlist.append(line[:-1]) #remove \n at the end of the line
file.close()

#transfers files from SplitAudioFiles directory
#and sorts audio files resulting from split_audio_files.py into folders 
for filename in os.listdir("C:\\lyrics_recognittion\\Lyrics-recognition\\audio_data"):
    word = filename.split("_")[0]
    if word[len(word)-1] == "." or word[len(word)-1] == "," or word[len(word)-1] == ":" or word[len(word)-1] == "!" or word[len(word)-1] == "?" or word[len(word)-1] == ";":
        word = word[0:len(word)-1]
    if word not in wordlist:
        continue
    old_file = "C:\\lyrics_recognition\\Lyrics-recognition\\SplitAudioFiles\\" + filename
    destination = "C:\\lyrics_recognition\\Lyrics-recognition\\audio_data\\"+word
    shutil.move(old_file, destination)