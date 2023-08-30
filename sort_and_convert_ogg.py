import os
from pydub import AudioSegment

unsorted_wordlist = {}

# Q22-eng-English is a dataset of spoken english words from https://lingualibre.org/datasets/
for directoryname in os.listdir("C:\\lyrics_recognition\\Lyrics-recognition\\Q22-eng-English"):
    for filename in os.listdir("C:\\lyrics_recognition\\Lyrics-recognition\\Q22-eng-English\\"+directoryname): 
        split_line = filename.split(".")
        word = str(split_line[0].lower())
        if unsorted_wordlist.get(word) is None:
            unsorted_wordlist.update({word:1})
        else:                   
            unsorted_wordlist.update({word:unsorted_wordlist.get(word)+1})


sorted_wordlist = {}
# very ineffective sorting algorithm
it = 0
while (it < len(unsorted_wordlist)):
    max = 0
    saved_word = ""
    for i, v in enumerate(unsorted_wordlist):
        if v not in sorted_wordlist and unsorted_wordlist.get(v) > max:
            max = unsorted_wordlist.get(v)
            saved_word = v
    sorted_wordlist.update({saved_word:max})
    it += 1

# write the sorted list into a new file
file = open("ogg_sorted_list.txt", "w") 
for i, v in enumerate(sorted_wordlist):  
    if sorted_wordlist.get(v) > 2:
        file.write(v + " " + str(sorted_wordlist.get(v)) + "\n")
file.close()



file = open("ogg_sorted_list.txt")
wordlist = []
for line in file:
    line = line.split(" ")
    wordlist.append(line[0])


i=0
# transfer and sort audio files from Q22-eng-English directory
for directoryname in os.listdir("C:\\lyrics_recognition\\Lyrics-recognition\\Q22-eng-English"):
    for filename in os.listdir("C:\\lyrics_recognition\\Lyrics-recognition\\Q22-eng-English\\"+directoryname): 
        i+=1
        try:         
            # filename[:-4] is word without .ogg
            if filename[:-4] in wordlist:
                dst = "C:\\lyrics_recognition\\Lyrics-recognition\\audio_data\\"+filename[:-4].lower()+"\\"+filename[:-4].lower()+ str(i) + ".wav"
                if not os.path.exists(dst):
                    # converts the ogg file to wav format
                    src = "C:\\lyrics_recognition\\Lyrics-recognition\\Q22-eng-English\\"+directoryname+"\\"+filename
                    sound = AudioSegment.from_ogg(src)
                    sound.export(dst, format="wav")
        except:
            continue