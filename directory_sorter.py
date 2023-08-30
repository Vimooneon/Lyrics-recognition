
#make dir
import os
import shutil
# f = open("cut_down_list.txt")
# i = 0
# for line in f:
#     i+=1
#     directory = line[:-1]
#     parent_dir = "C:\\lyrics_recognition\\Lyrics-recognition\\audio_data"
#     path = os.path.join(parent_dir, directory)
#     os.mkdir(path)
#     if i > 10:
#         break 

# os.getcwd()


f = open("cut_down_list_new.txt")
wordlist = []
for line in f:
    wordlist.append(line[:-1])
#print(wordlist)

for filename in os.listdir("C:\\lyrics_recognition\\Lyrics-recognition\\SplitAudioFiles2"):
    # try:
   # with open(os.path.join(os.getcwd(), filename), 'r') as f: # open in readonly mode
      # do your stuff
    word = filename.split("_")[0]
    if word is None or len(word)<1:
        continue
    if word[len(word)-1] == "." or word[len(word)-1] == "," or word[len(word)-1] == ":" or word[len(word)-1] == "!" or word[len(word)-1] == "?" or word[len(word)-1] == ";":
        word = word[0:len(word)-1]
    if word not in wordlist:
        continue
    old_file = "C:\\lyrics_recognition\\Lyrics-recognition\\SplitAudioFiles2\\" + filename
    destination = "C:\\lyrics_recognition\\Lyrics-recognition\\audio_data\\"+word
    shutil.move(old_file, destination)
    # except:
    #     print("uh oh, i did an oopsie")
    