

alphabet = "qwertyuiopasdfghjklzxcvbnm"
# this function checks if there are any non-standard symbols within a word
def check_for_special_symbols(word):
    for letter in word:
        if letter not in alphabet:
            return True
    return False


unsorted_wordlist = {}
# wordlistnew.txt consists of all the extracted words; this file was created after using NewSplitter.py (which was later incorporated into asrinterface)
# wordlistnew contains lines of words and many times they were encountered
file = open("wordlistnew.txt")
for line in file:
    split_line = line.split(" ")
    word = str(split_line[0].lower())
    # check if word is empty
    if word is None or len(word)<1:
        continue
    # remove the last symbol if it is non-standard, while keeping the word
    if word[len(word)-1] == "." or word[len(word)-1] == "," or word[len(word)-1] == ":" or word[len(word)-1] == "!" or word[len(word)-1] == "?" or word[len(word)-1] == ";":
        word = word[0:len(word)-1]
    # checks if there are any other non-standard symbols within a word
    if check_for_special_symbols(word):
        continue
    # count = how many times that word was extracted
    count = int(split_line[1])
    if unsorted_wordlist.get(word) is None:
        unsorted_wordlist.update({word:count})
    else:                   
        unsorted_wordlist.update({word:unsorted_wordlist.get(word)+count})
file.close()

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

file = open("sorted_list_new.txt", "w") 
# write the sorted list into a new file
for i, v in enumerate(sorted_wordlist):  
    if sorted_wordlist.get(v) > 2: # only write words that are encountered 3+ times
        file.write(v + " " + str(sorted_wordlist.get(v)) + "\n")

file.close()

# 3kwords is a list of the 3 thousand most popular english words, without names and proper nouns
file = open("3kwords.txt")
wordlist = []
for line in file:
    wordlist.append(line[0:-1])
file.close()

file = open("sorted_list_new.txt")
end_file = open("cut_down_list_new.txt", "w")
# cut out words that are not in 3kwords.txt
for line in file:
    split_line = line.split(" ")
    word = split_line[0]
    if word in wordlist:
        end_file.write(word + "\n")
file.close()
end_file.close()
# the end result, cut_down_list_new, is a sorted list of the words that are included in the vocabulary of our ASR system