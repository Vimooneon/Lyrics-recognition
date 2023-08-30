# Lyrics-recognition

# This repository contains a speech recognition/lyrics recognition model built using Tensorflow and all files associated with its creation and use. Although the end result is far from perfect, we hope that at 
# least some of its parts can be of use.

# Process of creating a model:
# 1. Acquire training data (e.g. data from Common Voice)
# 2. Use split_audio_files to separate larger files into word-sized chunks fit for training
# 3. Use create_wordlist to filter out uncommon or otherwise improper words
# 4. Use convert_mp3_to_wav
# 5. Use delete_unused files to delete the excess mp3 files
# 6. Use resample_audio to change the sample rate to the one required to train the model
# 7. Use model_training to train a model

# If your training data is from lingualibre (the files are in ogg format and already contain one word each), then use sort_and_convert_ogg instead of steps 1-5, then continue onto step 6.

# Process of using a model:
# Launch asr_interface.py, choose an audio file and use either the “Write” button to output the recognized words to a file or “Karaoke” to see them as the song is playing in the background.
