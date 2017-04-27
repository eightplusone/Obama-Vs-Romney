import os
import xlrd
import time
import random
import math

start = time.time()

print "\nGenerating data for folds... "

# Parameters
NUM_FOLD = 10

# Size of each dataset
oba_dataset_size = 0
rom_dataset_size = 0

# Read xlsx file
book = xlrd.open_workbook("data/training-Obama-Romney-tweets.xlsx")

print "  Calculating size of the two datasets... ",

oba_dataset_size = book.sheet_by_index(0).nrows - 3
rom_dataset_size = book.sheet_by_index(1).nrows - 3

oba_fold_size = oba_dataset_size / NUM_FOLD
rom_fold_size = rom_dataset_size / NUM_FOLD

print "done"

oba_input_file = "oba_tweets.txt"
rom_input_file = "rom_tweets.txt"

oba_input = open(oba_input_file, "r")
rom_input = open(rom_input_file, "r")

print "  Shuffling indices... ", 

oba_index = range(2, oba_dataset_size + 2)
rom_index = range(2, rom_dataset_size + 2)

random.shuffle(oba_index)
random.shuffle(rom_index)

print "done"

#
# Assign Obama tweets to sub datasets
#
print "  Assigning Obama tweets..."

print "    Creating files and folders...",
# Create a folder named 'tmp' if it does not exist
if not os.path.exists("tmp"):
  os.system("mkdir tmp")
if not os.path.exists("tmp/oba"):
  os.system("mkdir tmp/oba")

# Path prefix for all txt files
prefix = "tmp/oba/"

# File names are stored in lists
fold_train_file = [""] * NUM_FOLD
fold_test_file = [""] * NUM_FOLD

# Generate file names
for fold in range(NUM_FOLD):
  if fold < 9:
    fold_train_file[fold] = prefix + "tr_0" + str(fold+1) + ".txt"
    fold_test_file[fold] = prefix + "te_0" + str(fold+1) + ".txt"
  else:
    fold_train_file[fold] = prefix + "tr_" + str(fold+1) + ".txt"
    fold_test_file[fold] = prefix + "te_" + str(fold+1) + ".txt"

  # If the files exist, remove them
  if os.path.exists(fold_train_file[fold]):
    os.system("rm " + fold_train_file[fold])

  if os.path.exists(fold_test_file[fold]):
    os.system("rm " + fold_test_file[fold])

print "done"
print "  Assigning data to train and test subsets... ",

# Assign data to files
#print oba_dataset_size, len(oba_index), rom_dataset_size, len(rom_index)
for i in range(oba_dataset_size):
  idx = int(math.floor(i/oba_fold_size))

  for fold in range(NUM_FOLD):
    if idx < NUM_FOLD and fold != idx:    # Train data
      outfile = open(fold_train_file[fold], "a")

      tweetid = oba_index[i]
      tweet = book.sheet_by_index(0).cell_value(tweetid, 3)

       # Check if the tweet is valid
      if isinstance(tweet, basestring) and len(tweet) > 0:
        # Load label
        label = book.sheet_by_index(0).cell_value(tweetid, 4)

        # Check if label is valid. Also, ignore class #2
        if isinstance(label, float) and label < 2:
          label = int(label)

          # Re-format label so it takes exactly one letter
          # per line. "-1" takes two.
          if label == 1:
            label = "+"
          elif label == -1:
            label = "-"
          elif label == 0:
            label = "0"

          # Save to txt file
          outstring = label + "|||||" + tweet + "\n"
          outfile.write(outstring.encode("utf8"))

      outfile.close()
    else:                                # Test data
      outfile = open(fold_test_file[fold], "a")

      tweetid = oba_index[i]
      #print tweetid
      tweet = book.sheet_by_index(0).cell_value(tweetid, 3)

       # Check if the tweet is valid
      if isinstance(tweet, basestring) and len(tweet) > 0:
        # Load label
        label = book.sheet_by_index(0).cell_value(tweetid, 4)

        # Check if label is valid. Also, ignore class #2
        if isinstance(label, float) and label < 2:
          label = int(label)

          # Re-format label so it takes exactly one letter
          # per line. "-1" takes two.
          if label == 1:
            label = "+"
          elif label == -1:
            label = "-"
          elif label == 0:
            label = "0"

          # Save to txt file
          outstring = label + "|||||" + tweet + "\n"
          outfile.write(outstring.encode("utf8"))

      outfile.close()

print "done"


#
# Assign Romney tweets to sub datasets
#
print "  Assigning Romney tweets..."

print "    Creating files and folders...",
# Create a folder named 'tmp' if it does not exist
if not os.path.exists("tmp"):
  os.system("mkdir tmp")
if not os.path.exists("tmp/rom"):
  os.system("mkdir tmp/rom")

# Path prefix for all txt files
prefix = "tmp/rom/"

# File names are stored in lists
fold_train_file = [""] * NUM_FOLD
fold_test_file = [""] * NUM_FOLD

# Generate file names
for fold in range(NUM_FOLD):
  if fold < 9:
    fold_train_file[fold] = prefix + "tr_0" + str(fold+1) + ".txt"
    fold_test_file[fold] = prefix + "te_0" + str(fold+1) + ".txt"
  else:
    fold_train_file[fold] = prefix + "tr_" + str(fold+1) + ".txt"
    fold_test_file[fold] = prefix + "te_" + str(fold+1) + ".txt"

  # If the files exist, remove them
  if os.path.exists(fold_train_file[fold]):
    os.system("rm " + fold_train_file[fold])

  if os.path.exists(fold_test_file[fold]):
    os.system("rm " + fold_test_file[fold])

print "done"
print "  Assigning data to train and test subsets... ",

# Assign data to files
#print oba_dataset_size, len(oba_index), rom_dataset_size, len(rom_index)
for i in range(rom_dataset_size):
  idx = int(math.floor(i/rom_fold_size))

  for fold in range(NUM_FOLD):
    if idx < NUM_FOLD and fold != idx:    # Train data
      outfile = open(fold_train_file[fold], "a")

      tweetid = oba_index[i]
      tweet = book.sheet_by_index(1).cell_value(tweetid, 3)

       # Check if the tweet is valid
      if isinstance(tweet, basestring) and len(tweet) > 0:
        # Load label
        label = book.sheet_by_index(1).cell_value(tweetid, 4)

        # Check if label is valid. Also, ignore class #2
        if isinstance(label, float) and label < 2:
          label = int(label)

          # Re-format label so it takes exactly one letter
          # per line. "-1" takes two.
          if label == 1:
            label = "+"
          elif label == -1:
            label = "-"
          elif label == 0:
            label = "0"

          # Save to txt file
          outstring = label + "|||||" + tweet + "\n"
          outfile.write(outstring.encode("utf8"))

      outfile.close()
    else:                                # Test data
      outfile = open(fold_test_file[fold], "a")

      tweetid = rom_index[i]
      #print tweetid
      tweet = book.sheet_by_index(1).cell_value(tweetid, 3)

       # Check if the tweet is valid
      if isinstance(tweet, basestring) and len(tweet) > 0:
        # Load label
        label = book.sheet_by_index(1).cell_value(tweetid, 4)

        # Check if label is valid. Also, ignore class #2
        if isinstance(label, float) and label < 2:
          label = int(label)

          # Re-format label so it takes exactly one letter
          # per line. "-1" takes two.
          if label == 1:
            label = "+"
          elif label == -1:
            label = "-"
          elif label == 0:
            label = "0"

          # Save to txt file
          outstring = label + "|||||" + tweet + "\n"
          outfile.write(outstring.encode("utf8"))

      outfile.close()

print "done"


end = time.time()
print "Duration: ", end - start, " seconds"
print "1-generate_folds.py done!\n"
