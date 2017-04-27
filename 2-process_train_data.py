import os
import xlrd
import time
import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

start = time.time()

print "\nProcessing training data... "

# Parameters
NUM_FOLD = 10


#
# Obama
#
print "  Processing Obama tweets..."

# Remove old data
if os.path.exists("tmp/oba_processed"):
  os.system("rm -rf tmp/oba_processed")

os.system("mkdir tmp/oba_processed")

for fold in range(NUM_FOLD):
  infile_name = ""
  outfile_name = ""
  if fold < 9:
    infile_name = "tmp/oba/tr_0" + str(fold + 1) + ".txt"
    outfile_name = "tmp/oba_processed/tr_0" + str(fold + 1) + ".txt"
  else:
    infile_name = "tmp/oba/tr_" + str(fold + 1) + ".txt"
    outfile_name = "tmp/oba_processed/tr_" + str(fold + 1) + ".txt"

  infile = open(infile_name, "r")
  outfile = open(outfile_name, "a")
  print "    Processing then writing tweets from file " + infile.name + " to file " + outfile.name + " ... ",
  
  for line in infile:
    if "|||||" in line:

      # Get label and tweet
      label, tweet = line.split("|||||")

      # Process the tweet
      # Convert to lower case
      tweet = tweet.lower()
      # Convert www.* or https?://* to URL
      tweet = re.sub("((www\.[^\s]+)|(https?://[^\s]+))","URL",tweet)
      # Convert @username to AT_USER
      tweet = re.sub("@[^\s]+","AT_USER",tweet)
      # Remove AT_USER and URL
      tweet = re.sub("AT_USER", "", tweet)
      tweet = re.sub("URL", "", tweet)
      # Remove HTML tags
      tweet = re.sub("<.*?>", "", tweet)
      # Remove unicode symbols
      tweet = re.sub("&amp;", "", tweet)
      tweet = re.sub("&lt;", "", tweet)
      tweet = re.sub("&gt;", "", tweet)
      # Remove additional white spaces
      tweet = re.sub("[\s]+", " ", tweet)
      # Replace #word with word
      tweet = re.sub("#([^\s]+)", r"\1", tweet)
      # Strip punctuation
      #tweet = re.sub("\.|!|,|\(|\)|-|>|<|\?|:|%|&", "", tweet)
      # Remove quotation marks
      tweet = re.sub("\'|\"", "", tweet)

      # Write to outfile
      outstring = label + "|||||" + tweet + "\n"
      outfile.write(outstring)

  outfile.close()
  print "done"

  if fold < 9:
    infile_name = "tmp/oba/te_0" + str(fold + 1) + ".txt"
    outfile_name = "tmp/oba_processed/te_0" + str(fold + 1) + ".txt"
  else:
    infile_name = "tmp/oba/te_" + str(fold + 1) + ".txt"
    outfile_name = "tmp/oba_processed/te_" + str(fold + 1) + ".txt"

  infile = open(infile_name, "r")
  outfile = open(outfile_name, "a")
  print "    Processing then writing tweets from file " + infile.name + " to file " + outfile.name + " ... ",
  
  for line in infile:
    if "|||||" in line:

      # Get label and tweet
      label, tweet = line.split("|||||")

      # Process the tweet
      # Convert to lower case
      tweet = tweet.lower()
      # Convert www.* or https?://* to URL
      tweet = re.sub("((www\.[^\s]+)|(https?://[^\s]+))","URL",tweet)
      # Convert @username to AT_USER
      tweet = re.sub("@[^\s]+","AT_USER",tweet)
      # Remove AT_USER and URL
      tweet = re.sub("AT_USER", "", tweet)
      tweet = re.sub("URL", "", tweet)
      # Remove HTML tags
      tweet = re.sub("<.*?>", "", tweet)
      # Remove unicode symbols
      tweet = re.sub("&amp;", "", tweet)
      tweet = re.sub("&lt;", "", tweet)
      tweet = re.sub("&gt;", "", tweet)
      # Convert from unicode to ascii
      tweet.decode('utf-8').encode('ascii', errors='ignore')
      # Replace #word with word
      tweet = re.sub("#([^\s]+)", r"\1", tweet)
      # Strip non-letters and non-numbers
      # List given in https://arxiv.org/pdf/1509.01626.pdf
      tweet = re.sub("&", " ", tweet)
      tweet = re.sub("-", " ", tweet)
      tweet = re.sub(",", " ", tweet)
      tweet = re.sub(";", " ", tweet)
      tweet = re.sub(".", " ", tweet)
      tweet = re.sub("!", " ", tweet)
      tweet = re.sub("\?", " ", tweet)
      tweet = re.sub(":", " ", tweet)
      tweet = re.sub("'", " ", tweet)
      tweet = re.sub("\"", " ", tweet)
      tweet = re.sub("/", " ", tweet)
      tweet = re.sub("(\.)+", " ", tweet)
      tweet = re.sub("|", " ", tweet)
      tweet = re.sub("_", " ", tweet)
      tweet = re.sub("@", " ", tweet)
      tweet = re.sub("#", " ", tweet)
      tweet = re.sub("$", " ", tweet)
      tweet = re.sub("%", " ", tweet)
      tweet = re.sub("^", " ", tweet)
      tweet = re.sub("&", " ", tweet)
      tweet = re.sub("\*", " ", tweet)
      tweet = re.sub("~", " ", tweet)
      tweet = re.sub("`", " ", tweet)
      tweet = re.sub("\+", " ", tweet)
      tweet = re.sub("-", " ", tweet)
      tweet = re.sub("=", " ", tweet)
      tweet = re.sub("<", " ", tweet)
      tweet = re.sub(">", " ", tweet)
      tweet = re.sub("\(", " ", tweet)
      tweet = re.sub("\)", " ", tweet)
      tweet = re.sub("\[", " ", tweet)
      tweet = re.sub("\]", " ", tweet)
      tweet = re.sub("{", " ", tweet)
      tweet = re.sub("}", " ", tweet)
      # Remove additional white spaces
      tweet = re.sub("[\s]+", " ", tweet)
      # Write to outfile
      outstring = label + "|||||" + tweet + "\n"
      outfile.write(outstring)

  outfile.close()
  print "done"


#
# Romney
#
print "  Processing Romney tweets..."

# Remove old data
if os.path.exists("tmp/rom_processed"):
  os.system("rm -rf tmp/rom_processed")

os.system("mkdir tmp/rom_processed")

for fold in range(NUM_FOLD):
  infile_name = ""
  outfile_name = ""
  if fold < 9:
    infile_name = "tmp/rom/tr_0" + str(fold + 1) + ".txt"
    outfile_name = "tmp/rom_processed/tr_0" + str(fold + 1) + ".txt"
  else:
    infile_name = "tmp/rom/tr_" + str(fold + 1) + ".txt"
    outfile_name = "tmp/rom_processed/tr_" + str(fold + 1) + ".txt"

  infile = open(infile_name, "r")
  outfile = open(outfile_name, "a")
  print "    Processing then writing tweets from file " + infile.name + " to file " + outfile.name + " ... ",
  
  for line in infile:
    if "|||||" in line:

      # Get label and tweet
      label, tweet = line.split("|||||")

      # Process the tweet
      # Convert to lower case
      tweet = tweet.lower()
      # Convert www.* or https?://* to URL
      tweet = re.sub("((www\.[^\s]+)|(https?://[^\s]+))","URL",tweet)
      # Convert @username to AT_USER
      tweet = re.sub("@[^\s]+","AT_USER",tweet)
      # Remove AT_USER and URL
      tweet = re.sub("AT_USER", "", tweet)
      tweet = re.sub("URL", "", tweet)
      # Remove HTML tags
      tweet = re.sub("<.*?>", "", tweet)
      # Remove unicode symbols
      tweet = re.sub("&amp;", "", tweet)
      tweet = re.sub("&lt;", "", tweet)
      tweet = re.sub("&gt;", "", tweet)
      # Remove additional white spaces
      tweet = re.sub("[\s]+", " ", tweet)
      # Replace #word with word
      tweet = re.sub("#([^\s]+)", r"\1", tweet)
      # Strip punctuation
      #tweet = re.sub("\.|!|,|\(|\)|-|>|<|\?|:|%|&", "", tweet)
      # Remove quotation marks
      tweet = re.sub("\'|\"", "", tweet)

      # Write to outfile
      outstring = label + "|||||" + tweet + "\n"
      outfile.write(outstring)

  outfile.close()
  print "done"

  if fold < 9:
    infile_name = "tmp/rom/te_0" + str(fold + 1) + ".txt"
    outfile_name = "tmp/rom_processed/te_0" + str(fold + 1) + ".txt"
  else:
    infile_name = "tmp/rom/te_" + str(fold + 1) + ".txt"
    outfile_name = "tmp/rom_processed/te_" + str(fold + 1) + ".txt"

  infile = open(infile_name, "r")
  outfile = open(outfile_name, "a")
  print "    Processing then writing tweets from file " + infile.name + " to file " + outfile.name + " ... ",
  
  for line in infile:
    if "|||||" in line:

      # Get label and tweet
      label, tweet = line.split("|||||")

      # Process the tweet
      # Convert to lower case
      tweet = tweet.lower()
      # Convert www.* or https?://* to URL
      tweet = re.sub("((www\.[^\s]+)|(https?://[^\s]+))","URL",tweet)
      # Convert @username to AT_USER
      tweet = re.sub("@[^\s]+","AT_USER",tweet)
      # Remove AT_USER and URL
      tweet = re.sub("AT_USER", "", tweet)
      tweet = re.sub("URL", "", tweet)
      # Remove HTML tags
      tweet = re.sub("<.*?>", "", tweet)
      # Remove unicode symbols
      tweet = re.sub("&amp;", "", tweet)
      tweet = re.sub("&lt;", "", tweet)
      tweet = re.sub("&gt;", "", tweet)
      # Remove additional white spaces
      tweet = re.sub("[\s]+", " ", tweet)
      # Replace #word with word
      tweet = re.sub("#([^\s]+)", r"\1", tweet)
      # Strip punctuation
      #tweet = re.sub("\.|!|,|\(|\)|-|>|<|\?|:|%|&", "", tweet)
      # Remove quotation marks
      tweet = re.sub("\'|\"", "", tweet)

      # Write to outfile
      outstring = label + "|||||" + tweet + "\n"
      outfile.write(outstring)

  outfile.close()
  print "done"

end = time.time()
print "Duration: ", end - start, " seconds"
print "2-process_train_data.py done!\n"
