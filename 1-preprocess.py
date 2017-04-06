import os
import time
import re
import random
from nltk.corpus import stopwords

start = time.time()
print "Loading data..."

datafilename = "tweets.txt"

# Generate tweets.txt if it doesn't exist
if not os.path.exists(datafilename):
  print "\nSummoning 0-raw.py..."
  os.system("python 0-raw.py")

# Load data
datafile = open(datafilename, "r")

tweets = []
featureList = []

# ------------------------------------------------------------------------------
def getFeatureVector(tweet):
  # Initiate a list of feature vectors
  featureVector = []
  # Split tweet into words
  words = tweet.split()
  
  for w in words:
    # Replace two or more with two occurrences
    #w = replaceTwoOrMore(w)
    # Strip punctuation
    w = w.strip('\'"?,.')
    # Check if the word stats with an alphabet
    val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
    # Ignore if it is a stop word
    if(w in set(stopwords.words('english')) or w=="AT_USER" or w=="URL" or val is None):
      continue
    else:
      featureVector.append(w.lower())
  return featureVector
# ------------------------------------------------------------------------------
print "Reading tweets.txt..."
for line in datafile:
  # Get label and tweet
  label, tweet = line.split("|||")
  # Build a feature vector from the tweet
  featureVector = getFeatureVector(tweet)
  # Add features to featureVector
  for f in featureVector:
    if f not in featureList:
      featureList.append(f)
  # Add to dataset
  tweets.append((featureVector, label))

# Shuffle data
print "Shuffling data..."
random.shuffle(tweets)

# Output txt file
print "Writing into processedTweets.txt..."
processed_tweets_file = open("processedTweets.txt", "w")
for t in tweets:
  processed_tweets_file.write(str(t) + "\n")
processed_tweets_file.close()

print "Writing into featureList.txt..."
featureList_file = open("featureList.txt", "w")
for f in featureList:
  featureList_file.write(str(f) + "\n")
featureList_file.close()

datafile.close()
end = time.time()
print "Duration: ", end - start, " seconds"
print "1-preprocess.py done!\n"
