import os
import time
from datetime import datetime
import re
import random
import nltk.classify
from nltk import precision, recall, f_measure, accuracy
import collections

start = time.time()
print "Loading data..."

processedTweetsFile = "processedTweets.txt"
featureListFile = "featureList.txt"


# Generate tweets.txt if it doesn't exist
if (not os.path.exists(processedTweetsFile)) or (not os.path.exists(featureListFile)):
  print "\nSummoning 1-preprocess.py..."
  os.system("python 1-preprocess.py")

tweets = []
featureList = []

# Load tweets
print "Loading processed tweets..."
ptfile = open(processedTweetsFile, "r")

for line in ptfile:
  # Remove unnecessary characters
  line = re.sub("]", "", line[2:-2])
  line = re.sub("\'", "", line)
  line = re.sub("\)", "", line)
  line = re.sub("\'", "", line)

  # Label always takes the last two characters
  label = int(line[-2:])

  # Only remove white spaces when the label is already taken out. This makes
  # sure that the last two characters of the line belong to label
  line = re.sub(" ", "", line[:-2])

  if label != 2:
    # The rest is feature vector
    featureVector = line[:-2].split(",")
    # Remove empty tokens
    featureVector = filter(None, featureVector)
    # Convert label to text label
    if label == 1:
      label = "pos"
    elif label == 0:
      label = "neu"
    elif label == -1:
      label = "neg"
    
    tweets.append((featureVector, label))

ptfile.close()

# Load feature list
print "Loading feature list..."
flfile = open(featureListFile, "r")

for line in flfile:
  if len(line) > 0:
    featureList.append(re.sub("\n","",line))

flfile.close()

# ------------------------------------------------------------------------------
def extract_features(tweet):
  tweet_words = set(tweet)
  features = {}
  for word in featureList:
    features['contains(%s)' % word] = (word in tweet_words)
  return features
# ------------------------------------------------------------------------------

# Build a dataset from features of the tweets, what we will actually use for classification
print "Extracting features..."
data = nltk.classify.util.apply_features(extract_features, tweets)
print data[0]

# Preparation for 10-fold cross validation
print "Preparing k-fold..."
NUM_SAMPLES = len(tweets)
print "Number of samples =", NUM_SAMPLES
#NUM_SAMPLES = 200
NUM_FOLD = 10
fold_size = NUM_SAMPLES / NUM_FOLD

print "Opening results.txt..."
results = open("results.txt", "w")

print "Current time: ", str(datetime.now())

for fold in range(NUM_FOLD):
  results.write("\nfold #" + str(fold) + "\n")
  # Build train & test set
  print "\nBuilding datasets for fold #", fold

  train_set, test_set = [], []
  
  if fold < NUM_FOLD - 1:
    for idx in range(NUM_SAMPLES):
      if idx < fold_size * fold or idx > fold_size * (fold+1):
        train_set.append(data[idx])
    for idx in range(fold_size * fold, fold_size * (fold + 1)):
      test_set.append(data[idx])

  else:
    for idx in range(fold_size * fold):
      train_set.append(data[idx])
    for idx in range(fold_size * fold, NUM_SAMPLES):
      test_set.append(data[idx])
    
  print "Size of train set = " + str(len(train_set)) + ", size of test set = " + str(len(test_set))
  
  print "Training..."
  NBClassifier = nltk.classify.NaiveBayesClassifier.train(train_set)
  refsets = collections.defaultdict(set)
  testsets = collections.defaultdict(set)
  refsets['pos'] = set()
  refsets['neg'] = set()
  refsets['neu'] = set()
  testsets['pos'] = set()
  testsets['neg'] = set()
  testsets['neu'] = set()
  
  print "Testing..."
  for i, (feats, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = NBClassifier.classify(feats)
    testsets[observed].add(i)

  print "Saving results..."
  results.write('pos precision:' + str(precision(refsets["pos"], testsets["pos"])) + "\n")
  results.write('pos recall:' + str(recall(refsets["pos"], testsets["pos"])) + "\n")
  results.write('pos F-measure:' + str(f_measure(refsets["pos"], testsets["pos"])) + "\n")
  results.write('neg precision:' + str(precision(refsets["neg"], testsets["neg"])) + "\n")
  results.write('neg recall:' + str(recall(refsets["neg"], testsets["neg"])) + "\n")
  results.write('neg F-measure:' + str(f_measure(refsets["neg"], testsets["neg"])) + "\n")
  results.write('neu precision:' + str(precision(refsets["neutral"], testsets["neutral"])) + "\n")
  results.write('neu recall:' + str(recall(refsets["neutral"], testsets["neutral"])) + "\n")
  results.write('neu F-measure:' + str(f_measure(refsets["neutral"], testsets["neutral"])) + "\n")

  results.write("\nMost informative features:\n")
  mif = NBClassifier.most_informative_features()
  for f in mif:
    results.write(str(f) + "\n")

results.close()
end = time.time()
print "Duration: ", end - start, " seconds"
print "2-train.py done!"
