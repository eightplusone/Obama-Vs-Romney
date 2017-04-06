import os
import time
import re
#import random
from nltk import precision, recall, f_measure, accuracy
import collections
from sklearn import svm

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
    
    tweets.append((featureVector, label))

ptfile.close()

# Load feature list
print "Loading feature list..."
flfile = open(featureListFile, "r")

for line in flfile:
  if len(line) > 0:
    featureList.append(re.sub("\n","",line))

flfile.close()

# Build a dataset from features of the tweets, what we will actually use for classification
print "Extracting features..."
data = []
labels = []

for tweet in tweets:
  tweet_words = set(tweet[0])
  features = []
  for word in featureList:
    if word in tweet_words:
      features.append(1)
    else:
      features.append(0)
  data.append(features)
  labels.append(tweet[1])

# Preparation for 10-fold cross validation
print "Preparing k-fold..."
NUM_SAMPLES = len(labels)
print "Number of samples =", NUM_SAMPLES
#NUM_SAMPLES = 200
NUM_FOLD = 10
fold_size = NUM_SAMPLES / NUM_FOLD

print "Opening results.txt..."
results = open("results.txt", "w")

for fold in range(NUM_FOLD):
  results.write("\nfold #" + str(fold) + "\n")
  # Build train & test set
  print "\nBuilding datasets for fold #", fold

  train_set, test_set = [], []
  train_label, test_label = [], []
  
  if fold < NUM_FOLD - 1:
    for idx in range(NUM_SAMPLES):
      if idx < fold_size * fold or idx > fold_size * (fold+1):
        train_set.append(data[idx])
        train_label.append(labels[idx])
    for idx in range(fold_size * fold, fold_size * (fold + 1)):
      test_set.append(data[idx])
      test_label.append(labels[idx])

  else:
    for idx in range(fold_size * fold):
      train_set.append(data[idx])
      train_label.append(labels[idx])
    for idx in range(fold_size * fold, NUM_SAMPLES):
      test_set.append(data[idx])
      test_label.append(labels[idx])
    
  print "Size of train set = " + str(len(train_label)) + ", size of test set = " + str(len(test_label))
  
  print "Training..."
  classifier = svm.SVC(kernel="linear", decision_function_shape="ovr", probability=False)
  classifier.fit(train_set, train_label)
  
  print "Testing..."
  refsets = collections.defaultdict(set)
  testsets = collections.defaultdict(set)
  refsets[1] = set()
  refsets[-1] = set()
  refsets[0] = set()
  testsets[1] = set()
  testsets[-1] = set()
  testsets[0] = set()
  for i in range(len(test_set)):
    refsets[test_label[i]].add(i)
    observed = classifier.predict([test_set[i]])
    testsets[observed[0]].add(i)

  print "Saving results..."
  results.write('pos precision:' + str(precision(refsets[1], testsets[1])) + "\n")
  results.write('pos recall:' + str(recall(refsets[1], testsets[1])) + "\n")
  results.write('pos F-measure:' + str(f_measure(refsets[1], testsets[1])) + "\n")
  results.write('neg precision:' + str(precision(refsets[-1], testsets[-1])) + "\n")
  results.write('neg recall:' + str(recall(refsets[-1], testsets[-1])) + "\n")
  results.write('neg F-measure:' + str(f_measure(refsets[-1], testsets[-1])) + "\n")
  
results.close()
end = time.time()
print "Duration: ", end - start, " seconds"
print "2-train.py done!"
