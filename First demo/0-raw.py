from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import xlrd
import re
import time

start = time.time()

print "Loading data..."

# Parameters
NUM_FOLD = 1

# Read xlsx file
book = xlrd.open_workbook("rawData/training-Obama-Romney-tweets.xlsx")
# Output txt file
out_file = open("tweets.txt", "w")

# Size of each sheet
obama_num_entries = book.sheet_by_index(0).nrows - 3     # From row 2 to nrows-1. Row 0 and 1 
romney_num_entries = book.sheet_by_index(1).nrows - 3    # of each sheet do not contain data

# Size of the entire dataset
indices = [i for i in range(obama_num_entries + romney_num_entries)]
total_num_entries = len(indices)

vocab = {}

# NLP tasks
toker = RegexpTokenizer(r"\w+")
wordnet_lemmatizer = WordNetLemmatizer()

# ------------------------------------------------------------------------------
def add_to_vocab(tweetid, sheet):  
  # Load tweet
  tweet = sheet.cell_value(tweetid, 3)

  # Check if the tweet is valid
  if isinstance(tweet, basestring) and len(tweet) > 0:
    # Load label
    label = sheet.cell_value(tweetid, 4)

    # Check if label is valid
    if isinstance(label, float):
      label = int(label)

      # Process the tweet
      #Convert to lower case
      tweet = tweet.lower()
      #Convert www.* or https?://* to URL
      tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
      #Convert @username to AT_USER
      tweet = re.sub('@[^\s]+','AT_USER',tweet)
      #Remove additional white spaces
      tweet = re.sub('[\s]+', ' ', tweet)
      #Replace #word with word
      tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
      #trim
      tweet = tweet.strip('\'"')

      # Save to a file
      out_file.write(str(label))
      out_file.write("|||".encode('utf8'))
      out_file.write(tweet.encode('utf8'))
      out_file.write("\n".encode('utf8'))
      

# ------------------------------------------------------------------------------

# Read xlsx
print "Reading xlsx..."
for i in range(2, book.sheet_by_index(1).nrows):    # Obama
  add_to_vocab(i, book.sheet_by_index(1))

out_file.close()
end = time.time()
print "Duration: ", end - start, " seconds"
print "0-raw.py done!\n"
