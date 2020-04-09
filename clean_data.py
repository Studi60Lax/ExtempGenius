import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import download
from nltk import word_tokenize
from bs4 import BeautifulSoup
import re
import string
import unicodedata
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download("wordnet")


'''All of these are functions to normalize the data'''

def remove_between_square_brackets(text):
    '''uses regex to remove all of the square brackets since the result is a list'''
    return re.sub('\[[^]]*\]', '', text)


def tokenize(text):
    words = nltk.word_tokenize(text) # puts all of the words into a list
    return (words)



def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    new_words = []
    dirty_characters = "!@#$%^&*()[]{};:,./<>?\|`~-=_+"
    for word in words:
        if (word not in dirty_characters) and (word != "''" and word != "``") and ("'" not in word):
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    stop = set(stopwords.words('english'))
    new_words = []
    for word in words:
        if not word in stop:
            new_words.append(word)
    return new_words

def lemmatize(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    new_words = []
    for word in words:
        new_words.append(lemmatizer.lemmatize(word)) # lemmatizes the words in the list and adds them to the new list
    return new_words



def normalize(words):
    '''Applies all previous functions to isolate nonuseful data'''
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    words = lemmatize(words)
    return words

def getkeywords(words):
    common = open("common.txt").read().split('\n')
    worddict = {}

    for word in words:
        if word not in common:
            if word not in worddict:
                worddict[word] = 1
            if word in worddict:
                worddict[word] += 1
        elif word in common:
            continue
    word_frequency = sorted(worddict.items(),key = lambda kv:(kv[1], kv[0]), reverse = True)[0:]
    top_10 = word_frequency[0:10]

    return top_10

''' Main program'''
def full_clean(string):
    words = tokenize(remove_between_square_brackets(string))
    words = normalize(words)
    words = getkeywords(words)
    return words
