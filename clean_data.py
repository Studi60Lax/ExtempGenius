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
    with open("common.txt") as file:
        common_words = set(file.read().split('\n'))
    stop = stop|common_words
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
    worddict = {}

    for word in words:
        if word not in worddict:
            worddict[word] = 1
        if word in worddict:
            worddict[word] += 1

    word_frequency = sorted(worddict.items(),key = lambda kv:(kv[1], kv[0]), reverse = True)[0:]
    top_10 = word_frequency[0:10]

    return top_10



def split_into_sentences(text):
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

''' Main program'''
def full_clean(corpus):
    clean_corpus = []
    for c in corpus:
        words = tokenize(remove_between_square_brackets(c))
        words = normalize(words)
        clean_corpus.append(' '.join(words))
    return clean_corpus
