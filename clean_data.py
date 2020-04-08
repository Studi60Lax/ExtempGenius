from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import download

def full_clean(string):
    download('stopwords') # Downloads stopwords list if not already on computer
    lemmatizer = WordNetLemmatizer()
    stop = set(stopwords.words('english'))
    cleaned_string = "" # Cleaned text will be built here
    string = string.lower()
    # Lemmatize and remove stop words
    for word in string.split(" "):
        if not word in stop: # Removes stopwords
            cleaned_string += lemmatizer.lemmatize(word) # Lemmatization
            cleaned_string += " "
    cleaned_string = cleaned_string.translate ({ord(c): " " for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+"}) # Replace special chars

    return cleaned_string.strip()
