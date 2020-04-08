from nltk.stem import WordNetLemmatizer

def lemmatize(string):
    lemmatizer = WordNetLemmatizer()
    lemmatized_string = ""
    for word in string.split(" "):
        lemmatized_string += lemmatizer.lemmatize(word)
        lemmatized_string += "
    return lemmatized_string.strip()
