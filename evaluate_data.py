import scrape
import clean_data
from sklearn.feature_extraction.text import CountVectorizer

def vectorize_words(words, n):
    cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,n))
    return cv.fit_transform(words)

def get_top_n_grams(words, n, count):
    word_vector = vectorize_words(words, n)
    words_freq = [(word, word_vector.sum(axis=0)[0, idx]) for word, idx in word_vector.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:count]
