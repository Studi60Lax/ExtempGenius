import scrape
import clean_data
from sklearn.feature_extraction.text import CountVectorizer

def vectorize_words(corpus, n):
    cv=CountVectorizer(max_df=0.8, max_features=10000, ngram_range=(n,n))
    return [cv.fit_transform(corpus), cv]

def get_top_n_grams(corpus, n, count):

    word_vector, cv = vectorize_words(corpus, n)
    words_freq = [(word, word_vector.sum(axis=0)[0, idx]) for word, idx in cv.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:count]
