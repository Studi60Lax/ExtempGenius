import scrape
import clean_data
from sklearn.feature_extraction.text import CountVectorizer

def vectorize_words(corpus, n):
    cv=CountVectorizer(max_df=0.8, max_features=10000, ngram_range=(n,n))
    return [cv.fit_transform(corpus), cv]

def get_top_n_grams(corpus, n, count, remove_phrases=[]):
    remove_words = []
    for p in remove_phrases:
        remove_words += p.split(" ")

    word_vector, cv = vectorize_words(corpus, n)
    words_freq = [(word, word_vector.sum(axis=0)[0, idx]) for word, idx in cv.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

    # Removes phrases that are repeated from previous top words. Stops the #1 1-gram from being "states" and the #1 2-gram from being "united states"
    removed_repeats_word_freq = []
    for k in words_freq:
        repeat = False
        for word in remove_words:
            if word in k[0]:
                repeat = True
                break
        if not repeat:
            removed_repeats_word_freq.append(k[0])

    return removed_repeats_word_freq[:count]
