import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow_hub as hub
from nltk import pos_tag
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

os.environ['TFHUB_CACHE_DIR'] = './tf_cache'
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
label_enc = LabelEncoder()

data = pd.read_csv('sentence_classification_data.csv', header=0)
data = data.dropna()
X = data[['Sentence', 'Gram']]
y = data['Type']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state = 1)

def process_data(X):
    sentences = X['Sentence'].tolist()
    grams = X['Gram'].tolist()

    features = []

    pos_index_map = {
        'CC' : 0,
        'CD' : 1,
        'DT' : 2,
        'EX' : 3,
        'FW' : 4,
        'IN' : 5,
        'JJ' : 6,
        'JJR' : 7,
        'JJS' : 8,
        'LS' : 9,
        'MD' : 10,
        'NN' : 11,
        'NNS' : 12,
        'NNP' : 13,
        'NNPS' : 14,
        'PDT' : 15,
        'POS' : 16,
        'PRP' : 17,
        'PRP$' : 18,
        'RB' : 19,
        'RBR' : 20,
        'RP' : 21,
        'SYM' : 22,
        'TO' : 23,
        'UH' : 24,
        'VB' : 25,
        'VBD' : 26,
        'VBG' : 27,
        'VBN' : 28,
        'VBP' : 29,
        'VBZ' : 30,
        'WDT' : 31,
        'WP' : 32,
        'WP$' : 33,
        'WRB' : 34,
        'RBS' : 35
    }

    pos_phrase = []
    pos_sentence = []
    for i in range(35):
        pos_phrase.append(0)
        pos_sentence.append(0)

    for i in range(len(sentences)):
        phrase = grams[i]

        phrase_parts_of_speech = pos_tag(phrase)
        phrase_only_pos = []
        for p in phrase_parts_of_speech:
            phrase_only_pos.append(p[1])

        sentence = sentences[i]
        sentence_parts_of_speech = pos_tag(sentence)
        sentence_only_pos = []
        for p in sentence_parts_of_speech:
            sentence_only_pos.append(p[1])



        for p in phrase_only_pos:
            try:
                pos_phrase[pos_index_map[p]] += 1
            except:
                pass
        for p in sentence_only_pos:
            try:
                pos_sentence[pos_index_map[p]] += 1
            except:
                pass

        phrase_only_pos = ' '.join(phrase_only_pos)
        sentence_only_pos = ' '.join(sentence_only_pos)

        features.append([phrase, sentence, phrase_only_pos, sentence_only_pos] + pos_phrase + pos_sentence)

    return features

def encode_data(X_processed, y=None):
    embed_features = []

    for f in X_processed:
        gram = f[0]
        sentence = f[1]
        phrase_pos = f[2]
        sentence_pos = f[3]

        embeddings = embed([gram, sentence, phrase_pos, sentence_pos])
        new = np.array([embeddings[0], embeddings[1], embeddings[2], embeddings[3]])
        embed_features.append(new.flatten()) # We have to flatten because sklearn only supports 1D features :(

    if y.any():
        y_encoded = label_enc.fit_transform(y)

    return embed_features, y_encoded

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=1)

X_train_processed = process_data(X_train)
X_test_processed = process_data(X_test)
X_train_final,y_train_final = encode_data(X_train_processed, y_train)
X_test_final, y_test_final = encode_data(X_test_processed, y_test)

clf.fit(X_train_final, y_train_final)
preds = clf.predict(X_test_final)

preds_plaintext = label_enc.inverse_transform(preds)
y_test_plaintext = label_enc.inverse_transform(y_test_final)

print("Confusion Matrix:")
print(confusion_matrix(y_test_plaintext, preds_plaintext))
print()
print("Classification Report:")
print(classification_report(y_test_plaintext, preds_plaintext))
print("Accuracy:")
print(accuracy_score(y_test_plaintext, preds_plaintext))
