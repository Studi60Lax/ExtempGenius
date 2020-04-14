import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow_hub as hub
from nltk import pos_tag
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, percent_accuracy

os.environ['TFHUB_CACHE_DIR'] = './tf_cache'
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
label_enc = LabelEncoder()

data = pd.read_csv('sentence_classification_data.csv', header=0)
data = data.dropna()
X = data[['Sentence', 'Gram']]
y = data['Type']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.05,random_state = 1)

def process_data(X):
    sentences = X['Sentence'].tolist()
    grams = X['Gram'].tolist()

    features = []

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

        phrase_only_pos = ' '.join(phrase_only_pos)
        sentence_only_pos = ' '.join(sentence_only_pos)

        features.append([phrase, sentence, phrase_only_pos, sentence_only_pos])

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

print("Classification Report")
print(classification_report(y_test_plaintext, preds_plaintext))
