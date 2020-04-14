import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow_hub as hub
from nltk import pos_tag
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from datetime import datetime

start = datetime.now()

os.environ['TFHUB_CACHE_DIR'] = './tf_cache'
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
label_enc = LabelEncoder()

print("Gathering data")
data = pd.read_csv('sentence_classification_data.csv', header=0)
data = data.dropna()
X = data[['Sentence', 'Gram']]
y = data['Type']

label_enc.fit(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=1)


def process_data(X):
    sentences = X['Sentence'].tolist()
    grams = X['Gram'].tolist()

    features = []
    for i in range(len(sentences)):
        phrase = grams[i]
        sentence = sentences[i]
        features.append([phrase, sentence])
    return features

def encode_data(X_processed, y=None):
    embed_features = []

    for f in X_processed:
        gram = f[0]
        sentence = f[1]
        embeddings = embed([gram, sentence])
        new = np.array([embeddings[0], embeddings[1]])
        embed_features.append(new.flatten()) # We have to flatten because sklearn only supports 1D features :(

    if y.any():
        y_encoded = label_enc.transform(y)

    return embed_features, y_encoded

print("Processing.")
X_train_processed = process_data(X_train)
X_test_processed = process_data(X_test)
print("Encoding.")
X_train_final,y_train_final = encode_data(X_train_processed, y_train)
X_test_final, y_test_final = encode_data(X_test_processed, y_test)

print("Classifying")

clf = GradientBoostingClassifier(n_estimators=25, learning_rate=0.05, random_state=1)
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
acc = accuracy_score(y_test_plaintext, preds_plaintext)
print(acc)

end = datetime.now()

print("RUNTIME")
print(end-start)
