import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
import tensorflow_hub as hub
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")



data = pd.read_csv("sentence_classification_data.csv", names = ['Type','Gram','Sentence'])
data.head()
data['Gram'] = data['Gram'].apply(lambda x: embed(x))
data['Sentence'] = data['Sentence'].apply(lambda x: embed(x))


y = data.Type
x = train_data.drop(labels = "Type", axis = 1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state = 12)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(x_train, x_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.2f}".format(gb_clf.score(x_train, y_train)))
    print("Accuracy score (validation): {0:.2f}".format(gb_clf.score(x_test, y_test)))


gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(x_train, y_train)
predictions = gb_clf2.predict(X_val)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("Classification Report")
print(classification_report(y_test, predictions))




from xgboost import XGBClassifier
xgb_clf = XGBClassifier()
xgb_clf.fit(x_train, y_train)
score = xgb_clf.score(x_test, x_test)
print(score)
