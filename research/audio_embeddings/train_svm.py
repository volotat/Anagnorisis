from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

import utils

def train_and_evaluate(X_train, X_test, y_train, y_test):
  clf = svm.SVC()
  clf.fit(X_train, y_train)

  train_accuracy = accuracy_score(y_train, clf.predict(X_train))
  test_accuracy = accuracy_score(y_test, clf.predict(X_test))

  return train_accuracy, test_accuracy

utils.set_seed(42)

embeddings, reduced_embeddings, artists, titles = utils.load_data_from_csv("dataset.csv")
print("Dataset has been loaded")

# Split the data into a training set and a test set for full embeddings
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(embeddings, artists, test_size=0.2, random_state=42)

# Train the SVM on full embeddings and calculate accuracy
train_accuracy_full, test_accuracy_full = train_and_evaluate(X_train_full, X_test_full, y_train_full, y_test_full)
print(f'Train Accuracy on full embeddings: {train_accuracy_full*100:.2f}%, Test Accuracy on full embeddings: {test_accuracy_full*100:.2f}%')

# Split the data into a training set and a test set for reduced embeddings
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(reduced_embeddings, artists, test_size=0.2, random_state=42)

# Train the SVM on reduced embeddings and calculate accuracy
train_accuracy_reduced, test_accuracy_reduced = train_and_evaluate(X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced) 
print(f'Train Accuracy on reduced embeddings: {train_accuracy_reduced*100:.2f}%, Test Accuracy on reduced embeddings: {test_accuracy_reduced*100:.2f}%')