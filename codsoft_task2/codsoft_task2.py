import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder

# Load the dataset
link = 'https://raw.githubusercontent.com/uciml/sms-spam-collection-dataset/master/spam.csv'
data = pd.read_csv(link, encoding='latin-1')

# Clean and preprocess the data
data = data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])  # Drop unnecessary columns
data.columns = ['label', 'message']  # Rename columns for clarity

# Map labels to binary values: 'spam' -> 1, 'ham' -> 0
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Custom feature engineering (optional)
# Example: Length of messages
X_train_len = X_train.apply(len)
X_test_len = X_test.apply(len)

# Text preprocessing and vectorization using TF-IDF
tfidf = TfidfVectorizer(strip_accents='unicode', stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Function to evaluate and print model metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

# Initialize models in a pipeline for easier processing
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(kernel='linear', random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    print(f"Training {name}...")
    if name == 'Naive Bayes':
        pipe = Pipeline([('tfidf', tfidf), ('classifier', model)])
        pipe.fit(X_train, y_train)
        evaluate_model(pipe, X_test, y_test)
    else:
        pipe = Pipeline([('tfidf', tfidf), ('classifier', model)])
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__gamma': [1, 0.1, 0.01],
            'classifier__kernel': ['rbf', 'linear']
        }
        grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best Parameters: {grid_search.best_params_}")
        evaluate_model(best_model, X_test, y_test)
    print("="*50)

# Visualize ROC curves (for binary classification models)
plt.figure(figsize=(10, 8))
for name, model in models.items():
    if name == 'Naive Bayes':
        fpr, tpr, _ = roc_curve(y_test, pipe.predict_proba(X_test)[:, 1])
    else:
        fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
