import pandas as pd
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Function to parse the data
def parse_data(file_path, has_genre=True):
    info = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ::: ')
            if has_genre and len(parts) == 4:
                info.append(parts)
            elif not has_genre and len(parts) == 3:
                info.append(parts)
    return info

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Load and preprocess training data
training_data = parse_data('training_data.txt', has_genre=True)  # Replace with actual path
train_df = pd.DataFrame(training_data, columns=['ID', 'Title', 'Genre', 'Description'])
train_df['Description'] = train_df['Description'].apply(preprocess_text)

# Encode labels
label_encoder = LabelEncoder()
train_df['Genre'] = label_encoder.fit_transform(train_df['Genre'])

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_df['Description'], train_df['Genre'], test_size=0.2, random_state=42)

# Function to create and train the model
def create_and_train_model(model, param_grid, X_train, y_train):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', model)
    ])
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search

# Define model parameters for hyperparameter tuning
nb_param_grid = {
    'clf__alpha': [0.1, 0.5, 1.0]
}

lr_param_grid = {
    'clf__C': [0.1, 1, 10],
    'clf__max_iter': [100, 500, 1000]
}

svm_param_grid = {
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear', 'rbf']
}

# Train and evaluate Multinomial Naive Bayes
print("Training Multinomial Naive Bayes...")
nb_model = create_and_train_model(MultinomialNB(), nb_param_grid, X_train, y_train)
y_pred_nb = nb_model.predict(X_val)
print("Multinomial Naive Bayes")
print(classification_report(y_val, y_pred_nb))
print('Accuracy:', accuracy_score(y_val, y_pred_nb))

# Train and evaluate Logistic Regression
print("\nTraining Logistic Regression...")
lr_model = create_and_train_model(LogisticRegression(), lr_param_grid, X_train, y_train)
y_pred_lr = lr_model.predict(X_val)
print("Logistic Regression")
print(classification_report(y_val, y_pred_lr))
print('Accuracy:', accuracy_score(y_val, y_pred_lr))

# Train and evaluate Support Vector Machine
print("\nTraining Support Vector Machine...")
svm_model = create_and_train_model(SVC(), svm_param_grid, X_train, y_train)
y_pred_svm = svm_model.predict(X_val)
print("Support Vector Machine")
print(classification_report(y_val, y_pred_svm))
print('Accuracy:', accuracy_score(y_val, y_pred_svm))

# Select the best model based on validation performance
best_model = None
best_accuracy = 0
if accuracy_score(y_val, y_pred_nb) > best_accuracy:
    best_model = nb_model
    best_accuracy = accuracy_score(y_val, y_pred_nb)
if accuracy_score(y_val, y_pred_lr) > best_accuracy:
    best_model = lr_model
    best_accuracy = accuracy_score(y_val, y_pred_lr)
if accuracy_score(y_val, y_pred_svm) > best_accuracy:
    best_model = svm_model
    best_accuracy = accuracy_score(y_val, y_pred_svm)

print("\nBest model selected:", best_model)

# Load and preprocess test data
test_data = parse_data('test_data.txt', has_genre=False)  # Replace with actual path
test_df = pd.DataFrame(test_data, columns=['ID', 'Title', 'Description'])
test_df['Description'] = test_df['Description'].apply(preprocess_text)

# Predict the genres for the test data using the best model
test_df['Predicted_Genre'] = best_model.predict(test_df['Description'])

# Decode the predicted labels
test_df['Predicted_Genre'] = label_encoder.inverse_transform(test_df['Predicted_Genre'])

# Save the test data with predicted genres
test_df[['ID', 'Title', 'Description', 'Predicted_Genre']].to_csv('predicted_genres.csv', index=False)
print("Predictions saved to predicted_genres.csv")
