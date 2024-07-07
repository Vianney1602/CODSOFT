import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
import shap

# Load the dataset
set = pd.read_csv('Churn_Modelling.csv')

# Display first few rows and info
print("First few rows of the dataset:")
print(set.head())
print("\nDataset information:")
print(set.info())
print("\nDescriptive statistics of numerical columns:")
print(set.describe())
print("\nTarget variable counts:")
print(set['Exited'].value_counts())  # Assuming 'Exited' is the target variable

# Data Preprocessing
# Drop irrelevant columns
set = set.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode categorical variables
set = pd.get_dummies(set, drop_first=True)

# Custom Feature Engineering
set['BalanceSalaryRatio'] = set['Balance'] / set['EstimatedSalary']
set['TenureByAge'] = set['Tenure'] / set['Age']
set['CreditScoreGivenAge'] = set['CreditScore'] / set['Age']

# Separate features and target variable
X = set.drop('Exited', axis=1)
y = set['Exited']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# GridSearch for hyperparameter tuning
param_grid = {
    'classifier': [LogisticRegression(max_iter=1000, random_state=42),
                   RandomForestClassifier(random_state=42),
                   GradientBoostingClassifier(random_state=42)],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Evaluate the best model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

print("Evaluating Best Model...")
evaluate_model(best_model, X_test, y_test)

# Feature Importance using SHAP
explainer = shap.Explainer(best_model.named_steps['classifier'], X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)

# Feature Importance (Random Forest)
if isinstance(best_model.named_steps['classifier'], RandomForestClassifier):
    random_forest = best_model.named_steps['classifier']
    feature_importances = random_forest.feature_importances_
    features = X.columns
    importance_info = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_info)
    plt.title('Feature Importances from Random Forest')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
