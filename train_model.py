import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
dataset = pd.read_csv('creditcard.csv')

# Data Preprocessing
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Handle Imbalanced Data
ros = RandomOverSampler(random_state=0)
x_res, y_res = ros.fit_resample(X, y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.3, random_state=42)

# Train RandomForest Classifier
classifier = RandomForestClassifier(n_estimators=641, random_state=0)
classifier.fit(x_train, y_train)

# Save the trained model
joblib.dump(classifier, 'credit_card_fraud_model.pkl')
