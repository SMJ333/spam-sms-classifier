import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import string
import joblib
from google.colab import files

# Upload the file
uploaded = files.upload()

# Load the dataset (assuming the uploaded file name is 'spam.csv')
data = pd.read_csv('spam.csv', encoding='latin-1')

# Display the first few rows of the dataset
print(data.head())

# Drop unnecessary columns and rename the relevant ones
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert the labels to binary values: spam = 1, ham = 0
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Data cleaning function
def clean_text(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    text = text.lower()
    return text

data['message'] = data['message'].apply(clean_text)

# Display the first few cleaned messages
print(data.head())

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the training data, transform the test data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize and train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_tfidf)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Display the confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Save the model and vectorizer
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved to disk.")
