import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset (replace 'data.csv' with your dataset file path)
data = pd.read_csv("C:/Users/ayush/Downloads/BankNoteAuthentication.csv")

# Extract features (X) and target (y)
X = data[['variance', 'skewness', 'curtosis', 'entropy']]  # Input features
y = data['class']  # Target variable (1 for authentic, 0 for fake)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=499)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report and confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
