import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the dataset (replace 'data.tsv' with your file path)
data = pd.read_csv("C:/Users/ayush/Downloads/mammals.csv")

# Define independent variable (X) and dependent variable (y)
X = data[['body_wt']]  # Independent variable (body weight)
y = data['brain_wt']   # Dependent variable (brain weight)

# Split the data into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# Calculate the accuracy score (R-squared) on the test data
mse = mean_squared_error(y_test,y_pred)
print((mse))

# Plot the linear regression line and actual data points
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Linear Regression Model')
plt.xlabel('Body Weight')
plt.ylabel('Brain Weight')
plt.title('Linear Regression: Brain Weight vs Body Weight')
plt.legend()
plt.show()
