import pandas as pd       #File extraction and data manipulation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,root_mean_squared_error
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("C:/Users/ayush/Downloads/data.csv")
print(dataset.head())

x = dataset[['Height']]  #Independent variable  for multiple>>> [["Height","Gender"]]
y = dataset['Weight']    #Dependet variable
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
print(mse)

rmse=root_mean_squared_error(y_test,y_pred)
print(rmse)

