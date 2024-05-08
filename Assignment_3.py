import pandas as pd       #File extraction and data manipulation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,root_mean_squared_error
#from sklearn.preprocessing import LabelEncoder   #PREPROCESSING

dataset = pd.read_csv("C:/Users/ayush/Downloads/House Price India.csv/House Price India.csv")

x = dataset[['number of bedrooms','Area of the house(excluding basement)','Renovation Year','waterfront present','condition of the house']]  #Independent variable  for multiple>>> [["Height","Gender"]]
y = dataset['Price']    #Dependet variable

corr=dataset['number of bedrooms'].corr(dataset['Price'])
for col in x:
    corr = dataset[col].corr(dataset['Price'])
    print(corr)
x=dataset[['number of bedrooms','Area of the house(excluding basement)','waterfront present']]
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
print(mse)

rmse=root_mean_squared_error(y_test,y_pred)
print(rmse)

accuracy = model.score(x_test,y_test)
print(accuracy)
