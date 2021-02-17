# Libraries to import
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Reading file
dataset = pd.read_csv("C:\\Users\\Admin\\Desktop\\New folder\\try.csv")
dataset.head()

# Taking two rows of dataset as arrays
x = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values
print(x)
print(y)

# 2D to 1D
array = x.reshape(-1,1)
print (array)

# split data into test data and train data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(array,y,test_size=.40,random_state=3)

# to create the Linear Regression model
lm = LinearRegression()
lm.fit(x_train,y_train)
y_train_pred = lm.predict(x_train)
y_test_pred = lm.predict(x_test)

# Gives predicted and original marks
df = pd.DataFrame(y_test_pred,y_test)
print(df.head())

# mean square error
mse = mean_squared_error(y_test,y_test_pred)
print(mse)

# plotting in graph
plt.scatter(y_train, y_train_pred, c='blue', marker = 'o', label='Training data')
plt.scatter(y_test, y_test_pred, c='green', marker = '*', label='Test data')
plt.xlabel('Marks Obtained')
plt.ylabel('Predicted Marks')
plt.title('Obtained Marks vs Predicted Marks')
plt.legend(loc = 'upper left')
plt.plot()
plt.show()