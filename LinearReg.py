import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

hours_studied = np.array([[1], [2], [3], [4], [5]])
test_score = np.array([63, 80, 89, 91, 94])

model = LinearRegression()
model.fit(hours_studied, test_score)

hours_studied_input = np.array([[0.5], [1.5], [2.5], [3.5], [4.5]])
#y_test = np.array([6, 7, 8, 9, 10])

y_prediction = model.predict(hours_studied_input)

slope = model.coef_[0]
intercept = model.intercept_
#avg_squared_error = mean_squared_error, y_prediction)

print("Slope: ", round(slope,2))
print("Intercept: ", round(intercept,2))
print("Predicted Equation: y =  " + str(round(slope, 2)) + "x + " + str(round(intercept, 2)))
print("Predicted test scores: ",y_prediction)





