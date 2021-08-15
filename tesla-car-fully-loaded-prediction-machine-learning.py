from sklearn import linear_model
import numpy as np
import pandas

df = pandas.read_csv("tesla-emirhan-project-machine-learning.csv")

#According to Range and Base Price
X = df[['Base-price','Range']].values.astype(np.float)

#Fully Loaded
y = df['Fully-loaded'].values.astype(np.float)

regresyon = linear_model.LinearRegression()

regresyon.fit(X, y)

predictedFully_Loaded = regresyon.predict([[263,54490]])

print(predictedFully_Loaded)

#print(regresyon.coef_)