from sklearn import linear_model
import numpy as np
import pandas

df = pandas.read_csv("tesla-emirhan-project-machine-learning.csv")

X = df[['Range','Fully-loaded']].values.astype(np.float)

y = df['Base-price'].values.astype(np.float)

regresyon = linear_model.LinearRegression()

regresyon.fit(X, y)

#Coefficient
print(regresyon.coef_)