import numpy as np 
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from data.generate_data import generate_data
from sklearn.model_selection import cross_val_score

X,y = generate_data()

degrees = range(1,16)
errors = []

for d in degrees :
    model = Pipeline([
        ("poly",PolynomialFeatures(degree=0)),
        ("lr", LinearRegression())
    ])
    mse = -cross_val_score(
        model,
        X,
        y,
        scoring="neg_mean_squared_error",
        cv=5
    ).mean()

    errors.append(mse)

plt.figure()
plt.plot(degrees,errors,marker ="o")
plt.xlabel("Polynomial Deg")
plt.ylabel("Cross-Validated MSE")
plt.title("Model Complexity vs Genralizaton Error")
plt.show()