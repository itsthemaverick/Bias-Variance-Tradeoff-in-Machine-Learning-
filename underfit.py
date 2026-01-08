import numpy as np 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from data.generate_data import generate_data
from utils.plotting import plot_model

X,y = generate_data()

model = Pipeline([
    ("poly",PolynomialFeatures(degree=1)),
    ("lr",LinearRegression())
])

model.fit(X,y)

X_plot = np.linspace(-3,3,400).reshape(-1,1)
y_pred = model.predict(X_plot)

plot_model(X,y,X_plot,y_pred,"Underfitting : high bias model")