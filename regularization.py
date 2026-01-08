import numpy as np 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import Ridge,Lasso
from utils.plotting import plot_model
from data.generate_data import generate_data

X,y = generate_data()
X_plot = np.linspace(-3,3,400).reshape(-1,1)

ridge = Pipeline([
    ("poly",PolynomialFeatures(degree=15)),
    ("ridge",Ridge(alpha=0.1))
])

lasso = Pipeline([
    ("poly",PolynomialFeatures(degree=15)),
    ("lasso",Lasso(alpha=0.01,max_iter=40000))
])

ridge.fit(X,y)
lasso.fit(X,y)

plot_model(X,y,X_plot,ridge.predict(X_plot),"Ridge regularization L2")
plot_model(X,y,X_plot,lasso.predict(X_plot),"Lasso regularization L1")
