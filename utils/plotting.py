import matplotlib.pyplot as plt 
import numpy as np 

def plot_model(X,y,X_plot,y_pred,title):
    plt.figure()
    plt.scatter(X,y)
    plt.plot(X_plot,y_pred)
    plt.title(title)
    plt.show()