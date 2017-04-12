import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#import data
dataframe = pd.read_csv('data/challenge_dataset.csv')
x_values = dataframe[['x']]
y_values = dataframe[['y']]

#train model on data
model = LinearRegression()
model.fit(x_values, y_values)

#visualise results
plt.scatter(x_values, y_values)
plt.plot(x_values, model.predict(x_values))
plt.show()
