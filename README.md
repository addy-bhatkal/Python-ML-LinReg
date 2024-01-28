import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import r2_score

os.chdir("X:\\Files\\ABC\\Datasets")

df = pd.read_csv("BostonHousing.csv")
df.head()

x = df.iloc[:,0:13]
y = df['medv']

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=55, test_size = 0.8)

lm = LinearRegression()

model = lm.fit(x_train, y_train)

y_predict = lm.predict(x_test)

results = pd.DataFrame({'y_test' : y_test, 'y_predict' : y_predict })
results.head()

r2 = r2_score(y_test, y_predict)
r2
