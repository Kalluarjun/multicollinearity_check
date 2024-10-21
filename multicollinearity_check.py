import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import fetch_california_housing
df = fetch_california_housing()
#print(df) 
dataset= pd.DataFrame(df.data)
dataset.columns= df.feature_names 
#independent features and dependent features
X= dataset # or X= df.data
y= df.target
#multicollinearity check
X= sm.add_constant(X)
model= sm.OLS(y, X).fit()
print(model.summary())
print(X.iloc[:, 1:].corr())
