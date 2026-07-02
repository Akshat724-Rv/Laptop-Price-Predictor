import pickle
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import pandas as pd

# dummy data (just to create file)
df = pd.DataFrame({
    'Price': [1,2,3,4],
    'Feature': [10,20,30,40]
})

X = df[['Feature']]
y = df['Price']

pipe = Pipeline([
    ('model', LinearRegression())
])

pipe.fit(X, y)

pickle.dump(pipe, open('pipe.pkl','wb'))

print("✅ pipe.pkl created in same folder")