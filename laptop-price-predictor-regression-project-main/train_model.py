import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore',category=UserWarning)

# load data
df = pd.read_csv('laptop_data.csv')

# ---------------- FEATURE ENGINEERING ---------------- #

df['Ram'] = df['Ram'].str.replace('GB','').astype(int)
df['Weight'] = df['Weight'].str.replace('kg','').astype(float)

df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

df['X_res'] = df['ScreenResolution'].str.extract(r'(\d+)x(\d+)')[0].astype(int)
df['Y_res'] = df['ScreenResolution'].str.extract(r'(\d+)x(\d+)')[1].astype(int)

df['ppi'] = ((df['X_res']**2 + df['Y_res']**2)**0.5 / df['Inches'])

df['Cpu brand'] = df['Cpu'].apply(lambda x: x.split()[0])
df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])

# ---------------- MEMORY FIX ---------------- #

def extract_memory(x):
    x = str(x)
    hdd = 0
    ssd = 0

    if 'HDD' in x:
        try:
            hdd = int(x.split()[0])
        except:
            hdd = 0

    if 'SSD' in x:
        try:
            ssd = int(x.split()[0])
        except:
            ssd = 0

    return pd.Series([hdd, ssd])

df[['HDD','SSD']] = df['Memory'].apply(extract_memory)

# ---------------- FINAL DATA ---------------- #

df = df[['Company','TypeName','Ram','Weight','Touchscreen','Ips','ppi',
         'Cpu brand','HDD','SSD','Gpu brand','OpSys','Price']]

X = df.drop('Price', axis=1)
y = df['Price']

# ---------------- MODEL ---------------- #

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

categorical = ['Company','TypeName','Cpu brand','Gpu brand','OpSys']

step1 = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical)
], remainder='passthrough')

pipe = Pipeline([
    ('step1', step1),
    ('model', LinearRegression())
])

# train
pipe.fit(X, y)

# save
pickle.dump(pipe, open('pipe.pkl','wb'))
pickle.dump(df, open('df.pkl','wb'))

print("✅ MODEL TRAINED SUCCESSFULLY")