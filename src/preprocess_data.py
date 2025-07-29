import pickle
import pandas as pd
from utils import *
from columns_for_preprocessing import *

df=pd.read_csv('../datasets/cars.csv')

# Dropping and correcting columns
df['brand']=df['CarName'].apply(lambda word: word.split(' ')[0])
df=df.drop(columns_to_drop,axis=1)
df=correct_brands(df)

# Categorical encoding
df=encode_columns(df)

# Cleaning from anomalies
df=cleaning_anoms_with_iqr(df,['compressionratio'])

# Scaling
with open('standart_scaler.pkl','rb') as f:
    standart_scaler=pickle.load(f) 
df[numerical_columns]=standart_scaler.fit_transform(df[numerical_columns])

# spliting into train,test and validation datasets
split_data(df)
