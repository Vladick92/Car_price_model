import pickle
import pandas as pd
from columns_for_preprocessing import *
from utils import *

# Getting model and data
with open('car_model.pkl','rb') as f:
    model=pickle.load(f)
df=pd.read_csv('../datasets/cars_val_data.csv')

# Making predictions
preds=model.predict(df.drop('price',axis=1))
evaluate_model(df['price'],preds)

# Turning standartized numbers back into price
with open('standart_scaler.pkl','rb') as f:
    standart_scaler=pickle.load(f)
inversed_np_arr=standart_scaler.inverse_transform(df[numerical_columns])
inversed_df=pd.DataFrame(inversed_np_arr)
inversed_df.iloc[:,-1].to_csv('./predictions.csv',index=False)