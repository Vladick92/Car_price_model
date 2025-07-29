import pickle
import pandas as pd
from utils import *
from best_hyperparameters import params
from columns_for_preprocessing import *
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

df=pd.read_csv('../datasets/cars.csv')

# Data preprocessing

# Dropping and correcting columns
df=feauture_engineering(df)

# Categorical encoding
df=encode_columns(df)

# Cleaning from anomalies
df=cleaning_anoms_with_iqr(df,['compressionratio'])

# Scaling
standart_scaler=StandardScaler()
df[numerical_columns]=standart_scaler.fit_transform(df[numerical_columns])
with open('standart_scaler.pkl','wb') as f:
    pickle.dump(standart_scaler,f) 

# Splitting data
split_data(df)

train_data=pd.read_csv('../datasets/cars_train_data.csv')
test_data=pd.read_csv('../datasets/cars_test_data.csv')

# Training model
model=DecisionTreeRegressor(**params)
model.fit(train_data.drop('price',axis=1),train_data['price'])

# Evaluatin model
preds=model.predict(test_data.drop('price',axis=1))
evaluate_model(test_data['price'],preds)

# Saving model
with open('car_model.pkl','wb') as f:
    pickle.dump(model,f)