import pickle
import pandas as pd
from columns_for_preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn import metrics 

def correct_brands(dataframe):
    df=dataframe.copy()
    corrections={
        'maxda':'mazda',
        'porcshce':'porsche',
        'Nissan':'nissan',
        'toyouta':'toyota'
    }
    df['brand']=df['brand'].map(corrections).fillna(df['brand'])
    df['brand']=df['brand'].apply(lambda word: 'volkswagen' if word in ['vw','vokswagen'] else word)
    return df

def feauture_engineering(dataframe):
    df=dataframe.copy()
    df['brand']=df['CarName'].apply(lambda word: word.split(' ')[0])
    df=df.drop(columns_to_drop,axis=1)
    df=correct_brands(df)
    return df 

def encode_columns(dataframe):
    df=dataframe.copy()

    # dummy encoding
    for col in columns_for_dummy_encoding:
        df=pd.concat([df,pd.get_dummies(df[col],drop_first=True,prefix=col)],axis=1)
        df=df.drop(col,axis=1)

    # turning words into numbers
    cyl_nums={
    'four':4,
    'six':6,
    'five':5,
    'three':3,
    'twelve':12,
    'two':2,
    'eight':8
    }
    df['cylindernumber']=df['cylindernumber'].map(cyl_nums)

    # label encoding
    with open('label_encoder.pkl','rb') as f:
        label_encoder=pickle.load(f)
    for col in columns_for_label_encoding:
        df[col]=label_encoder.fit_transform(df[col])

    # count encoding
    df['brand']=df['brand'].map(df['brand'].value_counts())

    return df

def cleaning_anoms_with_iqr(dataframe,cols):
    df=dataframe.copy()
    for col in cols:
        q1=df[col].quantile(0.25)
        q3=df[col].quantile(0.75)
        iqr=q3-q1
        lower=iqr-1.5*q1
        upper=iqr+1.5*q3
        outliers=df[(df[col]<lower)|(df[col]>upper)]
        df=df.drop(outliers.index,axis=0)
    return df

def split_data(dataframe):
    df=dataframe.copy()
    x_train, x_temp, y_train, y_temp = train_test_split(df.drop('price', axis=1),df['price'],test_size=0.3,random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp,test_size=0.5,random_state=42)
    train_data=pd.concat([x_train,y_train],axis=1)
    train_data.to_csv('../datasets/cars_train_data.csv',index=False)
    test_data=pd.concat([x_test,y_test],axis=1)
    test_data.to_csv('../datasets/cars_test_data.csv',index=False)
    val_data=pd.concat([x_val,y_val],axis=1)
    val_data.to_csv('../datasets/cars_val_data.csv',index=False)

def evaluate_model(test,preds):
    print(f'MAE: {metrics.mean_absolute_error(test,preds)}')
    print(f'MSE: {metrics.mean_squared_error(test,preds)}')
    print(f'RMSE: {metrics.root_mean_squared_error(test,preds)}')
    print(f'R2 score: {metrics.r2_score(test,preds)}')