## Business needs
This model predicts price of car based in its parameters, like fuel type, wheel base, engine size, and lot more. Its regression task, because price is continuous variable. For getting prices you must pass all features that are in `cars.csv` dataset. Main stages of prepearing data and model training displayed in `car_notebook.ipynb` jupyer notebook.

## Requirements

    python 3.12
    pandas
    sklearn
    pickle

## Running: 

**For building model execute:**

    python train.py

Firstly repeare data for training and train model. Data preparation contains main stages of CRISP-DM methodology, such as feature engineering, imputing missing values, categorical encoding, anomalies cleansing and scaling. After these phases dataset will be splitted into training, test and validation datasets. All of which will be stored in folder 'datatsets'. After model training will be created `car_model.pkl` file with model`s parameters. If you have your data edit line 9:

    df=pd.read_csv('../datasets/{your_new_data}.csv')

**For getting predictions, execute:**

    python predict.py

This script will only load data, pass it into model and make predictions. For preprocessing data there is separate script `preprocess_data.py`. For making predictions with other new data edit line 9, e.g. get prediction for `cars_test_data.csv` dataset: 

    df=pd.read_csv('../datasets/{your_new_data}.csv')

**Metrics using test dataset(cars_test_data.csv):**
    MAE: 0.24189
    MSE: 0.13374
    RMSE: 0.36577
    R2 score: 0.91157

**Metrics using validational dataset(cars_val_data.csv):**
    MAE: 0.51396
    MSE: 0.92314
    RMSE: 0.96080
    R2 score: 0.31650

**For preprocessing data execute:**

    python preprocess_data.py

Possible numerical and categorical values displayed in `car_model.ipynb` in "Data preparation" part. New dataset must have all features, that is in `cars.csv`. Edit line 6 in `preprocess_data.py` for preprocessing and splitting data: 

    df=pd.read_csv('../datasets/{your_new_data}.csv')