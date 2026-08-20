# IDXExchange Housing Price Prediction
A full end-to-end regression pipeline which trains four ML models (Linear, Decision Tree, Random Forest, LightGBM) on 6 months of historical CRMLS data to predict a listing's closing price, `ClosePrice`. 

## Data Overview
**Sample:** 123,454 records, from 6 months of historical data (December 2025 - May 2026)

## Source
All data sourced from the *CRMLS* (CA Regional Multiple Listing Service)

## Data Preprocessing
Various steps were taken to preprocess the data before modeling, including but not limited to:
1. It was discovered that nearly half of all `ListingKeys` were duplicated, which led to duplicate records being dropped
2. Multiple columns were nearly or entirely missing, which led to them being dropped (this did not include boolean "-YN" columns, such as `WaterfrontYN` and `AttachedGarageYN`)
3. Outlier capping was performed on `ClosePrice`, `BathroomsTotalInteger`, `GarageSpaces`, `LivingArea`, `BedroomsTotal`, `ParkingTotal`, and `LotSizeSquareFeet`
4. To normalize the target variable `ClosePrice`, the target variable was log-transformed into `logClosePrice`
5. Categorical features, such as `PostalCode` and `City`, were target encoded with bayesian smoothing 
6. To determine the optimal number of months to use for the traning set, a Linear Regression was trained on all possible sets of training months, selecting the optimal period for the training set on R<sup>2</sup>.
7. All numerical features were scaled using a `StandardScaler`

Two verisons of the training and testing set pairs were saved—one pair with scaled numerical features, and the other pair with unscaled numerical features. 

## Models and Results
4 models were tested on the data: Linear Regression, Decision Tree, Random Forest, and Gradient Boosting (LightGBM) with hyperparameter tuning. Linear Regression used the scaled versions of the training and testing sets, while all other models used the unscaled versions. All models were evaluated on R<sup>2</sup>, MAPE, and MdAPE. 

Here are the results for all the models:
| **Model** | **R<sup>2</sup> score** | **MAPE** | **MdAPE**
|---|---|---|---|
|*Linear*| 0.7812 | 1.4234 | 1.0652 |
|*Decision Tree*| 0.7801 | 1.3382 | 0.8773 |
|*Random Forest*| 0.8766| 0.9689 | 0.6414 |
|*Gradient Boosting*| 0.9190 | 0.8853 | 0.6021 |

Gradient Boosting with hyperparameter tuning using `RandomizedSearchCV` performed the best, with the highest R<sup>2</sup> score and lowest MAPE and MdAPE among all the models. Among all the models, the MdAPE was smaller than the MAPE, indicating that outliers (luxury homes on the high end of `ClosePrice`) were skewing the models' predictions. The difference between the MAPE and MdAPE is the smallest with the Gradient Boosting model, indicating that while the models struggle with luxury outliers, the Gradient Boosting model is able to adapt to them best. 

## Tech Stack
* pandas
* numpy
* scikit-learn 
* matplotlib
* seaborn 
* lightGBM
* geopandas

## Instruction to Run Code
1. Clone repository on local machine 
2. 