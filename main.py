# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 03:56:55 2019

@author: user
"""

#DATA EXPLORATION FOR THESIS
#DONT RUN EVERYTHING FOR DATA REPRODUCTION, RUN EVERYTHING UNTIL THE KERNEL REG
#THEN IMPORT MODULES, RUN RF, GBM, NN FUNCTIONS AND CALLS
#THEN STACKED FUNCTION AND CALL

#import
import numpy as np
import pandas as pd
from subprocess import check_output
#read file
df = pd.read_json(r"C:\Users\user\Documents\Mannheim\THESIS\houses.json")
#inspect
print(df.shape)
print(df.columns)

#rename columns
df.rename(columns={"Haustyp":"HouseType",
                   "Angebotstyp": "OfferType", 
                   "Wohnflaeche": "LivingSpace",
                   "Zimmer": "NumRooms", 
                   "abtest": "Test", 
                   "plz": "PostCode", 
                   "Verkaeufer": "Seller", 
                   "Baujahr": "ConstructionYear", 
                   "Preis": "Price", 
                   "Grundstuecksflaeche__m²_": "LandArea", 
                   "Provision": "Commission", 
                   "kw": "Description", 
                   "ExactPreis": "ExactPrice",
                   "Aktuell_vermietet": "CurrentlyRented",
                   "Badewanne":"Bathtub",
                   "Balkon":"Balcony",
                   "Barrierefrei":"BarrierFree",
                   "Dachboden":"Attic", 
                   "Denkmalobjekt":"Monument", 
                   "Dusche":"Shower",
                   "Einbaukueche":"FittedKitchen", 
                   "Einliegerwohnung":"Apartment",  
                   "Gaeste_WC":"GuestWC",
                   "Garage/Stellplatz": "Garage/Pitch", 
                   "Garten/_mitnutzung": "Garden/SharedUse", 
                   "Haustiere_erlaubt":"PetsAllowed", 
                   "Heizkosten__in_€_":"HeatingCosts", 
                   "Heizungsart":"Heating",
                   "Kaution__in_€_":"DepositCost", 
                   "Keller":"Cellar", 
                   "Moebliert/Teilmoebliert":"Furnished/PartiallyFurnished",
                   "Nebenkosten__in_€_":"ExtraCosts",  
                   "Terrasse":"Terrace",
                   "Verfuegbar_ab_Jahr":"AvailableFromYear", 
                   "Verfuegbar_ab_Monat":"AvailableFromMonth",  
                   "WG_geeignet":"CommuneSuitable",
                   "Warmmiete__in_€_":"WarmRentCost", 
                   "Wohnflaeche__m²_":"LivingAreaM2s"}
                    ,inplace=True)
#check renamed columns
print(df.columns)
#rename categ values
df['HouseType'] = df['HouseType'].map({'andere':'other', 'bauernhaus':'farmhouse', 
                                       'doppelhaushaelfte':'semidetached', 
                                       'einfamilienhaus':'detachedhouse', 
                                       'mehrfamilienhaus':'apartmentbuilding', 
                                       'reihenhaus':'townhouse'})

#check
print(df.OfferType.value_counts()) #offer 20778, application 1622
print(df.HouseType.value_counts()) 
#detached: 13033, apartment: 2846, semidet: 1967, townhouse: 1427
#other: 1192, farmhouse: 639

#dropping unnecessary variables
df.drop(["posterid", "Description", "elasticSearch", "adid", "AvailableFromYear", "AvailableFromMonth"], axis=1)

#investigating missing values
df['ExactPrice'].isna().sum()
df = df[np.isfinite(df['ExactPrice'])]


#replacing null values

list_of_vars = ['Bathtub', 'Balcony', 'BarrierFree', 'Attic', 'Monument',
                'Shower', 'FittedKitchen', 'GuestWC', 'Garage/Pitch', 'Garden/SharedUse',
                'PetsAllowed', 'Cellar', 'Furnished/PartiallyFurnished', 
                'Terrace']

for var in list_of_vars:
    df[var].fillna(False, inplace=True)

df.isnull().any()
df.drop(["CurrentlyRented", "OfferType", "Apartment", "HeatingCosts", 
         "DepositCost", "Commission", "CommuneSuitable", "ExtraCosts", 
         "WarmRentCost", "Price"], axis=1)
df.isnull().sum()

#instead of using Heating, use one for Central Heating
df.Heating.value_counts()
df['CentralHeating'] = (df.Heating == "zentralheizung") | (df.Heating == "fussbodenheizung")
df['FloorHeating'] = (df.Heating == "fussbodenheizung")
df.CentralHeating.sum()
df.ConstructionYear.isna().sum()
df.HouseType.isna().sum()
df.HouseType.value_counts()
df['CentralHeating'].fillna(False, inplace=True)
df['FloorHeating'].fillna(False, inplace=True)

df.dropna(subset=['HouseType'], inplace=True)
df.HouseType.isna().sum()


#dropping outliers for sale price
df.ExactPrice.max()
df.ExactPrice.between(15000, 1000000, inclusive=True).sum()
df = df[df.ExactPrice.between(15000, 1000000, inclusive=True)]


#############################################################################
#FEATURE ENGINEERING
#PRICE
df['LogPrice'] = np.log(df['ExactPrice'])
df.LogPrice.min()
#LOCATION
print(len(df['PostCode'].unique()))

#read in the German Zip Codes file
plz_list = pd.read_csv(r"C:\Users\user\Documents\Mannheim\THESIS\de_postal_codes.csv", encoding='latin-1')
print(plz_list.columns)
print(plz_list['Place Name'].head())
plz_list.rename(columns={'Postal Code': 'PostCode'}, inplace=True)

#Adding the PLZ State and Geo information to the main dataset: df
df_main = pd.merge(df, plz_list, how='inner', on=['PostCode'])
df_main.head()
df_main.info()

print(df_main['State Abbreviation'].value_counts())
print(df_main['State Abbreviation'].describe())

#DROPPING WRONGLY CODED VALUES
df_main.loc[(df_main.Metro == True) & (df_main.City == True)]
df_main = df_main.drop(df_main[(df_main.LowGDP == True) & (df_main.MidGDP == True)].index)


#MAKING LOCATION INTERACTION DUMMIES
#the economic and city divisions exit in the excel file

df_main['MetroHighGDP'] = (df_main.HighGDP == 1) & (df_main.Metro == 1)
df_main['MetroMidGDP'] = (df_main.MidGDP == 1) & (df_main.Metro == 1)
df_main['MetroLowGDP'] = (df_main.LowGDP == 1) & (df_main.Metro == 1)
df_main['CityHighGDP'] = (df_main.HighGDP == 1) & (df_main.City == 1)
df_main['CityMidGDP'] = (df_main.MidGDP == 1) & (df_main.City == 1)
df_main['CityLowGDP'] = (df_main.LowGDP == 1) & (df_main.City == 1)
df_main['OtherHighGDP'] = (df_main.HighGDP == 1) & (df_main.Other == 1)
df_main['OtherMidGDP'] = (df_main.MidGDP == 1) & (df_main.Other == 1)
df_main['OtherLowGDP'] = (df_main.LowGDP == 1) & (df_main.Other == 1)

#convert the variables to booleon to avoid errors later on
tobool = ['HighGDP', 'MidGDP', 'LowGDP', 'Metro', 'City', 'Other', 'MetroHighGDP',
          'MetroMidGDP', 'MetroLowGDP', 'CityHighGDP', 'CityMidGDP', 
          'CityLowGDP', 'OtherHighGDP', 'OtherMidGDP',
          'OtherLowGDP', 'Bathtub', 'Balcony', 'BarrierFree', 'Attic',
          'FittedKitchen', 'GuestWC', 'Garage/Pitch', 'Garden/SharedUse',
          'PetsAllowed', 'Cellar', 'Furnished/PartiallyFurnished', 
          'Terrace']
for x in tobool:
    df_main[x] = df_main[x].astype('bool')
interactions = ['HighGDP', 'MidGDP', 'LowGDP', 'Metro', 'City', 'Other',
                'MetroHighGDP', 'MetroMidGDP', 'MetroLowGDP', 'CityHighGDP',
                'CityMidGDP', 'CityLowGDP', 'OtherHighGDP', 'OtherMidGDP', 'OtherLowGDP']
for i in interactions:
    print(i)
    print(df_main[i].value_counts())




#MAKING DUMMIES FOR HOUSETYPE
housetype_dummies = pd.get_dummies(df_main['HouseType'])
df_main = pd.concat([df_main, housetype_dummies], axis=1)
df_main.info()
df_main.isna().sum()

#transform NumRooms
df_main['NumRooms'] = df_main['NumRooms'].astype(float)
df_main = df_main.drop(df_main[df_main.NumRooms > 30.0].index)
df_main.NumRooms.value_counts()
df_main.info()

df_main.LivingAreaM2s.value_counts()
df_main.LandArea.isna().sum()

df_main.MetroHighGDP.value_counts()
df_main.OtherLowGDP.value_counts()
#LAND AREA
#slightly trimmed dataset as there are a lot of missing landArea values
df_main.LandArea.isna()

#drop outliers for LandArea
df_main2 = df_main[df_main.LandArea.isna() == False]
df_main2.info()
df_main2.LandArea.value_counts()
df_main2 = df_main2.drop(df_main2[df_main2.LandArea > 3000.0].index)
df_main2.LandArea[df_main2.LandArea < 10].count()
df_main2 = df_main2.drop(df_main2[df_main2.LandArea < 25.0].index)
df_main2.info()

#replace 0.0 numroom values
df_main2['NumRooms'].replace(to_replace=0.0, value=1.0, inplace=True)


df_main2['LogLandArea'] = np.log(df_main2['LandArea'])

df_main3 = df_main2[df_main2['State Abbreviation'] == 'NW']
df_main3.info()
#list of dep and indp variables
X_var = ['NumRooms', 'LivingAreaM2s', 'Bathtub', 'Balcony', 
         'BarrierFree', 'Attic', 'FittedKitchen', 
         'GuestWC', 'Garage/Pitch', 'Garden/SharedUse', 'PetsAllowed', 
         'Cellar', 'Furnished/PartiallyFurnished', 'Terrace', 'CentralHeating',
         'HighGDP', 'MidGDP', 'LowGDP', 'Metro', 'City', 'Other',
         'MetroHighGDP', 'MetroMidGDP', 'MetroLowGDP', 'CityHighGDP', 
         'CityMidGDP', 'CityLowGDP', 'OtherHighGDP', 
         'OtherMidGDP', 'OtherLowGDP', 'apartmentbuilding', 'detachedhouse',
         'farmhouse', 'semidetached', 'townhouse', 'other'] #IGNORE
X_var_LandArea = ['NumRooms', 'LandArea', 'Bathtub', 'Balcony', 
         'BarrierFree', 'Attic', 'FittedKitchen', 
         'GuestWC', 'Garage/Pitch', 'Garden/SharedUse', 'PetsAllowed', 
         'Cellar', 'Furnished/PartiallyFurnished', 'Terrace', 'CentralHeating',
         'HighGDP', 'MidGDP', 'LowGDP', 'Metro', 'City', 'Other',
         'MetroHighGDP', 'MetroMidGDP', 'MetroLowGDP', 'CityHighGDP',
         'CityMidGDP', 'CityLowGDP', 'OtherHighGDP', 
         'OtherMidGDP', 'OtherLowGDP', 'apartmentbuilding', 'detachedhouse',
         'farmhouse', 'semidetached', 'townhouse', 'other'] #IGNORE
X_var_both = ['NumRooms', 'LivingAreaM2s', 'LandArea', 'Bathtub', 'Balcony', 
         'BarrierFree', 'Attic', 'FittedKitchen', 
         'GuestWC', 'Garage/Pitch', 'Garden/SharedUse', 'PetsAllowed', 
         'Cellar', 'Furnished/PartiallyFurnished', 'Terrace', 'CentralHeating',
         'HighGDP', 'MidGDP', 'LowGDP', 'Metro', 'City', 'Other',
         'MetroHighGDP', 'MetroMidGDP', 'MetroLowGDP', 'CityHighGDP',
         'CityMidGDP', 'CityLowGDP', 'OtherHighGDP', 
         'OtherMidGDP', 'OtherLowGDP', 'apartmentbuilding', 'detachedhouse',
         'farmhouse', 'semidetached', 'townhouse', 'other'] #IGNORE
X_var_woSOwLogLA = ['NumRooms', 'LivingAreaM2s', 'LogLandArea', 'Bathtub', 'Balcony', 
         'Attic', 'FittedKitchen', 
         'GuestWC', 'Garage/Pitch', 'Garden/SharedUse', 
         'Cellar', 'Furnished/PartiallyFurnished', 'Terrace', 'CentralHeating',
         'HighGDP', 'MidGDP', 'LowGDP', 'Metro', 'City', 'Other',
         'MetroHighGDP', 'MetroMidGDP', 'MetroLowGDP', 'CityHighGDP',
         'CityMidGDP', 'CityLowGDP', 'OtherHighGDP', 
         'OtherMidGDP', 'OtherLowGDP', 'apartmentbuilding', 'detachedhouse',
         'farmhouse', 'semidetached', 'townhouse', 'other'] #USE
y_var = ['LogPrice']
X_var_woSO = ['NumRooms', 'LivingAreaM2s', 'LandArea', 'Bathtub', 'Balcony', 
         'Attic', 'FittedKitchen', 
         'GuestWC', 'Garage/Pitch', 'Garden/SharedUse', 
         'Cellar', 'Furnished/PartiallyFurnished', 'Terrace', 'CentralHeating',
         'HighGDP', 'MidGDP', 'LowGDP', 'Metro', 'City', 'Other',
         'MetroHighGDP', 'MetroMidGDP', 'MetroLowGDP', 'CityHighGDP',
         'CityMidGDP', 'CityLowGDP', 'OtherHighGDP', 
         'OtherMidGDP', 'OtherLowGDP', 'apartmentbuilding', 'detachedhouse',
         'farmhouse', 'semidetached', 'townhouse', 'other'] #IGNORE
X_var_1state = ['NumRooms', 'LivingAreaM2s', 'LandArea', 'Bathtub', 'Balcony', 
         'Attic', 'FittedKitchen', 
         'GuestWC', 'Garage/Pitch', 'Garden/SharedUse', 
         'Cellar', 'Furnished/PartiallyFurnished', 'Terrace', 'CentralHeating',
         'Metro', 'City', 'Other',
         'apartmentbuilding', 'detachedhouse',
         'farmhouse', 'semidetached', 'townhouse', 'other'] #FOR NRW 1 STATE
y_var = ['LogPrice']
y_varwoLog = ['ExactPrice']

X = df_main[X_var].values
y = df_main[y_var].values
X2 = df_main2[X_var_LandArea].values 
y2 = df_main2[y_var].values
X3 = df_main2[X_var_both].values
X4 = df_main2[X_var_woSO].values
X5 = df_main2[X_var_woSOwLogLA].values
yA = df_main2[y_varwoLog].values
X6 = df_main3[X_var_1state].values
y3 = df_main3[y_var].values

#SPLITTING DATA: INTO TRAINING AND TEST DATA
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=41)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.25,
                                                        random_state=41)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y2, test_size=0.25,
                                                        random_state=41)
X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y2, test_size=0.25,
                                                        random_state=41)
X_train5, X_test5, y_train5, y_test5 = train_test_split(X5, y2, test_size=0.25,
                                                        random_state=41)
X_train6, X_test6, y_train6, y_test6 = train_test_split(X6, y3, test_size=0.25,
                                                        random_state=41)
#USE RANDOM STATE=41 TO REPRODUCE PAPER RESULTS
#############################################################################
#BENCHMARK MODEL

from sklearn.linear_model import LinearRegression
from sklearn import metrics

def linearregression(xtrain, ytrain, xtest, ytest):
    linreg = LinearRegression()
    linreg.fit(xtrain, ytrain)
    y_pred = linreg.predict(xtest)
    print('MAE:', metrics.mean_absolute_error(ytest, y_pred))
    print('MSE:', metrics.mean_squared_error(ytest, y_pred))

#FOR MAIN MODEL
linearregression(X_train4, y_train4, X_test4, y_test4)

#FOR NRW 
linearregression(X_train6, y_train6, X_test6, y_test6)
#############################################################################
#KERNE RIDGE REGRESSION

from sklearn.kernel_ridge import KernelRidge

def kernelridge(xtrain, ytrain, xtest, ytest, alp):
    ridge = KernelRidge(alpha = alp)
    ridge.fit(xtrain, ytrain)
    y_pred = ridge.predict(xtest)
    print('MAE:', metrics.mean_absolute_error(ytest, y_pred))
    print('MSE:', metrics.mean_squared_error(ytest, y_pred))

for rate in list(np.arange(0.05, 0.9, 0.05)):
    print(rate)
    kernelridge(X_train5, y_train5, X_test5, y_test5, rate)  

#FOR MAIN MODEL
kernelridge(X_train4, y_train4, X_test4, y_test4, 0.1)
#FOR NRW
kernelridge(X_train6, y_train6, X_test6, y_test6, 0.2)

#############################################################################
#RANDOM FOREST REGRESSOR

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Hyperparameter Lists
min_ob_leaf = list(range(1,50))
max_tree_depth_rf = list(range(1,40))
#Parameter Grid for Random Forests
param_dist_rf = {"max_depth": max_tree_depth_rf,
                 "max_features": list(range(2,35)),
                 "min_samples_leaf": min_ob_leaf}
#Base Estimator
rfr = RandomForestRegressor(n_estimators = 120, random_state = 1111)
#RandomSearchCV to find best parameters
random_search_rfr = RandomizedSearchCV(estimator= rfr,
                                   param_distributions = param_dist_rf,
                                   n_iter = 100,
                                   cv = 5)
rsrfr_result = random_search_rfr.fit(X_train5, y_train5)
rsrfr_result.best_params_ 
#minsamleaf: 5, maxfeat: 2, maxdepth: 38 
#redoing Random Search might give different hyperparameters

#FUNCTION TO PERFORM RF
def randomforestreg(msl, mf, md, xtrain, ytrain, xtest, ytest):
    rfr_best = RandomForestRegressor(n_estimators=60, random_state=1111,
                                     max_depth=md, max_features=mf, min_samples_leaf=msl)
    rfr_best.fit(xtrain,ytrain)
    y_pred_rfr = rfr_best.predict(xtest)
    print('MAE:', metrics.mean_absolute_error(ytest, y_pred_rfr))
    print('MSE:', metrics.mean_squared_error(ytest, y_pred_rfr))


#FOR MAIN MODEL
randomforestreg(5,2,38, X_train4, y_train4, X_test4, y_test4) 
#FOR NRW
randomforestreg(5,2,38, X_train6, y_train6, X_test6, y_test6) 
#############################################################################
#GRADIENT BOOSTING REGRESSOR

from sklearn.ensemble import GradientBoostingRegressor

#initialize for RS
gbm = GradientBoostingRegressor(random_state = 1111)

#hyperparameters list
no_trees = list(range(20,80))
max_tree_depth_gbm = list(range(1,20))
learn_rate = list(np.arange(0.05,0.5,0.01))
sampling_rate = list(np.arange(0.75,0.9,0.01))
#randomized search
param_dist_gbm = {"max_depth": max_tree_depth_gbm,
                  "min_samples_leaf": min_ob_leaf,
                  "n_estimators": no_trees,
                  "max_features": list(range(2,35)),
                  "learning_rate": learn_rate,
                  "subsample": sampling_rate}
random_search_gbm = RandomizedSearchCV(estimator = gbm,
                                       param_distributions = param_dist_gbm,
                                       n_iter = 80,
                                       cv = 5)
rsgbm_result = random_search_gbm.fit(X_train, y_train)
rsgbm_result.best_params_
#maxdepth: 2, minsamleaf: 8, n: 30, maxfeat: 5, lr: 0.03, sr: 0.88

def gradientboostingmachine(md, msl, n, mf, lr, ss, xtrain, ytrain, xtest, ytest):
    gbm_best = GradientBoostingRegressor(n_estimators=n, random_state=1111,
                                         max_depth=md, max_features=mf, 
                                         min_samples_leaf=msl, learning_rate=lr,
                                         subsample=ss)
    gbm_best.fit(xtrain, ytrain)
    y_pred_gbm = gbm_best.predict(xtest)
    print('MAE:', metrics.mean_absolute_error(ytest, y_pred_gbm))
    print('MSE:', metrics.mean_squared_error(ytest, y_pred_gbm))

#FOR MAIN MODEL
gradientboostingmachine(2, 8, 30, 5, 0.03, 0.88, X_train4, y_train4, X_test4, y_test4)
#FOR NRW
gradientboostingmachine(2, 8, 30, 5, 0.03, 0.88, X_train6, y_train6, X_test6, y_test6)

#############################################################################
#NEURAL NETWORK
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint

#TESTINT NN WITHOUT HO
def neuralnetwork(optimizer = 'Adadelta'):
    model = Sequential()
    model.add(Dense(40, input_dim=X3.shape[1], activation='relu')) #Hidden Layer
    model.add(Dense(1)) #Output layer
    
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10,
                            verbose=1)
    return model

neuralnetwork_model = KerasRegressor(build_fn=neuralnetwork)

def nn_models(xtrain, ytrain, xtest, ytest):
    neuralnetwork_model.fit(xtrain, ytrain, validation_data=(xtest, ytest),
                            verbose=2,epochs=1000)
    ypred_nn = neuralnetwork_model.predict(xtest)
    print('MAE:', metrics.mean_absolute_error(ytest, ypred_nn))
    print('MSE:', metrics.mean_squared_error(ytest, ypred_nn))
    yt, yp = np.array(ytest), np.array(y_pred_nn)
    print(np.mean(np.abs((yt-yp)/yt))*100)

#HYPERPARAMETER OPTIMIZATION TO OPTIMIZE OUR OPTIMIZER
from keras.optimizers import Adadelta
def nn_hyp_32hu(rho, epsilon):
    model = Sequential()
    model.add(Dense(64, input_dim=X5.shape[1], activation='relu'))
    model.add(Dense(1))
    opti = Adadelta(rho=rho, epsilon=epsilon)
    model.compile(loss='mean_squared_error', optimizer=opti)
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10,
                            verbose=1)
    return model

rho_v = list(np.arange(0.92,0.99,0.001))
epsilons = [1e-7, 1e-6, 1e-5, 1e-4]

param_dist_nn = {"rho": rho_v,
                 "epsilon": epsilons}

neuralnetwork_hyp_32u = KerasRegressor(build_fn=nn_hyp_32hu)

random_search_nn = RandomizedSearchCV(estimator = neuralnetwork_hyp_32u,
                                      param_distributions = param_dist_nn,
                                      n_iter = 10,
                                      cv = 5)
rsnn_result = random_search_nn.fit(X_train5, y_train5)
rsnn_result.best_params_

#FOR 32U
#RHO: 0.984, EPSILON: 1e-07
#FOR 64U
#RHO: 0.96, EPSILON: 1e-07

#ADDING ARGS TO FUNCTION WAS GIVING ME ERRORS AGAIN AND AGAIN
#SO I MANUALLY ADJUSTED THE NUMBER OF LAYERS AND RHO IN THE SAME FUNC
#IN ORDER TO REPLICATE, PLEASE DO THE SAME

#function for NN
def finalneuralnetwork():
    model = Sequential()
    model.add(Dense(64, input_dim=X5.shape[1], activation='relu')) #hidden
    model.add(Dense(1)) #output
    opti = Adadelta(rho=0.984, epsilon=1e-07)
    model.compile(loss='mean_squared_error', optimizer=opti)
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-04, patience=10,
                            verbose=1)
    return model

nn_model_final = KerasRegressor(build_fn=finalneuralnetwork)

def nn_models_final(xtrain, ytrain, xtest, ytest):
    nn_model_final.fit(xtrain, ytrain, validation_data=(xtest, ytest),
                            verbose=2,epochs=1500)
    ypred_nn = nn_model_final.predict(xtest)
    print('MAE:', metrics.mean_absolute_error(ytest, ypred_nn))
    print('MSE:', metrics.mean_squared_error(ytest, ypred_nn))

#FOR MAIN MODEL
nn_models_final(X_train4, y_train4, X_test4, y_test4)
#FOR NRW
nn_models_final(X_train6, y_train6, X_test6, y_test6)


##############################################################################
#STACKED GENERALIZATION
from sklearn.linear_model import ElasticNet, Lasso

#function for stacking
def stackedgeneralization(xtrain, ytrain, xtest, ytest):
    x_training, x_valid, y_training, y_valid = train_test_split(xtrain, ytrain,
                                                                test_size=0.5,
                                                                random_state=42)
    #specify models
    model1 = KernelRidge(alpha = 0.1)
    model2 = RandomForestRegressor(n_estimators=60, random_state=1111,
                                   max_depth=35, max_features=2, 
                                   min_samples_leaf=5)
    model3 = GradientBoostingRegressor(n_estimators=30, random_state=1111,
                                         max_depth=2, max_features=5, 
                                         min_samples_leaf=8, learning_rate=0.03,
                                         subsample=0.88)
    model4 = KerasRegressor(build_fn=finalneuralnetwork)
    #fit models
    model1.fit(x_training, y_training)
    model2.fit(x_training, y_training)
    model3.fit(x_training, y_training)
    model4.fit(x_training, y_training,
               verbose=2, epochs=1500)
    #make pred on validation
    preds1 = model1.predict(x_valid)
    preds2 = model2.predict(x_valid)
    preds3 = model3.predict(x_valid)
    preds4 = model4.predict(x_valid)
    #make pred on test
    testpreds1 = model1.predict(xtest)
    testpreds2 = model2.predict(xtest)
    testpreds3 = model3.predict(xtest)
    testpreds4 = model4.predict(xtest)
    #form new dataset from valid and test
    stackedpredictions = np.column_stack((preds1, preds2, preds3, preds4))
    stackedtestpredictions = np.column_stack((testpreds1, testpreds2,
                                              testpreds3, testpreds4))
    #make meta model
    metamodel = LinearRegression()
    metamodel.fit(stackedpredictions, y_valid)
    final_predictions = metamodel.predict(stackedtestpredictions)
    print('MAE:', metrics.mean_absolute_error(ytest, final_predictions))
    print('MSE:', metrics.mean_squared_error(ytest, final_predictions))


#FOR MAIN MODEL
stackedgeneralization(X_train4, y_train4, X_test4, y_test4)
#FOR NRW
stackedgeneralization(X_train6, y_train6, X_test6, y_test6)


##############################################################################
#GRAPHS ############ GRAPHS ############### GRAPHS #################### GRAPHS
##############################################################################

import matplotlib.pyplot as plt

#to make histograms for Price and LogPrice
df_main2['ExactPrice'].hist(bins=25, grid=False, color='#86bf91')
plt.title("Price")
plt.xlabel("Price", weight="bold")
plt.ylabel("Number of Observations", weight="bold")

#visualizing RF
rfr_best = RandomForestRegressor(n_estimators=10, random_state=1111,
                                     max_depth=35, max_features=3, min_samples_leaf=5)
rfr_best.fit(X_train5,y_train5)
estimator_lim = rfr_best.estimators_[5]

from sklearn.tree import export_graphviz
export_graphviz(estimator_lim,
                out_file='tree.dot', rounded = True, 
                proportion = False, precision = 2)
with open("rfrlim.txt","w") as f:
    f = export_graphviz(estimator_lim, out_file=f, feature_names=X_var_woSOwLogLA,
                        rounded=True)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
from IPython.display import Image
Image(filename='tree.dot')

#for heatmap in appendix
import seaborn as sns
from matplotlib import pyplot
pyplot.figure(figsize=(40, 40))
corr_matr = df_main2[var_for_cm].corr()
hm = sns.heatmap(corr_matr, xticklabels=corr_matr.columns, 
            yticklabels=corr_matr.columns)
               
#################

print(df_main2['State Abbreviation'].value_counts())
#for summary stats
def sumstat(x):
    print('Min:', df_main2[x].min())
    print('Max:', df_main2[x].max())
    print('Mean:', df_main2[x].mean())
    print('Median:', df_main2[x].median())
    
sumstat('NumRooms')
sumstat('NumRooms')
sumstat('ExactPrice')



