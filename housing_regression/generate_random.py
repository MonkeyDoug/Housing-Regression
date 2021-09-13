# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 22:41:01 2021

@author: immor
"""

import os
import re
import shap
import pickle
import numpy as np
import pandas as pd
from git import Repo
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import median_absolute_error as mae
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from yellowbrick.regressor import ResidualsPlot
from housing_regression.shap_plot import plot_shap
# from yellowbrick.regressor import PredictionError

# Script to generate cleaned files for deployment

# Reads in and prepares data
def read(year,file_path):
    if year == '2020':
        # Reads the csv file starting at row 6 if the year is 2020 because of boilerplate at the start of the file
        df = pd.read_csv(file_path,header=6,encoding='unicode_escape') 
    else:
        # Reads the csv file starting at row 4 if the year is 2020 because of boilerplate at the start of the file
        df = pd.read_csv(file_path,header=4,encoding='unicode_escape')

    df.columns = df.columns.str.replace("\n","") # Removes all remaining next lines ( between words )
    df.columns = df.columns.str.replace('[^a-zA-Z]','',regex=True) # Removes everything that is not an alphabetic character or space
    df.columns = df.columns.str.replace("ASOFFINALROLL","ATPRESENT",regex=True) # Replaces AS OF FINAL ROLL with AT PRESENT to create consistent column names

    # Dropping columns that are useless
    drop_cols = ['BOROUGH','SALEDATE','ADDRESS','EASEMENT','APARTMENTNUMBER','TAXCLASSATTIMEOFSALE','BUILDINGCLASSATTIMEOFSALE','BUILDINGCLASSCATEGORY']
    df.drop(drop_cols,axis=1,inplace=True)
    df['SALEPRICE'] = df['SALEPRICE'].apply(lambda x : np.nan if '-' in str(x) else x)
    df.dropna(subset=[ 'SALEPRICE' ],inplace=True)
    df['SALEPRICE'] = df['SALEPRICE'].apply(lambda x : int( re.sub(r'[$, ]','',x) ))
    df = df[df['SALEPRICE'] > 10]
    return df

def clean(df):
    df.dropna(inplace=True)
    q1 = df['SALEPRICE'].quantile(.25)
    df = df[ df['SALEPRICE'] >= q1 ]
    df = df[df['TAXCLASSATPRESENT'] != ''] # Removes empty tax classes
    df = df[df['ZIPCODE'] != 0] # Removes invalid zipcodes
    df = df[df['YEARBUILT'] != 0] # Removes invalid years
    for col in df.columns:
        df[col] = df[col].apply(lambda x : re.sub(r'[$, ]','',str(x)))
        try:
            df[col] = df[col].apply(lambda x : np.nan if '-' in str(x) else float(x))
        except ValueError:
            df[col] = df[col].apply(lambda x : np.nan if '-' in str(x) else x)
    df.dropna(inplace=True)
    df.reset_index(inplace=True,drop=True)
    return df

def clean_p(df):
    df.dropna(inplace=True)
    q1 = df['SALEPRICE'].quantile(.25)
    q3 = df['SALEPRICE'].quantile(.75)
    df = df[ df['SALEPRICE'] >= q1 ]
    df = df[ df['SALEPRICE'] <= q3 ]
    df = df[df['TAXCLASSATPRESENT'] != ''] # Removes empty tax classes
    df = df[df['ZIPCODE'] != 0] # Removes invalid zipcodes
    df = df[df['YEARBUILT'] != 0] # Removes invalid years
    for col in df.columns:
        df[col] = df[col].apply(lambda x : re.sub(r'[$, ]','',str(x)))
        try:
            df[col] = df[col].apply(lambda x : np.nan if '-' in str(x) else float(x))
        except ValueError:
            df[col] = df[col].apply(lambda x : np.nan if '-' in str(x) else x)
    df.dropna(inplace=True)
    df.reset_index(inplace=True,drop=True)
    return df

def stat(df,borough_path):
    stat_path = os.path.join(borough_path,'stats.pickle')
    X = df.drop('SALEPRICE',axis=1)
    y = df['SALEPRICE'].tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True)
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(X_train,y_train)
    
    y_pred = model.predict(X_test)
    y_true = y_test
    
    # Stats for the model
    stats = {}
    MSE = mse(y_true,y_pred)
    stats['MSE'] = MSE
    stats['RMSE'] = np.sqrt(MSE)
    R_squared = r2_score(y_true,y_pred)
    stats['R_squared'] = R_squared
    stats['MAE'] = mae(y_true,y_pred)
    stats['adj_r2'] = 1-(1-R_squared)*(len(df.index)-1)/(len(df.index)-len(df.columns)-1)
    
    with open(stat_path,'wb') as f:
        pickle.dump(stats,f)
        
    # Residual
    plots = ['ResidualsPlot']
    for plot in plots:
        file_path = os.path.join(borough_path,f'{plot}.png')
        visualizer = eval(plot + '(model)')
        visualizer.fit(X_train,y_train)
        visualizer.score(X_test,y_test)
        visualizer.show(outpath=file_path)
    
# =============================================================================
#     # Residual + Prediction Plot
#     plots = ['ResidualsPlot','PredictionError']
#     for plot in plotsT
#         file_path = os.path.join(borough_path,f'{plot}.png')
#         visualizer = eval(plot + '(model)')
#         visualizer.fit(X_train,y_train)
#         visualizer.score(X_test,y_test)
#         visualizer.show(outpath=file_path)
# =============================================================================

def model(df,path):
    # Setting paths for files
    regressor_path = os.path.join(path,'regressor.pickle')
    explainer_path = os.path.join(path,'explainer.pickle')
    placeholder_path = os.path.join(path,'placeholder.pickle')

    # Separating data into target and features
    X = df.drop('SALE PRICE',axis=1)
    y = df['SALE PRICE'].tolist()
    
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(X,y)

    with open(placeholder_path,'wb') as f:
        x_head = X.iloc[0]
        data = {}
        for index,value in x_head.iteritems():
            if 'NEIGHBORHOOD' in index or 'BUILDING CLASS' in index or 'TAX CLASS' in index:
                data[index] = 0
            else:
                data[index] = value
        placeholder = pd.DataFrame(data,index=[0])
        pickle.dump(placeholder,f)

    with open(regressor_path,'wb') as f:
        pickle.dump(model,f)

    explainer = shap.TreeExplainer(model)
    with open(explainer_path,'wb') as f:
        pickle.dump(explainer,f)

    return X.head(1)

# Driver Code
def main():
    # Paths for file
    curr_path = os.path.dirname(os.path.realpath(__file__))
    asset_path = os.path.join(curr_path,'assets')
    
    # Downloads data if not present
    if not os.path.exists(asset_path):
        # Download data from github
        git_url = 'https://github.com/MonkeyDoug/Housing-Data.git'
        Repo.clone_from(git_url,'assets')
    
    # Initalize Borough & Year
    boroughs = ['bronx','brooklyn','manhattan','queens','statenisland']
    years = ['2015','2016','2017','2018','2019','2020']
    
    for borough in boroughs:
        df = pd.DataFrame()
        borough_path = os.path.join(asset_path,borough)
        for year in years:
            file_name = f'{year}_{borough}.csv' # Creates filename from the year and borough in the format of the csv
            file_path = os.path.join(borough_path,file_name) # Joins all the file names together to get the absolute path of the file
            tmp = read(year,file_path)
            df = pd.concat([df,tmp],ignore_index=True) # Concats all the csv files into dataframe for the entire borough
        mapper = {
                'TAXCLASSATPRESENT':'TAX CLASS AT PRESENT',
                'BUILDINGCLASSATPRESENT':'BUILDING CLASS AT PRESENT',
                'ZIPCODE':'ZIP CODE',
                'RESIDENTALUNITS':'RESIDENTIAL UNITS',
                'COMMERCIALUNITS':'COMMERCIAL UNITS',
                'TOTALUNITS':'TOTAL UNITS',
                'LANDSQUAREFEET':'LAND SQUARE FEET',
                'GROSSSQUAREFEET':'GROSS SQUARE FEET',
                'YEARBUILT':'YEAR BUILT',
                'SALEPRICE':'SALE PRICE'
                }
        with open(os.path.join(borough_path,'df.pickle'),'wb') as f:
            pickle.dump(df,f)
            
        clean_df = clean(df)
        clean_df = clean_df.rename(columns=mapper)
        with open(os.path.join(borough_path,'clean_df.pickle'),'wb') as f:
            pickle.dump(clean_df,f)

        clean_drop = clean_df.drop('SALE PRICE',axis=1)
        with open(os.path.join(borough_path,'clean_dropped.pickle'),'wb') as f:
            pickle.dump(clean_drop,f)

        clean_head = clean_drop.iloc[0]
        data = {}
        for index,value in clean_head.iteritems():
            data[index] = value
        clean_input = pd.DataFrame(data,index=[0])
        with open(os.path.join(borough_path,'input.pickle'),'wb') as f:
            pickle.dump(clean_input,f)
            
        encoded_df = pd.get_dummies(clean_df)
        with open(os.path.join(borough_path,'encoded_df.pickle'),'wb') as f:
            pickle.dump(encoded_df,f)
            
        input_df = model(encoded_df,borough_path)

        shap_path = os.path.join(borough_path,'shap_values.pickle')
        with open(shap_path,'rb') as f:
            shap_value = pickle.load(f)

        plot_shap(clean_df,input_df,shap_value,borough_path)
        
        # Generating Statistics
        clean_df_p = clean_p(df)
        encoded_df = pd.get_dummies(clean_df_p)
        stat(encoded_df,borough_path)

if __name__ == "__main__":
    main()
