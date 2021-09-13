from PIL import Image
import os
import streamlit as st
import pandas as pd
import pickle

cur_path = os.path.dirname(__file__)
asset_path = os.path.join(cur_path,'assets')

st.set_page_config(layout="wide")
main_col,stat_col = st.columns(2)

main_col.write("""
# House Price Prediction App
This app predicts the housing prices for the five boroughs of New York
""")
main_col.write('---')

st.sidebar.header('Specify Input Parameters')
boroughs = os.listdir(asset_path)
boroughs.remove('.git')
def get_borough():
    return st.sidebar.selectbox('Borough',boroughs)

borough_path = os.path.join(asset_path,get_borough())

files = {}
names = os.listdir(borough_path)
for name in names:
    if '.pickle' in name:
        file_path = os.path.join(borough_path,name)
        with open(file_path,'rb') as f:
            name = name.replace('.pickle','')
            files[name] = pickle.load(f)
            continue
    if '.png' in name:
        file_path = os.path.join(borough_path,name)
        name = name.replace('.png','')
        files[name] = Image.open(file_path)

def user_input_features(df):
    input_data = {}

    qual_df = df.select_dtypes(exclude='number')
    quant_df = df.select_dtypes(include='number')

    for col in qual_df:
        input_data[col] = st.sidebar.selectbox(col,qual_df[col].unique().tolist())
    for col in quant_df:
        input_data[col] = st.sidebar.text_input(col,quant_df[col].iloc[0])

    for col in files['placeholder'].columns:
        n = 'NEIGHBORHOOD_'
        n_test = n + input_data['NEIGHBORHOOD']
        bc = 'BUILDING CLASS AT PRESENT_'
        bc_test = bc +  input_data['BUILDING CLASS AT PRESENT']
        tc = 'TAX CLASS AT PRESENT_'
        tc_test = tc + input_data['TAX CLASS AT PRESENT']
        if col == n_test or col == bc_test or col ==tc_test:
            files['placeholder'][col].values[:] = 1
    for col in quant_df:
        files['placeholder'][col].values[:] = input_data[col]

    normal_df = pd.DataFrame(input_data,index=[0])
    return normal_df

display_df = user_input_features(files['clean_dropped'])
prediction = files['regressor'].predict(files['placeholder'])

main_col.header('Specified Input Parameters')
main_col.dataframe(display_df)

main_col.header('Predict House Price')
main_col.write(prediction[0])

main_col.metric(label='Adjusted R Squared',value = files['stats']['R_squared'])
main_col.metric(label='MSE',value = files['stats']['MSE'])
main_col.metric(label='RMSE',value = files['stats']['RMSE'])
main_col.metric(label='MAE',value = files['stats']['MAE'])

stat_col.header('Feature Importance')
stat_col.image(files['SHAP'])

stat_col.header('Feature Importance ( Bar )')
stat_col.image(files['SHAP_BAR'])

stat_col.header('Residual Plot')
stat_col.image(files['ResidualsPlot'])
