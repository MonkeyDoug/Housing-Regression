import os
import shap
import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_shap(clean_df,input_df,shap_value,borough_path):
    sum_col = ['NEIGHBORHOOD','TAXCLASS','BUILDINGCLASS']
    sums = []
    for col in sum_col:
        for column in input_df.columns:
            if ( col in column ):
                loc1 = input_df.columns.get_loc(column)
                if( col == 'NEIGHBORHOOD' ):
                    rloc1 = loc1
                break
        for column in input_df.columns[::-1]:
            if ( col in column ):
                loc2 = input_df.columns.get_loc(column)
                break

        sum_arr = []
        for i in range(shap_value.shape[0]):
            row_sum = np.sum(shap_value[i][loc1:loc2])
            sum_arr.append(row_sum)

        sums.append(sum_arr)

    sums = np.array(sums)
    main = shap_value[:,:rloc1]
    main = np.concatenate((main,sums.T),axis=1)
    clean_df.drop('SALE PRICE',axis=1,inplace=True)

    plt.figure(figsize=(20,10))
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(main,clean_df,show=False)
    plt.gcf().set_size_inches(20,10)
    plt.tight_layout()
    plt.savefig(f'{borough_path}\\SHAP.png')
    plt.clf()

    plt.figure(figsize=(25,10))
    plt.title('Feature importance based on SHAP values ( bar )')
    shap.summary_plot(main,clean_df,plot_type="bar",show=False)
    plt.gcf().set_size_inches(25,10)
    plt.tight_layout()
    plt.savefig(f'{borough_path}\\SHAP_BAR.png')
    plt.clf()
