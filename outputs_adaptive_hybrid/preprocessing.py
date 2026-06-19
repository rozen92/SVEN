import pandas as pd
import os
import sys
import pathlib as P
import numpy as np

def preprocess() : 
    dfs = [
        pd.read_csv(P.Path(__file__).parent / f, sep=',')
        for f in os.listdir(P.Path(__file__).parent)
        if f.endswith('.csv') and 'bem' not in f
    ]
    df = pd.concat(dfs, axis = 0, ignore_index = True)
    print("\nFusion terminée.\n")

    """
    for bye in deffec : 
        df.drop(df[df['TSR'] == bye].index, inplace = True)
    """
    df['theta'] = (df['theta'] + 90)%360 # Ajouter 90° et moduler par 360
    df['Ft'] = df['Ft']*-1 
    
    df = df.sort_values(by = 'Yaw')
    df1 = df.copy()

    for i, yaw in enumerate(sorted(df['Yaw'].unique())) :
        df_loc = df1[df['Yaw'] == yaw]
        df_loc = df_loc.sort_values(by = 'r')
        df_loc_copy = df_loc.copy()
        for j,r in enumerate(sorted(df['r'].unique())) :
            df_loc_loc = df_loc_copy[df_loc_copy['r'] == r]
            df_loc_loc = df_loc_loc.sort_values(by = 'theta')
            df_loc[72*j:72*(j+1)] = df_loc_loc
        df[2592*i:2592*(i+1)] = df_loc
    

    df = df.rename(columns = {'V_eff' : 'V_eff_SVEN', 'Alpha_deg' : 'alpha_SVEN', 'Fn' : 'Fn_SVEN', 'Ft' : 'Ft_SVEN'})    
    df = df.reset_index(drop = True)
    df.to_csv(P.Path(__file__).parent / 'dataset_sven.csv', sep=',', index=False)

    return

start_preprocess = True
if start_preprocess == True :
    preprocess()

