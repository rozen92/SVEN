import pandas as pd
import pathlib as P
from scipy.stats.qmc import LatinHypercube as lhc

sampler = lhc(2, strength = 1, seed = 42)
samples = sampler.random(n = 100)

samples[:,0] = samples[:,0]*8 + 4
samples[:,1] = samples[:,1]*60 - 30
print(f"{samples[0:10,0]}")

"""
file = P.Path(__file__).parent/'results_10.csv'


df = pd.read_csv(file, sep = ',')

df.insert(loc = 1, column = 'TSR', value = None)

df.loc[df['Yaw'] == df['Yaw'][0], 'TSR'] =    8.11563322   
df.loc[df['Yaw'] == df['Yaw'][2592], 'TSR'] = 7.53378951 

df.to_csv(file, sep = ',', index = False)

print("Job done.")
"""