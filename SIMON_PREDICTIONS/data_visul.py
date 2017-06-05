import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from tqdm import trange, tqdm
import pickle
import seaborn as sns

from hackathon.utils.utils import *

matplotlib.rcParams['figure.figsize'] = (5, 5)

power_df = load_dataframe("../data/merged_data/power.csv")

plt.figure(figsize=(20, 3))
power_df['NPWD2471'].plot(fontsize=16)
plt.show()

start_date, end_date = power_df.index[0], power_df.index[-1]

power_df_test = parse_data('../data/test_set/power.csv')

start_date_test, end_date_test = power_df_test.index[0], power_df_test.index[-1]

power_df_all = pd.concat([power_df, power_df_test])

dmop_24h_df = load_dataframe("../data/features/simple_dmop_24h.csv")

saaf_raw = pd.read_csv('../data/merged_data/saaf.csv', index_col=0, nrows=10)
saaf_df = load_dataframe("../data/merged_data/saaf.csv")
ltdata_df = load_dataframe("../data/merged_data/ltdata.csv")

evtf_df_raw = pd.read_csv('../data/merged_data/evtf.csv', index_col=0, nrows=10)
ftl_df_raw = pd.read_csv('../data/merged_data/ftl.csv')

method = 'nearest'
ltdata_df = align_to_power(ltdata_df, power_df_all, method=method)
saaf_df = align_to_power(saaf_df, power_df_all, method=method)
dmop_24h_df = align_to_power(dmop_24h_df, power_df_all, method=method)

all_data_df = pd.concat([power_df_all, ltdata_df, saaf_df, dmop_24h_df], axis=1)

all_data_df['usbx'] = np.cos(all_data_df['sx'])*np.sin(all_data_df['sa']) \
                        *np.sin(all_data_df['sz']) + np.sin(all_data_df['sx'])*np.cos(all_data_df['sa'])

all_data_df['usby'] = -np.sin(all_data_df['sx'])*np.sin(all_data_df['sa']) \
                        *np.sin(all_data_df['sz']) + np.cos(all_data_df['sx'])*np.cos(all_data_df['sa'])

all_data_df['usbz'] = np.cos(all_data_df['sz'])*np.sin(all_data_df['sa'])

train_data = all_data_df[all_data_df.index < end_date]
test_data = all_data_df.loc[power_df_test.index]

# returning back to unix time
train_data['ut_ms'] = to_utms(train_data.index)
train_data = train_data.set_index('ut_ms')
test_data['ut_ms'] = to_utms(test_data.index)
test_data = test_data.set_index('ut_ms')

train_data.to_csv("../data/train_data.csv")
test_data.to_csv("../data/test_data.csv")

names = power_df.columns.values

#for i, name in enumerate(names):
#    plt.figure(num=i)
#    (power_df[name]-power_df[name]).plot()
#    plt.title(name)

plt.show()
