#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### libraries

import pandas as pd
import numpy as np
import wfdb
import urllib.request
import matplotlib.pyplot as plt

### List of records
records = wfdb.io.get_record_list(db_dir = 'mitdb', records='all')


# In[6]:


### Dataframe of voltages

list_volt_0 = []
for i in records:
  image = wfdb.rdrecord(i, pb_dir='mitdb', sampto = 648000, channels=[0])
  signals_array, fields_dictionary = wfdb.rdsamp(i, pb_dir='mitdb', sampto = 648000, channels=[0])
  voltages = pd.DataFrame(signals_array)
  list_volt_0.append(voltages)

df_volt_0 = pd.concat(list_volt_0, axis=1) 
df_volt_0.columns = records 
df_volt_0.index.name = 'Time'
df_volt_0 = df_volt_0.reset_index()
df_volt_0 = pd.melt(df_volt_0,
                       id_vars = 'Time',
                       var_name="Patient",
                       value_name="Voltage_L2")
df_volt_0


# In[7]:


# adjusting df for merging at a later time
df_volt = df_volt_0
df_volt['Window'] = [i for i in range(8640) for _ in range(3600)] 
#8640 = 31,104,444(48 patients * sampleto number) / 3600

time_adj = pd.concat([pd.Series(range(3600))] * 8640, axis=0)
time_adj = pd.DataFrame(time_adj.reset_index(drop=True))
time_adj.columns = ['Time_Adj']
df_list = [df_volt, time_adj]
df_volt = pd.concat(df_list, axis=1)
df_volt


# In[70]:


### Annotations dataframe

list_ann = []
for i in records:
    ann = wfdb.rdann(i, 'atr', sampto = 648000, pb_dir='mitdb')
    labels_ann = ann.symbol
    annotated_time = ann.sample
    ann_df = pd.DataFrame({'Patient': i, 'Time':annotated_time, 'Annotation': labels_ann})
    list_ann.append(ann_df)
    
anns = pd.concat(list_ann, axis=0)
anns = anns.set_index(['Patient', 'Time'])['Annotation'].reset_index()
anns = anns.replace({'+':'N'})

anns['Results'] = np.where(anns['Annotation'] == 'x', 1, 0) #U is noisey data, labelled as 1
#anns


# In[71]:


total = anns['Results'].sum()
print(total)


# In[5]:


### Combining Windows Column from Voltage DF to Annotations DF

anns = df_volt.merge(anns, how='left', on = ['Patient', 'Time'])
anns = anns.pivot_table(values='Results', index='Window', aggfunc=np.sum)
anns.Results = np.where(anns['Results'] == 0, 0, 1)
# anns


# In[6]:


### Merge Voltages and Annotations
mitdb_df = df_volt.merge(anns, how='left', on = ['Window'])
# mitdb_df


# In[7]:


#### Dataframe by ECG Window
# Ignored Patient
# Results in 0 (normal ECG window) or 1 (abnormal ECG window)

df_by_window = mitdb_df.set_index(['Window','Patient','Results','Time_Adj'])['Voltage_L2'].unstack().reset_index()

# df_by_window
# df_by_window.to_csv('ECG_window_df.csv')


# In[8]:


df_by_window.to_csv('ECG_window_df.csv')


# In[9]:


df_by_window


# In[ ]:




