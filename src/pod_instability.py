import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


df = pd.read_csv('dataSynthetic.csv', parse_dates=['Timestamp']).sort_values('Timestamp')


df['day'] = (df['Timestamp'] - df['Timestamp'].min()).dt.days  
df['hour'] = df['Timestamp'].dt.hour  
df['minute'] = df['Timestamp'].dt.minute  


df['seconds_total'] = (df['Timestamp'] - df['Timestamp'].min()).dt.total_seconds()


df = df.groupby(['Pod Name', 'Node Name', 'Timestamp']).agg('last').reset_index()


df = df.sort_values(['Pod Name', 'Timestamp'])
df['time_since_last_record'] = (df.groupby('Pod Name')['seconds_total']
                                .diff().fillna(0))


df['Target'] = ((df['Pod Restarts'] > 0) | (df['Pod Status'] != 'Running')).astype(int)


features = ['CPU Usage (%)', 'Memory Usage (%)', 'Pod Restarts', 'Ready Containers', 'Total Containers',
            'time_since_last_record', 'day', 'hour', 'minute', 'Pod Status', 'Pod Event Type', 
            'Pod Event Reason', 'Node Name']


for col in ['Pod Status', 'Pod Event Type', 'Pod Event Reason', 'Node Name']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))


scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])


seq_length = 5
def create_sequences(data, feature_cols, target_col, seq_length):
    X, y = [], []
    for pod, group in data.groupby('Pod Name'):
        group = group.sort_values('Timestamp')
        feature_data = group[feature_cols].values
        target_data = group[target_col].values
        if len(group) > seq_length:  
            for i in range(len(group) - seq_length):
                X.append(feature_data[i:i + seq_length])      
                y.append(target_data[i + seq_length])         
    return np.array(X), np.array(y)

X, y = create_sequences(df, features, 'Target', seq_length)


train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

df.to_csv('processed_data.csv', index=False)
print(X)
print(y)



















































