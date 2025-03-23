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



df['Target'] = ((df['Network Receive Packets Dropped (p/s)'] > 0.07) | 
                (df['Network Transmit Packets Dropped (p/s)'] > 0.07)).astype(int)


features = ['Network Receive Bytes', 'Network Transmit Bytes', 'Network Receive Packets (p/s)', 
            'Network Transmit Packets (p/s)', 'Network Receive Packets Dropped (p/s)', 
            'Network Transmit Packets Dropped (p/s)', 'FS Reads Total (MB)', 'FS Writes Total (MB)', 
            'FS Reads/Writes Total (MB)', 'FS Reads Bytes Total (MB)', 'FS Writes Bytes Total (MB)', 
            'FS Reads/Writes Bytes Total (MB)', 'CPU Usage (%)', 'Memory Usage (%)', 'Memory Usage (Swap) (MB)', 
            'Pod Restarts', 'Ready Containers', 'Total Containers', 'time_since_last_record', 'day', 'hour', 'minute']
cat_features = ['Pod Status', 'Pod Event Type', 'Pod Event Reason', 'Event Reason', 'Node Name']


for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
features.extend(cat_features)


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


X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size=len(features), hidden_size=50, num_layers=2).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


model.eval()
with torch.no_grad():
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    outputs = model(X_test)
    predicted = (outputs.squeeze() > 0.5).float()
    accuracy = (predicted == y_test).float().mean()
    print(f'Test Accuracy: {accuracy.item():.4f}')

torch.save(model.state_dict(), 'network_disruptions_model.pth')