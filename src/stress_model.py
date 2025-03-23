import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("TrainData.csv")  


X = df[['CPU_Mean', 'PackRecv_Mean', 'PodsNumber_Mean']].values
y = df['StressRate_Mean'].values.reshape(-1, 1)  


scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)  


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

import torch.nn.functional as F

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


model = ANN()



model = ANN()
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)  


num_epochs = 400  
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:  
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)
    test_loss = criterion(y_test_pred, y_test).item()


y_test_pred_original = scaler_y.inverse_transform(y_test_pred.numpy())
y_test_original = scaler_y.inverse_transform(y_test.numpy())


print(f"Test Loss: {test_loss:.4f}")


for i in range(5):
    print(f"Predicted: {y_test_pred_original[i][0]:.2f}, Actual: {y_test_original[i][0]:.2f}")


torch.save(model.state_dict(), "stress_rate_model.pth")
print("Model saved as stress_rate_model.pth")



