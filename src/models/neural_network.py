"""
Deep Learning Models (MLP, CNN)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class MLPModel(nn.Module):
    """Multi-layer Perceptron"""
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32]):
        super(MLPModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_neural_model(X_train, y_train, X_test, y_test, epochs=50):
    """Train neural network"""
    
    # Convert to torch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create model
    model = MLPModel(input_size=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train_t)
        loss = criterion(y_pred, y_train_t)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_t).numpy()
    
    mae = np.mean(np.abs(y_pred_test - y_test.reshape(-1, 1)))
    print(f"\nNeural Network Test MAE: {mae:.2f} mg/dL")
    
    return model