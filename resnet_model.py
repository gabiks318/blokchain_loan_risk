import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)

    def forward(self, x):
        residual = x
        out = nn.functional.relu(self.fc1(x))
        out = self.fc2(out)
        out += residual
        out = nn.functional.relu(out)
        return out


# ResNet for Regression
class ResNetRegression(nn.Module):
    def __init__(self, input_size, num_blocks):
        super(ResNetRegression, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 64)
        self.blocks = self.make_layer(64, num_blocks)
        self.fc2 = nn.Linear(64, 1)  # Regression layer with 1 output

    def make_layer(self, out_features, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(out_features, out_features))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.functional.relu(self.fc1(x))
        out = self.blocks(out)
        out = self.fc2(out)  # Regression layer instead of classification
        return out

    def fit(self, train_loader, val_loader, num_epochs, lr, device):
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Move model to the device
        self.to(device)

        # Training loop
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, criterion, optimizer, device)
            val_loss = self.evaluate(val_loader, criterion, device)

            # Print progress
            print(
                f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    def train_epoch(self, train_loader, criterion, optimizer, device):
        self.train()
        train_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, targets.view(-1, 1))  # Targets are 1D for regression
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        return train_loss

    def evaluate(self, val_loader, criterion, device):
        self.eval()
        val_loss = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, targets.view(-1, 1))  # Targets are 1D for regression

                val_loss += loss.item()

        val_loss /= len(val_loader)

        return val_loss

    def predict(self, X: pd.DataFrame):
        self.eval()
        X = torch.tensor(X.values, dtype=torch.float32)
        with torch.no_grad():
            predictions = self(X)
        return predictions.squeeze().numpy()


# Example usage:
def get_trained_model(X: pd.DataFrame, y: pd.DataFrame):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32)  # Convert y to float32 for regression
    train_dataset = torch.utils.data.TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)  # Use a separate data loader for validation
    model = ResNetRegression(X.shape[1], 2)  # Modify the input_size based on the number of input features
    model.fit(train_loader, val_loader, 100, 10, device)  # Use the correct data loader for validation
    return model
