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
        out = self.fc1(x)
        out = self.fc2(out)
        out += residual
        out = nn.functional.relu(out)
        return out


# ResNetFCN
class ResNetFCN(nn.Module):
    def __init__(self, in_features, num_blocks, num_classes=10):
        super(ResNetFCN, self).__init__()
        self.in_features = 64
        self.fc1 = nn.Linear(in_features, 64)
        self.blocks = self.make_layer(64, num_blocks)
        self.fc2 = nn.Linear(64, num_classes)

    def make_layer(self, out_features, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(self.in_features, out_features))
            self.in_features = out_features
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.fc1(x)
        out = self.blocks(out)
        out = self.fc2(out)
        return out

    def fit(self, train_loader, val_loader, num_epochs, lr, device):
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)

        # Move model to the device
        self.to(device)

        # Training loop
        for epoch in range(num_epochs):
            train_loss, train_accuracy = self.train_epoch(train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = self.evaluate(val_loader, criterion, device)

            # Print progress
            print(
                f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

    def train_epoch(self, train_loader, criterion, optimizer, device):
        self.train()
        train_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        accuracy = 100.0 * correct / total

        return train_loss, accuracy

    def evaluate(self, val_loader, criterion, device):
        self.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        accuracy = 100.0 * correct / total

        return val_loss, accuracy

    def predict(self, X: pd.DataFrame):
        self.eval()
        X = torch.tensor(X.values, dtype=torch.float32)
        return self(X).argmax(1).numpy()


def get_trained_model(X: pd.DataFrame, y: pd.DataFrame):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.long)
    train_dataset = torch.utils.data.TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    model = ResNetFCN(X.shape[1], 2)
    model.fit(train_loader, train_loader, 10, 0.01, device)
    return model
