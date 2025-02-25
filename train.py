import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from model import CNNClassifier
from kaggle.api.kaggle_api_extended import KaggleApi

import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_data() -> None:
    """
    Download the Kaggle Credit Card Fraud dataset.
    Ensure your Kaggle API credentials are in ~/.kaggle/kaggle.json.
    """
    api = KaggleApi()
    api.authenticate()
    dataset = "mlg-ulb/creditcardfraud"
    data_dir = "data"
    
    # Create the directory if it doesn't exist; ignore if it already exists.
    try:
        os.makedirs(data_dir, exist_ok=True)
        print(f"Directory '{data_dir}' is ready.")
    except Exception as e:
        print(f"An error occurred while creating '{data_dir}': {e}")
    
    print("Downloading dataset...")
    api.dataset_download_files(dataset, path=data_dir, unzip=True)
    print("Download complete.")


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df: pd.DataFrame):
    """Preprocess the data: separate features/target and normalize features."""
    X = df.drop(columns=["Class"])
    y = df["Class"].values.astype(np.float32)
    # Normalize features using min-max scaling
    X = (X - X.min()) / (X.max() - X.min())
    return X.values.astype(np.float32), y

def create_dataloaders(X, y, batch_size: int = 64, test_size: float = 0.2):
    """Split data and create PyTorch DataLoaders."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, device, epochs: int = 20):
    model.train()
    train_losses = []
    train_accuracies = []
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            preds = (outputs >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    return train_losses, train_accuracies

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = (outputs >= 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())
    auc = roc_auc_score(all_targets, all_preds)
    print("Test ROC AUC Score:", auc)
    report = classification_report(all_targets, all_preds)
    print("Classification Report:\n", report)

    # Create the 'outputs' folder if it does not exist
    os.makedirs("outputs", exist_ok=True)
    # Save the classification report to a text file
    with open("outputs/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"ROC AUC Score: {auc}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    return all_preds, all_targets

def plot_training_curves(losses, accuracies):
    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, marker="o")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, marker="o")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()

    # Create the 'outputs' folder if it does not exist
    os.makedirs("outputs", exist_ok=True)
    # Save the figure
    plt.savefig("outputs/training_curves.png")
    plt.close()

def main():
    data_path = "data/creditcard.csv"
    if not os.path.exists(data_path):
        download_data()
    df = load_data(data_path)
    X, y = preprocess_data(df)
    train_loader, test_loader = create_dataloaders(X, y)
    
    num_features = X.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier(num_features).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting training...")
    losses, accuracies = train_model(model, train_loader, criterion, optimizer, device, epochs=20)
    plot_training_curves(losses, accuracies)
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
