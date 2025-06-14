import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from datetime import datetime
import pandas as pd

class GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        out, _ = self.gru(x)
        last_output = out[:, -1, :]
        return self.fc(last_output)

class TemperatureDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_length: int = 10):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
    def __len__(self) -> int:
        return len(self.features) - self.sequence_length
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length - 1]
        return torch.FloatTensor(x), torch.FloatTensor([y])

def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def check_convergence(losses: List[float], window_size: int = 5, threshold: float = 1e-4) -> bool:
    if len(losses) < window_size * 2:
        return False
    window1 = np.mean(losses[-window_size*2:-window_size])
    window2 = np.mean(losses[-window_size:])
    return abs(window1 - window2) < threshold

def calculate_mae(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_mae = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            # Приводим к одинаковой размерности
            outputs = outputs.view(-1)
            batch_y = batch_y.view(-1)
            mae = torch.abs(outputs - batch_y).sum().item()
            total_mae += mae
            total_samples += batch_y.size(0)
    return total_mae / total_samples if total_samples > 0 else 0.0

def calculate_accuracy(model: nn.Module, data_loader: DataLoader, device: torch.device, threshold: float = 0.05) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            outputs = outputs.view(-1)
            batch_y = batch_y.view(-1)
            correct += ((torch.abs(outputs - batch_y) < threshold).sum().item())
            total += batch_y.size(0)
    return correct / total if total > 0 else 0.0

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader], num_epochs: int, learning_rate: float, device: torch.device, early_stopping_patience: int = 10) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    train_losses = []
    val_losses = []
    maes = []
    lrs = []
    accuracies = []
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        if val_loader:
            val_loss = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'gru_model.pth')
            else:
                patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Раннее прекращение на эпохе {epoch + 1}')
                break
            if (epoch + 1) % 10 == 0:
                print(f'Эпоха [{epoch+1}/{num_epochs}], Потери: {train_loss:.6f}, Валидация: {val_loss:.6f}')
            mae = calculate_mae(model, val_loader, device)
            maes.append(mae)
            lr = optimizer.param_groups[0]['lr']
            lrs.append(lr)
            acc = calculate_accuracy(model, val_loader, device)
            accuracies.append(acc)
        else:
            if (epoch + 1) % 10 == 0:
                print(f'Эпоха [{epoch+1}/{num_epochs}], Потери: {train_loss:.6f}')
            maes.append(None)
            lrs.append(None)
            accuracies.append(None)
        if check_convergence(train_losses):
            print(f'Достигнута сходимость на эпохе {epoch + 1}')
            break
    return train_losses, val_losses, maes, lrs, accuracies

def plot_training_results(train_losses: List[float], val_losses: List[float], maes: List[float] = None, lrs: List[float] = None, accuracies: List[float] = None, early_stop_epoch: int = None):
    plt.figure(figsize=(14, 8))
    plt.plot(train_losses, label='MSE (train)')
    if val_losses:
        plt.plot(val_losses, label='MSE (val)')
    if maes is not None:
        plt.plot(maes, label='MAE (val)', linestyle='--')
    if lrs is not None:
        plt.plot(lrs, label='Learning rate', linestyle=':')
    if accuracies is not None:
        plt.plot(accuracies, label='Accuracy (val)', linestyle='-.')
    if early_stop_epoch is not None:
        plt.axvline(early_stop_epoch, color='red', linestyle='--', label='Early stopping')
    plt.title('Процесс обучения GRU')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка / Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'gru_training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()

def prepare_data(data_path: str, sequence_length: int = 10, val_split: float = 0.2) -> Tuple[DataLoader, DataLoader, MinMaxScaler, MinMaxScaler]:
    with open(data_path, 'r') as f:
        data = json.load(f)
    features = []
    targets = []
    for record in data:
        features.append([
            record['current_temp'],
            record['target_temp'],
            record['error'],
            record['integral'],
            record['derivative']
        ])
        targets.append([record['output_signal']])
    features = np.array(features)
    targets = np.array(targets)
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    features_normalized = feature_scaler.fit_transform(features)
    targets_normalized = target_scaler.fit_transform(targets)
    dataset = TemperatureDataset(features_normalized, targets_normalized, sequence_length)
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader, feature_scaler, target_scaler

def main():
    SEQUENCE_LENGTH = 10
    HIDDEN_SIZE = 32
    NUM_LAYERS = 2
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используется устройство: {device}')
    train_loader, val_loader, feature_scaler, target_scaler = prepare_data(
        'simulation_data_20250611_011638.json',
        SEQUENCE_LENGTH
    )
    gru = GRUModel(input_size=5, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    print("Обучение GRU...")
    gru_losses, gru_val_losses, gru_maes, gru_lrs, gru_accuracies = train_model(
        gru,
        train_loader,
        val_loader,
        NUM_EPOCHS,
        LEARNING_RATE,
        device
    )
    plot_training_results(gru_losses, gru_val_losses, gru_maes, gru_lrs, gru_accuracies)
    scaler_params = {
        'feature_scaler': {
            'data_min_': feature_scaler.data_min_.tolist(),
            'data_max_': feature_scaler.data_max_.tolist(),
            'data_range_': feature_scaler.data_range_.tolist(),
            'scale_': feature_scaler.scale_.tolist(),
            'min_': feature_scaler.min_.tolist(),
            'n_features_in_': feature_scaler.n_features_in_,
            'n_samples_seen_': feature_scaler.n_samples_seen_
        },
        'target_scaler': {
            'data_min_': target_scaler.data_min_.tolist(),
            'data_max_': target_scaler.data_max_.tolist(),
            'data_range_': target_scaler.data_range_.tolist(),
            'scale_': target_scaler.scale_.tolist(),
            'min_': target_scaler.min_.tolist(),
            'n_features_in_': target_scaler.n_features_in_,
            'n_samples_seen_': target_scaler.n_samples_seen_
        },
        'sequence_length': SEQUENCE_LENGTH
    }
    with open('gru_scaler_params.json', 'w') as f:
        json.dump(scaler_params, f)
    print('Обучение GRU завершено. Модель и параметры сохранены.')

if __name__ == '__main__':
    main() 