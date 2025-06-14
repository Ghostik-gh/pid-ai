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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.2, seq_len=10):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len+1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # берем последний таймстемп
        y = x[:, -1, :]
        return self.fc(y)

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

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader], num_epochs: int, learning_rate: float, device: torch.device, early_stopping_patience: int = 10) -> Tuple[List[float], List[float]]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    train_losses = []
    val_losses = []
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
                torch.save(model.state_dict(), 'transformer_model.pth')
            else:
                patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Раннее прекращение на эпохе {epoch + 1}')
                break
            if (epoch + 1) % 10 == 0:
                print(f'Эпоха [{epoch+1}/{num_epochs}], Потери: {train_loss:.6f}, Валидация: {val_loss:.6f}')
        else:
            if (epoch + 1) % 10 == 0:
                print(f'Эпоха [{epoch+1}/{num_epochs}], Потери: {train_loss:.6f}')
        if check_convergence(train_losses):
            print(f'Достигнута сходимость на эпохе {epoch + 1}')
            break
    return train_losses, val_losses

def plot_training_results(train_losses: List[float], val_losses: List[float]):
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Обучение')
    if val_losses:
        plt.plot(val_losses, label='Валидация')
    plt.title('Процесс обучения Transformer')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка (MSE)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'transformer_training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
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
    D_MODEL = 32
    NHEAD = 4
    NUM_LAYERS = 2
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используется устройство: {device}')
    train_loader, val_loader, feature_scaler, target_scaler = prepare_data(
        'simulation_data_20250611_011638.json',
        SEQUENCE_LENGTH
    )
    transformer = TimeSeriesTransformer(input_size=5, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, seq_len=SEQUENCE_LENGTH).to(device)
    print("Обучение Transformer...")
    transformer_losses, transformer_val_losses = train_model(
        transformer,
        train_loader,
        val_loader,
        NUM_EPOCHS,
        LEARNING_RATE,
        device
    )
    plot_training_results(transformer_losses, transformer_val_losses)
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
    with open('transformer_scaler_params.json', 'w') as f:
        json.dump(scaler_params, f)
    print('Обучение Transformer завершено. Модель и параметры сохранены.')

if __name__ == '__main__':
    main() 