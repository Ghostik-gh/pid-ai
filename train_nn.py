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

class PlantModel(nn.Module):
    """Модель объекта управления"""
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super(PlantModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Многослойная архитектура
        layers = []
        current_size = input_size
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_size = hidden_size
        
        layers.append(nn.Linear(hidden_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class LSTM(nn.Module):
    """LSTM модель для регулятора"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Расширенная выходная часть
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

class TemperatureDataset(Dataset):
    """Датасет с поддержкой валидации"""
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

def train_epoch(model: nn.Module, 
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device) -> float:
    """Обучение одной эпохи"""
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

def validate(model: nn.Module,
            val_loader: DataLoader,
            criterion: nn.Module,
            device: torch.device) -> float:
    """Валидация модели"""
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
    """Проверка сходимости обучения"""
    if len(losses) < window_size * 2:
        return False
    
    window1 = np.mean(losses[-window_size*2:-window_size])
    window2 = np.mean(losses[-window_size:])
    
    return abs(window1 - window2) < threshold

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    early_stopping_patience: int = 10
) -> Tuple[List[float], List[float]]:
    """Обучение модели с валидацией и ранней остановкой"""
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

                torch.save(model.state_dict(), 'model.pth')
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
        
        # Проверка сходимости
        if check_convergence(train_losses):
            print(f'Достигнута сходимость на эпохе {epoch + 1}')
            break
    
    return train_losses, val_losses

def plot_training_results(train_losses: List[float], val_losses: List[float]):
    """Визуализация процесса обучения"""
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Обучение')
    if val_losses:
        plt.plot(val_losses, label='Валидация')
    plt.title('Процесс обучения модели')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка (MSE)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()

def prepare_data(data_path: str, sequence_length: int = 10, val_split: float = 0.2) -> Tuple[DataLoader, DataLoader, MinMaxScaler, MinMaxScaler]:
    """Подготовка данных с разделением на обучающую и валидационную выборки"""
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
    
    # Разделение на обучающую и валидационную выборки
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, feature_scaler, target_scaler

def main():
    SEQUENCE_LENGTH = 10
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используется устройство: {device}')
    
    train_loader, val_loader, feature_scaler, target_scaler = prepare_data(
        'simulation_data_20250604_213252.json',
        SEQUENCE_LENGTH
    )
    
    # Создание и обучение модели объекта управления
    plant_model = PlantModel(input_size=5, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    print("Обучение модели объекта управления...")
    plant_losses, plant_val_losses = train_model(
        plant_model,
        train_loader,
        val_loader,
        NUM_EPOCHS,
        LEARNING_RATE,
        device
    )
    
    controller = LSTM(
        input_size=5,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    ).to(device)
    
    print("Предварительное обучение регулятора...")
    controller_losses, controller_val_losses = train_model(
        controller,
        train_loader,
        val_loader,
        NUM_EPOCHS,
        LEARNING_RATE,
        device
    )
    
    plot_training_results(plant_losses, plant_val_losses)
    plot_training_results(controller_losses, controller_val_losses)
    
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
    
    with open('scaler_params.json', 'w') as f:
        json.dump(scaler_params, f)
    
    print('Обучение завершено. Модели и параметры сохранены.')

if __name__ == '__main__':
    main() 