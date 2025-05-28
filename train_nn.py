import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import Tuple, List

class TemperatureDataset(Dataset):
    """Датасет для обучения нейронной сети"""
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

class LSTM(nn.Module):
    """LSTM модель для предсказания управляющего сигнала"""
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
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Ограничиваем выход в диапазоне [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Берем только последний выход LSTM
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

def prepare_data(data_path: str, sequence_length: int = 10) -> Tuple[Dataset, MinMaxScaler, MinMaxScaler]:
    """Подготовка данных для обучения"""
    # Загрузка данных
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Извлечение признаков и целевой переменной
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
    
    # Нормализация данных
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    features_normalized = feature_scaler.fit_transform(features)
    targets_normalized = target_scaler.fit_transform(targets)
    
    # Создание датасета
    dataset = TemperatureDataset(features_normalized, targets_normalized, sequence_length)
    
    return dataset, feature_scaler, target_scaler

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device
) -> List[float]:
    """Обучение модели"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    return losses

def plot_training_results(losses: List[float]):
    """Визуализация процесса обучения"""
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Процесс обучения модели')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка (MSE)')
    plt.grid(True)
    plt.savefig('training_results.png')
    plt.close()

def main():
    # Параметры
    SEQUENCE_LENGTH = 10
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # Определяем устройство для обучения
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используется устройство: {device}')
    
    # Подготовка данных
    dataset, feature_scaler, target_scaler = prepare_data(
        'simulation_data.json',
        SEQUENCE_LENGTH
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    
    # Создание и обучение модели
    model = LSTM(
        input_size=5,  # количество признаков
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    ).to(device)
    
    losses = train_model(
        model,
        train_loader,
        NUM_EPOCHS,
        LEARNING_RATE,
        device
    )
    
    # Визуализация результатов
    plot_training_results(losses)
    
    # Сохранение модели
    torch.save(model.state_dict(), 'model.pth')
    
    # Сохранение параметров скейлеров
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
    
    print('Обучение завершено. Модель и параметры сохранены.')

if __name__ == '__main__':
    main() 