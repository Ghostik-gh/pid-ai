import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from dataclasses import dataclass
from main import RoomModel, SystemState, PIDController
from train_nn import LSTM
from sklearn.preprocessing import MinMaxScaler

class HybridController:
    """Гибридный регулятор, объединяющий ПИД-регулятор и нейронную сеть"""
    def __init__(
        self,
        pid_params: Tuple[float, float, float],
        model_path: str,
        scaler_params_path: str,
        alpha: float = 0.5  # Коэффициент смешивания (0 - только ПИД, 1 - только НС)
    ):
        # Загрузка параметров скейлеров
        with open(scaler_params_path, 'r') as f:
            scaler_params = json.load(f)
        
        # Создание и настройка скейлеров
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        # Установка параметров для feature_scaler
        self.feature_scaler.data_min_ = np.array(scaler_params['feature_scaler']['data_min_'])
        self.feature_scaler.data_max_ = np.array(scaler_params['feature_scaler']['data_max_'])
        self.feature_scaler.data_range_ = np.array(scaler_params['feature_scaler']['data_range_'])
        self.feature_scaler.scale_ = np.array(scaler_params['feature_scaler']['scale_'])
        self.feature_scaler.min_ = np.array(scaler_params['feature_scaler']['min_'])
        self.feature_scaler.n_features_in_ = scaler_params['feature_scaler']['n_features_in_']
        self.feature_scaler.n_samples_seen_ = scaler_params['feature_scaler']['n_samples_seen_']
        
        # Установка параметров для target_scaler
        self.target_scaler.data_min_ = np.array(scaler_params['target_scaler']['data_min_'])
        self.target_scaler.data_max_ = np.array(scaler_params['target_scaler']['data_max_'])
        self.target_scaler.data_range_ = np.array(scaler_params['target_scaler']['data_range_'])
        self.target_scaler.scale_ = np.array(scaler_params['target_scaler']['scale_'])
        self.target_scaler.min_ = np.array(scaler_params['target_scaler']['min_'])
        self.target_scaler.n_features_in_ = scaler_params['target_scaler']['n_features_in_']
        self.target_scaler.n_samples_seen_ = scaler_params['target_scaler']['n_samples_seen_']
        
        self.sequence_length = scaler_params['sequence_length']
        
        # Инициализация LSTM модели
        self.model = LSTM(input_size=5, hidden_size=64, num_layers=2)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
        self.model.eval()
        
        # Инициализация ПИД-регулятора
        self.pid = PIDController(*pid_params)
        
        # Параметры
        self.alpha = alpha
        self.feature_buffer = []
        
    def update_buffer(self, state: List[float]):
        """Обновление буфера состояний для LSTM"""
        self.feature_buffer.append(state)
        if len(self.feature_buffer) > self.sequence_length:
            self.feature_buffer.pop(0)
    
    def get_nn_prediction(self, features: np.ndarray) -> float:
        """Получение предсказания от нейронной сети"""
        if len(self.feature_buffer) < self.sequence_length:
            return 0.0
        
        # Подготовка данных
        features_normalized = self.feature_scaler.transform(
            np.array(self.feature_buffer)
        )
        
        # Преобразование в тензор
        x = torch.FloatTensor(features_normalized).unsqueeze(0)  # добавляем размерность батча
        
        # Получение предсказания
        with torch.no_grad():
            output = self.model(x)
        
        # Обратное преобразование
        prediction = self.target_scaler.inverse_transform(
            output.numpy()
        )[0][0]
        
        return prediction
    
    def calculate(self, target: float, current: float) -> Tuple[float, float, float, float]:
        """Вычисление управляющего сигнала"""
        # Получаем сигнал от ПИД-регулятора
        pid_output, error, integral, derivative = self.pid.calculate(target, current)
        
        # Обновляем буфер состояний
        state = [current, target, error, integral, derivative]
        self.update_buffer(state)
        
        # Получаем предсказание от нейронной сети
        nn_output = self.get_nn_prediction(state)
        
        # Смешиваем сигналы
        output = (1 - self.alpha) * pid_output + self.alpha * nn_output
        
        return output, error, integral, derivative

class HybridSimulation:
    """Класс для запуска симуляции с гибридным регулятором"""
    def __init__(
        self,
        pid_params: Tuple[float, float, float],
        model_path: str,
        target_temp: float,
        alpha: float = 0.5,
        simulation_time: float = 300.0
    ):
        self.controller = HybridController(pid_params, model_path, 'scaler_params.json', alpha)
        self.room = RoomModel()
        self.target_temp = target_temp
        self.simulation_time = simulation_time
        self.dt = 0.1
        self.states: List[SystemState] = []
        
    def run(self) -> List[SystemState]:
        """Запускает симуляцию и собирает данные"""
        steps = int(self.simulation_time / self.dt)
        
        for step in range(steps):
            current_time = step * self.dt
            current_temp = self.room.current_temp
            
            # Получаем управляющий сигнал от гибридного регулятора
            output, error, integral, derivative = self.controller.calculate(
                self.target_temp, current_temp
            )
            
            # Обновляем температуру в помещении
            new_temp = self.room.update(output)
            
            # Сохраняем состояние системы
            state = SystemState(
                time=current_time,
                current_temp=current_temp,
                target_temp=self.target_temp,
                output_signal=output,
                error=error,
                integral=integral,
                derivative=derivative
            )
            self.states.append(state)
        
        return self.states
    
    def save_data(self, filename: str = "hybrid_simulation_data.json"):
        """Сохраняет данные симуляции в JSON файл"""
        data = [{
            "time": state.time,
            "current_temp": state.current_temp,
            "target_temp": state.target_temp,
            "output_signal": state.output_signal,
            "error": state.error,
            "integral": state.integral,
            "derivative": state.derivative
        } for state in self.states]
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    
    def plot_results(self, save_path: str = 'hybrid_simulation_results.png'):
        """Визуализирует результаты симуляции"""
        times = [state.time for state in self.states]
        temps = [state.current_temp for state in self.states]
        outputs = [state.output_signal for state in self.states]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # График температуры
        ax1.plot(times, temps, 'b-', label='Текущая температура')
        ax1.axhline(y=self.target_temp, color='r', linestyle='--', label='Целевая температура')
        ax1.set_xlabel('Время (с)')
        ax1.set_ylabel('Температура (°C)')
        ax1.legend()
        ax1.grid(True)
        
        # График управляющего сигнала
        ax2.plot(times, outputs, 'g-', label='Гибридный сигнал управления')
        ax2.set_xlabel('Время (с)')
        ax2.set_ylabel('Мощность охлаждения (0-1)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def main():
    # Параметры симуляции
    pid_params = (2.0, 0.05, 1.0)  # Те же параметры, что и при сборе данных
    target_temp = 20.0
    simulation_time = 300.0
    alpha = 0.9  # Равное влияние ПИД и нейронной сети
    
    # Создаем и запускаем симуляцию
    simulation = HybridSimulation(
        pid_params,
        'model.pth',
        target_temp,
        alpha,
        simulation_time
    )
    
    simulation.run()
    simulation.save_data()
    simulation.plot_results()
    
    print('Симуляция с гибридным регулятором завершена.')

if __name__ == '__main__':
    main() 