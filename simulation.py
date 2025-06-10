import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import json
import torch
from sklearn.preprocessing import MinMaxScaler
import argparse
from train_nn import LSTM
from datetime import datetime
import torch.nn as nn

@dataclass
class SystemState:
    """Класс для хранения состояния системы в каждый момент времени"""
    time: float
    current_temp: float
    target_temp: float
    output_signal: float
    error: float
    integral: float
    derivative: float

class PIDController:
    """ПИД-регулятор для управления системой охлаждения"""
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.dt = 0.1
        
        self.integral_min = -100.0
        self.integral_max = 100.0

    def calculate(self, target: float, current: float) -> Tuple[float, float, float, float]:
        """Вычисляет управляющий сигнал на основе текущей и целевой температуры"""
        error = target - current
        
        self.integral += error * self.dt
        
        # Ограничиваем интегральную составляющую для предотвращения интегрального насыщения
        self.integral = np.clip(self.integral, self.integral_min, self.integral_max)
        
        derivative = (error - self.prev_error) / self.dt if self.dt > 0 else 0
        
        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * derivative
        
        output = -(p_term + i_term + d_term)
        
        # Ограничиваем выходной сигнал в диапазоне [0, 1]
        output = np.clip(output, 0, 1)
        
        self.prev_error = error
        return output, error, self.integral, derivative

class PerlinNoiseGenerator:
    """Генератор шума Перлина для создания случайного теплового воздействия"""
    def __init__(self, octaves=3, persistence=0.5, lacunarity=2.0):
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.gradients: Dict[int, float] = {}
        np.random.seed()  
        
    def _get_gradient(self, ix: int) -> float:
        """Получает или генерирует градиент для заданной точки"""
        if ix not in self.gradients:
            self.gradients[ix] = np.random.rand() * 2 - 1
        return self.gradients[ix]
    
    def _interpolate(self, a0: float, a1: float, w: float) -> float:
        """Интерполяция значений с использованием кривой сглаживания"""
        return a0 + (a1 - a0) * (3.0 * w * w - 2.0 * w * w * w)
    
    def noise(self, x: float) -> float:
        """Генерирует одномерный шум Перлина"""
        x0 = int(np.floor(x))
        x1 = x0 + 1
        
        dx = x - x0
        
        g0 = self._get_gradient(x0)
        g1 = self._get_gradient(x1)
        
        v0 = g0 * dx
        v1 = g1 * (dx - 1)
        
        return self._interpolate(v0, v1, dx)
    
    def octave_noise(self, x: float) -> float:
        """Генерирует шум с несколькими октавами"""
        total = 0
        frequency = 1
        amplitude = 1
        max_value = 0
        
        for _ in range(self.octaves):
            total += self.noise(x * frequency) * amplitude
            max_value += amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity
        
        return total / max_value if max_value > 0 else 0

class ThermalDisturbance:
    """Класс для генерации случайного теплового воздействия"""
    def __init__(self, max_amplitude: float = 5.0, time_scale: float = 0.05, strong_events: List[Tuple[float, float, float]] = None):
        self.perlin = PerlinNoiseGenerator(octaves=4, persistence=0.5, lacunarity=2.0)
        self.max_amplitude = max_amplitude
        self.time_scale = time_scale
        self.offset = np.random.rand() * 1000
        self.strong_events = strong_events if strong_events is not None else []
    
    def get_disturbance(self, time: float) -> float:
        """Получает значение теплового воздействия для заданного момента времени"""
        # Нормализуем время для более плавного изменения
        scaled_time = time * self.time_scale + self.offset
        
        # Базовый шум Перлина
        base_noise = self.perlin.octave_noise(scaled_time)
        
        # Медленная модуляция (период примерно 10 минут)
        modulation = 0.5 * (1 + np.sin(time * 0.01))
        
        # Случайные всплески (каждые ~50 секунд)
        spikes = 0.2 * np.random.rand() * np.exp(-0.1 * (time % 50))
        
        # Комбинируем все эффекты и масштабируем
        disturbance = self.max_amplitude * (
            0.6 * base_noise + 
            0.3 * modulation + 
            0.1 * spikes
        )
        
        # Добавляем мощные возмущения, если время попадает в событие
        for start, duration, amplitude in self.strong_events:
            if start <= time < start + duration:
                disturbance += amplitude
        
        return disturbance

class RoomModel:
    """Модель помещения с тепловой инерцией"""
    def __init__(self, 
                 initial_temp: float = 25.0,
                 thermal_mass: float = 2000.0,  # Дж/К
                 heat_transfer_coef: float = 50.0,  # Вт/К
                 ambient_temp: float = 30.0,  # °C
                 strong_events: List[Tuple[float, float, float]] = None):
        self.current_temp = initial_temp
        self.thermal_mass = thermal_mass
        self.heat_transfer_coef = heat_transfer_coef
        self.ambient_temp = ambient_temp
        self.max_cooling_power = 2000  # Вт
        self.dt = 0.1  # Шаг времени (в секундах)
        self.time = 0.0  # Текущее время симуляции
        self.thermal_disturbance = ThermalDisturbance(max_amplitude=3.0, time_scale=0.05, strong_events=strong_events)

    def update(self, cooling_power: float) -> float:
        """Обновляет температуру в помещении на основе входного сигнала охлаждения"""
        actual_cooling = cooling_power * self.max_cooling_power
        
        # Добавляем случайное тепловое воздействие
        disturbance = self.thermal_disturbance.get_disturbance(self.time)
        
        # Базовый теплообмен с окружающей средой
        heat_flow = self.heat_transfer_coef * (self.ambient_temp - self.current_temp)
        
        # Добавляем тепловое воздействие к потоку тепла
        heat_flow += disturbance * 100  # Масштабируем воздействие
        
        # Изменение температуры с учетом всех факторов
        delta_temp = (heat_flow - actual_cooling) * self.dt / self.thermal_mass
        self.current_temp += delta_temp
        
        self.time += self.dt
        return self.current_temp

class HybridController:
    """Гибридный регулятор, объединяющий ПИД-регулятор и нейронную сеть"""
    def __init__(
        self,
        pid_params: Tuple[float, float, float],
        model_path: str,
        scaler_params_path: str,
        alpha: float = 0.5
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
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
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
        
        features_normalized = self.feature_scaler.transform(np.array(self.feature_buffer))
        x = torch.FloatTensor(features_normalized).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(x)
        
        prediction = self.target_scaler.inverse_transform(output.numpy())[0][0]
        return prediction
    
    def calculate(self, target: float, current: float) -> Tuple[float, float, float, float]:
        """Вычисление управляющего сигнала"""
        pid_output, error, integral, derivative = self.pid.calculate(target, current)
        
        state = [current, target, error, integral, derivative]
        self.update_buffer(state)
        
        nn_output = self.get_nn_prediction(state)
        output = (1 - self.alpha) * pid_output + self.alpha * nn_output
        
        return output, error, integral, derivative

class Simulation:
    """Универсальный класс для запуска симуляции"""
    def __init__(
        self,
        controller_type: str,
        pid_params: Tuple[float, float, float],
        target_temp: float = None,
        initial_temp: float = None,
        simulation_time: float = 300.0,
        model_path: str = None,
        scaler_params_path: str = None,
        alpha: float = 0.5,
        strong_disturbances: bool = False,
        num_strong_events: int = 3,
        strong_event_duration: float = 10.0,
        strong_event_amplitude: float = 10.0
    ):
        # Генерация случайных температур, если не указаны
        if target_temp is None:
            target_temp = np.random.uniform(18.0, 28.0)  # Случайная целевая температура
        if initial_temp is None:
            initial_temp = np.random.uniform(25.0, 35.0)  # Случайная начальная температура
        
        # Генерация мощных возмущений
        strong_events = []
        if strong_disturbances:
            max_time = simulation_time - strong_event_duration
            event_starts = np.sort(np.random.uniform(0, max_time, num_strong_events))
            for start in event_starts:
                strong_events.append((start, strong_event_duration, strong_event_amplitude))
        else:
            strong_events = None
        
        self.room = RoomModel(initial_temp=initial_temp, strong_events=strong_events)
        self.target_temp = target_temp
        self.simulation_time = simulation_time
        self.dt = 0.1
        self.states: List[SystemState] = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if controller_type == "pid":
            self.controller = PIDController(*pid_params)
        elif controller_type == "hybrid":
            if not model_path or not scaler_params_path:
                raise ValueError("Для гибридного контроллера необходимо указать путь к модели и параметрам скейлера")
            self.controller = HybridController(pid_params, model_path, scaler_params_path, alpha)
        else:
            raise ValueError("Неизвестный тип контроллера")
        
        self.controller_type = controller_type

    def run(self) -> List[SystemState]:
        """Запускает симуляцию и собирает данные"""
        steps = int(self.simulation_time / self.dt)
        
        for step in range(steps):
            current_time = step * self.dt
            current_temp = self.room.current_temp
            
            output, error, integral, derivative = self.controller.calculate(
                self.target_temp, current_temp
            )
            
            new_temp = self.room.update(output)
            
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

    def save_data(self):
        """Сохраняет данные симуляции в JSON файл"""
        base_filename = "simulation_data" if self.controller_type == "pid" else "hybrid_simulation_data"
        filename = f"{base_filename}_{self.timestamp}.json"
        
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
        return filename

    def plot_results(self):
        """Визуализирует результаты симуляции"""
        base_filename = "simulation_results" if self.controller_type == "pid" else "hybrid_simulation_results"
        filename = f"{base_filename}_{self.timestamp}.png"
        controller_label = "ПИД" if self.controller_type == "pid" else "Гибридный"
        
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
        ax2.plot(times, outputs, 'g-', label=f'{controller_label} сигнал управления')
        ax2.set_xlabel('Время (с)')
        ax2.set_ylabel('Мощность охлаждения (0-1)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        return filename

def main():
    parser = argparse.ArgumentParser(description='Симуляция системы охлаждения')
    parser.add_argument('--mode', type=str, choices=['pid', 'hybrid'], required=True,
                      help='Режим работы: pid - только ПИД-регулятор, hybrid - гибридный регулятор')
    parser.add_argument('--alpha', type=float, default=0.5,
                      help='Коэффициент смешивания для гибридного режима (0 - только ПИД, 1 - только НС)')
    parser.add_argument('--num_simulations', type=int, default=1,
                      help='Количество симуляций для сбора данных')
    parser.add_argument('--strong_disturbances', action='store_true',
                      help='Включить мощные тепловые возмущения во время симуляции')
    parser.add_argument('--num_strong_events', type=int, default=3,
                      help='Количество мощных возмущений за симуляцию (по умолчанию 3)')
    parser.add_argument('--strong_event_duration', type=float, default=10.0,
                      help='Длительность одного мощного возмущения (сек) (по умолчанию 10)')
    parser.add_argument('--strong_event_amplitude', type=float, default=10.0,
                      help='Амплитуда мощного возмущения (по умолчанию 10)')
    args = parser.parse_args()

    pid_params = (10.0, 0.5, 0.2)
    simulation_time = 300.0

    try:
        all_data = []
        
        for i in range(args.num_simulations):
            print(f"\nЗапуск симуляции {i+1}/{args.num_simulations}...")
            
            if args.mode == "pid":
                simulation = Simulation(
                    controller_type="pid",
                    pid_params=pid_params,
                    simulation_time=simulation_time,
                    strong_disturbances=args.strong_disturbances,
                    num_strong_events=args.num_strong_events,
                    strong_event_duration=args.strong_event_duration,
                    strong_event_amplitude=args.strong_event_amplitude
                )
            else:
                simulation = Simulation(
                    controller_type="hybrid",
                    pid_params=pid_params,
                    simulation_time=simulation_time,
                    model_path='model.pth',
                    scaler_params_path='scaler_params.json',
                    alpha=args.alpha,
                    strong_disturbances=args.strong_disturbances,
                    num_strong_events=args.num_strong_events,
                    strong_event_duration=args.strong_event_duration,
                    strong_event_amplitude=args.strong_event_amplitude
                )

            simulation.run()
            
            data = [{
                "time": state.time,
                "current_temp": state.current_temp,
                "target_temp": state.target_temp,
                "output_signal": state.output_signal,
                "error": state.error,
                "integral": state.integral,
                "derivative": state.derivative
            } for state in simulation.states]
            
            all_data.extend(data)
            
            plot_file = simulation.plot_results()
            print(f"График сохранен в: {plot_file}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_file = f"{'simulation' if args.mode == 'pid' else 'hybrid'}_data_{timestamp}.json"
        with open(data_file, 'w') as f:
            json.dump(all_data, f, indent=4)
        
        print(f"\nВсе симуляции завершены успешно.")
        print(f"Общий набор данных сохранен в: {data_file}")

    except Exception as e:
        print(f"Ошибка при выполнении симуляции: {str(e)}")

if __name__ == "__main__":
    main() 