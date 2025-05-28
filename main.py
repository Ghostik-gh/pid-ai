import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import json
from pathlib import Path

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
        self.kp = kp  # Пропорциональный коэффициент
        self.ki = ki  # Интегральный коэффициент
        self.kd = kd  # Дифференциальный коэффициент
        self.prev_error = 0.0
        self.integral = 0.0
        self.dt = 0.1  # Шаг времени (в секундах)
        
        # Добавляем ограничения для интегральной составляющей
        self.integral_min = -100.0
        self.integral_max = 100.0

    def calculate(self, target: float, current: float) -> Tuple[float, float, float, float]:
        """Вычисляет управляющий сигнал на основе текущей и целевой температуры"""
        error = target - current
        
        # Вычисление составляющих ПИД
        self.integral += error * self.dt
        
        # Ограничиваем интегральную составляющую для предотвращения интегрального насыщения
        self.integral = np.clip(self.integral, self.integral_min, self.integral_max)
        
        derivative = (error - self.prev_error) / self.dt if self.dt > 0 else 0
        
        # Расчет каждой составляющей отдельно для лучшего контроля
        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * derivative
        
        # Расчет управляющего сигнала
        output = -(p_term + i_term + d_term)  # Инвертируем сигнал, так как отрицательная ошибка должна включать охлаждение
        
        # Ограничиваем выходной сигнал в диапазоне [0, 1]
        output = np.clip(output, 0, 1)
        
        self.prev_error = error
        return output, error, self.integral, derivative

class RoomModel:
    """Модель помещения с тепловой инерцией"""
    def __init__(self, 
                 initial_temp: float = 25.0,
                 thermal_mass: float = 2000.0,  # Дж/К
                 heat_transfer_coef: float = 50.0,  # Вт/К
                 ambient_temp: float = 30.0):  # °C
        self.current_temp = initial_temp
        self.thermal_mass = thermal_mass
        self.heat_transfer_coef = heat_transfer_coef
        self.ambient_temp = ambient_temp
        self.max_cooling_power = 2000  # Вт
        self.dt = 0.1  # Шаг времени (в секундах)

    def update(self, cooling_power: float) -> float:
        """Обновляет температуру в помещении на основе входного сигнала охлаждения"""
        # Мощность охлаждения (пропорциональная входному сигналу)
        actual_cooling = cooling_power * self.max_cooling_power
        
        # Теплообмен с окружающей средой
        heat_flow = self.heat_transfer_coef * (self.ambient_temp - self.current_temp)
        
        # Изменение температуры
        delta_temp = (heat_flow - actual_cooling) * self.dt / self.thermal_mass
        self.current_temp += delta_temp
        
        return self.current_temp

class CoolingSimulation:
    """Класс для запуска симуляции и сбора данных"""
    def __init__(self, 
                 pid_params: Tuple[float, float, float],
                 target_temp: float,
                 simulation_time: float = 300.0):  # время в секундах
        self.controller = PIDController(*pid_params)
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
            
            # Получаем управляющий сигнал от ПИД-регулятора
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

    def save_data(self, filename: str = "simulation_data.json"):
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

    def plot_results(self):
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
        ax2.plot(times, outputs, 'g-', label='Сигнал управления')
        ax2.set_xlabel('Время (с)')
        ax2.set_ylabel('Мощность охлаждения (0-1)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('simulation_results.png')
        plt.close()

def main():
    # Параметры симуляции
    pid_params = (2.0, 0.05, 1.0)  # Kp, Ki, Kd - увеличили пропорциональную и дифференциальную составляющие
    target_temp = 20.0  # Целевая температура
    simulation_time = 300.0  # Время симуляции (секунды)

    # Создаем и запускаем симуляцию
    simulation = CoolingSimulation(pid_params, target_temp, simulation_time)
    simulation.run()
    
    # Сохраняем результаты
    simulation.save_data()
    simulation.plot_results()

if __name__ == "__main__":
    main()
