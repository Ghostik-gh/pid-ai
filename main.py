import tensorflow as tf
import numpy as np
from skfuzzy import control as ctrl
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

# Создание нечеткого контроллера
temp_error = ctrl.Antecedent(np.arange(-5, 5, 0.1), "error")
temp_rate = ctrl.Antecedent(np.arange(-2, 2, 0.1), "rate")
cooling_power = ctrl.Consequent(np.arange(0, 100, 0.1), "power")

# Определение функций принадлежности для ошибки
temp_error.automf(3, names=['negative', 'zero', 'positive'])

# Определение функций принадлежности для скорости изменения
temp_rate.automf(3, names=['negative', 'zero', 'positive'])

# Определение функций принадлежности для мощности охлаждения
cooling_power.automf(3, names=['low', 'medium', 'high'])

# Правила нечеткой логики
rules = [
    ctrl.Rule(temp_error['negative'] & temp_rate['negative'], cooling_power['low']),
    ctrl.Rule(temp_error['negative'] & temp_rate['zero'], cooling_power['low']),
    ctrl.Rule(temp_error['negative'] & temp_rate['positive'], cooling_power['medium']),
    
    ctrl.Rule(temp_error['zero'] & temp_rate['negative'], cooling_power['low']),
    ctrl.Rule(temp_error['zero'] & temp_rate['zero'], cooling_power['medium']),
    ctrl.Rule(temp_error['zero'] & temp_rate['positive'], cooling_power['medium']),
    
    ctrl.Rule(temp_error['positive'] & temp_rate['negative'], cooling_power['medium']),
    ctrl.Rule(temp_error['positive'] & temp_rate['zero'], cooling_power['high']),
    ctrl.Rule(temp_error['positive'] & temp_rate['positive'], cooling_power['high'])
]

# Создание и симуляция системы управления
cooling_ctrl = ctrl.ControlSystem(rules)
cooling_simulation = ctrl.ControlSystemSimulation(cooling_ctrl)

def fuzzy_controller(current_temp, target_temp, prev_temp):
    """Функция нечеткого управления"""
    error = current_temp - target_temp
    rate = current_temp - prev_temp
    cooling_simulation.input['error'] = error
    cooling_simulation.input['rate'] = rate
    cooling_simulation.compute()
    return cooling_simulation.output['power']

def simulate_temperature(current_temp, cooling_power):
    """Симуляция изменения температуры"""
    cooling_effect = cooling_power * 0.01
    natural_change = np.random.normal(0, 0.1)
    return current_temp - cooling_effect + natural_change

def create_sequence(temp_history, window_size=10):
    """Создание последовательности для LSTM"""
    if len(temp_history) < window_size:
        padding = [temp_history[0]] * (window_size - len(temp_history))
        temp_history = padding + temp_history
    return np.array([temp_history[-window_size:]])

# Параметры симуляции
time_steps = 200  # Увеличиваем количество шагов для лучшего обучения
target_temp = 25.0
window_size = 10

# Сбор данных с нечеткого контроллера
current_temp = 30.0
temps_fuzzy = [current_temp]
powers_fuzzy = []
errors_fuzzy = []
rates_fuzzy = []

# Симуляция работы нечеткого контроллера
for _ in range(time_steps):
    error = current_temp - target_temp
    rate = current_temp - temps_fuzzy[-1] if len(temps_fuzzy) > 1 else 0
    
    # Получаем управляющее воздействие
    power = fuzzy_controller(current_temp, target_temp, temps_fuzzy[-1])
    
    # Сохраняем результаты
    powers_fuzzy.append(power)
    errors_fuzzy.append(error)
    rates_fuzzy.append(rate)
    
    # Симулируем изменение температуры
    current_temp = simulate_temperature(current_temp, power)
    temps_fuzzy.append(current_temp)

# Подготовка данных для обучения LSTM
X_train = []
y_train = []

for i in range(len(temps_fuzzy) - window_size):
    X_train.append(temps_fuzzy[i:i+window_size])
    y_train.append(temps_fuzzy[i+window_size])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = X_train.reshape((X_train.shape[0], window_size, 1))

# Создание и обучение LSTM модели
inputs = tf.keras.Input(shape=(window_size, 1))
x = tf.keras.layers.LSTM(64)(inputs)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)

# Тестирование гибридной системы
current_temp = 30.0
temps_hybrid = [current_temp]
powers_hybrid = []
predictions_lstm = []
errors_hybrid = []
rates_hybrid = []

# Симуляция гибридной системы
for t in range(time_steps):
    error = current_temp - target_temp
    rate = current_temp - temps_hybrid[-1] if len(temps_hybrid) > 1 else 0
    
    # Получаем предсказание LSTM
    if len(temps_hybrid) >= window_size:
        sequence = create_sequence(temps_hybrid, window_size)
        lstm_pred = model.predict(sequence, verbose=0)[0][0]
    else:
        lstm_pred = current_temp
    
    # Получаем управляющее воздействие от нечеткого контроллера
    power = fuzzy_controller(current_temp, target_temp, temps_hybrid[-1])
    
    # Сохраняем результаты
    powers_hybrid.append(power)
    predictions_lstm.append(lstm_pred)
    errors_hybrid.append(error)
    rates_hybrid.append(rate)
    
    # Симулируем изменение температуры
    current_temp = simulate_temperature(current_temp, power)
    temps_hybrid.append(current_temp)

# Визуализация результатов
plt.rcParams['figure.figsize'] = [15, 12]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '--'

# Создаем фигуру с графиками
fig = plt.figure()
gs = GridSpec(4, 2, figure=fig)

# График температуры для нечеткого контроллера
ax1 = fig.add_subplot(gs[0, :])
time_points = np.arange(time_steps)
ax1.plot(time_points, temps_fuzzy[:-1], 'b-', label='Температура (нечеткий)')
ax1.plot(time_points, [target_temp] * time_steps, 'r--', label='Целевая температура')
ax1.set_title('Температура при нечетком управлении')
ax1.set_xlabel('Время')
ax1.set_ylabel('Температура')
ax1.legend()
ax1.grid(True)

# График температуры для гибридной системы
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(time_points, temps_hybrid[:-1], 'b-', label='Температура (гибридный)')
ax2.plot(time_points, [target_temp] * time_steps, 'r--', label='Целевая температура')
ax2.plot(time_points, predictions_lstm, 'g-', label='Предсказания LSTM')
ax2.set_title('Температура при гибридном управлении')
ax2.set_xlabel('Время')
ax2.set_ylabel('Температура')
ax2.legend()
ax2.grid(True)

# График мощности охлаждения
ax3 = fig.add_subplot(gs[2, :])
ax3.plot(time_points, powers_fuzzy, 'b-', label='Нечеткий контроллер')
ax3.plot(time_points, powers_hybrid, 'g-', label='Гибридная система')
ax3.set_title('Мощность охлаждения')
ax3.set_xlabel('Время')
ax3.set_ylabel('Мощность')
ax3.legend()
ax3.grid(True)

# График ошибки
ax4 = fig.add_subplot(gs[3, 0])
ax4.plot(time_points, errors_fuzzy, 'b-', label='Нечеткий')
ax4.plot(time_points, errors_hybrid, 'g-', label='Гибридный')
ax4.set_title('Ошибка регулирования')
ax4.set_xlabel('Время')
ax4.set_ylabel('Ошибка')
ax4.legend()
ax4.grid(True)

# График скорости изменения
ax5 = fig.add_subplot(gs[3, 1])
ax5.plot(time_points, rates_fuzzy, 'b-', label='Нечеткий')
ax5.plot(time_points, rates_hybrid, 'g-', label='Гибридный')
ax5.set_title('Скорость изменения температуры')
ax5.set_xlabel('Время')
ax5.set_ylabel('Скорость')
ax5.legend()
ax5.grid(True)

plt.tight_layout()
plt.show()

# Вывод статистики
print("\nСтатистика работы систем:")
print("Нечеткий контроллер:")
print(f"Средняя ошибка: {np.mean(np.abs(errors_fuzzy)):.2f}")
print(f"Максимальная ошибка: {np.max(np.abs(errors_fuzzy)):.2f}")
print(f"Средняя мощность охлаждения: {np.mean(powers_fuzzy):.2f}")

print("\nГибридная система:")
print(f"Средняя ошибка: {np.mean(np.abs(errors_hybrid)):.2f}")
print(f"Максимальная ошибка: {np.max(np.abs(errors_hybrid)):.2f}")
print(f"Средняя мощность охлаждения: {np.mean(powers_hybrid):.2f}")
print(f"Средняя ошибка предсказания LSTM: {np.mean(np.abs(np.array(predictions_lstm) - np.array(temps_hybrid[1:-1]))):.2f}")
