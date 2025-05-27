from threading import Timer
import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class PIDSimulator:
    def __init__(self, master):
        self.master = master
        master.title("PID Temperature Control Simulator")

        # Параметры системы
        self.running = True
        self.show_disturbance = True
        self.sim_time = 5  # Длительность отображения на графике

        # Инициализация состояния
        self.reset_state()

        # Создание GUI элементов
        self.create_widgets()

        # Запуск симуляции
        self.update_plot()

    def reset_state(self):
        self.y_current = 25.0
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.time_buffer = np.array([])
        self.temp_buffer = np.array([])
        self.disturbance_buffer = np.array([])
        self.start_time = np.datetime64("now")

    def create_widgets(self):
        # Создание фрейма для управления
        control_frame = ttk.Frame(self.master, padding=10)
        control_frame.grid(row=0, column=0, sticky="nsew")

        # Слайдеры
        self.kp_var = tk.DoubleVar(value=15.0)
        self.ki_var = tk.DoubleVar(value=2.0)
        self.kd_var = tk.DoubleVar(value=3.0)
        self.sp_var = tk.DoubleVar(value=40.0)

        ttk.Label(control_frame, text="PID Parameters").grid(
            row=0, column=0, columnspan=2
        )
        self.create_slider(control_frame, "Kp:", self.kp_var, 1, 100, 1)
        self.create_slider(control_frame, "Ki:", self.ki_var, 0.1, 10, 2)
        self.create_slider(control_frame, "Kd:", self.kd_var, 0.1, 10, 3)
        self.create_slider(control_frame, "Setpoint:", self.sp_var, 0, 100, 4)

        # Кнопки
        self.toggle_btn = ttk.Button(
            control_frame, text="Hide Disturbance", command=self.toggle_disturbance
        )
        self.toggle_btn.grid(row=5, column=0, pady=5)

        self.stop_btn = ttk.Button(
            control_frame, text="Stop Simulation", command=self.stop_simulation
        )
        self.stop_btn.grid(row=5, column=1, pady=5)

        # График
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        (self.line_temp,) = self.ax.plot([], [], label="Temperature (°C)")
        self.line_sp = self.ax.axhline(self.sp_var.get(), color="r", linestyle="--")
        (self.line_dist,) = self.ax.plot([], [], "g--", alpha=0.5, label="Disturbance")

        self.ax.set_ylim(-10, 60)
        self.ax.set_xlim(0, self.sim_time)
        self.ax.grid(True)
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")

        # Настройка размеров
        self.master.columnconfigure(1, weight=1)
        self.master.rowconfigure(0, weight=1)

    def create_slider(self, frame, label, var, min_val, max_val, row):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
        scale = ttk.Scale(
            frame,
            from_=min_val,
            to=max_val,
            variable=var,
            orient="horizontal",
            command=lambda e: var.set(float(scale.get())),
        )
        scale.grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(frame, textvariable=var).grid(row=row, column=2, padx=5)

    def toggle_disturbance(self):
        self.show_disturbance = not self.show_disturbance
        self.line_dist.set_visible(self.show_disturbance)
        self.toggle_btn.config(
            text="Show Disturbance" if not self.show_disturbance else "Hide Disturbance"
        )
        self.canvas.draw_idle()

    def stop_simulation(self):
        self.running = False
        self.master.destroy()

    def update_system(self):
        if not self.running:
            return

        current_time = (np.datetime64("now") - self.start_time).astype(float) / 1e9
        dt = 0.1

        # Расчет возмущения
        disturbance = 20 * np.sin(0.9 * current_time) + np.random.normal(1, 10)

        # PID расчет
        error = self.sp_var.get() - self.y_current
        self.integral_error += error * dt
        derivative_error = (error - self.previous_error) / dt

        u = (
            self.kp_var.get() * error
            + self.ki_var.get() * self.integral_error
            + self.kd_var.get() * derivative_error
        )

        # Модель системы
        self.y_current += (-self.y_current + u + disturbance) / 5.0 * dt
        self.previous_error = error

        # Обновление данных
        self.time_buffer = np.append(self.time_buffer, current_time)
        self.temp_buffer = np.append(self.temp_buffer, self.y_current)
        self.disturbance_buffer = np.append(self.disturbance_buffer, disturbance)

        # Обрезка старых данных
        keep_mask = self.time_buffer > (current_time - self.sim_time)
        self.time_buffer = self.time_buffer[keep_mask]
        self.temp_buffer = self.temp_buffer[keep_mask]
        self.disturbance_buffer = self.disturbance_buffer[keep_mask]

    def update_plot(self):
        if not self.running:
            return

        self.update_system()

        # Обновление линий
        self.line_temp.set_data(
            self.time_buffer - self.time_buffer[0], self.temp_buffer
        )
        self.line_sp.set_ydata([self.sp_var.get()])
        self.line_dist.set_data(
            self.time_buffer - self.time_buffer[0], self.disturbance_buffer
        )

        # Обновление границ
        # Вычисляем текущий временной диапазон
        if len(self.time_buffer) > 0:
            total_time = self.time_buffer[-1] - self.time_buffer[0]
            time_start = max(0, total_time - self.sim_time)
            time_end = total_time
        else:
            time_start = 0
            time_end = self.sim_time

        # Добавляем защиту от одинаковых значений и padding
        if time_start == time_end:
            padding = 0.1 * self.sim_time if self.sim_time > 0 else 1.0
            time_start = max(0, time_end - padding)

        self.ax.set_xlim(time_start, time_end)

        self.canvas.draw()

        # Планирование следующего обновления
        self.master.after(10, self.update_plot)


if __name__ == "__main__":
    root = tk.Tk()
    app = PIDSimulator(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_simulation)
    root.mainloop()
