# Імпортуємо необхідні бібліотеки
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Частина 1: Завантаження даних та первинний аналіз

# Завантажуємо історичні котирування акцій
ticker = 'AAPL'  # обрана компанія (наприклад, Apple)
data = yf.download(ticker, start="2023-11-01", end="2024-11-01")

# Перевірка на пропущені значення
print("Перевірка на пропущені значення:")
print(data.isnull().sum())

# Побудова графіка зміни ціни закриття
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Ціна закриття')
plt.title('Зміна ціни закриття акцій')
plt.xlabel('Дата')
plt.ylabel('Ціна ($)')
plt.legend()
plt.grid(True)
plt.show()

# Описова статистика
print("\nОписова статистика:")
print(data.describe())

# Частина 2: Аналіз компонентів часового ряду

# Декомпозиція часового ряду
from statsmodels.tsa.seasonal import seasonal_decompose

# Декомпозиція за допомогою ковзного середнього
decomposition = seasonal_decompose(data['Close'], model='additive', period=252)  # період 252 торгових дні на рік

# Графіки компонентів
decomposition.plot()
plt.show()

# Частина 3: Технічний аналіз

# Просте ковзне середнє (7 та 30 днів)
data['SMA7'] = data['Close'].rolling(window=7).mean()
data['SMA30'] = data['Close'].rolling(window=30).mean()

# Відносна сила (RSI)
def compute_rsi(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = compute_rsi(data['Close'], window=14)

# Волатильність (30-денна)
data['Volatility'] = data['Close'].rolling(window=30).std()

# Графіки технічних індикаторів
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Ціна закриття', color='blue')
plt.plot(data['SMA7'], label='7-денне SMA', color='red')
plt.plot(data['SMA30'], label='30-денне SMA', color='green')
plt.title('Прості ковзні середні')
plt.xlabel('Дата')
plt.ylabel('Ціна ($)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data['RSI'], label='RSI', color='purple')
plt.axhline(y=70, color='red', linestyle='--', label='Overbought')
plt.axhline(y=30, color='green', linestyle='--', label='Oversold')
plt.title('Індикатор відносної сили (RSI)')
plt.xlabel('Дата')
plt.ylabel('RSI')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data['Volatility'], label='30-денна волатильність', color='orange')
plt.title('30-денна волатильність')
plt.xlabel('Дата')
plt.ylabel('Волатильність')
plt.legend()
plt.grid(True)
plt.show()

# Частина 4: Прогнозування

# Розділяємо дані на навчальну та тестову вибірки
train_data, test_data = train_test_split(data['Close'], test_size=0.2, shuffle=False)

# Просте експоненційне згладжування
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model_exp_smooth = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=252)
exp_smooth_fit = model_exp_smooth.fit()
exp_smooth_forecast = exp_smooth_fit.forecast(len(test_data))

# ARIMA модель
from statsmodels.tsa.arima.model import ARIMA

model_arima = ARIMA(train_data, order=(5, 1, 0))
arima_fit = model_arima.fit()
arima_forecast = arima_fit.forecast(len(test_data))

# Оцінка якості прогнозу
exp_smooth_mse = mean_squared_error(test_data, exp_smooth_forecast)
exp_smooth_mae = mean_absolute_error(test_data, exp_smooth_forecast)

arima_mse = mean_squared_error(test_data, arima_forecast)
arima_mae = mean_absolute_error(test_data, arima_forecast)

print("\nОцінка моделей:")
print(f"Експоненційне згладжування - MSE: {exp_smooth_mse}, MAE: {exp_smooth_mae}")
print(f"ARIMA модель - MSE: {arima_mse}, MAE: {arima_mae}")

# Візуалізація результатів прогнозування
plt.figure(figsize=(10, 6))
plt.plot(train_data, label='Навчальна вибірка', color='blue')
plt.plot(test_data.index, test_data, label='Тестова вибірка', color='green')
plt.plot(test_data.index, exp_smooth_forecast, label='Прогноз (Exp. Smoothing)', color='red', linestyle='--')
plt.plot(test_data.index, arima_forecast, label='Прогноз (ARIMA)', color='orange', linestyle='--')
plt.title('Прогнозування цін закриття акцій')
plt.xlabel('Дата')
plt.ylabel('Ціна ($)')
plt.legend()
plt.grid(True)
plt.show()
