import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Завантажуємо датасет
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target  # Додаємо цільову змінну (ціна будинку)

# Частина 1: Дослідницький аналіз даних (EDA)
print("Описова статистика:")
print(df.describe())

print("\nПеревірка на пропущені значення:")
print(df.isnull().sum())

print("\nТипи даних:")
print(df.dtypes)

# Візуалізація: гістограми
df.hist(bins=50, figsize=(20,15))
plt.show()

# Boxplot для виявлення викидів
plt.figure(figsize=(10,6))
sns.boxplot(data=df)
plt.show()

# Кореляційна матриця
correlation_matrix = df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# Scatter plots для ціни
sns.pairplot(df, x_vars=df.columns[:-1], y_vars=['Target'])
plt.show()

# Частина 2: Підготовка даних
# Розподіляємо на тренувальні та тестові дані
X = df.drop(columns=['Target'])
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабування даних
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Зберігаємо скейлер для майбутнього використання
joblib.dump(scaler, 'scaler.pkl')

# Частина 3: Побудова моделей
# 1. Простий лінійний регресор
X_train_simple = X_train_scaled[:, 0].reshape(-1, 1)  # MedInc
X_test_simple = X_test_scaled[:, 0].reshape(-1, 1)

model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train)

y_pred_simple = model_simple.predict(X_test_simple)

mse_simple = mean_squared_error(y_test, y_pred_simple)
rmse_simple = np.sqrt(mse_simple)
r2_simple = r2_score(y_test, y_pred_simple)

print(f'\nSimple Model - MSE: {mse_simple}, RMSE: {rmse_simple}, R2: {r2_simple}')

# 2. Множинна лінійна регресія
model_multiple = LinearRegression()
model_multiple.fit(X_train_scaled, y_train)

y_pred_multiple = model_multiple.predict(X_test_scaled)

mse_multiple = mean_squared_error(y_test, y_pred_multiple)
rmse_multiple = np.sqrt(mse_multiple)
r2_multiple = r2_score(y_test, y_pred_multiple)

print(f'\nMultiple Model - MSE: {mse_multiple}, RMSE: {rmse_multiple}, R2: {r2_multiple}')

# Аналіз коефіцієнтів моделі
coefficients = pd.DataFrame(model_multiple.coef_, X.columns, columns=['Coefficient'])
print("\nКоефіцієнти моделі:")
print(coefficients)

# 3. Оптимізована модель (Lasso)
model_lasso = Lasso(alpha=0.1)
model_lasso.fit(X_train_scaled, y_train)

y_pred_lasso = model_lasso.predict(X_test_scaled)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f'\nLasso Model - MSE: {mse_lasso}, RMSE: {rmse_lasso}, R2: {r2_lasso}')

# Частина 4: Оцінка моделей
models = ['Simple', 'Multiple', 'Lasso']
mse_values = [mse_simple, mse_multiple, mse_lasso]
rmse_values = [rmse_simple, rmse_multiple, rmse_lasso]
r2_values = [r2_simple, r2_multiple, r2_lasso]

# Візуалізація результатів
plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.bar(models, mse_values, color='b', alpha=0.7, label='MSE')
plt.title('MSE Comparison')

plt.subplot(1, 2, 2)
plt.bar(models, r2_values, color='g', alpha=0.7, label='R2')
plt.title('R2 Comparison')

plt.show()

# Частина 5: Інтерпретація результатів
# Функція для прогнозування
def predict_price(features):
    scaler = joblib.load('scaler.pkl')
    
    features_scaled = scaler.transform([features])
    
    price = model_lasso.predict(features_scaled)
    
    return price[0]

# Приклад прогнозування
example_features = [6.5, 35, 6, 2, 1200, 3, 37.7, -122.5]  # Пример для MedInc, HouseAge, тощо
predicted_price = predict_price(example_features)
print(f'\nПрогнозована ціна: {predicted_price}')
