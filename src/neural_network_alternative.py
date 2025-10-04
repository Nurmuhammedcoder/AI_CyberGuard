import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import joblib
import sys
import locale
locale.getpreferredencoding = lambda: "UTF-8"
if sys.stdout.encoding != 'UTF-8':
    sys.stdout.reconfigure(encoding='utf-8')
print("Запуск обучения нейросетевой модели...")

# Загрузка данных
def load_data():
    try:
        data = pd.read_csv('data/creditcard.csv')
        print("Данные creditcard.csv загружены")
    except:
        # Создание синтетических данных
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=10000, n_features=30, n_classes=2, 
                                  weights=[0.95, 0.05], random_state=42)
        data = pd.DataFrame(X)
        data['Class'] = y
        data.to_csv('data/creditcard.csv', index=False)
        print("Созданы синтетические данные")
    
    return data

# Загрузка и подготовка данных
data = load_data()
X = data.drop('Class', axis=1)
y = data['Class']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Масштабирование данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание модели
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Обучение модели
print("Обучение модели...")
history = model.fit(X_train, y_train, 
                    epochs=30, 
                    batch_size=32, 
                    validation_split=0.2,
                    verbose=1)

# Оценка модели
y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)

print(f"Точность модели: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Сохранение модели
os.makedirs('models', exist_ok=True)
model.save('models/neural_network_model.h5')
joblib.dump(scaler, 'models/scaler_nn.pkl')

print("Модель сохранена в models/neural_network_model.h5")