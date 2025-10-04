import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def preprocess_data():
    # Путь к файлу с атаками
    data_dir = "D:/Datasets/CICIDS2017/MachineLearningCVE"
    file_name = "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    file_path = os.path.join(data_dir, file_name)
    
    # Загрузка данных
    print(f"Загрузка данных из {file_path}")
    
    # Читаем файл полностью
    df = pd.read_csv(file_path)
    print(f"Загружено {df.shape[0]} записей")
    
    # Проверяем метки
    print("Распределение меток до обработки:")
    print(df[' Label'].value_counts())
    
    # Кодируем метки: BENIGN -> 0, все остальное -> 1 (атака)
    df['is_attack'] = df[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    print("Распределение меток после обработки:")
    print(df['is_attack'].value_counts())
    
    # Выделяем признаки и целевую переменную
    X = df.drop([' Label', 'is_attack'], axis=1)
    y = df['is_attack']
    
    # Заполняем пропущенные значения
    X = X.fillna(X.median())
    
    # Обрабатываем бесконечные значения - заменяем их на максимальные/минимальные конечные значения
    for col in X.select_dtypes(include=[np.number]).columns:
        # Заменяем бесконечности на NaN
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        # Заполняем NaN медианными значениями
        X[col] = X[col].fillna(X[col].median())
    
    # Нормализуем признаки
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Разделяем на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Сохраняем обработанные данные
    output_dir = "D:/Datasets/CICIDS2017/Processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем масштабатор
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    
    # Сохраняем данные
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    print(f"Обработанные данные сохранены в {output_dir}")
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data()