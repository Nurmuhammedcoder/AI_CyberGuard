#!/usr/bin/env python3
"""
AI CyberGuard - Модуль обучения моделей
Исправленная версия без ошибок
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import json
from datetime import datetime

# Машинное обучение
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
from sklearn.datasets import make_classification
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow импорт
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.metrics import Precision, Recall, AUC
    TENSORFLOW_AVAILABLE = True
    print(f"✓ TensorFlow {tf.__version__} загружен успешно")
except Exception as e:
    print(f"⚠️ TensorFlow недоступен: {e}")
    TENSORFLOW_AVAILABLE = False

# Создание директорий
def create_directories():
    """Создание необходимых директорий"""
    directories = ['models', 'images', 'reports', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Директория {directory} создана")

def create_sample_data(n_samples=10000):
    """Создание синтетических данных для демонстрации"""
    print("Создание синтетических данных...")
    
    # Создаем реалистичные данные сетевого трафика
    np.random.seed(42)
    
    # Базовая классификация: нормальный трафик vs атаки
    X, y = make_classification(
        n_samples=n_samples,
        n_features=30,
        n_informative=20,
        n_redundant=10,
        n_classes=2,
        weights=[0.85, 0.15],  # 85% нормальный, 15% атаки
        random_state=42
    )
    
    # Создаем DataFrame с реалистичными именами признаков
    feature_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate'
    ]
    
    # Создаем DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Добавляем целевую переменную
    attack_types = ['normal', 'dos', 'probe', 'r2l', 'u2r']
    # Создаем типы атак
    attack_types = []
    for label in y:
        if label == 0:
            attack_types.append('normal')
        else:
            attack_types.append(np.random.choice(['dos', 'probe', 'r2l', 'u2r']))
    df['attack_type'] = attack_types
    df['is_attack'] = y
    
    print(f"Создано {n_samples} образцов данных")
    print(f"Нормальный трафик: {sum(y == 0)} ({sum(y == 0)/n_samples*100:.1f}%)")
    print(f"Атаки: {sum(y == 1)} ({sum(y == 1)/n_samples*100:.1f}%)")
    
    return df

def preprocess_data(data):
    """Предобработка данных"""
    print("Предобработка данных...")
    
    # Выбираем числовые признаки
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Исключаем целевые переменные
    if 'is_attack' in numeric_features:
        numeric_features.remove('is_attack')
    
    # Заполняем пропущенные значения
    X = data[numeric_features].fillna(0)
    y = data['is_attack'].values
    
    print(f"Размерность данных: {X.shape}")
    print(f"Количество признаков: {len(numeric_features)}")
    
    return X, y, numeric_features

class ImprovedRandomForest:
    """Улучшенная модель Random Forest"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Обучение модели"""
        print("Обучение Random Forest...")
        
        # Нормализация данных
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Создание модели с оптимальными параметрами
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Обучение
        start_time = time.time()
        self.model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        print(f"Обучение завершено за {training_time:.2f} секунд")
        
        return training_time
    
    def evaluate(self, X_test, y_test):
        """Оценка модели"""
        if self.model is None or self.scaler is None:
            raise ValueError("Модель не обучена!")
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }
        
        return metrics, y_pred, y_pred_proba

class ImprovedNeuralNetwork:
    """Улучшенная нейронная сеть"""
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
        self.scaler = None
        self.history = None
        
        if TENSORFLOW_AVAILABLE:
            self._build_model()
        else:
            raise ImportError("TensorFlow не доступен")
    
    def _build_model(self):
        """Создание архитектуры нейросети"""
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(self.input_shape,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dropout(0.1),
            
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
        )
        
        print("Архитектура нейронной сети создана")
        print(f"Параметров: {self.model.count_params()}")
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=128):
        """Обучение нейронной сети"""
        print("Обучение нейронной сети...")
        
        # Нормализация данных
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Callback'и для улучшения обучения
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Обучение
        start_time = time.time()
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        training_time = time.time() - start_time
        
        print(f"Обучение завершено за {training_time:.2f} секунд")
        
        return training_time
    
    def evaluate(self, X_test, y_test):
        """Оценка модели"""
        if self.model is None or self.scaler is None:
            raise ValueError("Модель не обучена!")
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred_proba = self.model.predict(X_test_scaled, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }
        
        return metrics, y_pred, y_pred_proba
    
    def plot_training_history(self):
        """Визуализация процесса обучения"""
        if self.history is None:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('images/neural_network_training.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig

def compare_models(X_train, X_test, y_train, y_test):
    """Сравнение моделей"""
    print("\nСРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 50)
    
    results = {}
    
    # 1. Random Forest
    print("\n1. Random Forest")
    print("-" * 30)
    rf_model = ImprovedRandomForest()
    rf_time = rf_model.train(X_train, y_train)
    rf_metrics, rf_pred, rf_proba = rf_model.evaluate(X_test, y_test)
    
    results['Random Forest'] = {
        'model': rf_model,
        'metrics': rf_metrics,
        'training_time': rf_time,
        'predictions': rf_pred,
        'probabilities': rf_proba
    }
    
    print(f"Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"Precision: {rf_metrics['precision']:.4f}")
    print(f"Recall: {rf_metrics['recall']:.4f}")
    print(f"F1-Score: {rf_metrics['f1']:.4f}")
    
    # 2. Neural Network (если доступен TensorFlow)
    if TENSORFLOW_AVAILABLE:
        print("\n2. Neural Network")
        print("-" * 30)
        
        try:
            # Разделяем training set на train и validation
            X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            nn_model = ImprovedNeuralNetwork(X_train.shape[1])
            nn_time = nn_model.train(X_train_nn, y_train_nn, X_val_nn, y_val_nn, epochs=50)
            nn_metrics, nn_pred, nn_proba = nn_model.evaluate(X_test, y_test)
            
            results['Neural Network'] = {
                'model': nn_model,
                'metrics': nn_metrics,
                'training_time': nn_time,
                'predictions': nn_pred,
                'probabilities': nn_proba
            }
            
            print(f"Accuracy: {nn_metrics['accuracy']:.4f}")
            print(f"Precision: {nn_metrics['precision']:.4f}")
            print(f"Recall: {nn_metrics['recall']:.4f}")
            print(f"F1-Score: {nn_metrics['f1']:.4f}")
            
            # Визуализация обучения нейросети
            nn_model.plot_training_history()
            
        except Exception as e:
            print(f"Ошибка обучения нейронной сети: {e}")
    
    return results

def create_visualizations(results, y_test):
    """Создание визуализаций сравнения"""
    print("\nСоздание визуализаций...")
    
    # Сравнение метрик
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        values = [results[model]['metrics'][metric] for model in models]
        
        bars = ax.bar(models, values, color=colors[:len(models)])
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(0, 1)
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('images/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Матрицы ошибок
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    for i, (model_name, result) in enumerate(results.items()):
        cm = result['metrics']['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{model_name}\nConfusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('images/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Визуализации сохранены в папку 'images/'")

def save_models_and_results(results):
    """Сохранение моделей и результатов"""
    print("\nСохранение моделей и результатов...")
    
    # Сохранение моделей
    for model_name, result in results.items():
        model = result['model']
        
        if model_name == 'Random Forest':
            joblib.dump(model.model, 'models/random_forest.pkl')
            joblib.dump(model.scaler, 'models/rf_scaler.pkl')
            print(f"✓ Random Forest модель сохранена")
        
        elif model_name == 'Neural Network' and TENSORFLOW_AVAILABLE:
            model.model.save('models/neural_network.h5')
            joblib.dump(model.scaler, 'models/nn_scaler.pkl')
            print(f"✓ Neural Network модель сохранена")
    
    # Сохранение результатов в JSON
    json_results = {}
    for model_name, result in results.items():
        json_results[model_name] = {
            'accuracy': float(result['metrics']['accuracy']),
            'precision': float(result['metrics']['precision']),
            'recall': float(result['metrics']['recall']),
            'f1': float(result['metrics']['f1']),
            'training_time': float(result['training_time'])
        }
    
    with open('reports/model_comparison.json', 'w') as f:
        json.dump(json_results, f, indent=4)
    
    # Создание текстового отчета
    report = "AI CYBERGUARD - ОТЧЕТ О СРАВНЕНИИ МОДЕЛЕЙ\n"
    report += "=" * 50 + "\n"
    report += f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for model_name, result in results.items():
        report += f"{model_name}:\n"
        report += f"  Время обучения: {result['training_time']:.2f} сек\n"
        report += f"  Accuracy: {result['metrics']['accuracy']:.4f}\n"
        report += f"  Precision: {result['metrics']['precision']:.4f}\n"
        report += f"  Recall: {result['metrics']['recall']:.4f}\n"
        report += f"  F1-Score: {result['metrics']['f1']:.4f}\n\n"
        report += "  Отчет классификации:\n"
        report += result['metrics']['classification_report'] + "\n\n"
    
    with open('reports/detailed_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✓ Результаты сохранены в папку 'reports/'")

def main():
    """Главная функция"""
    print("AI CYBERGUARD - СИСТЕМА ОБУЧЕНИЯ МОДЕЛЕЙ")
    print("=" * 60)
    
    # Создание директорий
    create_directories()
    
    try:
        # Создание данных
        print("\n1. СОЗДАНИЕ ДАННЫХ")
        print("-" * 30)
        data = create_sample_data(15000)
        
        # Предобработка
        print("\n2. ПРЕДОБРАБОТКА ДАННЫХ")
        print("-" * 30)
        X, y, feature_names = preprocess_data(data)
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Размер обучающей выборки: {X_train.shape}")
        print(f"Размер тестовой выборки: {X_test.shape}")
        
        # Сравнение моделей
        print("\n3. ОБУЧЕНИЕ И СРАВНЕНИЕ МОДЕЛЕЙ")
        print("-" * 30)
        results = compare_models(X_train, X_test, y_train, y_test)
        
        # Визуализация
        print("\n4. СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
        print("-" * 30)
        create_visualizations(results, y_test)
        
        # Сохранение
        print("\n5. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("-" * 30)
        save_models_and_results(results)
        
        print("\n" + "=" * 60)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("=" * 60)
        print("Результаты:")
        print("📁 models/ - Обученные модели")
        print("📁 images/ - Графики и визуализации") 
        print("📁 reports/ - Отчеты и метрики")
        
        if TENSORFLOW_AVAILABLE:
            print("\n✓ TensorFlow: Полностью функционален")
        else:
            print("\n⚠️ TensorFlow: Недоступен (использован только Random Forest)")
            
    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()