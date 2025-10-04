import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import json
from sklearn.datasets import make_classification
import sys
import locale
locale.getpreferredencoding = lambda: "UTF-8"
if sys.stdout.encoding != 'UTF-8':
    sys.stdout.reconfigure(encoding='utf-8')
# Проверка доступности TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.metrics import Precision, Recall, AUC
    TENSORFLOW_AVAILABLE = True
    print(f"TensorFlow успешно загружен. Версия: {tf.__version__}")
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    print(f"TensorFlow не доступен: {e}")
    print("Будет использоваться только Random Forest")

# Создание необходимых директорий
os.makedirs('../models', exist_ok=True)
os.makedirs('../images', exist_ok=True)
os.makedirs('../reports', exist_ok=True)
os.makedirs('../data', exist_ok=True)

def create_sample_data():
    """Создание синтетических данных для демонстрации"""
    print("Создание синтетических данных для демонстрации...")
    X, y = make_classification(
        n_samples=10000,
        n_features=30,
        n_informative=20,
        n_redundant=10,
        n_clusters_per_class=1,
        weights=[0.99, 0.01],
        flip_y=0.01,
        random_state=42
    )
    
    feature_names = [f'V{i}' for i in range(1, 31)]
    df = pd.DataFrame(X, columns=feature_names)
    df['Class'] = y
    
    return df

def load_data():
    """Загрузка данных"""
    try:
        # Сначала попробуем загрузить реальные данные
        try:
            data = pd.read_csv('../data/creditcard.csv')
            print("Загружены реальные данные creditcard.csv")
        except FileNotFoundError:
            print("Файл creditcard.csv не найден. Создаем синтетические данные...")
            data = create_sample_data()
            data.to_csv('../data/creditcard.csv', index=False)
        
        # Разделение на признаки и целевую переменную
        X = data.drop('Class', axis=1).values
        y = data['Class'].values
        
        # Разделение на тренировочную и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Нормализация данных
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Сохранение scaler
        joblib.dump(scaler, '../models/scaler.pkl')
        
        print(f"Размер обучающей выборки: {X_train.shape}")
        print(f"Размер тестовой выборки: {X_test.shape}")
        print(f"Распределение классов в обучающей выборке: {np.bincount(y_train)}")
        print(f"Распределение классов в тестовой выборке: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return None, None, None, None

def balance_data(X, y):
    """Балансировка классов с помощью взвешивания"""
    try:
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        print(f"Исходное распределение классов: {np.bincount(y)}")
        print(f"Веса классов: {weight_dict}")
        
        return X, y, weight_dict
    except Exception as e:
        print(f"Ошибка балансировки данных: {e}")
        return X, y, {0: 1, 1: 1}

def cross_validate_model(model, X, y, cv=5, model_type='rf'):
    """Кросс-валидация для модели"""
    try:
        if model_type == 'nn' and TENSORFLOW_AVAILABLE:
            # Для нейросетей используем свою реализацию кросс-валидации
            kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []
            
            for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
                print(f"Кросс-валидация нейросети: Фолд {fold+1}/{cv}")
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Разделяем на обучение и валидацию для ранней остановки
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
                )
                
                # Получаем веса классов
                _, _, class_weights = balance_data(X_train, y_train)
                
                # Создаем и обучаем модель
                nn_model = ImprovedNeuralNetwork(X_train.shape[1])
                nn_model.train(X_train, y_train, X_val, y_val, epochs=30, batch_size=256, class_weight=class_weights)
                
                # Оцениваем модель
                metrics, _ = nn_model.evaluate(X_test, y_test)
                
                accuracies.append(metrics['accuracy'])
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                f1_scores.append(metrics['f1'])
            
            return {
                'accuracy': (np.mean(accuracies), np.std(accuracies)),
                'precision': (np.mean(precisions), np.std(precisions)),
                'recall': (np.mean(recalls), np.std(recalls)),
                'f1': (np.mean(f1_scores), np.std(f1_scores))
            }
        else:
            # Для Random Forest используем стандартную кросс-валидацию
            scoring = ['accuracy', 'precision', 'recall', 'f1']
            results = {}
            
            for score in scoring:
                try:
                    cv_scores = cross_val_score(model, X, y, scoring=score, cv=cv, n_jobs=-1)
                    results[score] = (np.mean(cv_scores), np.std(cv_scores))
                except Exception as e:
                    print(f"Ошибка при вычислении {score}: {e}")
                    results[score] = (0.0, 0.0)
            
            return results
    except Exception as e:
        print(f"Ошибка кросс-валидации: {e}")
        return {
            'accuracy': (0.0, 0.0),
            'precision': (0.0, 0.0),
            'recall': (0.0, 0.0),
            'f1': (0.0, 0.0)
        }

def create_comprehensive_report(results, y_test):
    """Создание комплексного отчета сравнения моделей"""
    try:
        report = "ПОЛНОЕ СРАВНЕНИЕ УЛУЧШЕННЫХ МОДЕЛЕЙ\n"
        report += "=" * 60 + "\n\n"
        
        for model_name, result in results.items():
            report += f"{model_name}:\n"
            report += f"  Время обучения: {result['time']:.2f} секунд\n"
            report += f"  Accuracy: {result['metrics']['accuracy']:.4f}\n"
            report += f"  Precision: {result['metrics']['precision']:.4f}\n"
            report += f"  Recall: {result['metrics']['recall']:.4f}\n"
            report += f"  F1-Score: {result['metrics']['f1']:.4f}\n\n"
            
            report += "  Кросс-валидация (5-fold):\n"
            cv = result['cv_scores']
            report += f"    Accuracy: {cv['accuracy'][0]:.4f} (±{cv['accuracy'][1]:.4f})\n"
            report += f"    Precision: {cv['precision'][0]:.4f} (±{cv['precision'][1]:.4f})\n"
            report += f"    Recall: {cv['recall'][0]:.4f} (±{cv['recall'][1]:.4f})\n"
            report += f"    F1-Score: {cv['f1'][0]:.4f} (±{cv['f1'][1]:.4f})\n\n"
            
            report += "  Отчет классификации:\n"
            report += result['metrics']['classification_report'] + "\n"
            
            # Матрица ошибок
            report += "  Матрица ошибок:\n"
            report += f"  {result['metrics']['confusion_matrix']}\n\n"
        
        # Визуализация сравнения
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(results.keys())
        
        # Сравнение точности
        accuracies = [results[model_name]['metrics']['accuracy'] for model_name in models]
        cv_accuracies = [results[model_name]['cv_scores']['accuracy'][0] for model_name in models]
        cv_errors = [results[model_name]['cv_scores']['accuracy'][1] for model_name in models]
        
        x_pos = np.arange(len(models))
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        
        axes[0, 0].bar(x_pos, accuracies, color=colors[:len(models)])
        axes[0, 0].set_title('Сравнение точности моделей')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        
        # Добавление значений на столбцы
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        # Сравнение F1-Score
        f1_scores = [results[model_name]['metrics']['f1'] for model_name in models]
        
        axes[0, 1].bar(x_pos, f1_scores, color=colors[:len(models)])
        axes[0, 1].set_title('Сравнение F1-Score моделей')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
        
        # Добавление значений на столбцы
        for i, v in enumerate(f1_scores):
            axes[0, 1].text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        # Сравнение времени обучения
        times = [results[model_name]['time'] for model_name in models]
        axes[1, 0].bar(x_pos, times, color=colors[:len(models)])
        axes[1, 0].set_title('Сравнение времени обучения (секунды)')
        axes[1, 0].set_ylabel('Время (сек)')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
        
        # Добавление значений на столбцы
        for i, v in enumerate(times):
            axes[1, 0].text(i, v + 0.1, f'{v:.2f}', ha='center')
        
        # Сравнение кросс-валидации
        axes[1, 1].bar(x_pos, cv_accuracies, yerr=cv_errors, 
                      color=colors[:len(models)], capsize=10)
        axes[1, 1].set_title('Точность при кросс-валидации (5-fold)')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
        
        # Добавление значений на столбцы
        for i, v in enumerate(cv_accuracies):
            axes[1, 1].text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('../images/improved_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Сохранение отчета
        with open('../reports/improved_model_comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Сохранение результатов в JSON
        json_results = {}
        for model_name, result in results.items():
            json_results[model_name] = {
                'metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else str(v) 
                           for k, v in result['metrics'].items() if k != 'confusion_matrix'},
                'time': float(result['time']),
                'cv_scores': {k: [float(v[0]), float(v[1])] for k, v in result['cv_scores'].items()}
            }
        
        with open('../reports/improved_model_comparison.json', 'w') as f:
            json.dump(json_results, f, indent=4)
        
        return report, fig
    except Exception as e:
        print(f"Ошибка создания отчета: {e}")
        return f"Ошибка создания отчета: {e}", None

# Класс ImprovedNeuralNetwork (только если TensorFlow доступен)
if TENSORFLOW_AVAILABLE:
    class ImprovedNeuralNetwork:
        def __init__(self, input_shape):
            self.input_shape = input_shape
            self.model = self._build_model()
            self.history = None
            
        def _build_model(self):
            """Создание улучшенной нейросетевой архитектуры"""
            try:
                model = Sequential([
                    Dense(256, activation='relu', input_shape=(self.input_shape,)),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(128, activation='relu'),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(64, activation='relu'),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(32, activation='relu'),
                    Dropout(0.2),
                    Dense(1, activation='sigmoid')
                ])
                
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
                )
                
                return model
            except Exception as e:
                print(f"Ошибка создания модели: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=256, class_weight=None):
            """Обучение модели с callback'ами"""
            try:
                callbacks = []
                callbacks.append(EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1))
                callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1))
                
                validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
                
                self.history = self.model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=validation_data,
                    callbacks=callbacks,
                    verbose=1,
                    class_weight=class_weight
                )
                
                return self.history
            except Exception as e:
                print(f"Ошибка обучения нейросети: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        def evaluate(self, X_test, y_test):
            """Оценка модели"""
            try:
                y_pred_proba = self.model.predict(X_test, verbose=0)
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0),
                    'confusion_matrix': str(confusion_matrix(y_test, y_pred)),
                    'classification_report': classification_report(y_test, y_pred, zero_division=0)
                }
                
                return metrics, y_pred
            except Exception as e:
                print(f"Ошибка оценки модели: {e}")
                return {}, []
        
        def plot_training_history(self):
            """Визуализация процесса обучения"""
            if self.history is None:
                print("История обучения отсутствует. Сначала обучите модель.")
                return None
            
            try:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Loss
                axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
                if 'val_loss' in self.history.history:
                    axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
                axes[0, 0].set_title('Model Loss')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].legend()
                
                # Accuracy
                axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy')
                if 'val_accuracy' in self.history.history:
                    axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
                axes[0, 1].set_title('Model Accuracy')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].legend()
                
                # Precision
                if 'precision' in self.history.history:
                    axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
                if 'val_precision' in self.history.history:
                    axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
                axes[1, 0].set_title('Model Precision')
                axes[1, 0].set_ylabel('Precision')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].legend()
                
                # Recall
                if 'recall' in self.history.history:
                    axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
                if 'val_recall' in self.history.history:
                    axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
                axes[1, 1].set_title('Model Recall')
                axes[1, 1].set_ylabel('Recall')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].legend()
                
                plt.tight_layout()
                plt.savefig('../images/improved_nn_training.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                return fig
            except Exception as e:
                print(f"Ошибка визуализации истории обучения: {e}")
                return None

def compare_models_improved(X_train, y_train, X_test, y_test):
    """Сравнение улучшенных моделей"""
    results = {}
    
    try:
        # Получаем веса классов для балансировки
        _, _, class_weights = balance_data(X_train, y_train)
        
        # 1. Улучшенная нейросетевая модель (если TensorFlow доступен)
        if TENSORFLOW_AVAILABLE:
            try:
                print("Обучение улучшенной нейросетевой модели...")
                nn_model = ImprovedNeuralNetwork(X_train.shape[1])
                
                if nn_model.model is None:
                    print("Не удалось создать нейросетевую модель")
                else:
                    # Разделяем на обучение и валидацию для ранней остановки
                    X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
                        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
                    )
                    
                    start_time = time.time()
                    nn_model.train(X_train_nn, y_train_nn, X_val_nn, y_val_nn, epochs=50, batch_size=256, class_weight=class_weights)
                    nn_time = time.time() - start_time
                    
                    nn_metrics, y_pred_nn = nn_model.evaluate(X_test, y_test)
                    results['Improved Neural Network'] = {
                        'metrics': nn_metrics,
                        'time': nn_time,
                        'predictions': y_pred_nn,
                        'cv_scores': cross_validate_model(None, X_train, y_train, model_type='nn')
                    }
                    
                    # Визуализация обучения
                    nn_model.plot_training_history()
                    
                    # Сохранение нейросетевой модели
                    try:
                        nn_model.model.save('../models/improved_neural_network.h5')
                        print("Улучшенная нейросетевая модель сохранена.")
                    except Exception as e:
                        print(f"Ошибка сохранения нейросетевой модели: {e}")
            except Exception as e:
                print(f"Ошибка обучения нейросети: {e}")
        
        # 2. Random Forest с улучшенными параметрами
        try:
            print("Обучение улучшенного Random Forest...")
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # Балансировка классов
            )
            
            start_time = time.time()
            rf_model.fit(X_train, y_train)
            rf_time = time.time() - start_time
            
            y_pred_rf = rf_model.predict(X_test)
            rf_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_rf),
                'precision': precision_score(y_test, y_pred_rf, zero_division=0),
                'recall': recall_score(y_test, y_pred_rf, zero_division=0),
                'f1': f1_score(y_test, y_pred_rf, zero_division=0),
                'confusion_matrix': str(confusion_matrix(y_test, y_pred_rf)),
                'classification_report': classification_report(y_test, y_pred_rf, zero_division=0)
            }
            
            results['Improved Random Forest'] = {
                'metrics': rf_metrics,
                'time': rf_time,
                'predictions': y_pred_rf,
                'cv_scores': cross_validate_model(rf_model, X_train, y_train, model_type='rf')
            }
            
            # Сохранение Random Forest модели
            joblib.dump(rf_model, '../models/improved_random_forest.pkl')
            print("Улучшенная Random Forest модель сохранена.")
        except Exception as e:
            print(f"Ошибка обучения Random Forest: {e}")
        
        return results
    except Exception as e:
        print(f"Ошибка сравнения моделей: {e}")
        return {}

def print_system_info():
    """Вывод информации о системе и установленных пакетах"""
    print("СИСТЕМНАЯ ИНФОРМАЦИЯ")
    print("=" * 50)
    print(f"Python версия: {sys.version}")
    print(f"Операционная системы: {os.name}")
    
    # Проверяем версии основных библиотеки
    libraries_to_check = [
        'tensorflow', 'numpy', 'pandas', 'sklearn', 
        'matplotlib', 'seaborn', 'joblib'
    ]
    
    print("\nВерсии установленных библиотек:")
    for lib in libraries_to_check:
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'неизвестно')
            print(f"  {lib}: {version}")
        except ImportError:
            print(f"  {lib}: не установлено")
    
    print()

def main():
    """Основная функция"""
    try:
        print("ЗАПУСК СИСТЕМЫ СРАВНЕНИЯ МОДЕЛЕЙ")
        print("=" * 60)
        
        # Выводим системную информацию
        print_system_info()
        
        # Загрузка данных
        print("\nЗАГРУЗКА ДАННЫХ")
        print("-" * 30)
        X_train, X_test, y_train, y_test = load_data()
        
        if X_train is None:
            print("Не удалось загрузить данные. Завершение работы.")
            return
        
        # Сравнение улучшенных моделей
        print("\nСРАВНЕНИЕ МОДЕЛЕЙ")
        print("-" * 30)
        results = compare_models_improved(X_train, y_train, X_test, y_test)
        
        if results:
            # Создание отчета
            print("\nСОЗДАНИЕ ОТЧЕТА")
            print("-" * 30)
            report, fig = create_comprehensive_report(results, y_test)
            print(report)
            
            print("\n" + "=" * 60)
            print("СРАВНЕНИЕ УЛУЧШЕННЫх МОДЕЛЕЙ ЗАВЕРШЕНО!")
            print("=" * 60)
            print("Результаты сохранены в следующих папках:")
            print("  📁 models/ - Обученные модели")
            print("  📁 images/ - Графики и визуализации")
            print("  📁 reports/ - Текстовые отчеты и JSON")
            
            if TENSORFLOW_AVAILABLE:
                print("\n✓ TensorFlow/Keras: Полностью функциональны")
            else:
                print("\n⚠️  TensorFlow/Keras: Недоступны (используется только Random Forest)")
                
        else:
            print("Не удалось получить результаты сравнения моделей.")
            print("Проверьте логи выше для диагностики проблем.")
            
    except Exception as e:
        print(f"Критическая ошибка в основной функции: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()