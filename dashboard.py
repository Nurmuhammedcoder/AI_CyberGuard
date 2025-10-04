import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="AI CyberGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Импорт scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification

# TensorFlow импорт с обработкой ошибок
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
    print(f"✓ TensorFlow {tf.__version__} загружен")
except Exception as e:
    print(f"⚠️ TensorFlow недоступен: {e}")
    TENSORFLOW_AVAILABLE = False

# Создание директорий
def create_directories():
    """Создание необходимых директорий"""
    directories = ['models', 'data', 'reports', 'images']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

create_directories()

# Класс для работы с моделями
class CyberGuardModel:
    def __init__(self):
        self.rf_model = None
        self.nn_model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.is_trained = False
        
    def create_sample_data(self, n_samples=10000):
        """Создание синтетических данных для обучения"""
        st.info("Создание синтетических данных для демонстрации...")
        
        # Создаем данные похожие на сетевой трафик
        np.random.seed(42)
        
        # Базовые признаки
        X, y_binary = make_classification(
            n_samples=n_samples,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            weights=[0.8, 0.2],  # 80% нормальный трафик, 20% атаки
            random_state=42
        )
        
        # Создаем более реалистичные признаки
        data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        
        # Добавляем специфичные для кибербезопасности признаки
        data['packet_size'] = np.random.normal(500, 200, n_samples)
        data['duration'] = np.random.exponential(2, n_samples)
        data['src_port'] = np.random.randint(1024, 65536, n_samples)
        data['dst_port'] = np.random.choice([80, 443, 22, 23, 53, 25], n_samples)
        data['protocol'] = np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples)
        
        # Создаем типы атак
        attack_types = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
        attack_count = sum(y_binary)
        if attack_count > 0:
            y_multi = np.zeros_like(y_binary)
            attack_indices = np.where(y_binary == 1)[0]
            y_multi[attack_indices] = np.random.choice([1, 2, 3, 4], attack_count)
        else:
            y_multi = np.zeros_like(y_binary)
        data['is_attack'] = y_binary
        
        return data
    
    def preprocess_data(self, data):
        """Предобработка данных"""
        # Кодирование категориальных признаков
        if 'protocol' in data.columns:
            le_protocol = LabelEncoder()
            data['protocol_encoded'] = le_protocol.fit_transform(data['protocol'])
        
        # Выделяем числовые признаки
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'is_attack' in numeric_features:
            numeric_features.remove('is_attack')
        
        X = data[numeric_features].fillna(0)
        y = data['is_attack'] if 'is_attack' in data.columns else np.zeros(len(data))
        
        self.feature_names = numeric_features
        
        return X, y
    
    def train_models(self, data):
        """Обучение моделей"""
        X, y = self.preprocess_data(data)
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Нормализация
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # 1. Random Forest
        st.info("Обучение Random Forest модели...")
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        start_time = time.time()
        self.rf_model.fit(X_train_scaled, y_train)
        rf_time = time.time() - start_time
        
        rf_pred = self.rf_model.predict(X_test_scaled)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        results['Random Forest'] = {
            'accuracy': rf_accuracy,
            'time': rf_time,
            'predictions': rf_pred,
            'y_test': y_test
        }
        
        # 2. Neural Network (если TensorFlow доступен)
        if TENSORFLOW_AVAILABLE:
            st.info("Обучение нейронной сети...")
            try:
                self.nn_model = Sequential([
                    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(32, activation='relu'),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(16, activation='relu'),
                    Dropout(0.2),
                    Dense(1, activation='sigmoid')
                ])
                
                self.nn_model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                start_time = time.time()
                self.nn_model.fit(
                    X_train_scaled, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
                )
                nn_time = time.time() - start_time
                
                nn_pred_proba = self.nn_model.predict(X_test_scaled, verbose=0)
                nn_pred = (nn_pred_proba > 0.5).astype(int).flatten()
                nn_accuracy = accuracy_score(y_test, nn_pred)
                
                results['Neural Network'] = {
                    'accuracy': nn_accuracy,
                    'time': nn_time,
                    'predictions': nn_pred,
                    'y_test': y_test
                }
            except Exception as e:
                st.error(f"Ошибка обучения нейронной сети: {e}")
        
        self.is_trained = True
        self.save_models()
        
        return results
    
    def save_models(self):
        """Сохранение обученных моделей"""
        try:
            if self.rf_model is not None:
                joblib.dump(self.rf_model, 'models/random_forest.pkl')
            if self.scaler is not None:
                joblib.dump(self.scaler, 'models/scaler.pkl')
            if self.nn_model is not None and TENSORFLOW_AVAILABLE:
                self.nn_model.save('models/neural_network.h5')
            st.success("Модели сохранены успешно!")
        except Exception as e:
            st.error(f"Ошибка сохранения моделей: {e}")
    
    def load_models(self):
        """Загрузка обученных моделей"""
        try:
            if os.path.exists('models/random_forest.pkl'):
                self.rf_model = joblib.load('models/random_forest.pkl')
            if os.path.exists('models/scaler.pkl'):
                self.scaler = joblib.load('models/scaler.pkl')
            if os.path.exists('models/neural_network.h5') and TENSORFLOW_AVAILABLE:
                self.nn_model = load_model('models/neural_network.h5')
            
            if self.rf_model is not None or self.nn_model is not None:
                self.is_trained = True
                return True
        except Exception as e:
            st.error(f"Ошибка загрузки моделей: {e}")
        return False
    
    def predict(self, data):
        """Предсказание для новых данных"""
        if not self.is_trained:
            return None
        
        X, _ = self.preprocess_data(data)
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        predictions = {}
        
        if self.rf_model is not None:
            rf_pred = self.rf_model.predict(X_scaled)
            rf_proba = self.rf_model.predict_proba(X_scaled)[:, 1]
            predictions['Random Forest'] = {
                'predictions': rf_pred,
                'probabilities': rf_proba
            }
        
        if self.nn_model is not None and TENSORFLOW_AVAILABLE:
            nn_proba = self.nn_model.predict(X_scaled, verbose=0).flatten()
            nn_pred = (nn_proba > 0.5).astype(int)
            predictions['Neural Network'] = {
                'predictions': nn_pred,
                'probabilities': nn_proba
            }
        
        return predictions

# Инициализация модели
@st.cache_resource
def get_model():
    model = CyberGuardModel()
    model.load_models()
    return model

cyber_model = get_model()

# Главная страница
def main_page():
    st.title("🛡️ AI CyberGuard - Система обнаружения кибератак")
    
    st.markdown("""
    ### Добро пожаловать в AI CyberGuard!
    
    Это интеллектуальная система обнаружения кибератак, использующая машинное обучение 
    для анализа сетевого трафика и выявления потенциальных угроз.
    
    **Возможности системы:**
    - 🤖 Машинное обучение (Random Forest + Нейронные сети)
    - 📊 Анализ сетевого трафика в реальном времени
    - 📈 Визуализация угроз и статистики
    - 🔍 Детальная аналитика атак
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Модели", "2", "RF + NN")
    with col2:
        st.metric("TensorFlow", "✓" if TENSORFLOW_AVAILABLE else "✗", 
                 "Доступен" if TENSORFLOW_AVAILABLE else "Недоступен")
    with col3:
        st.metric("Статус", "🟢" if cyber_model.is_trained else "🔴",
                 "Готов" if cyber_model.is_trained else "Требует обучения")

def training_page():
    st.title("🎯 Обучение моделей")
    
    st.markdown("""
    На этой странице вы можете обучить модели машинного обучения 
    для обнаружения кибератак.
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        n_samples = st.number_input("Количество образцов", 1000, 50000, 10000, 1000)
        
        if st.button("🚀 Начать обучение", type="primary"):
            with st.spinner("Создание данных и обучение моделей..."):
                # Создание данных
                data = cyber_model.create_sample_data(n_samples)
                
                # Обучение моделей
                results = cyber_model.train_models(data)
                
                st.success("Обучение завершено!")
                
                # Показываем результаты
                st.subheader("📈 Результаты обучения")
                
                for model_name, result in results.items():
                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.metric(f"{model_name} - Точность", 
                                f"{result['accuracy']:.3f}",
                                f"{result['accuracy']*100:.1f}%")
                    with col_metric2:
                        st.metric(f"{model_name} - Время", 
                                f"{result['time']:.2f} сек")
                
                # Матрица ошибок
                fig, axes = plt.subplots(1, len(results), figsize=(12, 5))
                if len(results) == 1:
                    axes = [axes]
                
                for i, (model_name, result) in enumerate(results.items()):
                    cm = confusion_matrix(result['y_test'], result['predictions'])
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                    axes[i].set_title(f'{model_name}\nConfusion Matrix')
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('Actual')
                
                st.pyplot(fig)
    
    with col1:
        if cyber_model.is_trained:
            st.success("✅ Модели обучены и готовы к работе!")
            
            # Показываем информацию о моделях
            st.subheader("ℹ️ Информация о моделях")
            
            model_info = []
            if cyber_model.rf_model is not None:
                model_info.append({"Модель": "Random Forest", "Статус": "✅ Загружена", "Тип": "Ensemble"})
            if cyber_model.nn_model is not None:
                model_info.append({"Модель": "Neural Network", "Статус": "✅ Загружена", "Тип": "Deep Learning"})
            
            if model_info:
                st.dataframe(pd.DataFrame(model_info), use_container_width=True)
        else:
            st.info("⚠️ Модели не обучены. Нажмите 'Начать обучение' для создания моделей.")

def detection_page():
    st.title("🔍 Обнаружение атак")
    
    if not cyber_model.is_trained:
        st.error("❌ Модели не обучены! Перейдите на страницу 'Обучение моделей'.")
        return
    
    st.markdown("### Загрузите данные для анализа или создайте тестовые данные")
    
    tab1, tab2 = st.tabs(["📄 Загрузить файл", "🎲 Тестовые данные"])
    
    with tab1:
        uploaded_file = st.file_uploader("Выберите CSV файл", type=['csv'])
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            analyze_data(data)
    
    with tab2:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            test_samples = st.number_input("Количество тестовых записей", 10, 1000, 100)
            if st.button("Создать тестовые данные"):
                test_data = cyber_model.create_sample_data(test_samples)
                analyze_data(test_data)

def analyze_data(data):
    """Анализ загруженных данных"""
    st.subheader("📊 Анализ данных")
    
    # Показываем первые строки
    st.write("**Первые 5 строк данных:**")
    st.dataframe(data.head())
    
    # Получаем предсказания
    predictions = cyber_model.predict(data)
    
    if predictions:
        st.subheader("🎯 Результаты обнаружения")
        
        # Создаем сводную таблицу результатов
        results_df = data.copy()
        
        for model_name, pred_data in predictions.items():
            results_df[f'{model_name}_prediction'] = pred_data['predictions']
            results_df[f'{model_name}_probability'] = pred_data['probabilities']
        
        # Показываем статистику
        col1, col2, col3 = st.columns(3)
        
        total_records = len(data)
        
        for i, (model_name, pred_data) in enumerate(predictions.items()):
            attacks_detected = sum(pred_data['predictions'])
            attack_rate = attacks_detected / total_records * 100
            
            if i == 0:
                with col1:
                    st.metric("Всего записей", total_records)
            elif i == 1:
                with col2:
                    st.metric(f"{model_name} - Атаки", attacks_detected, f"{attack_rate:.1f}%")
            else:
                with col3:
                    st.metric(f"{model_name} - Атаки", attacks_detected, f"{attack_rate:.1f}%")
        
        # Визуализация результатов
        st.subheader("📈 Визуализация результатов")
        
        # График распределения вероятностей
        fig = go.Figure()
        
        for model_name, pred_data in predictions.items():
            fig.add_trace(go.Histogram(
                x=pred_data['probabilities'],
                name=f'{model_name}',
                opacity=0.7,
                nbinsx=30
            ))
        
        fig.update_layout(
            title="Распределение вероятностей атак",
            xaxis_title="Вероятность атаки",
            yaxis_title="Количество",
            barmode='overlay'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Показываем подробные результаты
        st.subheader("📋 Подробные результаты")
        
        # Фильтры
        show_attacks_only = st.checkbox("Показать только атаки")
        
        display_columns = ['packet_size', 'duration', 'src_port', 'dst_port']
        for model_name in predictions.keys():
            display_columns.extend([f'{model_name}_prediction', f'{model_name}_probability'])
        
        # Доступные колонки в данных
        available_columns = [col for col in display_columns if col in results_df.columns]
        
        filtered_df = results_df[available_columns]
        
        if show_attacks_only:
            # Показываем записи, где хотя бы одна модель предсказала атаку
            attack_mask = False
            for model_name in predictions.keys():
                pred_col = f'{model_name}_prediction'
                if pred_col in filtered_df.columns:
                    attack_mask = attack_mask | (filtered_df[pred_col] == 1)
            
            if attack_mask is not False:
                filtered_df = filtered_df[attack_mask]
        
        st.dataframe(filtered_df, use_container_width=True)

def dashboard_page():
    st.title("📊 Дашборд")
    
    if not cyber_model.is_trained:
        st.error("❌ Модели не обучены! Перейдите на страницу 'Обучение моделей'.")
        return
    
    # Создаем тестовые данные для демонстрации
    if st.button("🔄 Обновить данные"):
        demo_data = cyber_model.create_sample_data(1000)
        predictions = cyber_model.predict(demo_data)
        
        if predictions:
            st.subheader("🎯 Статистика обнаружения (последние 1000 записей)")
            
            # Основные метрики
            col1, col2, col3, col4 = st.columns(4)
            
            total_records = len(demo_data)
            normal_count = sum(demo_data['is_attack'] == 0)
            attack_count = sum(demo_data['is_attack'] == 1)
            
            with col1:
                st.metric("Всего записей", total_records)
            with col2:
                st.metric("Нормальный трафик", normal_count, f"{normal_count/total_records*100:.1f}%")
            with col3:
                st.metric("Атаки", attack_count, f"{attack_count/total_records*100:.1f}%")
            with col4:
                rf_accuracy = sum(predictions['Random Forest']['predictions'] == demo_data['is_attack']) / total_records
                st.metric("Точность RF", f"{rf_accuracy:.3f}", f"{rf_accuracy*100:.1f}%")
            
            # Графики
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart типов трафика
                fig = px.pie(
                    values=[normal_count, attack_count],
                    names=['Нормальный', 'Атаки'],
                    title="Распределение трафика"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # График типов атак
                if 'attack_type' in demo_data.columns:
                    attack_counts = demo_data['attack_type'].value_counts()
                    fig = px.bar(
                        x=attack_counts.index,
                        y=attack_counts.values,
                        title="Типы атак"
                    )
                    fig.update_xaxes(title="Тип атаки")
                    fig.update_yaxes(title="Количество")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Временная серия (симуляция)
            st.subheader("⏰ Атаки во времени (симуляция)")
            
            # Создаем временные данные
            time_data = pd.DataFrame({
                'time': pd.date_range(start='2024-01-01', periods=100, freq='H'),
                'attacks': np.random.poisson(2, 100)
            })
            
            fig = px.line(time_data, x='time', y='attacks', title="Количество атак по времени")
            st.plotly_chart(fig, use_container_width=True)

# Боковая панель навигации
def sidebar():
    st.sidebar.title("🛡️ AI CyberGuard")
    st.sidebar.markdown("---")
    
    pages = {
        "🏠 Главная": main_page,
        "🎯 Обучение моделей": training_page,
        "🔍 Обнаружение атак": detection_page,
        "📊 Дашборд": dashboard_page
    }
    
    selected_page = st.sidebar.selectbox("Выберите страницу", list(pages.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Статус системы")
    st.sidebar.write(f"**TensorFlow:** {'✅' if TENSORFLOW_AVAILABLE else '❌'}")
    st.sidebar.write(f"**Модели:** {'✅ Обучены' if cyber_model.is_trained else '❌ Не обучены'}")
    
    return pages[selected_page]

# Запуск приложения
def main():
    selected_page = sidebar()
    selected_page()
    
    # Футер
    st.markdown("---")
    st.markdown("🛡️ **AI CyberGuard** - Система обнаружения кибератак с использованием ИИ")

if __name__ == "__main__":
    main()