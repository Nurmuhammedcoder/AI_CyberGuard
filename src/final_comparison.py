import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from improved_neural_network import compare_models_improved, create_comprehensive_report
from train_model import load_data

def main():
    """Запуск финального сравнения моделей"""
    print("Запуск финального сравнения моделей...")
    
    # Загрузка данных
    X_train, X_test, y_train, y_test = load_data()
    
    if X_train is None:
        print("Не удалось загрузить данные.")
        return
    
    # Сравнение улучшенных моделей
    results = compare_models_improved(X_train, y_train, X_test, y_test)
    
    # Создание отчета
    report, fig = create_comprehensive_report(results, y_test)
    print(report)
    
    print("Финальное сравнение завершено! Результаты сохранены в папках reports/ и images/")

if __name__ == "__main__":
    main()