import os
import subprocess
import sys

def run_script(script_name):
    """Запуск Python скрипта"""
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            print(f"✓ {script_name} выполнен успешно")
            return True
        else:
            print(f"✗ Ошибка в {script_name}: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Не удалось запустить {script_name}: {e}")
        return False

def main():
    """Генерация всех отчетов"""
    scripts_to_run = [
        "train_model.py",
        "improved_neural_network.py",
        "final_comparison.py"
    ]
    
    print("Запуск генерации отчетов...")
    
    for script in scripts_to_run:
        if not run_script(script):
            print(f"Прерывание из-за ошибки в {script}")
            return
    
    print("Все отчеты успешно сгенерированы!")
    print("Проверьте папки models/, reports/ и images/")

if __name__ == "__main__":
    main()