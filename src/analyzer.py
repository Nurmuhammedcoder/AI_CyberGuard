import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class TrafficAnalyzer:
    def __init__(self, data_path):
        # Преобразуем относительный путь в абсолютный
        if not os.path.isabs(data_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            data_path = os.path.join(parent_dir, data_path)
            
        self.df = pd.read_csv(data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
    def basic_info(self):
        print("=" * 50)
        print("AI CYBERGUARD - АНАЛИЗ СЕТЕВОГО ТРАФИКА")
        print("=" * 50)
        print(f"Размер датасета: {self.df.shape}")
        print(f"Период данных: {self.df['timestamp'].min()} - {self.df['timestamp'].max()}")
        print("\nРаспределение меток:")
        print(self.df['label'].value_counts())
        print("\nТипы данных:")
        print(self.df.dtypes)
        
    def protocol_analysis(self):
        print("\n" + "=" * 50)
        print("АНАЛИЗ ПРОТОКОЛОВ")
        print("=" * 50)
        
        # Протоколы
        protocol_names = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}
        self.df['protocol_name'] = self.df['protocol'].map(protocol_names)
        
        protocol_counts = self.df['protocol_name'].value_counts()
        print("Распределение по протоколам:")
        print(protocol_counts)
        
        # Визуализация
        plt.figure(figsize=(10, 6))
        protocol_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Распределение сетевых протоколов')
        plt.ylabel('')
        plt.tight_layout()
        
        # Сохранение в папку images
        images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')
        os.makedirs(images_dir, exist_ok=True)
        plt.savefig(os.path.join(images_dir, 'protocol_distribution.png'))
        plt.show()
        
    def traffic_analysis(self):
        print("\n" + "=" * 50)
        print("АНАЛИЗ ТРАФИКА ПО ВРЕМЕНИ")
        print("=" * 50)
        
        # Группировка по часам
        self.df['hour'] = self.df['timestamp'].dt.hour
        hourly_traffic = self.df.groupby('hour').size()
        
        plt.figure(figsize=(12, 6))
        hourly_traffic.plot(kind='bar')
        plt.title('Сетевой трафик по часам')
        plt.xlabel('Час дня')
        plt.ylabel('Количество соединений')
        plt.tight_layout()
        
        # Сохранение в папку images
        images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')
        os.makedirs(images_dir, exist_ok=True)
        plt.savefig(os.path.join(images_dir, 'hourly_traffic.png'))
        plt.show()
        
    def attack_analysis(self):
        print("\n" + "=" * 50)
        print("АНАЛИЗ АТАК")
        print("=" * 50)
        
        # Только атаки
        attacks_df = self.df[self.df['label'] != 'Normal']
        
        if len(attacks_df) > 0:
            print("Распределение типов атак:")
            print(attacks_df['label'].value_counts())
            
            # Атаки по времени
            attacks_by_hour = attacks_df.groupby('hour').size()
            
            plt.figure(figsize=(12, 6))
            attacks_by_hour.plot(kind='bar', color='red')
            plt.title('Распределение атак по времени')
            plt.xlabel('Час дня')
            plt.ylabel('Количество атак')
            plt.tight_layout()
            
            # Сохранение в папку images
            images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')
            os.makedirs(images_dir, exist_ok=True)
            plt.savefig(os.path.join(images_dir, 'attacks_by_hour.png'))
            plt.show()
        else:
            print("Атаки не обнаружены в данных")
            
    def generate_report(self):
        self.basic_info()
        self.protocol_analysis()
        self.traffic_analysis()
        self.attack_analysis()
        
        # Сохранение отчета
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path = os.path.join(reports_dir, f'traffic_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"AI CYBERGUARD - ОТЧЕТ О СЕТЕВОМ ТРАФИКЕ\n")
            f.write(f"Сгенерировано: {report_time}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Размер датасета: {self.df.shape}\n")
            f.write(f"Период данных: {self.df['timestamp'].min()} - {self.df['timestamp'].max()}\n\n")
            
            f.write("РАСПРЕДЕЛЕНИЕ МЕТОК:\n")
            f.write(str(self.df['label'].value_counts()) + "\n\n")
            
            f.write("РАСПРЕДЕЛЕНИЕ ПРОТОКОЛОВ:\n")
            protocol_counts = self.df['protocol_name'].value_counts()
            f.write(str(protocol_counts) + "\n\n")
            
            if len(self.df[self.df['label'] != 'Normal']) > 0:
                f.write("ОБНАРУЖЕННЫЕ АТАКИ:\n")
                attacks_df = self.df[self.df['label'] != 'Normal']
                f.write(str(attacks_df['label'].value_counts()) + "\n")
                
        print(f"\nОтчет сохранен: {report_path}")

if __name__ == "__main__":
    analyzer = TrafficAnalyzer('data/network_traffic_demo.csv')
    analyzer.generate_report()