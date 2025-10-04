import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_network_traffic_data(num_records=50000):
    """
    Генерация реалистичных сетевых данных для демонстрации
    """
    np.random.seed(42)
    random.seed(42)
    
    # Списки для генерации правдоподобных данных
    protocols = [6, 17, 1]  # TCP, UDP, ICMP
    services = ['http', 'https', 'ftp', 'ssh', 'smtp', 'dns', 'dhcp']
    flags = ['SF', 'S0', 'REJ', 'RSTO', 'RSTR', 'SH', 'S1', 'S2', 'S3']
    normal_ips = ['192.168.1.{}'.format(i) for i in range(1, 50)]
    server_ips = ['10.0.0.{}'.format(i) for i in range(1, 20)]
    external_ips = ['203.0.113.{}'.format(i) for i in range(1, 100)]
    
    # Генерация временных меток
    start_time = datetime.now() - timedelta(days=1)
    timestamps = [start_time + timedelta(seconds=random.randint(0, 86400)) 
                 for _ in range(num_records)]
    timestamps.sort()
    
    data = {
        'timestamp': timestamps,
        'duration': np.random.exponential(0.3, num_records),
        'protocol': np.random.choice(protocols, num_records, p=[0.7, 0.2, 0.1]),
        'service': np.random.choice(services, num_records),
        'flag': np.random.choice(flags, num_records, p=[0.6, 0.1, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02, 0.05]),
        'src_bytes': np.random.lognormal(5, 2, num_records),
        'dst_bytes': np.random.lognormal(5, 2, num_records),
        'count': np.random.poisson(5, num_records),
        'srv_count': np.random.poisson(3, num_records),
        'dst_host_count': np.random.poisson(10, num_records),
        'dst_host_srv_count': np.random.poisson(5, num_records),
    }
    
    # Генерация IP-адресов
    src_ips = []
    dst_ips = []
    
    for i in range(num_records):
        # 80% трафика - внутренний, 20% - внешний
        if random.random() < 0.8:
            src_ips.append(random.choice(normal_ips))
            dst_ips.append(random.choice(server_ips))
        else:
            src_ips.append(random.choice(external_ips))
            dst_ips.append(random.choice(normal_ips))
    
    data['src_ip'] = src_ips
    data['dst_ip'] = dst_ips
    
    # Генерация портов
    data['src_port'] = [random.randint(1024, 65535) for _ in range(num_records)]
    data['dst_port'] = [random.choice([80, 443, 22, 21, 25, 53, 67, 68]) for _ in range(num_records)]
    
    df = pd.DataFrame(data)
    
    # Добавление меток атак (5% записей - атаки)
    labels = ['Normal'] * num_records
    num_attacks = int(num_records * 0.05)
    attack_indices = random.sample(range(num_records), num_attacks)
    
    # Типы атак
    attack_types = ['DDoS', 'PortScan', 'BruteForce', 'Malware', 'WebAttack']
    
    for i in attack_indices:
        labels[i] = random.choice(attack_types)
        
        # Добавляем аномалии в данные для атак
        if labels[i] == 'DDoS':
            df.loc[i, 'src_bytes'] *= 100  # Увеличиваем объем данных
            df.loc[i, 'count'] *= 10       # Увеличиваем количество соединений
        elif labels[i] == 'PortScan':
            df.loc[i, 'dst_port'] = random.randint(1000, 65535)  # Нестандартный порт
            df.loc[i, 'srv_count'] *= 5    # Увеличиваем количество сервисов
        elif labels[i] == 'BruteForce':
            df.loc[i, 'count'] *= 20       # Много попыток
            df.loc[i, 'dst_bytes'] *= 0.1  # Маленькие ответы
    
    df['label'] = labels
    
    # Сохранение - ИСПРАВЛЕННЫЙ БЛОК КОДА
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, 'network_traffic_demo.csv')
    df.to_csv(data_path, index=False)
    
    print(f"Сгенерирован демонстрационный набор данных с {num_records} записями")
    print(f"Файл сохранен: {os.path.abspath(data_path)}")
    print("Распределение меток:")
    print(pd.Series(labels).value_counts())
    
    return df

if __name__ == "__main__":
    generate_network_traffic_data()