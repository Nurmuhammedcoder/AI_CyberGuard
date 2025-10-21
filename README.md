# 🛡️ AI CyberGuard - ML-Based Network Intrusion Detection System

An educational machine learning project demonstrating cybersecurity threat detection using ensemble methods.

## 📊 Current Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 95.1% | 93.6% | 73.1% | 82.1% |
| Neural Network | 99.2% | 98.7% | 95.9% | 97.2% |

*Results on synthetic dataset (15,000 samples)*

## 🎯 Features

- **Dual ML Architecture**: Random Forest + Deep Neural Network
- **Interactive Dashboard**: Real-time visualization with Streamlit
- **Automated Training**: Complete pipeline from data to deployment
- **Model Comparison**: Side-by-side performance analysis
- **Export Capabilities**: Save trained models and reports

## 🛠️ Technical Stack

- **Python 3.10**
- **TensorFlow 2.13** - Neural network implementation
- **Scikit-learn 1.3** - Traditional ML models
- **Streamlit 1.28** - Web interface
- **Pandas/NumPy** - Data processing

## 📁 Project Structure
```
AI_CyberGuard/
├── dashboard.py              # Main Streamlit application
├── src/
│   ├── train_model.py        # Model training pipeline
│   ├── data_generator.py     # Synthetic data generation
│   └── analyzer.py           # Data analysis utilities
├── models/                   # Trained model files
│   ├── random_forest.pkl
│   ├── neural_network.h5
│   └── scaler.pkl
├── images/                   # Visualizations and charts
├── reports/                  # Generated reports
├── data/                     # Dataset storage
└── notebooks/                # Jupyter notebooks for analysis
```

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- 4GB+ RAM recommended

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Nurmuhammedcoder/AI_CyberGuard.git
cd AI_CyberGuard
```

2. **Create virtual environment**
```bash
python -m venv my_cyber_env

# Windows
my_cyber_env\Scripts\activate

# Linux/macOS
source my_cyber_env/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Dashboard
```bash
streamlit run dashboard.py
```

Open browser to `http://localhost:8501`

### Training Models
```bash
cd src
python train_model.py
```

This will:
- Generate synthetic training data
- Train both Random Forest and Neural Network models
- Create visualizations
- Save models to `models/` directory
- Generate performance reports

### Dashboard Features

- **Home** - System overview and status
- **Train Models** - Interactive model training interface
- **Detect Attacks** - Real-time threat analysis
- **Dashboard** - Performance metrics and visualizations

## 📈 Model Details

### Random Forest Classifier

- 200 decision trees
- Balanced class weights
- Max depth: 15
- Training time: ~2.4 seconds

### Neural Network

- Architecture: 128→64→32→16→1 neurons
- Activation: ReLU + Sigmoid output
- Optimizer: Adam (lr=0.001)
- Regularization: Dropout + BatchNormalization
- Training time: ~19 seconds

## 🎓 Educational Purpose

This project is designed for learning and demonstration purposes. It uses synthetic data to simulate network traffic patterns and attack behaviors.

### Current Limitations

- **Synthetic Data**: Uses generated data, not real network captures
- **Simplified Features**: 30 numerical features vs. complex real-world scenarios
- **Binary Classification**: Normal vs. Attack (real systems need multi-class)
- **No Real-time Processing**: Batch processing only
- **Educational Scope**: Not production-ready

## 🗺️ Development Roadmap

### Phase 1 (Completed)
- ✅ Basic ML models implementation
- ✅ Streamlit dashboard
- ✅ Model training pipeline
- ✅ Visualization system

### Phase 2 (In Progress)
- 🔄 CICIDS2017 dataset integration
- 🔄 Multi-class attack classification
- 🔄 Statistical validation framework
- 🔄 Jupyter notebook tutorials

### Phase 3 (Planned)
- 📋 Real-time PCAP file processing
- 📋 REST API for model inference
- 📋 Docker containerization
- 📋 Comparative analysis with existing IDS

## 📊 Performance Visualizations

The project generates several visualizations:

- Confusion matrices
- Training history plots
- Feature importance charts
- Model comparison graphs

All saved to `images/` directory.

## 🤝 Contributing

This is an educational project. Suggestions and improvements are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🔗 Resources

- [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)

## 👤 Author

**Nurmukhammed**

- GitHub: [@Nurmuhammedcoder](https://github.com/Nurmuhammedcoder)
- Email: nekulov@internet.ru

## 🙏 Acknowledgments

- Dataset inspiration: Canadian Institute for Cybersecurity
- ML frameworks: TensorFlow and Scikit-learn teams
- Streamlit for excellent visualization tools

---

⭐ **If you find this project useful, please consider giving it a star!**

⚠️ **Disclaimer**: This is an educational project. Not intended for production cybersecurity applications.
```
