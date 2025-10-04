"""
Comparative Analysis - AI CyberGuard vs Existing Solutions
Research-grade benchmarking implementation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class ComparativeAnalysis:
    """Comprehensive comparison with existing cybersecurity solutions"""
    
    def __init__(self):
        self.baseline_models = {
            'Naive_Bayes': GaussianNB(),
            'SVM_Linear': SVC(kernel='linear', random_state=42),
            'Random_Forest_Basic': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42),
        }
        self.results = {}
        
    def benchmark_against_baselines(self, X_train, X_test, y_train, y_test):
        """Compare AI CyberGuard against baseline models"""
        
        print("Running comparative analysis against baseline models...")
        
        for name, model in self.baseline_models.items():
            print(f"Training {name}...")
            
            # Training time
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Inference time
            start_time = time.time()
            y_pred = model.predict(X_test)
            inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
            
            # Metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'training_time': training_time,
                'inference_time_ms': inference_time
            }
            
            self.results[name] = metrics
            
        return self.results
    
    def statistical_significance_test(self, ai_cyberguard_predictions, baseline_predictions, y_true):
        """Perform McNemar's test for statistical significance"""
        
        results = {}
        
        for model_name, pred in baseline_predictions.items():
            # Create contingency table for McNemar's test
            ai_correct = (ai_cyberguard_predictions == y_true)
            baseline_correct = (pred == y_true)
            
            # McNemar's test
            b = np.sum(ai_correct & ~baseline_correct)  # AI correct, baseline wrong
            c = np.sum(~ai_correct & baseline_correct)  # AI wrong, baseline correct
            
            if b + c > 0:
                mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
                p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
            else:
                mcnemar_stat = 0
                p_value = 1.0
            
            # Effect size (Cohen's d)
            ai_acc = np.mean(ai_correct)
            baseline_acc = np.mean(baseline_correct)
            pooled_std = np.sqrt((np.var(ai_correct) + np.var(baseline_correct)) / 2)
            cohens_d = (ai_acc - baseline_acc) / pooled_std if pooled_std > 0 else 0
            
            results[model_name] = {
                'mcnemar_statistic': mcnemar_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'ai_accuracy': ai_acc,
                'baseline_accuracy': baseline_acc,
                'significant': p_value < 0.05
            }
            
        return results
    
    def create_comparison_report(self, ai_cyberguard_metrics):
        """Generate comprehensive comparison report"""
        
        # Add AI CyberGuard results
        self.results['AI_CyberGuard'] = ai_cyberguard_metrics
        
        # Create comparison DataFrame
        df = pd.DataFrame(self.results).T
        
        # Sort by F1 score
        df = df.sort_values('f1', ascending=False)
        
        # Create visualizations
        self._create_comparison_plots(df)
        
        # Generate text report
        report = self._generate_text_report(df)
        
        return df, report
    
    def _create_comparison_plots(self, df):
        """Create comparison visualizations"""
        
        # Performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        models = df.index
        accuracies = df['accuracy']
        colors = ['red' if model == 'AI_CyberGuard' else 'lightblue' for model in models]
        
        axes[0, 0].bar(range(len(models)), accuracies, color=colors)
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(range(len(models)))
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        
        # Add values on bars
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # F1-Score comparison
        f1_scores = df['f1']
        axes[0, 1].bar(range(len(models)), f1_scores, color=colors)
        axes[0, 1].set_title('F1-Score Comparison')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_xticks(range(len(models)))
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
        
        for i, v in enumerate(f1_scores):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Training time comparison
        training_times = df['training_time']
        axes[1, 0].bar(range(len(models)), training_times, color=colors)
        axes[1, 0].set_title('Training Time Comparison (seconds)')
        axes[1, 0].set_ylabel('Training Time (s)')
        axes[1, 0].set_xticks(range(len(models)))
        axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
        
        # Inference time comparison
        inference_times = df['inference_time_ms']
        axes[1, 1].bar(range(len(models)), inference_times, color=colors)
        axes[1, 1].set_title('Inference Time Comparison (ms per sample)')
        axes[1, 1].set_ylabel('Inference Time (ms)')
        axes[1, 1].set_xticks(range(len(models)))
        axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('images/comparative_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance vs Speed scatter plot
        plt.figure(figsize=(10, 6))
        for i, model in enumerate(models):
            x = df.loc[model, 'inference_time_ms']
            y = df.loc[model, 'f1']
            color = 'red' if model == 'AI_CyberGuard' else 'blue'
            size = 100 if model == 'AI_CyberGuard' else 60
            plt.scatter(x, y, c=color, s=size, alpha=0.7, label=model)
            plt.annotate(model, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.xlabel('Inference Time (ms per sample)')
        plt.ylabel('F1-Score')
        plt.title('Performance vs Speed Trade-off')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('images/performance_vs_speed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, df):
        """Generate detailed text report"""
        
        report = "COMPARATIVE ANALYSIS REPORT - AI CYBERGUARD\n"
        report += "=" * 60 + "\n\n"
        
        # Overall ranking
        report += "PERFORMANCE RANKING (by F1-Score):\n"
        report += "-" * 35 + "\n"
        for i, (model, row) in enumerate(df.iterrows(), 1):
            report += f"{i}. {model}: {row['f1']:.4f}\n"
        
        # Key findings
        report += "\nKEY FINDINGS:\n"
        report += "-" * 15 + "\n"
        
        best_model = df.index[0]
        if best_model == 'AI_CyberGuard':
            improvement = df.loc['AI_CyberGuard', 'f1'] - df.iloc[1]['f1']
            report += f"✓ AI CyberGuard achieves BEST performance with {improvement:.3f} F1-score improvement\n"
        else:
            gap = df.iloc[0]['f1'] - df.loc['AI_CyberGuard', 'f1']
            report += f"⚠ AI CyberGuard trails best model by {gap:.3f} F1-score points\n"
        
        # Speed analysis
        ai_speed = df.loc['AI_CyberGuard', 'inference_time_ms']
        fastest_speed = df['inference_time_ms'].min()
        if ai_speed <= fastest_speed * 1.1:  # Within 10% of fastest
            report += f"✓ AI CyberGuard offers competitive speed: {ai_speed:.2f}ms per sample\n"
        else:
            report += f"⚠ AI CyberGuard slower than fastest model: {ai_speed:.2f}ms vs {fastest_speed:.2f}ms\n"
        
        # Detailed metrics table
        report += "\nDETAILED METRICS:\n"
        report += "-" * 20 + "\n"
        report += df.round(4).to_string()
        
        return report

def main():
    """Run comparative analysis"""
    
    print("Starting comprehensive comparative analysis...")
    
    # This would typically load your trained models and test data
    # For demonstration, we'll create synthetic results
    
    analyzer = ComparativeAnalysis()
    
    # Example usage:
    # results = analyzer.benchmark_against_baselines(X_train, X_test, y_train, y_test)
    # stat_results = analyzer.statistical_significance_test(ai_pred, baseline_pred, y_test)
    # df, report = analyzer.create_comparison_report(ai_metrics)
    
    print("Comparative analysis framework ready!")
    print("Use this script to benchmark AI CyberGuard against existing solutions.")

if __name__ == "__main__":
    main()