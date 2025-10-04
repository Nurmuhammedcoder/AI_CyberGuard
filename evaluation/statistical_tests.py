"""
Statistical Tests and Validation for AI CyberGuard
Research-grade statistical analysis framework
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, mcnemar
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class StatisticalValidator:
    """Statistical validation and significance testing for model comparisons"""
    
    def __init__(self):
        self.results = {}
        self.significance_level = 0.05
    
    def mcnemar_test(self, y_true, y_pred1, y_pred2, model1_name="Model 1", model2_name="Model 2"):
        """
        McNemar's test for comparing two models on the same test set
        
        Parameters:
        y_true: True labels
        y_pred1: Predictions from first model
        y_pred2: Predictions from second model
        
        Returns:
        dict: Test results including statistic, p-value, and interpretation
        """
        
        # Create contingency table
        correct1 = (y_pred1 == y_true)
        correct2 = (y_pred2 == y_true)
        
        # McNemar's test contingency table
        # |          | Model2 Correct | Model2 Wrong |
        # |----------|----------------|--------------|
        # |Model1 Correct|     a       |      b      |
        # |Model1 Wrong  |     c       |      d      |
        
        a = np.sum(correct1 & correct2)    # Both correct
        b = np.sum(correct1 & ~correct2)   # Model1 correct, Model2 wrong
        c = np.sum(~correct1 & correct2)   # Model1 wrong, Model2 correct
        d = np.sum(~correct1 & ~correct2)  # Both wrong
        
        # McNemar's test statistic
        if b + c == 0:
            mcnemar_stat = 0
            p_value = 1.0
        else:
            mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        
        # Effect size (Cohen's h for proportions)
        p1 = np.mean(correct1)
        p2 = np.mean(correct2)
        cohens_h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
        
        result = {
            'test_type': 'McNemar Test',
            'model1_name': model1_name,
            'model2_name': model2_name,
            'contingency_table': {
                'both_correct': a,
                'model1_only': b,
                'model2_only': c,
                'both_wrong': d
            },
            'mcnemar_statistic': mcnemar_stat,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'model1_accuracy': p1,
            'model2_accuracy': p2,
            'accuracy_difference': p1 - p2,
            'cohens_h': cohens_h,
            'effect_size': self._interpret_cohens_h(abs(cohens_h)),
            'interpretation': self._interpret_mcnemar_result(p_value, p1, p2, model1_name, model2_name)
        }
        
        return result
    
    def paired_t_test(self, scores1, scores2, model1_name="Model 1", model2_name="Model 2"):
        """
        Paired t-test for comparing cross-validation scores
        
        Parameters:
        scores1: Cross-validation scores from first model
        scores2: Cross-validation scores from second model
        
        Returns:
        dict: Test results
        """
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        # Effect size (Cohen's d for paired samples)
        diff = scores1 - scores2
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        
        result = {
            'test_type': 'Paired t-test',
            'model1_name': model1_name,
            'model2_name': model2_name,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'model1_mean': np.mean(scores1),
            'model2_mean': np.mean(scores2),
            'mean_difference': np.mean(scores1) - np.mean(scores2),
            'cohens_d': cohens_d,
            'effect_size': self._interpret_cohens_d(abs(cohens_d)),
            'interpretation': self._interpret_ttest_result(t_stat, p_value, np.mean(scores1), np.mean(scores2), model1_name, model2_name)
        }
        
        return result
    
    def cross_validation_comparison(self, model1, model2, X, y, cv_folds=5, 
                                   model1_name="Model 1", model2_name="Model 2"):
        """
        Compare two models using cross-validation with statistical testing
        """
        
        # Stratified k-fold for balanced splits
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        scores1 = cross_val_score(model1, X, y, cv=skf, scoring='accuracy')
        scores2 = cross_val_score(model2, X, y, cv=skf, scoring='accuracy')
        
        # Paired t-test on CV scores
        ttest_result = self.paired_t_test(scores1, scores2, model1_name, model2_name)
        
        # Additional metrics
        f1_scores1 = cross_val_score(model1, X, y, cv=skf, scoring='f1')
        f1_scores2 = cross_val_score(model2, X, y, cv=skf, scoring='f1')
        
        result = {
            'cv_folds': cv_folds,
            'accuracy_scores': {
                model1_name: scores1,
                model2_name: scores2
            },
            'f1_scores': {
                model1_name: f1_scores1,
                model2_name: f1_scores2
            },
            'statistical_test': ttest_result,
            'summary': {
                'model1_accuracy_mean': np.mean(scores1),
                'model1_accuracy_std': np.std(scores1),
                'model2_accuracy_mean': np.mean(scores2),
                'model2_accuracy_std': np.std(scores2),
                'model1_f1_mean': np.mean(f1_scores1),
                'model2_f1_mean': np.mean(f1_scores2)
            }
        }
        
        return result
    
    def confidence_interval(self, data, confidence=0.95):
        """Calculate confidence interval for data"""
        
        n = len(data)
        mean = np.mean(data)
        std_err = stats.sem(data)
        
        # t-distribution for small samples
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_error = t_val * std_err
        
        return {
            'mean': mean,
            'confidence_level': confidence,
            'lower_bound': mean - margin_error,
            'upper_bound': mean + margin_error,
            'margin_error': margin_error
        }
    
    def statistical_power_analysis(self, effect_size, sample_size, alpha=0.05):
        """
        Calculate statistical power for given effect size and sample size
        """
        from scipy.stats import norm
        
        # For two-tailed test
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(effect_size * np.sqrt(sample_size/2) - z_alpha)
        power = norm.cdf(z_beta)
        
        return {
            'effect_size': effect_size,
            'sample_size': sample_size,
            'alpha': alpha,
            'power': power,
            'adequate_power': power >= 0.8  # Conventional threshold
        }
    
    def create_statistical_report(self, results_dict):
        """Generate comprehensive statistical report"""
        
        report = "STATISTICAL VALIDATION REPORT - AI CYBERGUARD\n"
        report += "=" * 60 + "\n\n"
        
        for test_name, result in results_dict.items():
            report += f"Test: {test_name}\n"
            report += "-" * 40 + "\n"
            
            if result['test_type'] == 'McNemar Test':
                report += f"Models: {result['model1_name']} vs {result['model2_name']}\n"
                report += f"McNemar Statistic: {result['mcnemar_statistic']:.4f}\n"
                report += f"P-value: {result['p_value']:.6f}\n"
                report += f"Significant: {'Yes' if result['significant'] else 'No'}\n"
                report += f"Effect Size (Cohen's h): {result['cohens_h']:.4f} ({result['effect_size']})\n"
                report += f"Interpretation: {result['interpretation']}\n\n"
            
            elif result['test_type'] == 'Paired t-test':
                report += f"Models: {result['model1_name']} vs {result['model2_name']}\n"
                report += f"T-statistic: {result['t_statistic']:.4f}\n"
                report += f"P-value: {result['p_value']:.6f}\n"
                report += f"Significant: {'Yes' if result['significant'] else 'No'}\n"
                report += f"Effect Size (Cohen's d): {result['cohens_d']:.4f} ({result['effect_size']})\n"
                report += f"Interpretation: {result['interpretation']}\n\n"
        
        return report
    
    def _interpret_cohens_h(self, h):
        """Interpret Cohen's h effect size"""
        if h < 0.2:
            return "Small"
        elif h < 0.5:
            return "Medium"
        else:
            return "Large"
    
    def _interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return "Small"
        elif d < 0.5:
            return "Medium"
        elif d < 0.8:
            return "Large"
        else:
            return "Very Large"
    
    def _interpret_mcnemar_result(self, p_value, acc1, acc2, model1_name, model2_name):
        """Interpret McNemar's test result"""
        if p_value >= self.significance_level:
            return f"No significant difference between {model1_name} and {model2_name}"
        else:
            better_model = model1_name if acc1 > acc2 else model2_name
            return f"{better_model} performs significantly better (p < {self.significance_level})"
    
    def _interpret_ttest_result(self, t_stat, p_value, mean1, mean2, model1_name, model2_name):
        """Interpret t-test result"""
        if p_value >= self.significance_level:
            return f"No significant difference between {model1_name} and {model2_name}"
        else:
            better_model = model1_name if mean1 > mean2 else model2_name
            return f"{better_model} performs significantly better (p < {self.significance_level})"

def validate_model_performance(model_results, significance_level=0.05):
    """
    Validate that model performance meets research standards
    
    Parameters:
    model_results: dict with model performance metrics
    significance_level: alpha level for statistical tests
    
    Returns:
    dict: Validation results
    """
    
    validation_results = {
        'meets_accuracy_threshold': False,
        'meets_precision_threshold': False,
        'meets_recall_threshold': False,
        'meets_f1_threshold': False,
        'overall_validation': False
    }
    
    # Research-grade thresholds
    thresholds = {
        'accuracy': 0.95,
        'precision': 0.90,
        'recall': 0.90,
        'f1': 0.90
    }
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        if metric in model_results:
            value = model_results[metric]
            threshold = thresholds[metric]
            validation_results[f'meets_{metric}_threshold'] = value >= threshold
            validation_results[f'{metric}_value'] = value
            validation_results[f'{metric}_threshold'] = threshold
    
    # Overall validation
    validation_results['overall_validation'] = all([
        validation_results['meets_accuracy_threshold'],
        validation_results['meets_precision_threshold'],
        validation_results['meets_recall_threshold'],
        validation_results['meets_f1_threshold']
    ])
    
    return validation_results

# Example usage
if __name__ == "__main__":
    print("Statistical Validation Framework for AI CyberGuard")
    print("=" * 50)
    
    # Create sample data for demonstration
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred1 = np.random.randint(0, 2, 1000)
    y_pred2 = np.random.randint(0, 2, 1000)
    
    # Initialize validator
    validator = StatisticalValidator()
    
    # Run McNemar's test
    mcnemar_result = validator.mcnemar_test(y_true, y_pred1, y_pred2, "Random Forest", "Neural Network")
    print("McNemar's Test Result:")
    print(f"P-value: {mcnemar_result['p_value']:.6f}")
    print(f"Significant: {mcnemar_result['significant']}")
    print(f"Interpretation: {mcnemar_result['interpretation']}")
    
    print("\nStatistical validation framework ready for use!")