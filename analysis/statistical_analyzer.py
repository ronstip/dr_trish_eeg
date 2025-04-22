import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.power import TTestPower
from statsmodels.stats.diagnostic import lilliefors
import statsmodels.api as sm
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

class StatisticalAnalyzer:
    """
    A class for performing comprehensive statistical analysis on EEG data.
    
    This class provides methods for data exploration, normality testing,
    parametric and non-parametric tests, bootstrap confidence intervals,
    and power analysis.
    """
    
    def __init__(self, data_path: str = "data/processed/processed_eeg_data.csv"):
        """
        Initialize the StatisticalAnalyzer with data path and visualization settings.
        
        Args:
            data_path (str): Path to the processed EEG data CSV file
        """
        # Set visualization style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("muted")
        sns.set_context("notebook", font_scale=1.2)
        
        # For reproducibility
        np.random.seed(42)
        
        # Store parameters
        self.data_path = Path(data_path)
        self.data = None
        self.z_scores = None
        self.field_name = None
        self.results = {}
        
    def load_data(self, field_name: str, sd_threshold: float = 2.5) -> np.ndarray:
        """
        Load data and handle outliers for the specified column.
        
        Args:
            column (str): Name of the column to analyze
            sd_threshold (float): Standard deviation threshold for outlier removal
            
        Returns:
            np.ndarray: Processed z-scores with outliers removed
        """
        self.field_name = field_name
        df = pd.read_csv(self.data_path)
        
        # Get the samples and remove outliers
        samples = df[field_name]
        mean = samples.mean()
        std = samples.std()
        outliers = samples[(samples - mean).abs() > sd_threshold * std]
        print(f"Number of outliers removed: {outliers.shape[0]}")
        
        # Store cleaned z-scores
        self.z_scores = samples.drop(outliers.index).values
        
        # Store field name
        self.results['field_name'] = field_name
        
        # store data
        self.data = df
        
        return self.z_scores
    
    def explore_data(self) -> Dict[str, float]:
        """
        Perform exploratory data analysis on the z-scores.
        
        Returns:
            Dict[str, float]: Dictionary containing summary statistics
        """
        if self.z_scores is None:
            raise ValueError("Please load data first using load_data() method")
            
        # Calculate basic statistics
        stats_dict = {
            'n': len(self.z_scores),
            'mean': np.mean(self.z_scores),
            'std': np.std(self.z_scores, ddof=1),
            'median': np.median(self.z_scores),
            'q1': np.percentile(self.z_scores, 25),
            'q3': np.percentile(self.z_scores, 75),
            'min': np.min(self.z_scores),
            'max': np.max(self.z_scores)
        }
        
        # Store results
        self.results['summary_stats'] = stats_dict
        
        # Create visualization
        # self._plot_exploratory_analysis()
        
        return stats_dict
    
    def check_normality(self) -> Dict[str, Dict[str, float]]:
        """
        Check normality assumption using multiple methods.
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary containing test results
        """
        if self.z_scores is None:
            raise ValueError("Please load data first using load_data() method")
            
        # Perform normality tests
        shapiro_stat, shapiro_p = stats.shapiro(self.z_scores)
        ks_stat, ks_p = lilliefors(self.z_scores)
        k2, dagostino_p = stats.normaltest(self.z_scores)
        anderson_result = stats.anderson(self.z_scores, dist='norm')
        
        normality_results = {
            'shapiro': {'statistic': shapiro_stat, 'p_value': shapiro_p},
            'lilliefors': {'statistic': ks_stat, 'p_value': ks_p},
            'dagostino': {'statistic': k2, 'p_value': dagostino_p},
            'anderson': {
                'statistic': anderson_result.statistic,
                'critical_values': anderson_result.critical_values.tolist(),
                'significance_levels': [15, 10, 5, 2.5, 1]
            }
        }
        
        self.results['normality_tests'] = normality_results
        return normality_results
    
    def perform_t_test(self) -> Dict[str, float]:
        """
        Perform one-sample t-test and calculate effect size.
        
        Returns:
            Dict[str, float]: Dictionary containing test results
        """
        if self.z_scores is None:
            raise ValueError("Please load data first using load_data() method")
            
        # Perform t-test
        t_stat, p_value = stats.ttest_1samp(self.z_scores, popmean=0)
        mean_z = np.mean(self.z_scores)
        cohen_d = mean_z / 1.0  # Since we're using z-scores, SD of reference population is 1
        
        t_test_results = {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohen_d,
            'effect_size_interpretation': self._interpret_effect_size(cohen_d)
        }
        
        self.results['t_test'] = t_test_results
        # self._plot_t_test_results()
        
        return t_test_results
    
    def perform_wilcoxon_test(self) -> Dict[str, float]:
        """
        Perform Wilcoxon signed-rank test as non-parametric alternative.
        
        Returns:
            Dict[str, float]: Dictionary containing test results
        """
        if self.z_scores is None:
            raise ValueError("Please load data first using load_data() method")
            
        # Perform Wilcoxon test
        w_stat, p_value = stats.wilcoxon(self.z_scores)
        z_value = stats.norm.isf(p_value / 2)
        n = len(self.z_scores)
        r = z_value / np.sqrt(n)
        
        wilcoxon_results = {
            'w_statistic': w_stat,
            'p_value': p_value,
            'effect_size_r': r,
            'effect_size_interpretation': self._interpret_effect_size_r(r)
        }
        
        self.results['wilcoxon'] = wilcoxon_results
        return wilcoxon_results
    
    def calculate_bootstrap_ci(self, n_bootstrap: int = 10000, ci: float = 0.95) -> Dict[str, float]:
        """
        Calculate bootstrap confidence interval for the mean.
        
        Args:
            n_bootstrap (int): Number of bootstrap samples
            ci (float): Confidence interval level (0-1)
            
        Returns:
            Dict[str, float]: Dictionary containing confidence interval bounds
        """
        if self.z_scores is None:
            raise ValueError("Please load data first using load_data() method")
            
        n = len(self.z_scores)
        bootstrap_means = np.zeros(n_bootstrap)
        
        for i in range(n_bootstrap):
            bootstrap_sample = np.random.choice(self.z_scores, size=n, replace=True)
            bootstrap_means[i] = np.mean(bootstrap_sample)
        
        lower = np.percentile(bootstrap_means, 100 * (1 - ci) / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 + ci) / 2)
        
        bootstrap_results = {
            'lower_bound': lower,
            'upper_bound': upper,
            'confidence_level': ci
        }
        
        self.results['bootstrap'] = bootstrap_results
        # self._plot_bootstrap_distribution(bootstrap_means, lower, upper, ci)
        
        return bootstrap_results
    
    def analyze_power(self, alpha: float = 0.05) -> Dict[str, float]:
        """
        Perform post-hoc power analysis for the one-sample t-test.
        
        Args:
            alpha (float): Significance level
            
        Returns:
            Dict[str, float]: Dictionary containing power analysis results
        """
        if self.z_scores is None:
            raise ValueError("Please load data first using load_data() method")
            
        mean_z = np.mean(self.z_scores)
        std_z = np.std(self.z_scores, ddof=1)
        n = len(self.z_scores)
        effect_size = mean_z
        
        power_calculator = TTestPower()
        power = power_calculator.power(effect_size=effect_size, nobs=n, alpha=alpha)
        
        sample_size_80 = power_calculator.solve_power(effect_size=effect_size, power=0.8, alpha=alpha)
        sample_size_90 = power_calculator.solve_power(effect_size=effect_size, power=0.9, alpha=alpha)
        
        power_results = {
            'effect_size': effect_size,
            'achieved_power': power,
            'sample_size_80': int(np.ceil(sample_size_80)),
            'sample_size_90': int(np.ceil(sample_size_90)),
            'current_sample_size': n,
            'alpha': alpha
        }
        
        self.results['power'] = power_results
        # self._plot_power_curve(power_calculator, effect_size, n, sample_size_80, sample_size_90, alpha)
        
        return power_results
    
    def run_full_analysis(self, column: str, sd_threshold: float = 2.5) -> Dict[str, Any]:
        """
        Run complete statistical analysis pipeline.
        
        Args:
            column (str): Column name to analyze
            sd_threshold (float): Standard deviation threshold for outlier removal
            
        Returns:
            Dict[str, Any]: Complete analysis results
        """
        self.load_data(column, sd_threshold)
        self.explore_data()
        self.check_normality()
        self.perform_t_test()
        self.perform_wilcoxon_test()
        self.calculate_bootstrap_ci()
        self.analyze_power()
        self.calculate_demographic_info()
        
        return self.results
    
    def calculate_demographic_info(self):
        """
        Calculate demographic information from the data.
        
        using the columns:         
        self.COLS_METADATA = [
            'date', 'age', 'gender', 'first_ghq_score', 'assessment_id',
            'patient_id', 'recordId', 'patientId'
        ]
        """
        if self.data is None:
            raise ValueError("Please load data first using load_data() method")
        
        # Calculate demographic information
        self.results['demographic_info'] = {
            'mean_age': self.data['age'].mean(),
            'median_age': self.data['age'].median(),
            'std_age': self.data['age'].std(),
            'min_age': self.data['age'].min(),
            'max_age': self.data['age'].max(),
            'gender_counts': self.data['gender'].value_counts().to_dict(),
            'first_ghq_score_mean': self.data['first_ghq_score'].mean(),
            'first_ghq_score_median': self.data['first_ghq_score'].median(),
            'first_ghq_score_std': self.data['first_ghq_score'].std(),
            'first_ghq_score_min': self.data['first_ghq_score'].min(),
            'first_ghq_score_max': self.data['first_ghq_score'].max(),
            'date_min': self.data['date'].min(),
            'date_max': self.data['date'].max(),
        }
        

    
    def _interpret_effect_size(self, d: float) -> str:
        """Helper method to interpret Cohen's d effect size."""
        if abs(d) < 0.2:
            return "negligible"
        elif abs(d) < 0.5:
            return "small"
        elif abs(d) < 0.8:
            return "medium"
        return "large"
    
    def _interpret_effect_size_r(self, r: float) -> str:
        """Helper method to interpret r effect size."""
        if abs(r) < 0.1:
            return "negligible"
        elif abs(r) < 0.3:
            return "small"
        elif abs(r) < 0.5:
            return "medium"
        return "large"
    
    def _plot_exploratory_analysis(self):
        """Create exploratory analysis plots."""
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Histogram with kernel density estimate
        ax1 = fig.add_subplot(221)
        sns.histplot(self.z_scores, kde=True, ax=ax1, color='skyblue')
        ax1.axvline(x=0, color='red', linestyle='--', label='Normal Population Mean')
        ax1.axvline(x=np.mean(self.z_scores), color='blue', linestyle='-', 
                   label='Abnormal Population Mean')
        ax1.set_title(f'Distribution of Z-scores: {self.field_name}')
        ax1.set_xlabel('Z-score')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # 2. Q-Q plot
        ax2 = fig.add_subplot(222)
        sm.qqplot(self.z_scores, line='45', ax=ax2)
        ax2.set_title('Q-Q Plot of Z-scores')
        
        # 3. Box plot
        ax3 = fig.add_subplot(223)
        sns.boxplot(x=self.z_scores, ax=ax3, color='skyblue')
        ax3.axvline(x=0, color='red', linestyle='--', label='Normal Population Mean')
        ax3.set_title('Box Plot of Z-scores')
        ax3.set_xlabel('Z-score')
        ax3.legend()
        
        # 4. Violin plot
        ax4 = fig.add_subplot(224)
        sns.violinplot(x=self.z_scores, ax=ax4, color='skyblue')
        ax4.axvline(x=0, color='red', linestyle='--', label='Normal Population Mean')
        ax4.set_title('Violin Plot of Z-scores')
        ax4.set_xlabel('Z-score')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
    
    def _plot_t_test_results(self):
        """Create t-test results visualization."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.histplot(self.z_scores, kde=True, ax=ax, color='skyblue', 
                    alpha=0.6, label='Abnormal Population')
        
        x = np.linspace(-4, 4, 1000)
        y = stats.norm.pdf(x, 0, 1)
        ax.plot(x, y * len(self.z_scores) * (max(self.z_scores) - min(self.z_scores)) / 30,
                color='red', linestyle='--', label='Normal Population')
        
        ax.axvline(x=0, color='red', linestyle='-', label='Normal Population Mean')
        ax.axvline(x=np.mean(self.z_scores), color='blue', linestyle='-', 
                  label='Abnormal Population Mean')
        
        t_results = self.results['t_test']
        ax.text(0.7, 0.9, 
                f"t = {t_results['t_statistic']:.3f}\n"
                f"p = {t_results['p_value']:.6f}\n"
                f"d = {t_results['cohens_d']:.3f}",
                transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_title(f'T-Test Results: {self.field_name}')
        ax.set_xlabel('Z-score')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def _plot_bootstrap_distribution(self, bootstrap_means: np.ndarray, 
                                  lower: float, upper: float, ci: float):
        """Create bootstrap distribution visualization."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.histplot(bootstrap_means, kde=True, ax=ax, color='skyblue')
        ax.axvline(x=lower, color='red', linestyle='--',
                  label=f'{(1-ci)/2*100:.1f}th percentile')
        ax.axvline(x=upper, color='red', linestyle='--',
                  label=f'{(1+ci)/2*100:.1f}th percentile')
        ax.axvline(x=0, color='green', linestyle='-',
                  label='Normal Population Mean')
        
        ax.set_title(f'Bootstrap Distribution: {self.field_name}')
        ax.set_xlabel('Mean Z-score')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def _plot_power_curve(self, power_calculator: TTestPower, effect_size: float,
                         n: int, sample_size_80: float, sample_size_90: float,
                         alpha: float):
        """Create power analysis visualization."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sample_sizes = np.arange(10, max(500, int(sample_size_90*1.2)), 10)
        powers = [power_calculator.power(effect_size=effect_size, nobs=ss, alpha=alpha)
                 for ss in sample_sizes]
        
        ax.plot(sample_sizes, powers, '-', color='blue')
        ax.axhline(y=0.8, color='red', linestyle='--', label='80% Power')
        ax.axhline(y=0.9, color='green', linestyle='--', label='90% Power')
        ax.axvline(x=n, color='purple', linestyle='-',
                  label=f'Current Sample Size ({n})')
        
        ax.plot(sample_size_80, 0.8, 'ro')
        ax.text(sample_size_80 + 5, 0.78,
                f'n = {int(np.ceil(sample_size_80))}', color='red')
        
        ax.plot(sample_size_90, 0.9, 'go')
        ax.text(sample_size_90 + 5, 0.88,
                f'n = {int(np.ceil(sample_size_90))}', color='green')
        
        ax.set_title(f'Power Analysis: {self.field_name}')
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('Statistical Power')
        ax.set_ylim(0, 1.05)
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def visualize_results(self):
        """
        Create a comprehensive visualization of all analysis results.
        This includes summary statistics, normality tests, hypothesis tests,
        confidence intervals, and power analysis in a single figure.
        """
        if not self.results:
            raise ValueError("No results to visualize. Please run the analysis first.")
            
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2)
        
        # 1. Distribution Plot with Summary Statistics
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(self.z_scores, kde=True, ax=ax1, color='skyblue')
        ax1.axvline(x=0, color='red', linestyle='--', label='Normal Population Mean')
        ax1.axvline(x=np.mean(self.z_scores), color='blue', linestyle='-', 
                   label='Abnormal Population Mean')
        
        # Add summary statistics as text
        stats = self.results['summary_stats']
        stats_text = (f"n = {stats['n']}\n"
                     f"Mean = {stats['mean']:.4f}\n"
                     f"SD = {stats['std']:.4f}\n"
                     f"Median = {stats['median']:.4f}\n"
                     f"Q1 = {stats['q1']:.4f}\n"
                     f"Q3 = {stats['q3']:.4f}")
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        ax1.set_title(f'Distribution of Z-scores: {self.field_name}')
        ax1.set_xlabel('Z-score')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # 2. Q-Q Plot
        ax2 = fig.add_subplot(gs[0, 1])
        sm.qqplot(self.z_scores, line='45', ax=ax2)
        ax2.set_title('Q-Q Plot of Z-scores')
        
        # 3. T-Test Results
        ax3 = fig.add_subplot(gs[1, 0])
        t_test = self.results['t_test']
        bootstrap = self.results['bootstrap']
        
        # Plot the distribution with confidence interval
        sns.histplot(self.z_scores, kde=True, ax=ax3, color='skyblue', alpha=0.6)
        ax3.axvline(x=0, color='red', linestyle='--', label='Normal Population Mean')
        ax3.axvline(x=np.mean(self.z_scores), color='blue', linestyle='-', 
                   label='Abnormal Population Mean')
        ax3.axvline(x=bootstrap['lower_bound'], color='green', linestyle='--',
                   label=f"{bootstrap['confidence_level']*100:.0f}% CI Lower")
        ax3.axvline(x=bootstrap['upper_bound'], color='green', linestyle='--',
                   label=f"{bootstrap['confidence_level']*100:.0f}% CI Upper")
        
        # Add t-test results as text
        t_test_text = (f"t = {t_test['t_statistic']:.4f}\n"
                      f"p = {t_test['p_value']:.6f}\n"
                      f"Cohen's d = {t_test['cohens_d']:.4f}\n"
                      f"Effect size: {t_test['effect_size_interpretation']}")
        ax3.text(0.02, 0.98, t_test_text, transform=ax3.transAxes,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        ax3.set_title('T-Test Results and Confidence Interval')
        ax3.set_xlabel('Z-score')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # 4. Normality Tests
        ax4 = fig.add_subplot(gs[1, 1])
        normality = self.results['normality_tests']
        
        # Create a table of normality test results
        test_names = ['Shapiro-Wilk', 'Lilliefors', "D'Agostino-Pearson"]
        statistics = [normality['shapiro']['statistic'],
                     normality['lilliefors']['statistic'],
                     normality['dagostino']['statistic']]
        p_values = [normality['shapiro']['p_value'],
                   normality['lilliefors']['p_value'],
                   normality['dagostino']['p_value']]
        
        # Plot the table
        cell_text = [[f"{stat:.4f}", f"{p:.6f}"] for stat, p in zip(statistics, p_values)]
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(cellText=cell_text,
                         colLabels=['Statistic', 'p-value'],
                         rowLabels=test_names,
                         loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax4.set_title('Normality Test Results')
        
        # 5. Power Analysis
        ax5 = fig.add_subplot(gs[2, 0])
        power = self.results['power']
        
        # Create power analysis visualization
        power_calculator = TTestPower()
        sample_sizes = np.arange(10, max(500, int(power['sample_size_90']*1.2)), 10)
        powers = [power_calculator.power(effect_size=power['effect_size'],
                                       nobs=ss, alpha=power['alpha'])
                 for ss in sample_sizes]
        
        ax5.plot(sample_sizes, powers, '-', color='blue')
        ax5.axhline(y=0.8, color='red', linestyle='--', label='80% Power')
        ax5.axhline(y=0.9, color='green', linestyle='--', label='90% Power')
        ax5.axvline(x=power['current_sample_size'], color='purple', linestyle='-',
                   label=f'Current Sample Size ({power["current_sample_size"]})')
        
        # Add power analysis results as text
        power_text = (f"Current power: {power['achieved_power']:.4f}\n"
                     f"Required n for 80% power: {power['sample_size_80']}\n"
                     f"Required n for 90% power: {power['sample_size_90']}")
        ax5.text(0.02, 0.98, power_text, transform=ax5.transAxes,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        ax5.set_title('Power Analysis')
        ax5.set_xlabel('Sample Size')
        ax5.set_ylabel('Statistical Power')
        ax5.set_ylim(0, 1.05)
        ax5.grid(True)
        ax5.legend()
        
        # 6. Non-parametric Test Results
        ax6 = fig.add_subplot(gs[2, 1])
        wilcoxon = self.results['wilcoxon']
        
        # Create a table of non-parametric test results
        cell_text = [[f"{wilcoxon['w_statistic']:.4f}",
                     f"{wilcoxon['p_value']:.6f}",
                     f"{wilcoxon['effect_size_r']:.4f}",
                     wilcoxon['effect_size_interpretation']]]
        
        ax6.axis('tight')
        ax6.axis('off')
        table = ax6.table(cellText=cell_text,
                         colLabels=['W-statistic', 'p-value', 'Effect size (r)', 'Interpretation'],
                         rowLabels=['Wilcoxon Test'],
                         loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax6.set_title('Non-parametric Test Results')
        
        plt.tight_layout()
        plt.show()
        
        # format results
        self.print_results()
        
    
    def print_results(self):
        """
        Print the results of the statistical analysis.
        """
        formatted_result = format_statistical_results(self.results)
        print(formatted_result)
        


def format_statistical_results(results: Dict[str, Any]) -> str:
    """
    Format statistical results according to APA style guidelines.
    
    Args:
        results: A dictionary containing statistical analysis results
        
    Returns:
        A formatted string with results in APA style
    """
    # Extract field name and format it nicely
    field_name = results['field_name'].replace('_', ' ').title()
    
    # Initialize the output string with a title
    output = f"Statistical Analysis Results for {field_name}\n"
    output += "=" * len(output) + "\n\n"
    
    # Descriptive Statistics
    stats = results['summary_stats']
    output += "Descriptive Statistics\n"
    output += "-" * 22 + "\n"
    output += f"N = {stats['n']}\n"
    output += f"M = {stats['mean']:.2f}, SD = {stats['std']:.2f}\n"
    output += f"Mdn = {stats['median']:.2f}, IQR = [{stats['q1']:.2f}, {stats['q3']:.2f}]\n"
    output += f"Range: {stats['min']:.2f} to {stats['max']:.2f}\n\n"
    
    # Normality Tests
    output += "Normality Tests\n"
    output += "-" * 15 + "\n"
    
    # Shapiro-Wilk Test
    shapiro = results['normality_tests']['shapiro']
    output += f"Shapiro-Wilk: W({stats['n'] - 2}) = {shapiro['statistic']:.3f}, "
    output += f"p {format_p_value(shapiro['p_value'])}\n"
    
    # Lilliefors Test
    lilliefors = results['normality_tests']['lilliefors']
    output += f"Lilliefors: D = {lilliefors['statistic']:.3f}, "
    output += f"p {format_p_value(lilliefors['p_value'])}\n"
    
    # D'Agostino Test
    dagostino = results['normality_tests']['dagostino']
    output += f"D'Agostino: K² = {dagostino['statistic']:.3f}, "
    output += f"p {format_p_value(dagostino['p_value'])}\n"
    
    # Anderson-Darling Test
    anderson = results['normality_tests']['anderson']
    output += "Anderson-Darling: "
    output += f"A² = {anderson['statistic']:.3f}, "
    
    # Determine significance based on critical values
    if anderson['statistic'] > anderson['critical_values'][-1]:
        output += f"p < {anderson['significance_levels'][-1]/100:.2f}\n\n"
    else:
        for i, crit_val in enumerate(anderson['critical_values']):
            if anderson['statistic'] <= crit_val:
                output += f"p > {anderson['significance_levels'][i]/100:.2f}\n\n"
                break
    
    # Hypothesis Tests
    output += "Hypothesis Tests\n"
    output += "-" * 16 + "\n"
    
    # t-test
    t_test = results['t_test']
    output += f"One-sample t-test: t({stats['n'] - 1}) = {t_test['t_statistic']:.3f}, "
    output += f"p {format_p_value(t_test['p_value'])}, "
    output += f"d = {t_test['cohens_d']:.2f} ({t_test['effect_size_interpretation']} effect)\n"
    
    # Wilcoxon test
    wilcoxon = results['wilcoxon']
    output += f"Wilcoxon signed-rank test: W = {wilcoxon['w_statistic']:.1f}, "
    output += f"p {format_p_value(wilcoxon['p_value'])}, "
    output += f"r = {wilcoxon['effect_size_r']:.2f} ({wilcoxon['effect_size_interpretation']} effect)\n\n"
    
    # Confidence Intervals
    bootstrap = results['bootstrap']
    output += f"95% CI for Mean: [{bootstrap['lower_bound']:.2f}, {bootstrap['upper_bound']:.2f}]\n\n"
    
    # Statistical Power
    power = results['power']
    output += "Statistical Power Analysis\n"
    output += "-" * 25 + "\n"
    output += f"Effect size (d): {power['effect_size']:.2f}\n"
    output += f"Achieved power: {power['achieved_power']:.3f}\n"
    output += f"Required sample size for 80% power: {power['sample_size_80']}\n"
    output += f"Required sample size for 90% power: {power['sample_size_90']}\n"
    output += f"Current sample size: {power['current_sample_size']}\n\n"
    
    # Demographic Information
    demo = results['demographic_info']
    output += "Demographic Information\n"
    output += "-" * 22 + "\n"
    output += f"Age: M = {demo['mean_age']:.2f}, SD = {demo['std_age']:.2f}, "
    output += f"Range: {demo['min_age']:.1f}-{demo['max_age']:.1f} years\n"
    
    # Format gender counts
    gender_str = ", ".join([f"{k}: {v}" for k, v in demo['gender_counts'].items()])
    output += f"Gender: {gender_str}\n"
    
    output += f"GHQ Score: M = {demo['first_ghq_score_mean']:.2f}, "
    output += f"SD = {demo['first_ghq_score_std']:.2f}, "
    output += f"Range: {demo['first_ghq_score_min']:.1f}-{demo['first_ghq_score_max']:.1f}\n"
    
    # Format date range
    date_min = demo['date_min'].split(' ')[0]
    date_max = demo['date_max'].split(' ')[0]
    output += f"Data collection period: {date_min} to {date_max}\n"
    
    return output

def format_p_value(p_value: float) -> str:
    """Format p-value according to APA guidelines"""
    if p_value < 0.001:
        return "< .001"
    elif p_value >= 0.001 and p_value < 0.01:
        return f"= .{str(p_value).split('.')[-1][:3]}"
    else:
        return f"= {p_value:.3f}"
