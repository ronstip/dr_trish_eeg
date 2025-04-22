import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure pandas display options at module level
pd.set_option('display.max_columns', None)

class EEGDataProcessor:
    def __init__(self, raw_data_path, test_group_path, users_meta_path):
        """
        Initialize the EEG data processor.
        
        Args:
            raw_data_path (str): Path to raw EEG data CSV
            test_group_path (str): Path to test group list CSV
            users_meta_path (str): Path to users metadata CSV
        """
        self.raw_data_path = Path(raw_data_path)
        self.test_group_path = Path(test_group_path)
        self.users_meta_path = Path(users_meta_path)
        
        # Data attributes
        self.raw_data = None
        self.test_group_ids = None
        self.processed_data = None
        self.users_meta = None
        self.test_group_names = None
        
        # Column constants
        self.COLS_QEEG = [
            'QEEG_alpha_eyesClosed_Cz', 'QEEG_alpha_eyesClosed_F3', 'QEEG_alpha_eyesClosed_F4',
            'QEEG_alpha_eyesClosed_Fp1', 'QEEG_alpha_eyesClosed_Fp2', 'QEEG_alpha_eyesClosed_Fz',
            'QEEG_alpha_eyesClosed_O1', 'QEEG_alpha_eyesClosed_Tp10', 'QEEG_alpha_eyesClosed_Tp9',
            'QEEG_alpha_eyesOpened_Cz', 'QEEG_alpha_eyesOpened_F3', 'QEEG_alpha_eyesOpened_F4',
            'QEEG_alpha_eyesOpened_Fp1', 'QEEG_alpha_eyesOpened_Fp2', 'QEEG_alpha_eyesOpened_Fz',
            'QEEG_alpha_eyesOpened_O1', 'QEEG_alpha_eyesOpened_Tp10', 'QEEG_alpha_eyesOpened_Tp9',
            'QEEG_beta_eyesClosed_Cz', 'QEEG_beta_eyesClosed_F3', 'QEEG_beta_eyesClosed_F4',
            'QEEG_beta_eyesClosed_Fp1', 'QEEG_beta_eyesClosed_Fp2', 'QEEG_beta_eyesClosed_Fz',
            'QEEG_beta_eyesClosed_O1', 'QEEG_beta_eyesClosed_Tp10', 'QEEG_beta_eyesClosed_Tp9',
            'QEEG_beta_eyesOpened_Cz', 'QEEG_beta_eyesOpened_F3', 'QEEG_beta_eyesOpened_F4',
            'QEEG_beta_eyesOpened_Fp1', 'QEEG_beta_eyesOpened_Fp2', 'QEEG_beta_eyesOpened_Fz',
            'QEEG_beta_eyesOpened_O1', 'QEEG_beta_eyesOpened_Tp10', 'QEEG_beta_eyesOpened_Tp9',
            'QEEG_delta_eyesClosed_Cz', 'QEEG_delta_eyesClosed_F3', 'QEEG_delta_eyesClosed_F4',
            'QEEG_delta_eyesClosed_Fp1', 'QEEG_delta_eyesClosed_Fp2', 'QEEG_delta_eyesClosed_Fz',
            'QEEG_delta_eyesClosed_O1', 'QEEG_delta_eyesClosed_Tp10', 'QEEG_delta_eyesClosed_Tp9',
            'QEEG_delta_eyesOpened_Cz', 'QEEG_delta_eyesOpened_F3', 'QEEG_delta_eyesOpened_F4',
            'QEEG_delta_eyesOpened_Fp1', 'QEEG_delta_eyesOpened_Fp2', 'QEEG_delta_eyesOpened_Fz',
            'QEEG_delta_eyesOpened_O1', 'QEEG_delta_eyesOpened_Tp10', 'QEEG_delta_eyesOpened_Tp9',
            'QEEG_high_alpha_eyesClosed_Cz', 'QEEG_high_alpha_eyesClosed_F3', 'QEEG_high_alpha_eyesClosed_F4',
            'QEEG_high_alpha_eyesClosed_Fp1', 'QEEG_high_alpha_eyesClosed_Fp2', 'QEEG_high_alpha_eyesClosed_Fz',
            'QEEG_high_alpha_eyesClosed_O1', 'QEEG_high_alpha_eyesClosed_Tp10', 'QEEG_high_alpha_eyesClosed_Tp9',
            'QEEG_high_alpha_eyesOpened_Cz', 'QEEG_high_alpha_eyesOpened_F3', 'QEEG_high_alpha_eyesOpened_F4',
            'QEEG_high_alpha_eyesOpened_Fp1', 'QEEG_high_alpha_eyesOpened_Fp2', 'QEEG_high_alpha_eyesOpened_Fz',
            'QEEG_high_alpha_eyesOpened_O1', 'QEEG_high_alpha_eyesOpened_Tp10', 'QEEG_high_alpha_eyesOpened_Tp9',
            'QEEG_high_beta_eyesClosed_Cz', 'QEEG_high_beta_eyesClosed_F3', 'QEEG_high_beta_eyesClosed_F4',
            'QEEG_high_beta_eyesClosed_Fp1', 'QEEG_high_beta_eyesClosed_Fp2', 'QEEG_high_beta_eyesClosed_Fz',
            'QEEG_high_beta_eyesClosed_O1', 'QEEG_high_beta_eyesClosed_Tp10', 'QEEG_high_beta_eyesClosed_Tp9',
            'QEEG_high_beta_eyesOpened_Cz', 'QEEG_high_beta_eyesOpened_F3', 'QEEG_high_beta_eyesOpened_F4',
            'QEEG_high_beta_eyesOpened_Fp1', 'QEEG_high_beta_eyesOpened_Fp2', 'QEEG_high_beta_eyesOpened_Fz',
            'QEEG_high_beta_eyesOpened_O1', 'QEEG_high_beta_eyesOpened_Tp10', 'QEEG_high_beta_eyesOpened_Tp9',
            'QEEG_low_alpha_eyesClosed_Cz', 'QEEG_low_alpha_eyesClosed_F3', 'QEEG_low_alpha_eyesClosed_F4',
            'QEEG_low_alpha_eyesClosed_Fp1', 'QEEG_low_alpha_eyesClosed_Fp2', 'QEEG_low_alpha_eyesClosed_Fz',
            'QEEG_low_alpha_eyesClosed_O1', 'QEEG_low_alpha_eyesClosed_Tp10', 'QEEG_low_alpha_eyesClosed_Tp9',
            'QEEG_low_alpha_eyesOpened_Cz', 'QEEG_low_alpha_eyesOpened_F3', 'QEEG_low_alpha_eyesOpened_F4',
            'QEEG_low_alpha_eyesOpened_Fp1', 'QEEG_low_alpha_eyesOpened_Fp2', 'QEEG_low_alpha_eyesOpened_Fz',
            'QEEG_low_alpha_eyesOpened_O1', 'QEEG_low_alpha_eyesOpened_Tp10', 'QEEG_low_alpha_eyesOpened_Tp9',
            'QEEG_low_beta_eyesClosed_Cz', 'QEEG_low_beta_eyesClosed_F3', 'QEEG_low_beta_eyesClosed_F4',
            'QEEG_low_beta_eyesClosed_Fp1', 'QEEG_low_beta_eyesClosed_Fp2', 'QEEG_low_beta_eyesClosed_Fz',
            'QEEG_low_beta_eyesClosed_O1', 'QEEG_low_beta_eyesClosed_Tp10', 'QEEG_low_beta_eyesClosed_Tp9',
            'QEEG_low_beta_eyesOpened_Cz', 'QEEG_low_beta_eyesOpened_F3', 'QEEG_low_beta_eyesOpened_F4',
            'QEEG_low_beta_eyesOpened_Fp1', 'QEEG_low_beta_eyesOpened_Fp2', 'QEEG_low_beta_eyesOpened_Fz',
            'QEEG_low_beta_eyesOpened_O1', 'QEEG_low_beta_eyesOpened_Tp10', 'QEEG_low_beta_eyesOpened_Tp9',
            'QEEG_slow_eyesClosed_Cz', 'QEEG_slow_eyesClosed_F3', 'QEEG_slow_eyesClosed_F4',
            'QEEG_slow_eyesClosed_Fp1', 'QEEG_slow_eyesClosed_Fp2', 'QEEG_slow_eyesClosed_Fz',
            'QEEG_slow_eyesClosed_O1', 'QEEG_slow_eyesClosed_Tp10', 'QEEG_slow_eyesClosed_Tp9',
            'QEEG_slow_eyesOpened_Cz', 'QEEG_slow_eyesOpened_F3', 'QEEG_slow_eyesOpened_F4',
            'QEEG_slow_eyesOpened_Fp1', 'QEEG_slow_eyesOpened_Fp2', 'QEEG_slow_eyesOpened_Fz',
            'QEEG_slow_eyesOpened_O1', 'QEEG_slow_eyesOpened_Tp10', 'QEEG_theta_eyesClosed_Cz',
            'QEEG_theta_eyesClosed_F3', 'QEEG_theta_eyesClosed_F4', 'QEEG_theta_eyesClosed_Fp1',
            'QEEG_theta_eyesClosed_Fp2', 'QEEG_theta_eyesClosed_Fz', 'QEEG_theta_eyesClosed_O1',
            'QEEG_theta_eyesClosed_Tp10', 'QEEG_theta_eyesClosed_Tp9', 'QEEG_theta_eyesOpened_Cz',
            'QEEG_theta_eyesOpened_F3', 'QEEG_theta_eyesOpened_F4', 'QEEG_theta_eyesOpened_Fp1',
            'QEEG_theta_eyesOpened_Fp2', 'QEEG_theta_eyesOpened_Fz', 'QEEG_theta_eyesOpened_O1',
            'QEEG_theta_eyesOpened_Tp10', 'QEEG_theta_eyesOpened_Tp9',
        ]
        
        self.COLS_RATIOS = [
            'Theta Symmetry; Frontal Left and Right (F3, F4); Theta (F3) / theta (F4)',
            'Theta/Alpha Ratio; Frontal Left (F3); Theta / alpha',
            'Theta/Alpha Ratio; Frontal Right (F4); Theta / alpha',
            'Theta/Beta Ratio Response While Counting; Central (Cz); [TBR(UT) - TBR(EC)] / TBR(EC)',
            'Theta/Beta Ratio Response; Posterior (O1); [ [Theta / beta] (EC) - [Theta / beta] (EO)] ] / [Theta / beta] (EO)',
            'Theta/Beta Ratio Symmetry; Frontal Left and Right (F3, F4); [Theta / beta] (F3) / [theta / beta] (F4)',
            'Theta/Beta Ratio While Counting; Central (Cz); Theta / beta',
            'Theta/Beta Ratio; Central (Cz); Theta / beta',
            'Theta/Beta Ratio; Frontal Left (F3); Theta / beta',
            'Theta/Beta Ratio; Frontal Right (F4); Theta / beta',
            'Theta/Beta Ratio; Posterior (O1); Theta / beta',
            'Theta/Low-Beta Ratio; Central (Cz); Theta / SMR',
            'Alpha Recovery; Central (Cz); [Alpha(EO) after - Alpha(EO) before] / Alpha(EO) before',
            'Alpha Recovery; Posterior (O1); [Alpha(EO) after - Alpha(EO) before] / Alpha(EO) before',
            'Alpha Response; Central (Cz); [Alpha(EC) - Alpha(EO)] / Alpha(EO)',
            'Alpha Response; Posterior (O1); [Alpha(EC) - Alpha(EO)] / Alpha(EO)',
            'Alpha Symmetry; Frontal Left and Right (F3, F4); Alpha (F3) / alpha (F4)',
            'Beta Balance; Frontal (Fz); HiBeta/Beta',
            'Beta Response While Counting; Central (Cz); [Beta(UT) - Beta(EC)] / Beta(EC)',
            'Beta Symmetry; Frontal Left and Right (F3, F4); Beta (F3) / beta (F4)',
            'Low-alpha/High-alpha Ratio; Frontal (Fz); Low-alpha (8-9Hz) / high-alpha (11-12Hz)',
            'Peak Alpha; Posterior (O1); The maximum amplitude value in the EEG frequency spectrum between 8 and 12 Hz',
        ]
        
        self.COLS_METADATA = [
            'date', 'age', 'gender', 'first_ghq_score', 'assessment_id',
            'patient_id', 'recordId', 'patientId'
        ]
        
        self.COLS_SIGNAL_QUALITY = [
            'clean_signal_length', 'state', 'train_location', 'reference_location',
            'diffsum_1', 'diffsum_2', 'max_abs_amplitude', 'noise_ratio',
            'raw_signal_length', 'total_power'
        ]
        
        self.COLS_HZ = [f'power_{i}Hz' for i in range(1, 31)]
        
        self.COLS_BANDS = [
            'delta', 'theta', 'slow', 'low_alpha', 'high_alpha',
            'alpha', 'low_beta', 'high_beta', 'beta',
            'alpha_peak', 'alpha_weighted_peak'
        ]
        
        self.STATES = ['eyesOpened', 'eyesClosed']
        self.FRONTAL_CHANNELS = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4']
        self.PREFRONTAL_CHANNELS = ['Fp1', 'Fp2']
        self.FRONTAL_LEFT_CHANNELS = ['Fp1', 'F3', 'Fz']
        self.FRONTAL_RIGHT_CHANNELS = ['Fp2', 'F4']
        self.POSTERIOR_CHANNELS = ['O1']
        self.CENTRAL_CHANNELS = ['Cz']
        self.TEMPORAL_CHANNELS = ['Tp9', 'Tp10']
            
        
    def feature_engineering(self):
        """
        Perform feature engineering on the processed data.
        """
        try:
            logger.info("Performing feature engineering...")
            
            # Create new features
            for band in [b for b in self.COLS_BANDS if b not in ['alpha_peak', 'alpha_weighted_peak']]:
                for state in self.STATES:
                    frontal_left = [f'QEEG_{band}_{state}_{channel}' for channel in self.FRONTAL_LEFT_CHANNELS]
                    frontal_right = [f'QEEG_{band}_{state}_{channel}' for channel in self.FRONTAL_RIGHT_CHANNELS]
                    posterior = [f'QEEG_{band}_{state}_{channel}' for channel in self.POSTERIOR_CHANNELS]
                    central = [f'QEEG_{band}_{state}_{channel}' for channel in self.CENTRAL_CHANNELS]
                    temporal = [f'QEEG_{band}_{state}_{channel}' for channel in self.TEMPORAL_CHANNELS]
                    frontal = [f'QEEG_{band}_{state}_{channel}' for channel in self.FRONTAL_CHANNELS]
                    prefrontal = [f'QEEG_{band}_{state}_{channel}' for channel in self.PREFRONTAL_CHANNELS]
                    
                    self.processed_data[f'QEEG_{band}_{state}_Frontal_left'] = self.processed_data[frontal_left].mean(axis=1)
                    self.processed_data[f'QEEG_{band}_{state}_Frontal_right'] = self.processed_data[frontal_right].mean(axis=1)
                    self.processed_data[f'QEEG_{band}_{state}_Posterior'] = self.processed_data[posterior].mean(axis=1)
                    self.processed_data[f'QEEG_{band}_{state}_Central'] = self.processed_data[central].mean(axis=1)
                    # self.processed_data[f'QEEG_{band}_{state}_Temporal'] = self.processed_data[temporal].mean(axis=1)
                    self.processed_data[f'QEEG_{band}_{state}_Frontal'] = self.processed_data[frontal].mean(axis=1)
                    self.processed_data[f'QEEG_{band}_{state}_Prefrontal'] = self.processed_data[prefrontal].mean(axis=1)
            
            # Alpha response: [Alpha(EC) - Alpha(EO)] / Alpha(EO)
            self.processed_data['alpha_response_Cz'] = (self.processed_data['QEEG_alpha_eyesClosed_Cz'] - self.processed_data['QEEG_alpha_eyesOpened_Cz']) / self.processed_data['QEEG_alpha_eyesOpened_Cz']
            self.processed_data['low_alpha_response_Cz'] = (self.processed_data['QEEG_low_alpha_eyesClosed_Cz'] - self.processed_data['QEEG_low_alpha_eyesOpened_Cz']) / self.processed_data['QEEG_low_alpha_eyesOpened_Cz']
            self.processed_data['high_alpha_response_Cz'] = (self.processed_data['QEEG_high_alpha_eyesClosed_Cz'] - self.processed_data['QEEG_high_alpha_eyesOpened_Cz']) / self.processed_data['QEEG_high_alpha_eyesOpened_Cz']
            
            logger.info("Feature engineering completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error during feature engineering: {str(e)}")
            return False    
    
    
    def validate_data(self):
        """
        Validate the loaded data for required columns, data types, and value ranges.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            logger.info("Validating data...")
            
            # Check required columns
            required_cols = set(self.COLS_METADATA + self.COLS_QEEG)
            missing_cols = required_cols - set(self.raw_data.columns)
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Check data types
            numeric_cols = self.COLS_QEEG + ['age', 'first_ghq_score']
            for col in numeric_cols:
                if col in self.raw_data.columns and not pd.api.types.is_numeric_dtype(self.raw_data[col]):
                    logger.error(f"Column {col} should be numeric but is {self.raw_data[col].dtype}")
                    return False
            
            # Check value ranges
            if 'age' in self.raw_data.columns:
                if (self.raw_data['age'] < 0).any() or (self.raw_data['age'] > 120).any():
                    logger.error("Age values out of expected range (0-120)")
                    return False
            
            # Check missing data threshold (e.g., no more than 20% missing values per column)
            missing_threshold = 0.2
            missing_percentages = self.raw_data[self.COLS_QEEG].isnull().mean()
            problematic_cols = missing_percentages[missing_percentages > missing_threshold]
            if not problematic_cols.empty:
                logger.warning(f"Columns with more than {missing_threshold*100}% missing values: {problematic_cols}")
            
            logger.info("Data validation completed")
            return True
        except Exception as e:
            logger.error(f"Error during data validation: {str(e)}")
            return False
    
    def load_data(self):
        """
        Load and validate raw data from CSV files.
        
        Returns:
            bool: True if data loading was successful, False otherwise
        """
        try:
            logger.info("Loading raw data...")
            self.raw_data = pd.read_csv(self.raw_data_path)
            self.users_meta = pd.read_csv(self.users_meta_path)
            self.test_group_names = pd.read_csv(self.test_group_path)
            
            logger.info(f"Loaded {len(self.raw_data)} rows of raw data")
            
            # Validate the loaded data
            return self.validate_data()
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def visualize_data(self):
        """
        Generate various visualizations of the data.
        
        Returns:
            list: List of matplotlib figures
        """
        try:
            logger.info("Generating visualizations...")
            figures = []
            
            # 1. Missing values heatmap
            fig1 = plt.figure(figsize=(20, 10))
            sns.heatmap(self.processed_data.isnull(), cbar=False)
            plt.title("Heatmap of missing values in processed data")
            figures.append(fig1)
            
            # 2. Records per patient
            records_per_patient = self.raw_data.groupby("patient_id").size()
            fig2 = plt.figure(figsize=(10, 6))
            sns.histplot(records_per_patient)
            plt.title("Distribution of records per patient")
            figures.append(fig2)
            
            # 3. Noise ratio analysis
            noise_data = self.raw_data[self.COLS_BANDS + self.COLS_SIGNAL_QUALITY]
            noise_data = noise_data[noise_data['alpha'].isna()]
            
            fig3 = plt.figure(figsize=(10, 6))
            sns.histplot(noise_data['noise_ratio'])
            plt.title("Histogram of noise_ratio for rows with missing alpha values")
            figures.append(fig3)
            
            # 4. QEEG feature analysis
            for feature in ['QEEG_alpha_eyesOpened_Fp1', 'QEEG_beta_eyesClosed_Cz']:
                if feature in self.processed_data.columns:
                    fig4, axs = plt.subplots(2, 2, figsize=(20, 10))
                    
                    # Histogram
                    sns.histplot(self.processed_data[feature], ax=axs[0,0])
                    axs[0,0].set_title(f"Histogram of {feature}")
                    
                    # Boxplot
                    sns.boxplot(data=self.processed_data, y=feature, ax=axs[0,1])
                    axs[0,1].set_title(f"Boxplot of {feature}")
                    
                    # Violin plot
                    sns.violinplot(data=self.processed_data, x='age', y=feature, ax=axs[1,0])
                    axs[1,0].set_title(f"Violin plot of {feature} by age")
                    
                    # Scatter plot
                    sns.scatterplot(data=self.processed_data, x='age', y=feature, ax=axs[1,1])
                    axs[1,1].set_title(f"Scatter plot of {feature} vs age")
                    
                    plt.tight_layout()
                    figures.append(fig4)
            
            logger.info("Visualizations generated successfully")
            return figures
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            return []
    
    def filter_test_group(self):
        """
        Filter data for test group patients.
        
        Returns:
            bool: True if filtering was successful, False otherwise
        """
        try:
            logger.info("Filtering test group data...")
            # Get patient_id of all first_name and last_name in patient_list
            self.test_group_ids = self.users_meta[
                self.users_meta['first_name'].isin(self.test_group_names['first_name']) & 
                self.users_meta['last_name'].isin(self.test_group_names['last_name'])
            ]['patient_id']
            
            rows_before = len(self.raw_data)
            self.raw_data = self.raw_data[self.raw_data['patient_id'].isin(self.test_group_ids)]
            rows_after = len(self.raw_data)
            
            logger.info(f"Filtered {rows_before - rows_after} rows. Remaining rows: {rows_after}")
            return True
        except Exception as e:
            logger.error(f"Error filtering test group: {str(e)}")
            return False
    
    def process_first_assessment(self):
        """
        Process the first assessment for each patient.
        
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            logger.info("Processing first assessment data...")
            # Group by patient_id and take the first assessment by date
            self.processed_data = (
                self.raw_data
                .sort_values('date')
                .groupby(['patient_id'])
                .first()
                .reset_index()#[self.COLS_METADATA + self.COLS_QEEG]
                .set_index('patient_id')
            )
            
            logger.info(f"Processed first assessment data for {len(self.processed_data)} patients")
            return True
        except Exception as e:
            logger.error(f"Error processing first assessment: {str(e)}")
            return False
    
    def analyze_data_quality(self):
        """
        Analyze signal quality and missing data.
        
        Returns:
            bool: True if analysis was successful, False otherwise
        """
        try:
            logger.info("Analyzing data quality...")
            
            # Ensure reports/figures directory exists
            output_dir = Path('reports/figures')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a figure for the heatmap
            plt.figure(figsize=(20, 10))
            sns.heatmap(self.processed_data.isnull(), cbar=False)
            plt.title("Heatmap of missing values in processed data")
            plt.savefig(output_dir / 'missing_values_heatmap.png')
            plt.close()
            
            # Analyze records per patient
            records_per_patient = self.raw_data.groupby("patient_id").size()
            logger.info(f"Records per patient statistics:\n{records_per_patient.describe()}")
            
            return True
        except Exception as e:
            logger.error(f"Error analyzing data quality: {str(e)}")
            return False
    
    def save_processed_data(self, output_path):
        """
        Save processed data to CSV file.
        
        Args:
            output_path (str): Path to save processed data
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        try:
            logger.info(f"Saving processed data to {output_path}...")
            self.processed_data.to_csv(output_path)
            logger.info("Data saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            return False
    
    def process_data(self, output_path):
        """
        Orchestrate the entire data processing pipeline.
        
        Args:
            output_path (str): Path to save processed data
            
        Returns:
            bool: True if all steps were successful, False otherwise
        """
        steps = [
            self.load_data,
            self.filter_test_group,
            self.process_first_assessment,
            self.feature_engineering,
            self.analyze_data_quality,
            lambda: self.visualize_data(),
            lambda: self.save_processed_data(output_path)
        ]
        
        for step in steps:
            if not step():
                logger.error("Data processing pipeline failed")
                return False
        
        logger.info("Data processing pipeline completed successfully")
        return True

if __name__ == "__main__":
    # Example usage
    processor = EEGDataProcessor(
        raw_data_path="data/raw/trish_all_users_data.csv",
        test_group_path="data/raw/test_group_list.csv",
        users_meta_path="data/raw/users_meta_data.csv"
    )
    
    processor.process_data("data/processed/processed_eeg_data.csv")
