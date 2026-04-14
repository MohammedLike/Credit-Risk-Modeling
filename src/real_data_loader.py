"""
Real Data Loader for Credit Risk Modeling
Supports LendingClub, Kaggle, and other real datasets
"""
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from config import DATA_DIR


class RealDataLoader:
    """Load and preprocess real credit datasets."""
    
    @staticmethod
    def load_lending_club(filepath):
        """
        Load LendingClub dataset.
        
        Expected columns:
        - loan_status (1=Charged Off/Default, 0=Fully Paid)
        - loan_amnt, funded_amnt
        - term, int_rate
        - grade
        - emp_length
        - home_ownership
        - annual_inc
        - dti
        - fico_range_low, fico_range_high
        - open_acc
        - pub_rec
        - revol_bal
        - revol_util
        - total_acc
        
        Returns:
            df: Cleaned DataFrame
        """
        print("Loading LendingClub dataset...")
        df = pd.read_csv(filepath, low_memory=False)
        
        # Target variable
        df['default'] = (df['loan_status'].isin(['Charged Off', 'Default'])).astype(int)
        
        # Select relevant features
        features = [
            'loan_amnt', 'int_rate', 'term', 'grade', 'emp_length', 'home_ownership',
            'annual_inc', 'dti', 'fico_range_low', 'open_acc', 'pub_rec',
            'revol_bal', 'revol_util', 'total_acc', 'default'
        ]
        
        df = df[features].dropna()
        
        # Convert term to numeric
        df['term'] = df['term'].str.extract('(\d+)').astype(int)
        
        # Convert int_rate to numeric
        df['int_rate'] = df['int_rate'].str.rstrip('%').astype(float) / 100
        
        # Convert emp_length to numeric
        def convert_emp_length(x):
            if pd.isna(x) or x == '< 1 year':
                return 0
            elif x == '10+ years':
                return 10
            else:
                return int(x.split()[0])
        
        df['emp_length'] = df['emp_length'].apply(convert_emp_length)
        
        # Encode categorical variables
        le_grade = LabelEncoder()
        le_home = LabelEncoder()
        
        df['grade'] = le_grade.fit_transform(df['grade'])
        df['home_ownership'] = le_home.fit_transform(df['home_ownership'])
        
        # Calculate FICO average
        df['fico'] = (df['fico_range_low'] + df['fico_range_high']) / 2
        df = df.drop(['fico_range_low', 'fico_range_high'], axis=1)
        
        # Rename to standard names
        df = df.rename(columns={
            'loan_amnt': 'loan_amount',
            'int_rate': 'interest_rate',
            'annual_inc': 'income',
            'revol_bal': 'revolving_balance',
            'revol_util': 'utilization',
            'open_acc': 'open_accounts',
            'total_acc': 'total_accounts',
            'pub_rec': 'public_records',
            'home_ownership': 'home_owner',
        })
        
        print(f"  Loaded: {len(df):,} records | Default rate: {df['default'].mean():.2%}")
        return df
    
    @staticmethod
    def load_kaggle_credit_default(filepath):
        """
        Load Kaggle Credit Default dataset.
        
        Source: https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset
        
        Returns:
            df: Cleaned DataFrame
        """
        print("Loading Kaggle Credit Default dataset...")
        df = pd.read_csv(filepath, index_col=0)
        
        # Target is typically 'default.payment.next.month'
        if 'default.payment.next.month' in df.columns:
            df['default'] = df['default.payment.next.month']
            df = df.drop('default.payment.next.month', axis=1)
        
        # Clean column names
        df.columns = [col.replace('.', '_').lower() for col in df.columns]
        
        print(f"  Loaded: {len(df):,} records | Default rate: {df['default'].mean():.2%}")
        return df
    
    @staticmethod
    def load_synthetic_or_real(use_real_data=False, real_data_path=None, dataset_name="lending_club"):
        """
        Flexible loader: use real data if available, fall back to synthetic.
        
        Args:
            use_real_data: If True, try to load real data
            real_data_path: Path to real dataset
            dataset_name: "lending_club" or "kaggle_credit"
        
        Returns:
            df: Loaded DataFrame
            data_source: "real" or "synthetic"
        """
        if use_real_data and real_data_path and os.path.exists(real_data_path):
            try:
                if dataset_name.lower() == "lending_club":
                    df = RealDataLoader.load_lending_club(real_data_path)
                elif dataset_name.lower() == "kaggle_credit":
                    df = RealDataLoader.load_kaggle_credit_default(real_data_path)
                else:
                    df = pd.read_csv(real_data_path)
                
                return df, "real"
            except Exception as e:
                print(f"  Warning: Could not load real data ({e}). Using synthetic data.")
                from src.data_generator import generate_credit_data, add_derived_features
                df = generate_credit_data()
                df = add_derived_features(df)
                return df, "synthetic"
        else:
            from src.data_generator import generate_credit_data, add_derived_features
            df = generate_credit_data()
            df = add_derived_features(df)
            return df, "synthetic"
    
    @staticmethod
    def download_lending_club_sample():
        """
        Instructions to download LendingClub data.
        
        Note: Actual download requires authentication at https://www.lendingclub.com/info/download-data.action
        """
        instructions = """
        To download LendingClub data:
        
        1. Go to: https://www.lendingclub.com/info/download-data.action
        2. Create a free account
        3. Download "Loan Data" CSV files
        4. Save to: output/data/lending_club_data.csv
        
        Alternative (Kaggle mirror):
        - kaggle datasets download -d wordsforthewise/lending-club
        
        LendingClub datasets range from ~100k to 2M+ loans covering 2007-2018+
        """
        return instructions
    
    @staticmethod
    def generate_sample_configs():
        """Generate configuration examples for different datasets."""
        configs = {
            "lending_club": {
                "path": "output/data/lending_club_data.csv",
                "name": "lending_club",
                "description": "LendingClub loan data",
            },
            "kaggle_credit": {
                "path": "output/data/kaggle_credit_default.csv",
                "name": "kaggle_credit",
                "description": "Kaggle credit card default dataset",
            },
            "synthetic": {
                "path": None,
                "name": None,
                "description": "Generated synthetic data",
            },
        }
        return configs
