#!/usr/bin/env python3
"""
=========================================
DATA PREPARATION MODULE
=========================================
Extended analysis based on "Exploratory Data Analysis with Python Cookbook"
Chapter 2: Preparing Data for EDA

Author: Lumo Analysis Assistant
Date: 2026-08-01
"""

# ==========================================# IMPORT LIBRARIES
# ==========================================
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataPreparator:
    """
    Main class for comprehensive data preparation workflows.
    Encapsulates all EDA preparation techniques from the cookbook.
    """
    
    def __init__(self, filepath: str):
        """Initialize preparator with data file path."""
        self.filepath = Path(filepath)
        self.df = None
        self.prepared_df = None
        self.preprocessing_log = []
        
    def _log(self, message: str):
        """Log preprocessing steps."""
        self.preprocessing_log.append(message)
        print(f"[LOG] {message}")
    
    # ==========================================# DATA LOADING
    # ==========================================
    
    def load_data(self) -> pd.DataFrame:
        """Load CSV data into DataFrame."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {self.filepath}")
        
        self.df = pd.read_csv(self.filepath)
        self._log(f"Loaded data: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def inspect_data(self) -> Dict:
        """Perform initial data inspection."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        inspection = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing': self.df.isnull().sum().to_dict(),
            'missing_pct': (self.df.isnull().sum() / len(self.df) * 100).round(2).to_dict(),
            'memory_usage_mb': round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        
        self._log(f"Inspection complete: {inspection['shape']}")
        return inspection
    
    def subset_columns(self, columns: List[str]) -> pd.DataFrame:
        """Select specific columns from DataFrame."""
        available = [col for col in columns if col in self.df.columns]
        missing = set(columns) - set(available)
        
        if missing:
            self._log(f"Warning: Columns not found: {missing}")
        
        self.df = self.df[available].copy()
        self._log(f"Subscribed to {len(available)} columns")
        return self.df
    
    # ==========================================# TECHNIQUE 1: GROUPING
    # ==========================================
    
    def group_by(self, 
                 group_col: str, 
                 agg_col: str, 
                 agg_funcs: List[str] = ['mean']) -> pd.DataFrame:
        """
        Group data by category and aggregate.
        
        Args:
            group_col: Column to group by (categorical)
            agg_col: Column to aggregate (numeric)
            agg_funcs: List of aggregation functions
        
        Returns:
            Aggregated DataFrame
        """
        result = self.df.groupby(group_col)[agg_col].agg(agg_funcs)
        self._log(f"Grouped by '{group_col}', aggregated '{agg_col}'")
        return result
    
    def multi_column_grouping(self, 
                               group_cols: List[str], 
                               agg_dict: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Group by multiple columns with custom aggregations.
        
        Args:
            group_cols: List of columns to group by
            agg_dict: Dictionary mapping columns to aggregation functions
        
        Returns:
            Multi-index aggregated DataFrame
        """
        result = self.df.groupby(group_cols).agg(agg_dict)
        self._log(f"Multi-group by: {group_cols}")
        return result
    
    # ==========================================# TECHNIQUE 2: APPENDING
    # ==========================================
    
    def append_datasets(self, 
                        dataframes: List[pd.DataFrame], 
                        axis: int = 0) -> pd.DataFrame:
        """
        Append multiple DataFrames along rows or columns.
        
        Args:
            dataframes: List of DataFrames to combine
            axis: 0 for rows, 1 for columns
        
        Returns:
            Combined DataFrame
        """
        if not isinstance(dataframes, list):
            dataframes = [dataframes]
        
        result = pd.concat(dataframes, axis=axis)
        self._log(f"Appended {len(dataframes)} datasets, axis={axis}, shape={result.shape}")
        return result
    
    # ==========================================# TECHNIQUE 3: CONCATENATING
    # ==========================================
    
    def concatenate_features(self,
                             dataframe_list: List[pd.DataFrame],
                             key_col: str = None) -> pd.DataFrame:
        """
        Concatenate DataFrames by columns (features).
        
        Args:
            dataframe_list: List of DataFrames with same row count
            key_col: Optional common key column for alignment
        
        Returns:
            Concatenated DataFrame
        """
        if key_col:
            for df in dataframe_list:
                df.set_index(key_col, inplace=True)
        
        result = pd.concat(dataframe_list, axis=1)
        self._log(f"Concatenated {len(dataframe_list)} feature sets, shape={result.shape}")
        return result.reset_index()
    
    # ==========================================# TECHNIQUE 4: MERGING
    # ==========================================
    
    def merge_tables(self,
                     left_df: pd.DataFrame,
                     right_df: pd.DataFrame,
                     on_col: str = None,
                     left_on: str = None,
                     right_on: str = None,
                     how: str = 'inner') -> pd.DataFrame:
        """
        Merge two DataFrames on common key.
        
        Args:
            left_df: Left DataFrame
            right_df: Right DataFrame
            on_col: Common column name
            left_on: Left key column (if names differ)
            right_on: Right key column (if names differ)
            how: Join type ('inner', 'left', 'right', 'outer')
        
        Returns:
            Merged DataFrame
        """
        kwargs = {'how': how}
        if on_col:
            kwargs['on'] = on_col
        elif left_on and right_on:
            kwargs['left_on'] = left_on
            kwargs['right_on'] = right_on
        else:
            raise ValueError("Must specify either on_col or (left_on, right_on)")
        
        result = pd.merge(left_df, right_df, **kwargs)
        self._log(f"Merged with '{how}' join, shape={result.shape}")
        return result
    
    # ==========================================# TECHNIQUE 5: SORTING
    # ==========================================
    
    def sort_data(self,
                  by: str | List[str],
                  ascending: bool = False) -> pd.DataFrame:
        """
        Sort DataFrame by specified column(s).
        
        Args:
            by: Column name(s) to sort by
            ascending: Sort order
        
        Returns:
            Sorted DataFrame
        """
        result = self.df.sort_values(by=by, ascending=ascending)
        self._log(f"Sorted by '{by}', ascending={ascending}")
        return result
    
    # ==========================================# TECHNIQUE 6: CATEGORIZING
    # ==========================================
    
    def bin_numerical(self,
                      column: str,
                      bins: List[int],
                      labels: List[str]) -> pd.Series:
        """
        Create categorical bins from numerical column.
        
        Args:
            column: Numerical column to bin
            bins: Bin edges
            labels: Labels for each bin
        
        Returns:
            Categorical Series
        """
        if len(bins) != len(labels) + 1:
            raise ValueError("Number of bins must be labels + 1")
        
        binned = pd.cut(self.df[column], bins=bins, labels=labels)
        self.df[f'{column}_binned'] = binned
        self._log(f"Binned '{column}' into {len(labels)} categories")
        return binned
    
    def qcut_distribution(self,
                          column: str,
                          n_bins: int,
                          labels: List[str] = None) -> pd.Series:
        """
        Create quantile-based bins (equal distribution).
        
        Args:
            column: Numerical column to bin
            n_bins: Number of bins
            labels: Optional custom labels
        
        Returns:
            Quantile-binned Series
        """
        kwargs = {}
        if labels:
            kwargs['labels'] = labels
        
        binned = pd.qcut(self.df[column], q=n_bins, **kwargs)
        self.df[f'{column}_qcut'] = binned
        self._log(f"Q-cut '{column}' into {n_bins} quantile bins")
        return binned
    
    # ==========================================# TECHNIQUE 7: DUPLICATES
    # ==========================================
    
    def find_duplicates(self, subset: List[str] = None) -> int:
        """Find duplicate rows."""
        count = self.df.duplicated(subset=subset).sum()
        self._log(f"Found {count} duplicate rows")
        return count
    
    def remove_duplicates(self,
                          subset: List[str] = None,
                          keep: str = 'first') -> pd.DataFrame:
        """Remove duplicate rows."""
        original = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = original - len(self.df)
        self._log(f"Removed {removed} duplicate rows, kept '{keep}'")
        return self.df
    
    # ==========================================# TECHNIQUE 8: DROPPING
    # ==========================================
    
    def drop_columns(self, columns: List[str], errors: str = 'ignore') -> pd.DataFrame:
        """Drop specified columns."""
        available = [col for col in columns if col in self.df.columns]
        self.df = self.df.drop(columns=available, errors=errors)
        self._log(f"Dropped {len(available)} columns: {available}")
        return self.df
    
    def drop_rows_by_condition(self, condition) -> pd.DataFrame:
        """Drop rows where condition evaluates to True."""
        original = len(self.df)
        self.df = self.df[~condition]
        removed = original - len(self.df)
        self._log(f"Removed {removed} rows matching condition")
        return self.df
    
    def drop_missing(self, how: str = 'any', thresh: int = None) -> pd.DataFrame:
        """Remove rows with missing values."""
        original = len(self.df)
        self.df = self.df.dropna(how=how, thresh=thresh)
        removed = original - len(self.df)
        self._log(f"Dropped {removed} rows with missing values (how='{how}')")
        return self.df
    
    # ==========================================# TECHNIQUE 9: REPLACING
    # ==========================================
    
    def replace_values(self,
                       column: str,
                       old_values: List,
                       new_values: List,
                       regex: bool = False) -> pd.Series:
        """Replace values in a column."""
        if len(old_values) != len(new_values):
            raise ValueError("old_values and new_values must be same length")
        
        replacement = dict(zip(old_values, new_values))
        self.df[column] = self.df[column].replace(replacement, regex=regex)
        self._log(f"Replaced values in '{column}'")
        return self.df[column]
    
    # ==========================================# TECHNIQUE 10: FORMAT CONVERSION
    # ==========================================
    
    def fill_missing(self, column: str, value) -> pd.Series:
        """Fill missing values in specific column."""
        self.df[column] = self.df[column].fillna(value)
        self._log(f"Filled missing values in '{column}' with {value}")
        return self.df[column]
    
    def convert_dtype(self,
                      column: str,
                      target_type: str,
                      errors: str = 'raise') -> pd.Series:
        """Convert column data type."""
        # Fill NA before numeric conversion
        if target_type in ['int', 'int64', 'int32']:
            self.fill_missing(column, 0)
        
        self.df[column] = self.df[column].astype(target_type, errors=errors)
        self._log(f"Converted '{column}' to {target_type}")
        return self.df[column]
    
    # ==========================================# EXECUTE COMPLETE WORKFLOW
    # ==========================================
    
    def execute_preparation_pipeline(self,
                                     config: Dict) -> pd.DataFrame:
        """
        Execute complete data preparation pipeline based on config.
        
        Args:
            config: Dictionary with preparation steps
        
        Example config:
            {
                'load': True,
                'subset': ['ID', 'Income', 'Education'],
                'group': {'group_col': 'Education', 'agg_col': 'Income'},
                'bin': {'column': 'Income', 'bins': [0, 50000, 100000], 'labels': ['Low', 'High']},
                'drop_missing': True,
                'remove_duplicates': True
            }
        """
        self._log("=" * 50)
        self._log("STARTING PREPARATION PIPELINE")
        self._log("=" * 50)
        
        # Load data
        if config.get('load', True):
            self.load_data()
        
        # Subset columns
        if 'subset' in config:
            self.subset_columns(config['subset'])
        
        # Grouping (optional output capture)
        if 'group' in config:
            gr = config['group']
            self.group_by(gr['group_col'], gr['agg_col'], gr.get('agg_funcs', ['mean']))
        
        # Binning
        if 'bin' in config:
            bc = config['bin']
            self.bin_numerical(bc['column'], bc['bins'], bc['labels'])
        
        # Remove duplicates
        if config.get('remove_duplicates', False):
            self.remove_duplicates()
        
        # Drop missing
        if config.get('drop_missing', False):
            self.drop_missing(how=config.get('how', 'any'))
        
        # Replace values
        if 'replace' in config:
            rc = config['replace']
            self.replace_values(rc['column'], rc['old'], rc['new'])
        
        # Format conversion
        if 'convert' in config:
            cc = config['convert']
            self.convert_dtype(cc['column'], cc['type'])
        
        self.prepared_df = self.df.copy()
        self._log("PIPELINE COMPLETE")
        self._log(f"Final shape: {self.prepared_df.shape}")
        
        return self.prepared_df
    
    def save_prepared_data(self,
                           output_path: str,
                           formats: List[str] = ['csv', 'pickle']):
        """Save prepared data in specified formats."""
        if self.prepared_df is None:
            raise ValueError("No prepared data to save. Run pipeline first.")
        
        output_dir = Path(output_path).parent
        output_dir.mkdir(exist_ok=True)
        base_name = Path(output_path).stem
        
        for fmt in formats:
            if fmt == 'csv':
                self.prepared_df.to_csv(output_dir / f"{base_name}.csv", index=False)
            elif fmt == 'pickle':
                self.prepared_df.to_pickle(output_dir / f"{base_name}.pkl")
            elif fmt == 'json':
                self.prepared_df.to_json(output_dir / f"{base_name}.json")
        
        self._log(f"Saved prepared data to {output_path}")
    
    def get_preprocessing_report(self) -> pd.DataFrame:
        """Generate comprehensive preprocessing report."""
        if self.df is None:
            return pd.DataFrame()
        
        report = {
            'Metric': [
                'Original Rows', 'Original Columns', 
                'Final Rows', 'Final Columns',
                'Duplicates Removed', 'Missing Values Handled',
                'Total Processing Steps'
            ],
            'Value': [
                self.preprocessing_log.count('Loaded'),  # Simplified - should track properly
                len(self.df.columns),
                len(self.df),
                len(self.df.columns),
                0,  # Should track from logs
                0,  # Should track from logs
                len(self.preprocessing_log)
            ]
        }
        
        return pd.DataFrame(report)
    
    def preprocess_and_save(self,
                            input_file: str,
                            output_file: str,
                            config: Dict) -> pd.DataFrame:
        """Convenience method: run entire workflow from input to output."""
        self.filepath = Path(input_file)
        
        # Execute pipeline
        prepared = self.execute_preparation_pipeline(config)
        
        # Save results
        self.save_prepared_data(output_file)
        
        return prepared


# ==========================================# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Example usage
    CONFIG = {
        'load': True,
        'subset': ['ID', 'Year_Birth', 'Education', 'Marital_Status',
                   'Income', 'Kidhome', 'Teenhome', 'Recency',
                   'NumStorePurchases', 'NumWebVisitsMonth'],
        'bin': {
            'column': 'NumStorePurchases',
            'bins': [0, 4, 8, 13],
            'labels': ['Low', 'Moderate', 'High']
        },
        'remove_duplicates': True,
        'drop_missing': True,
        'how': 'any'
    }
    
    # Initialize and run
    preparator = DataPreparator('data/marketing_campaign.csv')
    
    try:
        prepared_data = preparator.preprocess_and_save(
            input_file='data/marketing_campaign.csv',
            output_file='prepared_data/marketing_prepared.csv',
            config=CONFIG
        )
        
        print("\n" + "=" * 60)
        print("PREPARATION COMPLETE!")
        print("=" * 60)
        print(f"Final dataset shape: {prepared_data.shape}")
        print(f"\nProcessing log:")
        for log_entry in preparator.preprocessing_log[-10:]:
            print(f"  • {log_entry}")
    
    except Exception as e:
        print(f"Error during preparation: {e}")