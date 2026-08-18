#!/usr/bin/env python3
"""
=========================================
EDA ANALYSIS TEMPLATE
Reusable framework for data exploration
=========================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class EDATemplate:
    """
    Template class for reusable EDA workflows.
    Copy this class and customize for each new dataset.
    """
    
    def __init__(self, data_file: str, project_name: str = "Analysis"):
        self.data_file = Path(data_file)
        self.project_name = project_name
        self.df = None
        self.analyzed_at = datetime.now().isoformat()
        
    def load_data(self) -> pd.DataFrame:
        """Load data from file."""
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        self.df = pd.read_csv(self.data_file)
        print(f"✅ Loaded {self.data_file.name}: {self.df.shape}")
        return self.df
    
    def quick_inspect(self) -> Dict:
        """Perform quick data inspection."""
        inspection = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_pct': (self.df.isnull().sum() / len(self.df) * 100).round(2).to_dict(),
            'numerical_cols': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_cols': self.df.select_dtypes(include=['object']).columns.tolist()
        }
        
        print(f"\n📊 Quick Inspection Report for {self.project_name}")
        print("-" * 50)
        print(f"Total rows: {inspection['shape'][0]}")
        print(f"Total columns: {inspection['shape'][1]}")
        print(f"Numerical columns: {len(inspection['numerical_cols'])}")
        print(f"Categorical columns: {len(inspection['categorical_cols'])}")
        
        return inspection
    
    def prepare_basic(self, 
                      columns: List[str] = None,
                      remove_na: bool = True,
                      remove_duplicates: bool = True) -> pd.DataFrame:
        """
        Basic data preparation workflow.
        
        Args:
            columns: List of columns to keep (None = keep all)
            remove_na: Whether to remove rows with missing values
            remove_duplicates: Whether to remove duplicate rows
        """
        df_prep = self.df.copy()
        
        # Select columns if specified
        if columns:
            available = [col for col in columns if col in df_prep.columns]
            df_prep = df_prep[available].copy()
            print(f"✔ Selected {len(available)} columns")
        
        # Remove duplicates
        if remove_duplicates:
            original = len(df_prep)
            df_prep = df_prep.drop_duplicates()
            print(f"✔ Removed {original - len(df_prep)} duplicates")
        
        # Remove missing values
        if remove_na:
            original = len(df_prep)
            df_prep = df_prep.dropna()
            print(f"✔ Removed {original - len(df_prep)} rows with missing values")
        
        self.df = df_prep
        return df_prep
    
    def add_age_feature(self, birth_year_col: str = 'Year_Birth') -> pd.DataFrame:
        """Calculate age from birth year column."""
        if birth_year_col not in self.df.columns:
            print(f"⚠ Column '{birth_year_col}' not found, skipping age calculation")
            return self.df
        
        current_year = datetime.now().year
        self.df['Age'] = current_year - self.df[birth_year_col]
        print(f"✔ Added 'Age' column from '{birth_year_col}'")
        return self.df
    
    def categorize_continuous(self,
                              column: str,
                              bins: List[int],
                              labels: List[str]) -> pd.DataFrame:
        """Create categorical bins from continuous variable."""
        if len(bins) != len(labels) + 1:
            raise ValueError("bins list must have 1 more element than labels")
        
        self.df[f'{column}_category'] = pd.cut(
            self.df[column],
            bins=bins,
            labels=labels
        )
        print(f"✔ Created category '{column}_category' with {len(labels)} bins")
        return self.df
    
    def export_results(self, output_dir: str = "results") -> Dict:
        """Export all results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = self.analyzed_at.replace(':', '-')[:19]
        
        exported = {}
        
        # Export data
        csv_file = output_path / f"{self.project_name}_{timestamp}.csv"
        self.df.to_csv(csv_file, index=False)
        exported['csv'] = csv_file
        
        # Export metadata
        metadata = {
            'project': self.project_name,
            'analyzed_at': self.analyzed_at,
            'data_file': str(self.data_file),
            'final_shape': list(self.df.shape),
            'columns': list(self.df.columns)
        }
        
        meta_file = output_path / f"{self.project_name}_metadata.json"
        with open(meta_file, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        exported['metadata'] = meta_file
        
        print(f"✔ Exported results to {output_path}")
        return exported


# ==========================================
# QUICK START FUNCTION
# ==========================================

def quick_eda_analysis(data_file: str, columns: List[str] = None) -> pd.DataFrame:
    """
    One-liner for quick exploratory data analysis.
    
    Usage:
        df = quick_eda_analysis('my_data.csv', ['col1', 'col2', 'col3'])
    """
    eda = EDATemplate(data_file, "QuickEDAnalysis")
    eda.load_data()
    eda.quick_inspect()
    eda.prepare_basic(columns=columns)
    eda.export_results()
    
    return eda.df


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__