"""
Data Cleaning Cookbook: Selecting Rows
--------------------------------------
Extended from 'Python Data Cleaning Cookbook' (Chapter 3: Selecting Rows).
Includes standard slicing, indexing, boolean indexing, as well as modern
chaining (.query), loc/iloc best practices, and enterprise-grade PySpark methods.
"""

import os
import pandas as pd
import numpy as np


def load_nls_data(filepath: str) -> pd.DataFrame:
    """Loads NLS97 data and sets the primary key index."""
    if not os.path.exists(filepath):
        # Create synthetic data fallback matching the schema for standalone testing
        print(f"[INFO] File '{filepath}' not found. Generating synthetic NLS DataFrame...")
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'personid': np.arange(100000, 100000 + n),
            'gender': np.random.choice(['Male', 'Female'], n),
            'birthmonth': np.random.randint(1, 13, n),
            'birthyear': np.random.randint(1980, 1985, n),
            'highestgradecompleted': np.random.choice([np.nan, 12, 13, 14, 16, 17], n),
            'maritalstatus': np.random.choice(['Never-married', 'Married', np.nan], n),
            'nightlyhrssleep': np.random.choice([3, 4, 5, 6, 7, 8, np.nan], n, p=[0.05, 0.05, 0.1, 0.3, 0.3, 0.15, 0.05]),
            'childathome': np.random.randint(0, 6, n),
            'colenroct15': np.random.choice(['1. Not enrolled', '2. Enrolled'], n)
        })
        df.set_index('personid', inplace=True)
        return df

    df = pd.read_csv(filepath)
    df.set_index("personid", inplace=True)
    return df


def demonstrate_basic_slicing(df: pd.DataFrame) -> None:
    """Basic positional slicing using Python bracket operators."""
    print("\n=== 1. Basic Positional Slicing ===")
    # 1. Start at 1001st row (idx 1000) to 1004th row
    print("Rows 1000 to 1004 (Transposed):")
    print(df[1000:1004].T)

    # 2. Slicing with a step of 2
    print("\nRows 1000 to 1004 with step=2:")
    print(df[1000:1004:2].T)

    # 3. First and last 3 rows using bracket operator vs head/tail
    print("\nHead(3) vs Bracket [:3]:")
    print(df[:3].T)
    
    print("\nTail(3) vs Bracket [-3:]:")
    print(df[-3:].T)


def demonstrate_loc_and_iloc(df: pd.DataFrame) -> None:
    """Label-based (.loc) vs Integer-based (.iloc) selection."""
    print("\n=== 2. Modern Indexing with .loc and .iloc ===")
    
    # Selecting specific row labels
    sample_ids = df.index[:3].tolist()
    print(f"Selecting by explicit index labels using .loc {sample_ids}:")
    print(df.loc[sample_ids].T)

    # Range selection via .loc (inclusive of end index)
    print("\nRange selection via .loc:")
    print(df.loc[sample_ids[0]:sample_ids[-1]].T)

    # Integer positional selection with .iloc
    print("\nFirst 3 rows using .iloc[[0, 1, 2]]:")
    print(df.iloc[[0, 1, 2]].T)

    print("\nLast 3 rows using .iloc[-3:]:")
    print(df.iloc[-3:].T)


def demonstrate_boolean_indexing(df: pd.DataFrame) -> None:
    """Multi-condition boolean indexing and selection[cite: 1]."""
    print("\n=== 3. Boolean Indexing & Conditional Filtering ===")
    
    # Finding low sleep (<= 4 hours)
    low_sleep_mask = df['nightlyhrssleep'] <= 4
    low_sleep_df = df.loc[low_sleep_mask]
    print(f"Total respondents with <= 4 hrs sleep: {len(low_sleep_df)}")

    # Multi-condition: low sleep AND 3+ children at home
    multi_cond = (df['nightlyhrssleep'] <= 4) & (df['childathome'] >= 3)
    complex_subset = df.loc[multi_cond, ['nightlyhrssleep', 'childathome']]
    print(f"\nRespondents with <= 4 hrs sleep AND >= 3 children (Shape: {complex_subset.shape}):")
    print(complex_subset.head())


# =====================================================================
# EXTENSION SECTIONS (NEW ADDITIONS TO THE COOKBOOK)
# =====================================================================

def demonstrate_modern_query_and_eval(df: pd.DataFrame) -> None:
    """EXTENSION: String-based evaluation with .query() for dynamic readability."""
    print("\n=== 4. Modern Method Chaining with .query() (NEW) ===")
    
    # 1. Querying with dynamic variables
    max_hours = 4
    min_kids = 3
    
    # High readability syntax without requiring full DataFrame indexing repetitive calls
    queried_df = df.query("nightlyhrssleep <= @max_hours and childathome >= @min_kids")[
        ['nightlyhrssleep', 'childathome']
    ]
    print(f"Queried subset shape: {queried_df.shape}")
    print(queried_df.head(3))

    # 2. String operations inside query
    married_low_sleep = df.query("maritalstatus == 'Married' and nightlyhrssleep <= 4")
    print(f"Married & Low Sleep count: {len(married_low_sleep)}")


def demonstrate_isin_and_string_matching(df: pd.DataFrame) -> None:
    """EXTENSION: Selection via list membership (.isin) and regex string matching."""
    print("\n=== 5. List Matching (.isin) & Regex Row Filtering (NEW) ===")
    
    # 1. Categorical inclusion with .isin()
    valid_years = [1980, 1981, 1982]
    isin_df = df.loc[df['birthyear'].isin(valid_years)]
    print(f"Rows matching birthyear in {valid_years}: {len(isin_df)}")

    # 2. Regex string filtering on categorical columns (handling NaNs safely)
    enrolled_mask = df['colenroct15'].str.contains('Not enrolled', na=False)
    print(f"Count of non-enrolled individuals (regex match): {enrolled_mask.sum()}")


def demonstrate_pyspark_equivalent() -> None:
    """EXTENSION: Production big-data equivalent using PySpark DataFrame filtering."""
    print("\n=== 6. Enterprise PySpark Equivalent (NEW) ===")
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F

        spark = SparkSession.builder.appName("RowSelectionCookbook").getOrCreate()
        
        # Synthetic Spark DF
        data = [(100061, "Female", 1980, 4, 3), 
                (100139, "Male", 1983, 7, 1), 
                (100284, "Male", 1984, 3, 4)]
        columns = ["personid", "gender", "birthyear", "nightlyhrssleep", "childathome"]
        
        spark_df = spark.createDataFrame(data, columns)
        
        print("PySpark DataFrame Filter Output:")
        spark_df.filter((F.col("nightlyhrssleep") <= 4) & (F.col("childathome") >= 3)).show()
        
        spark.stop()
    except ImportError:
        print("[SKIP] PySpark is not installed in the active environment.")


def main():
    filepath = "data/nls97.csv"
    nls97 = load_nls_data(filepath)
    
    # Original Cookbook Recipes
    demonstrate_basic_slicing(nls97)
    demonstrate_loc_and_iloc(nls97)
    demonstrate_boolean_indexing(nls97)
    
    # Extended Recipes
    demonstrate_modern_query_and_eval(nls97)
    demonstrate_isin_and_string_matching(nls97)
    demonstrate_pyspark_equivalent()


if __name__ == "__main__":
    main()