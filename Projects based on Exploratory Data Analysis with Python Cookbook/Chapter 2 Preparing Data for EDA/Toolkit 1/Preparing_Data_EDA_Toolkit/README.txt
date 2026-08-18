Preparing Data for EDA Toolkit
==============================
Extended from Chapter 2 of "Exploratory Data Analysis with Python Cookbook" (Packt).

Contents
--------
scripts/
  data_preparation.py      – Reusable library (group, append, concat, merge, sort,
                             bin, dedupe, drop, dtype, replace, missing values + pipeline)
  run_data_preparation.py  – CLI demo / smoke test

notebooks/
  02_preparing_data_for_eda_extended.ipynb  – Full walkthrough with enhancements

templates/
  data_prep_template.py / .ipynb  – Copy, edit CONFIG, run on your data

reports/
  Preparing_Data_for_EDA_Report.docx  – Features, purpose, how to use every file

Preparing_Data_Toolkit.xlsx  – Google-Sheets-compatible spreadsheet demos

data/
  marketing_campaign*.csv  – Synthetic sample data matching the book schema

Quick start
-----------
  python scripts/run_data_preparation.py
  # or open the notebook / spreadsheet / template

Requirements: Python 3.8+, pandas, numpy
