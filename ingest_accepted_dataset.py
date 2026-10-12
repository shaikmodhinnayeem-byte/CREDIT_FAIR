"""
CreditFair ETL: Ingest Accepted LendingClub Dataset
Extracts completed loans from accepted_2007_to_2018Q4.csv (1.238M+ rows),
applies standardized data transformations, and persists to dataset/data.csv.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)
TARGET_DATASET = DATASET_DIR / "data.csv"
BACKUP_DATASET = DATASET_DIR / "data_sample_15k.csv"

# Resolve accepted_2007_to_2018Q4 path
CANDIDATE_PATHS = [
    Path(r"C:\Users\shaik\Downloads\accepted_2007_to_2018Q4.csv\accepted_2007_to_2018Q4.csv"),
    Path(r"C:\Users\shaik\Downloads\accepted_2007_to_2018Q4.csv"),
]

SOURCE_FILE = None
for p in CANDIDATE_PATHS:
    if p.is_file() and p.stat().st_size > 1000:
        SOURCE_FILE = p
        break

if SOURCE_FILE is None:
    # Check directory
    dir_p = Path(r"C:\Users\shaik\Downloads\accepted_2007_to_2018Q4.csv")
    if dir_p.is_dir():
        for f in dir_p.glob("*.csv"):
            if f.stat().st_size > 1000:
                SOURCE_FILE = f
                break

if SOURCE_FILE is None:
    raise FileNotFoundError("Could not find uncompressed accepted_2007_to_2018Q4.csv")

print(f"[ETL] Source dataset resolved: {SOURCE_FILE} ({SOURCE_FILE.stat().st_size / (1024**3):.2f} GB)")

# Backup old sample if not already backed up
if TARGET_DATASET.exists() and not BACKUP_DATASET.exists():
    import shutil
    shutil.copyfile(TARGET_DATASET, BACKUP_DATASET)
    print(f"[ETL] Backed up original sample to {BACKUP_DATASET.name}")


def parse_emp_length(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    if "< 1" in s:
        return 0.0
    if "10+" in s:
        return 10.0
    digits = "".join([c for c in s if c.isdigit()])
    return float(digits) if digits else np.nan


def parse_pct(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace("%", "").strip()
    try:
        return float(s)
    except:
        return np.nan


def run_ingestion():
    t_start = time.time()
    
    needed_cols = [
        "loan_amnt", "term", "int_rate", "installment", "emp_length",
        "home_ownership", "annual_inc", "verification_status", "purpose",
        "addr_state", "dti", "fico_range_low", "fico_range_high",
        "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
        "mort_acc", "pub_rec_bankruptcies", "loan_status"
    ]
    
    valid_statuses = {"Fully Paid", "Charged Off", "Default"}
    chunk_size = 250000
    chunks = []
    total_processed = 0
    total_extracted = 0

    print("[ETL] Reading and streaming chunks...")
    for chunk in pd.read_csv(SOURCE_FILE, usecols=needed_cols, chunksize=chunk_size, low_memory=False):
        total_processed += len(chunk)
        
        # Filter completed loans
        completed = chunk[chunk["loan_status"].isin(valid_statuses)].copy()
        if len(completed) == 0:
            continue
            
        completed["loan_status"] = completed["loan_status"].replace({"Default": "Charged Off"})
        total_extracted += len(completed)
        chunks.append(completed)
        print(f"  Processed {total_processed:,} rows -> Extracted {total_extracted:,} completed loans...")

    print(f"[ETL] Concatenating {len(chunks)} chunks ({total_extracted:,} rows)...")
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    
    print("[ETL] Computing derived features and standardizations...")
    
    # Term months
    df["term_months"] = df["term"].astype(str).str.extract(r"(\d+)")[0].astype(float).fillna(36).astype(int)
    
    # Numeric parsings
    df["int_rate"] = df["int_rate"].apply(parse_pct)
    df["revol_util"] = df["revol_util"].apply(parse_pct)
    df["emp_length_years"] = df["emp_length"].apply(parse_emp_length)
    
    # FICO average score
    df["fico_score"] = ((df["fico_range_low"] + df["fico_range_high"]) / 2.0).round(1)
    
    # Home ownership standardization
    df["home_ownership"] = df["home_ownership"].replace({
        "NONE": "OTHER",
        "ANY": "OTHER"
    }).fillna("OTHER")
    
    # Annual income quantile bracket
    med_inc = df["annual_inc"].median()
    df["annual_inc"] = df["annual_inc"].fillna(med_inc)
    df["income_bracket"] = pd.qcut(
        df["annual_inc"],
        q=4,
        labels=["Low", "Lower-Mid", "Upper-Mid", "High"]
    ).astype(str)
    
    # Column ordering to match exact CreditFair schema
    desired_cols = [
        "loan_amnt", "term", "term_months", "int_rate", "installment",
        "emp_length", "emp_length_years", "home_ownership", "annual_inc",
        "income_bracket", "verification_status", "purpose", "addr_state",
        "dti", "fico_range_low", "fico_range_high", "fico_score",
        "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
        "mort_acc", "pub_rec_bankruptcies", "loan_status"
    ]
    
    df = df[desired_cols]
    
    print(f"[ETL] Writing final cleaned dataset to {TARGET_DATASET}...")
    df.to_csv(TARGET_DATASET, index=False)
    
    file_size_mb = TARGET_DATASET.stat().st_size / (1024 * 1024)
    duration = time.time() - t_start
    print("=" * 80)
    print(f"ETL COMPLETE: Extracted {len(df):,} loans ({file_size_mb:.2f} MB) in {duration:.1f}s")
    print(f"Target Distribution:\n{df['loan_status'].value_counts(normalize=True).mul(100).round(2)}")
    print("=" * 80)
    return df


if __name__ == "__main__":
    run_ingestion()
