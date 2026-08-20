# Data Analyst Project Cheat Sheet
*Copy-paste reference for scrape → clean → store → query → visualize projects*

---

## 1. Imports

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import requests
import os
import re
from dotenv import load_dotenv
```

---

## 2. Data Acquisition

### 2a. Scrape an HTML table
```python
tables = pd.read_html(url)          # returns a LIST of DataFrames
df = tables[0]
```

### 2b. Call a paginated REST API
```python
load_dotenv()
API_KEY = os.getenv("MY_API_KEY")

url = "https://api.example.com/endpoint"
params = {"api_key": API_KEY, "limit": 1000, "offset": 0}

frames = []
for i in range(n_pages):
    params["offset"] = i * 1000
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()["data"]     # adjust key to API's response shape
        frames.append(pd.DataFrame(data))
    else:
        print(f"Failed at offset {params['offset']}: {response.status_code}")

df = pd.concat(frames, ignore_index=True)
```

---

## 3. Cleaning

### Text standardization
```python
special_char_pattern = r"[^A-Z0-9 ]"

df["name"] = df["name"].str.upper()
df["name"] = df["name"].str.replace(special_char_pattern, "", regex=True)
df["name"] = df["name"].str.replace("  ", " ")   # collapse double spaces
```

### Dates
```python
df["date_col"] = pd.to_datetime(df["date_col"])
```

### Split / parse compound text fields
```python
df["part2"] = df["compound_col"].str.split("/").str[1].str.strip()
```

### Map ordinal categories to numbers (e.g. "$$$" price levels)
```python
possible_levels = ["$$$$$", "$$$$", "$$$", "$$", "$", " - No ratings yet"]
for i, level in enumerate(possible_levels):
    df["price_level"] = df["price_level"].str.replace(level, str(5 - i))
```

### Extract structured data with regex
```python
df["zip_code"] = df["address"].str.extract(r"(\d{5})")
```

### Correct dtypes (use nullable Int64 when NaNs are present)
```python
df["price_level"] = df["price_level"].astype("Int64")
df["zip_code"] = df["zip_code"].astype("Int64")
```

### Duplicates
```python
df = df.drop_duplicates()
```

---

## 4. Data Quality / Validation

```python
# range check
df = df[(df["zip_code"] >= 10001) & (df["zip_code"] <= 11697)]

# missing value audit
df.isna().sum()

# cross-source sanity check
set(df_a["key"]) - set(df_b["key"])   # keys in A but not B
```

---

## 5. Store in SQLite

```python
conn = sqlite3.connect("project_data.db")

df_clean.to_sql("table_name", conn, if_exists="replace", index=False)

# always close when done
conn.close()
```

### Quick read-back test
```python
conn = sqlite3.connect("project_data.db")
pd.read_sql_query("SELECT * FROM table_name LIMIT 5", conn)
```

---

## 6. SQL Query Patterns

### Group + count + average
```sql
SELECT category,
       COUNT(*) AS total_records,
       AVG(metric) AS avg_metric
FROM table_name
GROUP BY category
ORDER BY avg_metric;
```

### Join two tables on multiple keys
```sql
SELECT DISTINCT a.col1, a.col2, b.col3
FROM table_a a
JOIN table_b b
  ON a.key1 = b.key1 AND a.key2 = b.key2
WHERE a.col1 IS NOT NULL
ORDER BY a.col2 DESC;
```

### Conditional aggregation (pass rate / percentage)
```sql
SELECT category,
       COUNT(*) AS total,
       AVG(CASE WHEN score <= 13 THEN 1 ELSE 0 END) * 100 AS pass_rate,
       AVG(score) AS avg_score
FROM table_name
GROUP BY category
ORDER BY pass_rate DESC;
```

### Run from Python
```python
query = """
SELECT ...
"""
result_df = pd.read_sql_query(query, conn)
```

---

## 7. Visualization

### Bar plot (compare category averages)
```python
sns.barplot(data=df, x="avg_metric", y="category")
plt.title("Metric by Category")
plt.show()
```

### Boxplot (spread across groups)
```python
sns.boxplot(data=df, x="group_col", y="value_col")
plt.title("Distribution by Group")
plt.show()
```

### Custom color-coded categories (e.g. grade-based coloring)
```python
def assign_grade(score):
    if score < 14: return "A"
    elif score <= 27: return "B"
    else: return "C"

df["grade"] = df["avg_score"].apply(assign_grade)
palette = {"A": "green", "B": "yellow", "C": "red"}
sns.barplot(data=df, x="avg_score", y="category", hue="grade", palette=palette, dodge=False)
```

### Choropleth map (geographic pattern)
```python
import geopandas as gpd

shapefile = gpd.read_file("path/to/shapefile.shp")
shapefile.rename(columns={"ZCTA5CE20": "zip_code"}, inplace=True)
shapefile["zip_code"] = pd.to_numeric(shapefile["zip_code"])

merged = shapefile.merge(df, on="zip_code", how="left")
merged.plot(column="metric", cmap="RdBu", legend=True)
plt.title("Metric by Zip Code")
plt.show()
```

### Rotate x-axis labels for long category names
```python
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
```

---

## Quick Reference: Common Gotchas

| Symptom | Likely fix |
|---|---|
| `pd.read_html` returns multiple tables | Index into the list: `tables[0]`, inspect each with `.head()` |
| API returns 200 but empty data | Check the actual JSON key holding records — don't assume `"data"` |
| `.astype(int)` fails on column with NaN | Use `.astype("Int64")` (capital I = nullable) instead |
| SQL `JOIN` returns fewer rows than expected | Check both join keys are the *same dtype* (e.g., both int, not int vs. str) |
| Regex extract returns all NaN | Test the pattern separately with `re.search()` on one sample string first |
| Choropleth map is blank/wrong colors | Confirm the merge key dtype matches on both sides before merging |
