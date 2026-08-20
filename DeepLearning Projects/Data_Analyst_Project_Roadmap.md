# Strategy Roadmap: End-to-End Data Analyst Projects
*(Based on the pattern used in the NYC Restaurant Inspections capstone)*

This roadmap generalizes to **any** project where you combine multiple messy data sources, clean them, store them, query them, and turn the results into a visual story.

---

## The 7-Stage Pattern

```
1. FRAME THE QUESTION
        ↓
2. ACQUIRE DATA (scrape / API / files)
        ↓
3. CLEAN & STANDARDIZE
        ↓
4. VALIDATE (data quality checks)
        ↓
5. STORE (SQL database)
        ↓
6. QUERY & AGGREGATE (SQL)
        ↓
7. VISUALIZE & COMMUNICATE
```

Treat this as a checklist you re-run on every new dataset, regardless of domain (restaurants, real estate, sports, finance, etc.).

---

## Stage 1 — Frame the Question

Before touching data, write down:
- Who is the audience? (client, exec, yourself)
- What decision will this analysis inform?
- What are 2–3 concrete questions you need to answer? (e.g., "Does cuisine type predict inspection failure?")

**Skill to build:** turning a vague business ask into a measurable question.

---

## Stage 2 — Acquire Data

You'll typically combine **at least two sources** so you have something to join/compare.

| Source type | Tool | When to use |
|---|---|---|
| HTML table on a page | `pd.read_html(url)` | Static table, no login needed |
| Full webpage scrape | `BeautifulSoup` + `requests` | Data isn't in a clean `<table>` |
| REST API | `requests.get(url, params=...)` | Structured data, needs auth/pagination |
| Local files | `pd.read_csv`, `pd.read_excel`, `pd.read_json` | You already have the files |

**Pagination pattern to memorize:**
```python
all_data = []
offset = 0
limit = 1000
for _ in range(n_pages):
    params = {"api_key": API_KEY, "limit": limit, "offset": offset}
    r = requests.get(url, params=params)
    if r.status_code == 200:
        all_data.append(pd.DataFrame(r.json()["data"]))
    offset += limit
df = pd.concat(all_data, ignore_index=True)
```

**Skill to build:** reading API docs, secrets via `.env` + `python-dotenv`, defensive checks on `status_code`.

---

## Stage 3 — Clean & Standardize

Checklist to run on every new dataset:
1. **Text fields:** uppercase/lowercase consistently, strip special characters with regex, collapse double spaces.
2. **Dates:** convert every date-like column with `pd.to_datetime()`.
3. **Categorical → numeric:** map ordinal categories (like `$$$` price tiers) to numbers explicitly, in order.
4. **Extract sub-fields:** use `.str.extract(regex)` to pull structured pieces (zip codes, IDs) out of free text.
5. **Dtypes:** cast to the *correct* final dtype (`Int64` for nullable integers, `category` for repeated strings, `datetime64` for dates).
6. **Duplicates:** `.drop_duplicates()` — especially important when data was collected via multiple paginated queries.

**Skill to build:** regex, pandas string accessor (`.str`), pandas dtype system (especially nullable `Int64`/`boolean` for data with missing values).

---

## Stage 4 — Validate (Data Quality)

Ask: "Does every row make physical/business sense?"
- Range checks (e.g., valid zip code ranges, valid dates, valid score ranges)
- Cross-source sanity checks (do join keys actually match between tables?)
- Missing value audit: `df.isna().sum()`

**Skill to build:** thinking like a skeptic about your own data before you analyze it.

---

## Stage 5 — Store in SQL

Even for modest-sized data, SQLite is a fast, dependency-free way to enable proper querying:

```python
import sqlite3
conn = sqlite3.connect("project_data.db")
df_clean.to_sql("table_name", conn, if_exists="replace", index=False)
```

**Skill to build:** knowing *when* SQL beats pandas (large joins, aggregations, sharing a dataset as a portable file).

---

## Stage 6 — Query & Aggregate with SQL

Core patterns you'll reuse constantly:

```sql
-- Group + aggregate
SELECT category, COUNT(*) AS n, AVG(metric) AS avg_metric
FROM table
GROUP BY category
ORDER BY avg_metric;

-- Join two tables on multiple keys
SELECT DISTINCT a.col1, b.col2
FROM table_a a
JOIN table_b b ON a.key1 = b.key1 AND a.key2 = b.key2
WHERE a.col1 IS NOT NULL;

-- Conditional aggregation (rate/percentage)
SELECT category,
       COUNT(*) AS total,
       AVG(CASE WHEN metric <= threshold THEN 1 ELSE 0 END) * 100 AS pass_rate
FROM table
GROUP BY category;
```

**Skill to build:** `GROUP BY` + aggregate functions, multi-key `JOIN`, `CASE WHEN` for rates/buckets.

---

## Stage 7 — Visualize & Communicate

Match the chart to the question:

| Question shape | Chart |
|---|---|
| Compare a metric across categories | `sns.barplot` |
| Show spread/outliers across groups | `sns.boxplot` |
| Show geographic patterns | `geopandas` choropleth |
| Show a trend over time | line plot |
| Show relationship between two numeric vars | scatterplot |

End every project with a **plain-language summary**: 3–5 bullet points a non-technical stakeholder could act on.

---

## Meta-Learning Path (if you want to get systematically better at this pattern)

1. **Python + pandas fundamentals** — filtering, grouping, merging, string methods
2. **Regex** — enough to clean text and extract fields
3. **APIs** — reading docs, auth, pagination, error handling
4. **SQL** — `SELECT`, `GROUP BY`, `JOIN`, `CASE WHEN`, subqueries
5. **sqlite3 / SQLAlchemy** — moving between pandas and SQL
6. **matplotlib/seaborn** — the ~6 plot types that cover 90% of use cases
7. **geopandas** (optional) — only if your project has a spatial dimension
8. **Storytelling** — translating charts into a recommendation

Work through them roughly in this order, but the fastest way to learn is exactly what this capstone does: pick a real messy dataset and force yourself through all 7 stages.
