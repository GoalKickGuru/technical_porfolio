# Cheat Sheet: Explore the 1985 Cars Dataset

A reference for every R / tidyverse command you need for this project,
grouped by task.

## Setup

| Goal | Command |
|---|---|
| Load a package | `library(readr)` |
| Load multiple packages | `library(readr)` then `library(dplyr)` |
| Read a CSV into a dataframe | `cars <- read_csv("cars85.csv")` |

`readr::read_csv()` is preferred over base R's `read.csv()` — it's faster,
keeps strings as strings (not factors), and prints a column-type summary.

## Inspecting Data

| Goal | Command |
|---|---|
| Preview first 6 rows | `head(cars)` |
| Preview first N rows | `head(cars, 10)` |
| Column-by-column stats (min/max/quartiles/counts) | `summary(cars)` |
| List column names | `colnames(cars)` |
| Dataframe dimensions | `dim(cars)` |
| Structure / types of each column | `str(cars)` |

## Selecting / Dropping Columns

| Goal | Command |
|---|---|
| Keep only certain columns | `select(cars, make, price)` |
| Drop one column | `select(cars, -normalized_losses)` |
| Drop multiple columns | `select(cars, -c(col1, col2))` |
| Pipe version | `cars <- cars %>% select(-normalized_losses)` |

The `-` sign in `select()` means "everything except this."

## Renaming Columns

| Goal | Command |
|---|---|
| Rename a column | `rename(cars, new_name = old_name)` |
| Pipe version | `cars <- cars %>% rename(risk_factor = symboling)` |

`rename()` syntax is always `new_name = old_name` (new name goes first).

## Adding / Transforming Columns

| Goal | Command |
|---|---|
| Add a new column from a calculation | `mutate(cars, new_col = some_expression)` |
| Example: distance from a threshold | `cars <- cars %>% mutate(mpg_diff_from_threshold = highway_mpg - mpg_threshold)` |
| Add multiple columns at once | `mutate(cars, a = x + 1, b = y * 2)` |

`mutate()` adds a new column while keeping all existing ones. It's
vectorized, so `highway_mpg - mpg_threshold` is computed row by row
automatically.

## Filtering Rows

| Goal | Command |
|---|---|
| Keep rows matching a condition | `filter(cars, mpg_diff_from_threshold > 0)` |
| Filter on equality | `filter(cars, make == "volvo")` |
| Filter on multiple conditions (AND) | `filter(cars, make == "volvo", price < 20000)` |
| Filter on multiple conditions (OR) | `filter(cars, make == "volvo" \| make == "audi")` |

Use `==` for equality (not `=`), and always quote string values:
`make == "audi"`, not `make == audi`.

## Arranging / Sorting Rows

| Goal | Command |
|---|---|
| Sort ascending (smallest first) | `arrange(cars, engine_size)` |
| Sort descending (largest first) | `arrange(cars, desc(engine_size))` |
| Sort by multiple columns | `arrange(cars, desc(make), price)` |

## Chaining Steps with the Pipe (`%>%`)

The pipe operator passes the result on its left into the first argument of
the function on its right. This lets you chain several dplyr verbs
readably:

```r
mpg_exceeds_threshold <- cars %>%
  filter(mpg_diff_from_threshold > 0) %>%
  arrange(desc(mpg_diff_from_threshold))
```

is equivalent to the more nested (and harder-to-read):

```r
mpg_exceeds_threshold <- arrange(
  filter(cars, mpg_diff_from_threshold > 0),
  desc(mpg_diff_from_threshold)
)
```

## Viewing a Dataframe

| Goal | Command |
|---|---|
| Print in console / notebook | just type the dataframe's name, e.g. `cars` |
| Open in spreadsheet-style viewer (RStudio) | `View(cars)` |

## Full Task-to-Command Map

| Task | Command(s) |
|---|---|
| 1. Load libraries | `library(readr)`, `library(dplyr)` |
| 2. Load data | `cars <- read_csv("cars85.csv")` |
| 3. Inspect data | `head(cars)`, `summary(cars)` |
| 4. Drop `normalized_losses` | `cars <- cars %>% select(-normalized_losses)` |
| 5. Print column names | `colnames(cars)` |
| 6. Rename `symboling` | `cars <- cars %>% rename(risk_factor = symboling)` |
| 7. Set threshold | `mpg_threshold <- 30` |
| 8. Add diff column | `cars <- cars %>% mutate(mpg_diff_from_threshold = highway_mpg - mpg_threshold)` |
| 9. Filter above threshold | `mpg_exceeds_threshold <- cars %>% filter(mpg_diff_from_threshold > 0)` |
| 10. Sort by diff desc | `mpg_exceeds_threshold <- mpg_exceeds_threshold %>% arrange(desc(mpg_diff_from_threshold))` |
| 11. Sort by engine size desc | `ordered_by_engine_size <- cars %>% arrange(desc(engine_size))` |
| 12. Choose a make | `chosen_make <- "volvo"` |
| 13. Filter to that make | `chosen_make_details <- cars %>% filter(make == chosen_make)` |
| 14. Sort that make by engine size | `chosen_make_details <- chosen_make_details %>% arrange(desc(engine_size))` |

## Valid Values for `make` in This Dataset

`alfa-romero`, `audi`, `bmw`, `chevrolet`, `dodge`, `honda`, `isuzu`,
`jaguar`, `mazda`, `mercedes-benz`, `mercury`, `mitsubishi`, `nissan`,
`peugot`, `plymouth`, `porsche`, `renault`, `saab`, `subaru`, `toyota`,
`volkswagen`, `volvo`

## Common Pitfalls

- **`select()` with a minus sign** only works on column *names*, not
  values — `select(cars, -normalized_losses)` drops the column, it does
  not filter rows.
- **`filter()` vs `select()`** — `filter()` picks *rows*, `select()` picks
  *columns*. It's easy to reach for the wrong one.
- **`=` vs `==`** — inside `mutate()`/`select()`/`rename()` you use a
  single `=` to *name* something; inside `filter()` you use `==` to *test*
  equality.
- **Quoting strings** — `chosen_make <- "volvo"` needs quotes; without
  them R looks for an object named `volvo` and errors.
- **Overwriting vs. new variable** — tasks that say "save your new
  dataframe to `cars`" mean overwrite `cars <- cars %>% ...`; tasks that
  say "save to `mpg_exceeds_threshold`" mean create a *new* object.
