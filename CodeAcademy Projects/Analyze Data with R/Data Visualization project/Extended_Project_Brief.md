# Citi Bike Extended Analysis Project

## Origin
This project extends Codecademy's "Explore Citi Bike Data" off-platform project.
The original walks through: loading `january_trips_subset.csv`, a spatial heat map
of start stations, engineering `age`, `distance` (Haversine), and `speed`, then
visualizing mean speed by age, by age+gender, and a stacked age/gender bar chart.

## Why extend it
The original project answers one question well ("do younger riders bike faster?")
but stops there. A real analytics engagement for a bike-share operator would keep
going: When do bikes actually move? Which stations run empty? Does membership type
change behavior? Is our speed estimate even trustworthy? The extended project below
turns the original notebook into a small portfolio piece that touches temporal
patterns, station-level operations, data-quality reasoning, and a stakeholder-ready
recommendation — the skills that separate a tutorial exercise from applied work.

## Business framing
You are a data analyst for Citi Bike's operations team. Leadership wants to know:
(1) who rides, (2) when and where demand happens, (3) whether current
infrastructure (stations, dock counts) matches demand, and (4) whether any rider
segment shows a safety-relevant behavior pattern worth a policy response.

## Dataset
- `january_trips_subset.csv` (or full `january_trips.csv`) — same schema as the
  original project: `tripduration`, `starttime`, `stoptime`, station id/name/lat/lon
  for start and end, `bikeid`, `usertype`, `birth.year`, `gender`.
- Optional enrichment (extended sections only, all free/public):
  - NOAA / Visual Crossing daily weather for NYC, January 2020 (temp, precip).
  - Citi Bike "System Data" station status feed (if working with live/recent data).

## Extended scope (modules)
1. **Data audit** — schema check, missing values, duplicate trips, implausible
   values (0-second trips, negative duration, ages > 100, stations with lat/lon
   of 0). Document every assumption before modeling anything.
2. **Core EDA (original project)** — spatial heat map of start stations; engineer
   `age`, `distance` (Haversine), `speed`; mean speed by age; by age × gender;
   stacked age/gender bar chart.
3. **Temporal patterns (new)** — parse `starttime`; derive hour-of-day,
   day-of-week, weekday vs. weekend; plot trip volume by hour, faceted by
   weekday/weekend, to find commute vs. leisure signatures.
4. **User-type comparison (new)** — compare `usertype` (Subscriber vs. Customer)
   on trip duration, speed, and time-of-day profile. Subscribers and casual
   riders usually behave very differently — this is often the highest-value
   finding for a bike-share operator.
5. **Station-level flow / imbalance (new)** — for each station, count departures
   and arrivals; compute net flow (arrivals − departures) to flag stations that
   drain or flood over the day — a classic "rebalancing" operations question.
6. **Distance-method sensitivity (new)** — compare straight-line Haversine
   distance against a scaled "riding distance" proxy (e.g., multiply by a
   detour factor, or bucket trips into short/medium/long and check how the
   age-vs-speed relationship holds up under each assumption). This addresses
   the original project's own caveat about distance accuracy.
7. **Weather overlay (optional/new)** — join a small daily weather table by date;
   check whether ridership or average trip duration correlates with temperature
   or precipitation.
8. **Recommendation memo (new)** — 3/4-page narrative for the operations team:
   what you found, how confident you are, and what data or field you'd get next
   to raise that confidence.

## Deliverables in this bundle
| File | Purpose |
|---|---|
| `Citibike_Extended_Skeleton.ipynb` | Instructions for every step above, empty code cells — you do the work |
| `Citibike_Extended_Solution.ipynb` | Fully worked reference solution |
| `Citibike_Cheatsheet.ipynb` | Every function/pattern used, organized by task, with mini-examples |
| `EDA_Viz_Strategy_Playbook.ipynb` | A general step-by-step method for tackling *any* similar exploratory/visualization dataset |
| `Reusable_EDA_Template.ipynb` | Fill-in-the-blank scaffold you can point at a new CSV in minutes |

## Suggested grading rubric (if using this for teaching)
- Data audit documented before analysis (10%)
- Correct feature engineering: age, distance, speed (20%)
- At least 4 distinct, correctly-labeled visualizations (30%)
- At least one new (non-original) analytical angle explored with a clear finding (25%)
- Written takeaway/recommendation grounded in the actual numbers, including caveats (15%)
