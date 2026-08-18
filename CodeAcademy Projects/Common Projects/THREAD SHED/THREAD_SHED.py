"""
============================================================
  THREAD SHED — Daily Sales Analyzer (Extended Edition)
============================================================

You've recently been hired as a cashier at the local sewing
hobby shop, Thread Shed. Your register system stores every
transaction in a single unwieldy string. This script parses,
cleans, and analyses that data end-to-end.

Extended features beyond the original project:
  • Transaction objects (dicts) for cleaner access
 • Per-customer sales aggregation
 • Colour frequency analysis
 • Top spenders ranking
 • Average sale calculation
 • Multi-colour vs single-colour purchase breakdown
  • Formatted terminal report
 • Easily extensible data pipeline
"""

# ------------------------------------------------------------------
# 0. RAW DATA
# ------------------------------------------------------------------
daily_sales = \
"""Edith Mcbride   ;,;$1.21   ;,;   white ;,; 
09/15/17   ,Herbert Tran   ;,;   $7.29;,; 
white&blue;,;   09/15/17 ,Paul Clarke ;,;$12.52 
;,;   white&blue ;,; 09/15/17 ,Lucille Caldwell   
;,;   $5.13   ;,; white   ;,; 09/15/17,
Eduardo George   ;,;$20.39;,; white&yellow 
;,;09/15/17   ,   Danny Mclaughlin;,;$30.82;,;   
purple ;,;09/15/17 ,Stacy Vargas;,; $1.85   ;,; 
purple&yellow ;,;09/15/17,   Shaun Brock;,; 
$17.98;,;purple&yellow ;,; 09/15/17 , 
Erick Harper ;,;$17.41;,; blue ;,; 09/15/17, 
Michelle Howell ;,;$28.59;,; blue;,;   09/15/17   , 
Carroll Boyd;,; $14.51;,;   purple&blue   ;,;   
09/15/17   , Teresa Carter   ;,; $19.64 ;,; 
white;,;09/15/17   ,   Jacob Kennedy ;,; $11.40   
;,; white&red   ;,; 09/15/17, Craig Chambers;,; 
$8.79 ;,; white&blue&red   ;,;09/15/17   , Peggy Bell;,; $8.65 ;,;blue   ;,; 09/15/17,   Kenneth Cunningham ;,;   $10.53;,;   green&blue   ;,; 
09/15/17   ,   Marvin Morgan;,;   $16.49;,; 
green&blue&red   ;,;   09/15/17 ,Marjorie Russell 
;,; $6.55 ;,;   green&blue&red;,;   09/15/17 ,
Israel Cummings;,;   $11.86   ;,;black;,;  
09/15/17,   June Doyle   ;,;   $22.29 ;,;  
black&yellow ;,;09/15/17 , Jaime Buchanan   ;,;   
$8.35;,;   white&black&yellow   ;,;   09/15/17,   
Rhonda Farmer;,;$2.91 ;,;   white&black&yellow   
;,;09/15/17, Darren Mckenzie ;,;$22.94;,;green 
;,;09/15/17,Rufus Malone;,;$4.70   ;,; green&yellow 
;,; 09/15/17   ,Hubert Miles;,;   $3.59   
;,;green&yellow&blue;,;   09/15/17   , Joseph Bridges  ;,;$5.66   ;,; green&yellow&purple&blue 
;,;   09/15/17 , Sergio Murphy   ;,;$17.51   ;,;   
black   ;,;   09/15/17 , Audrey Ferguson ;,; 
$5.54;,;black&blue   ;,;09/15/17 ,Edna Williams ;,; 
$17.13;,; black&blue;,;   09/15/17,   Randy Fleming;,;   $21.13 ;,;black ;,;09/15/17 ,Elisa Hart;,; $0.35   ;,; black&purple;,;   09/15/17   ,
Ernesto Hunt ;,; $13.91   ;,;   black&purple ;,;   
09/15/17,   Shannon Chavez   ;,;$19.26   ;,; 
yellow;,; 09/15/17   , Sammy Cain;,; $5.45;,;   
yellow&red ;,;09/15/17 ,   Steven Reeves ;,;$5.50   
;,;   yellow;,;   09/15/17, Ruben Jones   ;,; 
$14.56 ;,;   yellow&blue;,;09/15/17 , Essie Hansen;,;   $7.33   ;,;   yellow&blue&red
;,; 09/15/17   ,   Rene Hardy   ;,; $20.22   ;,; 
black ;,;   09/15/17 ,   Lucy Snyder   ;,; $8.67   
;,;black&red  ;,; 09/15/17 ,Dallas Obrien ;,;   
$8.31;,;   black&red ;,;   09/15/17,   Stacey Payne 
;,;   $15.70   ;,;   white&black&red ;,;09/15/17   
,   Tanya Cox   ;,;   $6.74   ;,;yellow   ;,; 
09/15/17 , Melody Moran ;,;   $30.84   
;,;yellow&black;,;   09/15/17 , Louise Becker   ;,; 
$12.31 ;,; green&yellow&black;,;   09/15/17 ,
Ryan Webster;,;$2.94 ;,; yellow ;,; 09/15/17 
,Justin Blake ;,; $22.46   ;,;white&yellow ;,;   
09/15/17,   Beverly Baldwin ;,;   $6.60;,;   
white&yellow&black ;,;09/15/17   ,   Dale Brady   
;,;   $6.27 ;,; yellow   ;,;09/15/17 ,Guadalupe Potter ;,;$21.12   ;,; yellow;,; 09/15/17   , 
Desiree Butler ;,;$2.10   ;,;white;,; 09/15/17  
,Sonja Barnett ;,; $14.22 ;,;white&black;,;   
09/15/17, Angelica Garza;,;$11.60;,;white&black   
;,;   09/15/17   ,   Jamie Welch   ;,; $25.27   ;,; 
white&black&red ;,;09/15/17   ,   Rex Hudson   
;,;$8.26;,;   purple;,; 09/15/17 ,   Nadine Gibbs 
;,;   $30.80 ;,;   purple&yellow   ;,; 09/15/17   , 
Hannah Pratt;,;   $22.61   ;,;   purple&yellow   
;,;09/15/17,Gayle Richards;,;$22.19 ;,; 
green&purple&yellow ;,;09/15/17   ,Stanley Holland 
;,; $7.47   ;,; red ;,; 09/15/17 , Anna Dean;,;$5.49 ;,; yellow&red ;,;   09/15/17   ,
Terrance Saunders ;,;   $23.70  ;,;green&yellow&red 
;,; 09/15/17 ,   Brandi Zimmerman ;,; $26.66 ;,; 
red   ;,;09/15/17 ,Guadalupe Freeman ;,; $25.95;,; 
green&red ;,;   09/15/17   ,Irving Patterson 
;,;$19.55 ;,; green&white&red ;,;   09/15/17 ,Karl Ross;,;   $15.68;,;   white ;,;   09/15/17 , Brandy Cortez ;,;$23.57;,;   white&red   ;,;09/15/17, 
Mamie Riley   ;,;$29.32;,; purple;,;09/15/17 ,Mike Thornton   ;,; $26.44 ;,;   purple   ;,; 09/15/17, 
Jamie Vaughn   ;,; $17.24;,;green ;,; 09/15/17   , 
Noah Day ;,;   $8.49   ;,;green   ;,;09/15/17   
,Josephine Keller ;,;$13.10 ;,;green;,;   09/15/17 ,   Tracey Wolfe;,;$20.39 ;,; red   ;,; 09/15/17 ,
Ignacio Parks;,;$14.70   ;,; white&red ;,;09/15/17 
, Beatrice Newman ;,;$22.45   ;,;white&purple&red 
;,;   09/15/17, Andre Norris   ;,;   $28.46   ;,;   
red;,;   09/15/17 ,   Albert Lewis ;,; $23.89;,;   
black&red;,; 09/15/17,   Javier Bailey   ;,;   
$24.49   ;,; black&red ;,; 09/15/17   , Everett Lyons ;,;$1.81;,;   black&red ;,; 09/15/17 ,   
Abraham Maxwell;,; $6.81   ;,;green;,;   09/15/17   
,   Traci Craig ;,;$0.65;,; green&yellow;,; 
09/15/17 , Jeffrey Jenkins   ;,;$26.45;,; 
green&yellow&blue   ;,;   09/15/17,   Merle Wilson 
;,;   $7.69 ;,; purple;,; 09/15/17,Janis Franklin   
;,;$8.74   ;,; purple&black   ;,;09/15/17 ,  
Leonard Guerrero ;,;   $1.86   ;,;yellow  
;,;09/15/17,Lana Sanchez;,;$14.75   ;,; yellow;,;   
09/15/17   ,Donna Ball ;,; $28.10  ;,; 
yellow&blue;,;   09/15/17   , Terrell Barber   ;,; 
$9.91   ;,; green ;,;09/15/17   ,Jody Flores;,; 
$16.34 ;,; green ;,;   09/15/17,   Daryl Herrera 
;,;$27.57;,; white;,;   09/15/17   , Miguel Mcguire;,;$5.25;,; white&blue   ;,;   09/15/17 ,   
Rogelio Gonzalez;,; $9.51;,;   white&black&blue   
;,;   09/15/17   ,   Lora Hammond ;,;$20.56 ;,; 
green;,;   09/15/17,Owen Ward;,; $21.64   ;,;   
green&yellow;,;09/15/17,Malcolm Morales ;,;   
$24.99   ;,;   green&yellow&black;,; 09/15/17 ,   
Eric Mcdaniel ;,;$29.70;,; green ;,; 09/15/17 
,Madeline Estrada;,;   $15.52;,;green;,;   09/15/17 
, Leticia Manning;,;$15.70 ;,; green&purple;,; 
09/15/17 ,   Mario Wallace ;,; $12.36 ;,;green ;,; 
09/15/17,Lewis Glover;,;   $13.66   ;,;   
green&white;,;09/15/17,   Gail Phelps   ;,;$30.52   
;,; green&white&blue   ;,; 09/15/17 , Myrtle Morris 
;,;   $22.66   ;,; green&white&blue;,;09/15/17"""

# ------------------------------------------------------------------
#  PART 1 — PARSE THE RAW STRING INTO CLEAN STRUCTURES
# ------------------------------------------------------------------

# Step 1: Replace the ;,; artifact so commas won't split transactions
daily_sales_replaced = daily_sales.replace(';,;', '|')

# Step 2: Split into individual transaction strings
daily_transactions = daily_sales_replaced.split(',')

# Step 3: Split each transaction into its data points
daily_transactions_split = []
for transaction in daily_transactions:
    daily_transactions_split.append(transaction.split('|'))

# Step 4: Strip whitespace from every data point
transactions_clean = []
for transaction in daily_transactions_split:
    cleaned = [data_point.strip() for data_point in transaction]
    transactions_clean.append(cleaned)

print("=== Raw parsed transactions (first 3) ===")
for t in transactions_clean[:3]:
    print(t)

# ------------------------------------------------------------------
#  PART 2 — EXTRACT INTO SEPARATE LISTS
# ------------------------------------------------------------------

customers = []
sales = []
thread_sold = []
dates = []                       # EXTENDED — also capture the date

for transaction in transactions_clean:
    # Guard against malformed rows (fewer/more than 4 fields)
    if len(transaction) >= 4:
        name, amount, threads, date = transaction[0], transaction[1], transaction[2], transaction[3]
    else:
        continue
    customers.append(name)
    sales.append(amount)
    thread_sold.append(threads)
    dates.append(date)

print("\n=== First 5 customers ===")
print(customers[:5])
print("\n=== First 5 sales ===")
print(sales[:5])
print("\n=== First 5 thread colours ===")
print(thread_sold[:5])
print("\n=== First 5 dates ===")
print(dates[:5])

# ------------------------------------------------------------------
#  PART 3 — CALCULATE TOTAL SALES
# ------------------------------------------------------------------

total_sales = 0.0
for sale in sales:
    # Remove the '$' sign and convert to float
    amount = float(sale.strip('$'))
    total_sales += amount

print(f"\n{'=' * 50}")
print(f"  TOTAL SALES: ${total_sales:.2f}")
print(f"{'=' * 50}")

# ------------------------------------------------------------------
#  PART 4 — COUNT SALES BY THREAD COLOUR
# ------------------------------------------------------------------

# Step A: Flatten thread_sold into individual colours
thread_sold_split = []
for item in thread_sold:
    if '&' in item:
        for colour in item.split('&'):
            thread_sold_split.append(colour.strip())
    else:
        thread_sold_split.append(item.strip())

# Step B: Generic colour-counting function
def color_count(color):
    """Return how many times *color* appears in thread_sold_split."""
    count = 0
    for c in thread_sold_split:
        if c == color:
            count += 1
    return count

# Step C: Test the function
print(f"\nTest — color_count('white') = {color_count('white')}")

# Step D: Full colour report
colors = ['red', 'yellow', 'green', 'white', 'black', 'blue', 'purple']

print("\n=== Thread Colour Report ===")
for color in colors:
    print("{}: {} threads sold today.".format(color, color_count(color)))

# ==================================================================
#  EXTENDED ANALYSIS (beyond the original project)
# ==================================================================

# ------------------------------------------------------------------
#  E1. Build structured transaction records (list of dicts)
# ------------------------------------------------------------------

transactions = []
for i in range(len(customers)):
    transactions.append({
        'customer': customers[i],
        'amount': float(sales[i].strip('$')),
        'colours_raw': thread_sold[i],
        'colours': [c.strip() for c in thread_sold[i].split('&')],
        'date': dates[i],
        'num_colours': len(thread_sold[i].split('&')),
    })

# ------------------------------------------------------------------
#  E2. Number of transactions & average sale
# ------------------------------------------------------------------

num_transactions = len(transactions)
average_sale = total_sales / num_transactions if num_transactions else 0

print(f"\n{'=' * 50}")
print(f"  EXTENDED REPORT")
print(f"{'=' * 50}")
print(f"  Total transactions : {num_transactions}")
print(f"  Total revenue      : ${total_sales:.2f}")
print(f"  Average sale       : ${average_sale:.2f}")
print(f"  Total threads sold : {len(thread_sold_split)}")

# ------------------------------------------------------------------
#  E3. Per-customer aggregation
# ------------------------------------------------------------------

customer_totals = {}
customer_visits = {}
for t in transactions:
    name = t['customer']
    customer_totals[name] = customer_totals.get(name, 0) + t['amount']
    customer_visits[name] = customer_visits.get(name, 0) + 1

print(f"\n--- Customer Summary (sorted by total spent) ---")
sorted_customers = sorted(customer_totals.items(), key=lambda x: x[1], reverse=True)
for name, spent in sorted_customers:
    visits = customer_visits[name]
    print(f"  {name:<25}  ${spent:>7.2f}  ({visits} visit{'s' if visits > 1 else ''})")

# ------------------------------------------------------------------
#  E4. Top 5 spenders
# ------------------------------------------------------------------

print(f"\n--- Top 5 Spenders ---")
for rank, (name, spent) in enumerate(sorted_customers[:5], 1):
    print(f"  #{rank}  {name:<20}  ${spent:.2f}")

# ------------------------------------------------------------------
#  E5. Smallest & largest single sale
# ------------------------------------------------------------------

amounts = [t['amount'] for t in transactions]
min_sale = min(amounts)
max_sale = max(amounts)
min_customer = transactions[amounts.index(min_sale)]['customer']
max_customer = transactions[amounts.index(max_sale)]['customer']

print(f"\n--- Sale Extremes ---")
print(f"  Smallest sale : ${min_sale:.2f}  ({min_customer})")
print(f"  Largest sale  : ${max_sale:.2f}  ({max_customer})")

# ------------------------------------------------------------------
#  E6. Single-colour vs multi-colour purchases
# ------------------------------------------------------------------

single_colour = sum(1 for t in transactions if t['num_colours'] == 1)
multi_colour = num_transactions - single_colour

print(f"\n--- Purchase Composition ---")
print(f"  Single-colour purchases : {single_colour}")
print(f"  Multi-colour purchases  : {multi_colour}")

# ------------------------------------------------------------------
#  E7. Colour popularity ranking (descending)
# ------------------------------------------------------------------

colour_counts = {c: color_count(c) for c in colors}
# Also catch any colours not in the predefined list
for c in thread_sold_split:
    if c not in colour_counts:
        colour_counts[c] = colour_counts.get(c, 0) + 1

sorted_colours = sorted(colour_counts.items(), key=lambda x: x[1], reverse=True)

print(f"\n--- Colour Popularity (all colours, ranked) ---")
for colour, count in sorted_colours:
    pct = (count / len(thread_sold_split)) * 100
    print(f"  {colour:<10}  {count:>3} threads  ({pct:5.1f}%)")

# ------------------------------------------------------------------
#  E8. Revenue contribution by dominant colour
#     (assigns each transaction's revenue to its first-listed colour)
# ------------------------------------------------------------------

revenue_by_colour = {}
for t in transactions:
    primary = t['colours'][0]
    revenue_by_colour[primary] = revenue_by_colour.get(primary, 0) + t['amount']

print(f"\n--- Revenue by Primary Colour (first listed) ---")
for colour, rev in sorted(revenue_by_colour.items(), key=lambda x: x[1], reverse=True):
    pct = (rev / total_sales) * 100
    print(f"  {colour:<10}  ${rev:>7.2f}  ({pct:5.1f}% of total)")

# ------------------------------------------------------------------
#  E9. Histogram of sale amounts (text-based)
# ------------------------------------------------------------------

print(f"\n--- Sale Amount Distribution ---")
bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35)]
for lo, hi in bins:
    count = sum(1 for a in amounts if lo <= a < hi)
    bar = '#' * count
    print(f"  ${lo:>2}-${hi:<2}: {bar} ({count})")

# ------------------------------------------------------------------
#  E10. Functions for reuse
# ------------------------------------------------------------------

def get_transactions_on_date(date_str):
    """Return all transactions matching a given date string."""
    return [t for t in transactions if t['date'] == date_str]

def get_customer_purchases(customer_name):
    """Return all transactions for a given customer."""
    return [t for t in transactions if t['customer'] == customer_name]

def colour_buyers(colour):
    """Return a list of unique customers who bought a given colour."""
    buyers = set()
    for t in transactions:
        if colour in t['colours']:
            buyers.add(t['customer'])
    return sorted(buyers)

# Quick demo of helper functions
print(f"\n--- Function Demos ---")
print(f"  Buyers of 'purple': {colour_buyers('purple')[:5]} ... ({len(colour_buyers('purple'))} total)")
print(f"  Purchases by 'Guadalupe Potter': {get_customer_purchases('Guadalupe Potter')}")

# ------------------------------------------------------------------
#  FINAL SUMMARY CARD
# ------------------------------------------------------------------

print(f"\n{'=' * 50}")
print(f"  DAILY SUMMARY — Thread Shed")
print(f"{'=' * 50}")
print(f"  Date             : {dates[0]}")
print(f"  Transactions     : {num_transactions}")
print(f"  Unique customers : {len(customer_totals)}")
print(f"  Total revenue    : ${total_sales:.2f}")
print(f"  Average sale     : ${average_sale:.2f}")
print(f"  Threads sold     : {len(thread_sold_split)}")
print(f"  Most popular     : {sorted_colours[0][0]} ({sorted_colours[0][1]} sold)")
print(f"  Least popular    : {sorted_colours[-1][0]} ({sorted_colours[-1][1]} sold)")
print(f"  Top spender      : {sorted_customers[0][0]} (${sorted_customers[0][1]:.2f})")
print(f"{'=' * 50}")