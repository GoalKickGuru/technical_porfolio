import numpy as np, pandas as pd

def make_finviz_like_raw(n=400, seed=42):
    """Generates a synthetic dataset that mimics a raw finviz.com screener CSV export:
    numeric fields are stored as messy strings with %, $, and commas, exactly like the
    real export, so the 'cleaning' step in the lab has real work to do."""
    rng = np.random.default_rng(seed)
    sectors = {
        "Basic Materials": ["Gold","Chemicals","Steel"],
        "Conglomerates": ["Diversified"],
        "Consumer Goods": ["Electronic Equipment","Beverages","Apparel"],
        "Financial": ["Property & Casualty Insurance","Regional Banks","Asset Management","REIT - Office"],
        "Healthcare": ["Medical Laboratories & Research","Drug Manufacturers - Major","Diagnostic Substances"],
        "Industrial Goods": ["General Building Materials","Industrial Equipment & Components"],
        "Services": ["Major Airlines","Auto Parts Stores","Trucking","Business Services"],
        "Technology": ["Semiconductor - Integrated Circuits","Software - Application"],
        "Utilities": ["Diversified Utilities"]
    }
    countries_pool = ["USA","USA","USA","Canada","Switzerland","UK"]
    countries_p = [0.6,0.1,0.1,0.1,0.05,0.05]
    rows = []
    tickers_used = set()
    base_price_by_sector = {"Financial":40, "Technology":55, "Healthcare":45, "Industrial Goods":38,
                  "Consumer Goods":42, "Services":33, "Basic Materials":30, "Conglomerates":44,
                  "Utilities":34}
    for i in range(n):
        sector = rng.choice(list(sectors.keys()))
        industry = rng.choice(sectors[sector])
        country = rng.choice(countries_pool, p=countries_p)
        while True:
            t = "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), size=rng.integers(3,5)))
            if t not in tickers_used:
                tickers_used.add(t); break
        base_price = base_price_by_sector[sector]
        price = max(1, rng.lognormal(mean=np.log(base_price), sigma=0.7))
        pe = np.nan if rng.random() < 0.05 else round(max(-5, rng.normal(22,15)),2)
        peg = np.nan if (pd.isna(pe) or rng.random()<0.1) else round(max(0.1, rng.normal(2,1.5)),2)
        ps = round(max(0.05, rng.lognormal(np.log(2),0.8)),2)
        pb = round(max(0.1, rng.lognormal(np.log(3),0.9)),2)
        pcash = round(max(0.1, rng.lognormal(np.log(10),1.0)),2)
        eps_ttm = round(rng.normal(1.5,2.5),2)
        eps_next_year_pct = round(rng.normal(8,10),2)
        eps_next_5y_pct = round(rng.normal(10,12),2)
        debt_equity = round(max(0, rng.lognormal(np.log(0.5),0.9)),2)
        beta = round(max(0, rng.normal(1.0,0.5)),2)
        inst_own_pct = round(min(100,max(0, rng.normal(55,25))),1)
        volume = int(max(100, rng.lognormal(np.log(500000),1.2)))
        market_cap_m = max(5, rng.lognormal(np.log(1500), 1.5))  # in $ millions

        # One deliberate extreme outlier to mimic Berkshire Hathaway in the book
        if i == 0:
            sector, industry, country = "Financial", "Property & Casualty Insurance", "USA"
            price = 172000.00
            pe = 15.4; peg = np.nan

        def fmt_money_m(v):
            # finviz shows market cap like "1.23B" or "456.70M"
            if v >= 1000:
                return f"{v/1000:.2f}B"
            return f"{v:.2f}M"

        rows.append({
            "No.": i+1,
            "Ticker": t,
            "Company": f"{t} Holdings Inc." if i else "Berkshire Hathaway Inc.",
            "Sector": sector,
            "Industry": industry,
            "Country": country,
            "Market Cap": fmt_money_m(market_cap_m),
            "P/E": "-" if pd.isna(pe) else f"{pe:.2f}",
            "Forward P/E": "-" if pd.isna(pe) else f"{pe*rng.uniform(0.7,1.0):.2f}",
            "PEG": "-" if pd.isna(peg) else f"{peg:.2f}",
            "P/S": f"{ps:.2f}",
            "P/B": f"{pb:.2f}",
            "P/Cash": f"{pcash:.2f}",
            "EPS (ttm)": f"{eps_ttm:.2f}",
            "EPS growth next year": f"{eps_next_year_pct:.2f}%",
            "EPS growth next 5 years": f"{eps_next_5y_pct:.2f}%",
            "Total Debt/Equity": f"{debt_equity:.2f}",
            "Beta": f"{beta:.2f}",
            "Institutional Ownership": f"{inst_own_pct:.1f}%",
            "Price": f"{price:,.2f}",
            "Volume": f"{volume:,}",
        })
    return pd.DataFrame(rows)

def simulate_price_history(tickers, start="2018-01-01", end="2023-12-29", seed=7):
    """Simulates daily adjusted-close price history via Geometric Brownian Motion,
    standing in for a Yahoo Finance / yfinance historical download."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    frames = []
    for k, t in enumerate(tickers):
        s0 = rng.uniform(15, 90)
        mu = rng.uniform(-0.05, 0.20)      # annual drift
        sigma = rng.uniform(0.15, 0.45)    # annual volatility
        dt = 1/252
        n = len(dates)
        shocks = rng.normal((mu - 0.5*sigma**2)*dt, sigma*np.sqrt(dt), n)
        log_path = np.cumsum(shocks)
        prices = s0 * np.exp(log_path)
        daily_range = prices * rng.uniform(0.005, 0.02, n)
        open_ = prices * (1 + rng.normal(0, 0.003, n))
        high = np.maximum(open_, prices) + daily_range/2
        low = np.minimum(open_, prices) - daily_range/2
        vol = rng.integers(200_000, 3_000_000, n)
        frames.append(pd.DataFrame({
            "Symbol": t, "Date": dates, "Open": open_.round(2), "High": high.round(2),
            "Low": low.round(2), "Close": prices.round(2), "Volume": vol,
            "AdjClose": prices.round(2)
        }))
    return pd.concat(frames, ignore_index=True)

if __name__ == "__main__":
    df = make_finviz_like_raw()
    print(df.shape)
    print(df.head(3).to_string())
    hist = simulate_price_history(["AAA","BBB"])
    print(hist.head())
