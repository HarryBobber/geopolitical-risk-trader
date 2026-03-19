"""
price_ingestion.py
------------------
Fetches end-of-day price data for geopolitically sensitive
financial instruments using yfinance (free, no API key needed).

Tracks sectors most affected by geopolitical events:
- Defense ETFs and stocks
- Energy ETFs and oil proxies
- Safe haven assets (gold, volatility)
- Regional market ETFs
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


# --- Watchlist Configuration ---

WATCHLIST = {
    "Defense ETFs": ["ITA", "XAR"],
    "Defense Stocks": ["LMT", "RTX", "NOC", "GD", "LHX"],
    "Energy ETFs": ["XLE", "USO"],
    "Energy Stocks": ["XOM", "CVX"],
    "Safe Haven": ["GLD", "SLV"],
    "Volatility": ["VXX"],
    "Regional Markets": ["EIS", "TUR", "RSX"],  # Israel, Turkey, Russia ETFs
}

# Flatten watchlist into a single list for batch downloading
ALL_TICKERS = [ticker for group in WATCHLIST.values() for ticker in group]


def fetch_price_data(days_back: int = 5) -> pd.DataFrame:
    """
    Fetches EOD price data for all watchlist instruments.
    Gets the past N days to provide context on recent trends.

    Args:
        days_back: How many trading days of history to fetch

    Returns:
        DataFrame with closing prices for all tickers
    """
    start_date = (datetime.now() - timedelta(days=days_back + 5)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"[PriceData] Fetching EOD prices for {len(ALL_TICKERS)} instruments...")

    try:
        raw = yf.download(
            tickers=ALL_TICKERS,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )

        # Extract closing prices only
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]]

        # Drop any columns with all NaN (tickers that failed)
        prices = prices.dropna(axis=1, how="all")

        print(f"[PriceData] Successfully fetched {len(prices.columns)} instruments.")
        return prices

    except Exception as e:
        print(f"[PriceData] Error fetching price data: {e}")
        return pd.DataFrame()


def calculate_performance(prices: pd.DataFrame) -> dict:
    """
    Calculates 1-day and 5-day percentage change for each instrument.
    This tells the model what the market has already priced in.

    Args:
        prices: DataFrame of closing prices

    Returns:
        Dictionary of performance metrics per ticker
    """
    performance = {}

    for ticker in prices.columns:
        series = prices[ticker].dropna()

        if len(series) < 2:
            continue

        latest_price = series.iloc[-1]
        prev_price = series.iloc[-2]

        # 1-day change
        day_change = ((latest_price - prev_price) / prev_price) * 100

        # 5-day change (or however many days we have)
        if len(series) >= 5:
            five_day_price = series.iloc[-5]
            five_day_change = ((latest_price - five_day_price) / five_day_price) * 100
        else:
            five_day_change = None

        performance[ticker] = {
            "latest_price": round(float(latest_price), 2),
            "1d_change_pct": round(float(day_change), 2),
            "5d_change_pct": round(float(five_day_change), 2) if five_day_change is not None else "N/A",
        }

    return performance


def flag_unusual_moves(performance: dict, threshold: float = 2.0) -> list[str]:
    """
    Flags any instruments with unusually large single-day moves.
    Large moves in defense or energy often precede or confirm geopolitical events.

    Args:
        performance: Dictionary of performance metrics
        threshold: Percentage change threshold to flag (default: 2%)

    Returns:
        List of flagged ticker strings with context
    """
    flags = []

    for ticker, data in performance.items():
        change = data["1d_change_pct"]
        if abs(change) >= threshold:
            direction = "UP" if change > 0 else "DOWN"
            flags.append(
                f"!! {ticker} moved {direction} {abs(change):.2f}% -- "
                f"current price: ${data['latest_price']}"
            )

    return flags


def format_for_prompt(performance: dict, flags: list[str]) -> str:
    """
    Formats price data and flags into a clean text block
    ready to be inserted into the Claude prompt.

    Args:
        performance: Dictionary of performance metrics
        flags: List of flagged unusual moves

    Returns:
        Formatted string for use in AI prompt
    """
    lines = ["=== MARKET PRICE DIGEST ===\n"]

    # Group by sector for readability
    for sector, tickers in WATCHLIST.items():
        lines.append(f"[ {sector} ]")
        for ticker in tickers:
            if ticker in performance:
                data = performance[ticker]
                five_d = data['5d_change_pct']
                five_d_str = five_d if isinstance(five_d, str) else f"{five_d:+.2f}%"
                lines.append(
                    f"  {ticker}: ${data['latest_price']} | "
                    f"1D: {data['1d_change_pct']:+.2f}% | "
                    f"5D: {five_d_str}"
                )
            else:
                lines.append(f"  {ticker}: Data unavailable")
        lines.append("")

    # Add flags section
    if flags:
        lines.append("[ UNUSUAL MOVES -- POTENTIAL SIGNAL ]")
        for flag in flags:
            lines.append(f"  {flag}")
    else:
        lines.append("[ No unusual moves detected today ]")

    return "\n".join(lines)


def get_market_digest() -> str:
    """
    Main function — fetches prices, calculates performance,
    flags unusual moves, and returns formatted prompt-ready string.

    Returns:
        Formatted market digest string
    """
    prices = fetch_price_data(days_back=5)

    if prices.empty:
        return "Market data unavailable."

    performance = calculate_performance(prices)
    flags = flag_unusual_moves(performance, threshold=2.0)

    return format_for_prompt(performance, flags)


# --- Quick test when run directly ---
if __name__ == "__main__":
    digest = get_market_digest()
    print(digest)