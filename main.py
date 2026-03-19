"""
main.py
-------
The main pipeline for the geopolitical risk trader.
Orchestrates the full workflow:
1. Fetch and score geopolitical news
2. Fetch EOD market prices
3. Send combined digest to Claude for analysis
4. Save and display the structured briefing

Run this script once daily before market open for best results.
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Import pipeline modules
from data.news_ingestion import get_top_articles, format_for_prompt as format_news
from data.price_ingestion import get_market_digest
from analysis.analyst import run_analysis, format_briefing

load_dotenv()

# --- Configuration ---
OUTPUT_DIR = "output/briefings"
DAYS_BACK = 1  # How many days of news to fetch (increase during active conflicts)


def ensure_output_dir():
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[Main] Created output directory: {OUTPUT_DIR}")


def save_briefing(briefing: dict, formatted: str):
    """
    Saves both the raw JSON and formatted briefing to disk.
    This is important for the backtester later — it caches
    model outputs so you don't re-call the API on reruns.

    Args:
        briefing: Raw JSON briefing dictionary
        formatted: Human readable formatted string
    """
    today = datetime.now().strftime("%Y-%m-%d")

    # Save raw JSON for programmatic use / backtesting
    json_path = os.path.join(OUTPUT_DIR, f"{today}_briefing.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(briefing, f, indent=2)

    # Save formatted report for easy reading
    txt_path = os.path.join(OUTPUT_DIR, f"{today}_briefing.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(formatted)

    print(f"[Main] Briefing saved to {json_path}")
    print(f"[Main] Formatted report saved to {txt_path}")


def print_header():
    """Prints a clean startup header."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 60)
    print("       GEOPOLITICAL RISK TRADER")
    print(f"       Run Date: {now}")
    print("=" * 60)
    print()


def run_pipeline(days_back: int = DAYS_BACK) -> dict:
    """
    Runs the full pipeline:
    1. News ingestion
    2. Price ingestion
    3. AI analysis
    4. Save and display output

    Args:
        days_back: How many days of news to fetch

    Returns:
        The raw briefing dictionary
    """
    print_header()
    ensure_output_dir()

    # --- Step 1: News Ingestion ---
    print("[Main] Step 1/3 -- Fetching geopolitical news...")
    try:
        articles = get_top_articles(days_back=days_back, top_n=10)
        news_digest = format_news(articles)
        print(f"[Main] News digest ready. ({len(articles)} articles)\n")
    except Exception as e:
        print(f"[Main] ERROR in news ingestion: {e}")
        news_digest = "News data unavailable for this run."

    # --- Step 2: Price Ingestion ---
    print("[Main] Step 2/3 -- Fetching market prices...")
    try:
        market_digest = get_market_digest()
        print("[Main] Market digest ready.\n")
    except Exception as e:
        print(f"[Main] ERROR in price ingestion: {e}")
        market_digest = "Market data unavailable for this run."

    # --- Step 3: AI Analysis ---
    print("[Main] Step 3/3 -- Running AI analysis...")
    try:
        briefing = run_analysis(news_digest, market_digest)
        formatted = format_briefing(briefing)
        print("[Main] Analysis complete.\n")
    except Exception as e:
        print(f"[Main] ERROR in AI analysis: {e}")
        print("[Main] Check your API key and credit balance at console.anthropic.com")
        return {}

    # --- Output ---
    save_briefing(briefing, formatted)
    print("\n" + formatted)

    return briefing


# --- Entry point ---
if __name__ == "__main__":
    run_pipeline(days_back=DAYS_BACK)