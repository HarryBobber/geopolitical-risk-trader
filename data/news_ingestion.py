"""
news_ingestion.py
-----------------
Fetches and filters geopolitical news from NewsAPI.
Targets keywords relevant to conflict, sanctions, military activity,
and diplomatic breakdown that could affect financial markets.
"""

import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# --- Configuration ---

# Keywords that signal potential geopolitical risk
GEOPOLITICAL_KEYWORDS = [
    "military strike",
    "sanctions",
    "nuclear",
    "conflict",
    "invasion",
    "missile",
    "war",
    "diplomatic crisis",
    "IAEA",
    "oil supply",
    "Strait of Hormuz",
    "NATO",
    "escalation",
    "ceasefire",
    "regime",
]

# Financial sectors most affected by geopolitical events
MARKET_KEYWORDS = [
    "oil price",
    "crude",
    "defense spending",
    "gold rally",
    "safe haven",
    "energy stocks",
    "defense contractor",
]

# Countries/regions to monitor
REGIONS = [
    "Iran",
    "Israel",
    "Russia",
    "Ukraine",
    "China",
    "Taiwan",
    "North Korea",
    "Middle East",
    "South China Sea",
]


def build_query() -> str:
    """
    Builds a NewsAPI query string combining geopolitical
    and regional keywords for maximum signal relevance.
    """
    # Combine region names with top geopolitical keywords
    region_query = " OR ".join(REGIONS)
    keyword_query = " OR ".join(GEOPOLITICAL_KEYWORDS[:6])  # Top 6 to stay within API limits
    return f"({region_query}) AND ({keyword_query})"


def fetch_news(days_back: int = 1) -> list[dict]:
    """
    Fetches top geopolitical news articles from the past N days.

    Args:
        days_back: How many days back to search (default: 1 for daily runs)

    Returns:
        List of relevant article dictionaries
    """
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY not found. Check your .env file.")

    # Calculate date range
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")

    query = build_query()

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 20,  # Top 20 most relevant articles
        "apiKey": NEWS_API_KEY,
    }

    print(f"[NewsAPI] Fetching articles from {from_date} to {to_date}...")

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"[NewsAPI] Error: {response.status_code} - {response.text}")
        return []

    data = response.json()
    articles = data.get("articles", [])
    print(f"[NewsAPI] Found {len(articles)} articles.")
    return articles


def parse_articles(articles: list[dict]) -> list[dict]:
    """
    Cleans and extracts only the relevant fields from raw API response.

    Args:
        articles: Raw article list from NewsAPI

    Returns:
        List of cleaned article dictionaries
    """
    parsed = []
    for article in articles:
        # Skip articles with missing key fields
        if not article.get("title") or not article.get("description"):
            continue

        parsed.append({
            "title": article["title"],
            "description": article["description"],
            "source": article.get("source", {}).get("name", "Unknown"),
            "published_at": article.get("publishedAt", ""),
            "url": article.get("url", ""),
        })

    return parsed


def score_article(article: dict) -> int:
    """
    Simple relevancy scoring based on keyword presence.
    Higher score = more geopolitically significant.

    Args:
        article: Cleaned article dictionary

    Returns:
        Integer relevancy score
    """
    score = 0
    text = (article["title"] + " " + article["description"]).lower()

    for keyword in GEOPOLITICAL_KEYWORDS:
        if keyword.lower() in text:
            score += 2  # Geopolitical keywords worth more

    for keyword in MARKET_KEYWORDS:
        if keyword.lower() in text:
            score += 1  # Market keywords worth less but still relevant

    for region in REGIONS:
        if region.lower() in text:
            score += 1

    return score


def get_top_articles(days_back: int = 1, top_n: int = 10) -> list[dict]:
    """
    Main function — fetches, parses, scores, and returns
    the top N most relevant geopolitical articles.

    Args:
        days_back: How many days back to search
        top_n: Number of top articles to return

    Returns:
        Sorted list of top N scored articles
    """
    raw_articles = fetch_news(days_back=days_back)
    parsed = parse_articles(raw_articles)

    # Score and sort by relevancy
    for article in parsed:
        article["relevancy_score"] = score_article(article)

    sorted_articles = sorted(parsed, key=lambda x: x["relevancy_score"], reverse=True)

    top_articles = sorted_articles[:top_n]
    print(f"[NewsAPI] Returning top {len(top_articles)} scored articles.")
    return top_articles


def format_for_prompt(articles: list[dict]) -> str:
    """
    Formats the top articles into a clean text block
    ready to be inserted into the Claude prompt.

    Args:
        articles: List of scored article dictionaries

    Returns:
        Formatted string for use in AI prompt
    """
    if not articles:
        return "No relevant geopolitical news found for this period."

    lines = ["=== GEOPOLITICAL NEWS DIGEST ===\n"]
    for i, article in enumerate(articles, 1):
        lines.append(f"[{i}] {article['title']}")
        lines.append(f"    Source: {article['source']} | Published: {article['published_at']}")
        lines.append(f"    Summary: {article['description']}")
        lines.append(f"    Relevancy Score: {article['relevancy_score']}")
        lines.append(f"    URL: {article['url']}\n")

    return "\n".join(lines)


# --- Quick test when run directly ---
if __name__ == "__main__":
    articles = get_top_articles(days_back=1, top_n=10)
    print("\n" + format_for_prompt(articles))