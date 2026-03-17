"""
analyst.py
----------
The AI reasoning layer of the geopolitical risk trader.
Takes in the news digest and market digest, sends them to
Claude, and returns a structured geopolitical risk briefing
with stock recommendations.
"""

import os
import json
import anthropic
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- System Prompt ---
# This is what makes Claude behave like a specialized
# geopolitical risk analyst every single time.

SYSTEM_PROMPT = """
You are a senior geopolitical risk analyst specializing in financial markets.
Your job is to analyze a daily digest of geopolitical news and market price data,
then produce a structured investment briefing.

You must always respond in valid JSON format with exactly this structure:

{
  "risk_level": "low" | "medium" | "high" | "critical",
  "summary": "2-3 sentence overview of the current geopolitical environment",
  "top_risks": [
    {
      "title": "Short risk title",
      "description": "What is happening and why it matters",
      "probability": "low" | "medium" | "high",
      "timeframe": "e.g. 1-2 weeks, 30 days, 3-6 months"
    }
  ],
  "recommendations": [
    {
      "ticker": "e.g. LMT",
      "action": "buy" | "sell" | "watch",
      "conviction": "low" | "medium" | "high",
      "rationale": "Specific reason grounded in the provided data",
      "sector": "e.g. Defense, Energy, Safe Haven"
    }
  ],
  "signals_to_watch": [
    "Specific event or data point to monitor in the next 24-48 hours"
  ],
  "thesis_breakers": [
    "Specific event that would invalidate the current risk thesis"
  ]
}

Critical rules:
- Never fabricate data. Only reference facts present in the provided digest.
- Every recommendation must include a rationale tied to the digest.
- Be direct and specific. Avoid vague language.
- Always return valid JSON. No preamble, no markdown, no extra text.
"""


def build_user_message(news_digest: str, market_digest: str) -> str:
    """
    Combines the news and market digests into a single
    structured message for Claude.

    Args:
        news_digest: Formatted string from news_ingestion.py
        market_digest: Formatted string from price_ingestion.py

    Returns:
        Combined prompt string
    """
    return f"""
Please analyze the following data and produce your geopolitical risk briefing.

{news_digest}

{market_digest}

Return your analysis as valid JSON only.
"""


def run_analysis(news_digest: str, market_digest: str) -> dict:
    """
    Sends the combined digest to Claude and returns
    the parsed JSON briefing.

    Args:
        news_digest: Formatted news string
        market_digest: Formatted market price string

    Returns:
        Parsed briefing as a Python dictionary
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not found. Check your .env file.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    print("[Analyst] Sending digest to Claude for analysis...")

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": build_user_message(news_digest, market_digest)
            }
        ],
        temperature=0.2,  # Low temperature for consistent, grounded analysis
    )

    raw_response = message.content[0].text

    # Parse and validate JSON response
    try:
        briefing = json.loads(raw_response)
        print("[Analyst] Successfully received and parsed briefing.")
        return briefing
    except json.JSONDecodeError:
        print("[Analyst] Warning: Response was not valid JSON. Returning raw text.")
        return {"raw_response": raw_response}


def format_briefing(briefing: dict) -> str:
    """
    Formats the JSON briefing into a human-readable
    report for display or saving.

    Args:
        briefing: Parsed briefing dictionary

    Returns:
        Formatted string report
    """
    if "raw_response" in briefing:
        return f"Raw Response:\n{briefing['raw_response']}"

    lines = []
    lines.append("=" * 60)
    lines.append("       GEOPOLITICAL RISK BRIEFING")
    lines.append(f"       Risk Level: {briefing.get('risk_level', 'N/A').upper()}")
    lines.append("=" * 60)

    # Summary
    lines.append("\n📋 SITUATION SUMMARY")
    lines.append(briefing.get("summary", "No summary available."))

    # Top Risks
    lines.append("\n⚠️  TOP GEOPOLITICAL RISKS")
    for i, risk in enumerate(briefing.get("top_risks", []), 1):
        lines.append(f"\n  [{i}] {risk.get('title', 'Unknown')}")
        lines.append(f"      {risk.get('description', '')}")
        lines.append(
            f"      Probability: {risk.get('probability', 'N/A')} | "
            f"Timeframe: {risk.get('timeframe', 'N/A')}"
        )

    # Recommendations
    lines.append("\n💹 STOCK RECOMMENDATIONS")
    for rec in briefing.get("recommendations", []):
        action = rec.get("action", "").upper()
        ticker = rec.get("ticker", "")
        conviction = rec.get("conviction", "")
        sector = rec.get("sector", "")
        rationale = rec.get("rationale", "")
        lines.append(f"\n  {action} {ticker} | Conviction: {conviction} | Sector: {sector}")
        lines.append(f"  → {rationale}")

    # Signals to Watch
    lines.append("\n🔭 SIGNALS TO WATCH (Next 24-48 Hours)")
    for signal in briefing.get("signals_to_watch", []):
        lines.append(f"  • {signal}")

    # Thesis Breakers
    lines.append("\n🚫 THESIS BREAKERS")
    for breaker in briefing.get("thesis_breakers", []):
        lines.append(f"  • {breaker}")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


# --- Quick test when run directly ---
if __name__ == "__main__":
    # Test with mock data so you can verify without running full pipeline
    mock_news = """=== GEOPOLITICAL NEWS DIGEST ===

[1] Iran Signals It Won't Accept US Nuclear Proposal
    Source: Reuters | Published: 2025-06-09
    Summary: Iranian officials rejected the latest US framework for nuclear negotiations.
    Relevancy Score: 12

[2] US Military Buildup in Middle East Continues
    Source: Wall Street Journal | Published: 2025-06-10
    Summary: The Pentagon confirms additional carrier groups deployed to the Persian Gulf.
    Relevancy Score: 10
"""

    mock_market = """=== MARKET PRICE DIGEST ===

[ Defense ETFs ]
  ITA: $158.20 | 1D: +1.80% | 5D: +3.20%

[ Energy ETFs ]
  XLE: $89.50 | 1D: +2.30% | 5D: +4.10%

[ Safe Haven ]
  GLD: $298.40 | 1D: +0.90% | 5D: +1.50%

[ UNUSUAL MOVES — POTENTIAL SIGNAL ]
  ⚠️  XLE moved UP 2.30%
"""

    briefing = run_analysis(mock_news, mock_market)
    print(format_briefing(briefing))