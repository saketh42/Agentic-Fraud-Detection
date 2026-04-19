# Reddit Data Scraping

## Overview

The data collection phase involves scraping Reddit posts from subreddits known to contain financial scam content. This creates a real-world dataset of fraud patterns.

## Target Sources

### Subreddits
- r/Scams
- r/Fraud
- r/PersonalFinance (scam-related posts)
- r/CryptoScams
- r/InvestmentScams

### Keywords Used
```
scam, fraud, fake, cheated, scammer, cryptocurrency scam, 
investment fraud, ponzi, pyramid scheme, upi fraud, payment scam,
social engineering, phishing, fake profile, lottery scam
```

## Scraping Process

1. **Time Window**: Trailing 3-month window for recent fraud patterns
2. **Filters**: 
   - English language
   - Keyword matching
   - Minimum engagement threshold
3. **Data Extracted**:
   - Post title and body
   - Timestamp
   - User comments (for context)
   - Post metadata (score, comments count)

## Output Format

```json
{
  \"post_id\": \"abc123\",
  \"title\": \"I was scammed by...\",
  \"body\": \"Full post text...\",
  \"timestamp\": \"2024-01-15T10:30:00Z\",
  \"comments\": [...],
  \"subreddit\": \"r/Scams\",
  \"keywords_matched\": [\"scam\", \"cheated\"]
}
```

## Files

- `Scrapper.py` - Main scraping script
- `config.json` - Configuration (subreddits, keywords, limits)
- `scrape_cache.json` - Cached results

## References

1. Reddit API terms of service compliance
2. Rate limiting (100 requests/10 min)
3. Respect robots.txt and no-scrape headers