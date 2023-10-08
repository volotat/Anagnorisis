import json
import feedparser
from time import mktime
from datetime import datetime, timezone


def get(rss_url = 'https://n1info.rs/feed'):
  # Parse the RSS feed
  feed = feedparser.parse(rss_url)

  news = []
  for entry in feed.entries:
    timestamp = int(mktime(entry.published_parsed))
    news_data = {
      "title": entry.title,
      "text": entry.summary,
      "datetime": timestamp,
      "link": entry.link,
      "source": "N1 info"
    }

    news.append(news_data)

  return news