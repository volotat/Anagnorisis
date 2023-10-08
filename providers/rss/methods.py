import json
import feedparser
from time import mktime
from datetime import datetime, timezone


def get(rss_url):
  # Parse the RSS feed
  feed = feedparser.parse(rss_url)

  news = []
  for entry in feed.entries:
    timestamp = int(mktime(entry.published_parsed))
    news_data = {
      "title": entry.title,
      "text": entry.summary,
      "datetime": timestamp,
    }

    news.append(news_data)

  return news