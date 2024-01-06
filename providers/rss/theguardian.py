import json
import feedparser
from time import mktime
from datetime import datetime, timezone


def get():
  rss_urls = [
    'https://www.theguardian.com/world/rss',
    #'https://www.theguardian.com/uk/rss',
    'https://www.theguardian.com/us-news/rss',
    #'https://www.theguardian.com/world/europe-news/rss',
    #'https://www.theguardian.com/uk/money/rss',
    #'https://www.theguardian.com/uk/technology/rss',
    #'https://www.theguardian.com/environment/energy/rss',
    #'https://www.theguardian.com/science/rss',
    #'https://www.theguardian.com/uk/business/rss',
    #'https://www.theguardian.com/music/rss',
    #'https://www.theguardian.com/games/rss'
  ]

  news = []

  for rss_url in rss_urls:
    # Parse the RSS feed
    feed = feedparser.parse(rss_url)
    
    for entry in feed.entries:
      timestamp = int(mktime(entry.published_parsed))
      news_data = {
        "title": entry.title,
        "text": entry.description,
        "datetime": timestamp,
        "link": entry.link,
        "source": "The Guardian"
      }

      news.append(news_data)

  return news