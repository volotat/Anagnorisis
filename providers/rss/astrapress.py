import json
import feedparser
from time import mktime
from datetime import datetime, timezone
import re


def get(rss_url='https://rsshub.app/telegram/channel/astrapress'):
  # Parse the RSS feed
  feed = feedparser.parse(rss_url)
  
  news = []
  for entry in feed.entries:
    timestamp = int(mktime(entry.published_parsed))

    text = entry.summary
    pattern = r'<div class="rsshub-quote"><blockquote>.*?</blockquote></div>'
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)

    pattern = r'<b>(.*?)<br /><br />'
    match = re.search(pattern, cleaned_text)

    title = "No title"
    if match:
      title = match.group(1)
      cleaned_text = re.sub(pattern, '', cleaned_text, count=1)

    news_data = {
      "raw_text": entry.summary,
      "title": title,
      "text": cleaned_text,
      "datetime": timestamp,
      "link": entry.link,
      "source": "TG/AstraPress"
    }

    news.append(news_data)
  return news

if __name__ == "__main__":
  news = get()
  print(json.dumps(news))