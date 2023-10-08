import meduza
import json

def get(number = 5):
  news = []
  for article in meduza.section('news', n=number, lang='ru'):
    news_data = {
      #"raw_data": article,
      "title": article['title'],
      "datetime": article['datetime'],
      "link": "https://meduza.io/" + article["url"],
      "source": "Meduza"
    }

    #print('Art og:', article['og'])
    news_text = ''
    #news_text += f"{article['title']}\n"
    if 'second_title' in article:
      news_text += f"{article['second_title']}\n"
    news_text += f"\n"
    
    # TODO: add footnotes to the news text
    if 'blocks' in article['content']:
      blocks = article['content']['blocks']
      
      for block in blocks:
        if block['type'] == 'p':
          news_text += f"{block['data']}\n"
        if block['type'] == 'embed' and 'caption' in block['data']:
          news_text += f"{block['data']['caption']}\n"

    news_data['text'] = news_text
    news.append(news_data)

  return news

if __name__ == "__main__":
  news = get()
  print(json.dumps(news))