import llm_engine
from datetime import datetime
import re
from tqdm import tqdm
import hashlib
import src.db_models

def format_timestamp(timestamp):
    dt_object = datetime.fromtimestamp(timestamp)
    formatted_date = dt_object.strftime('%A, %B %d, %Y at %I:%M:%S %p')
    return formatted_date

def hash_string(data):
    sha256_hash = hashlib.sha256(data.encode()).hexdigest()
    return sha256_hash

from transformers import AutoTokenizer
def format_rate_news_entry_prompt(news_data, with_response=False):
  rate_name = ['insignificant', 'negligible', 'trivial', 'minor', 'slightly important', 'moderate', 
               'significant', 'considerably important', 'very important', 'highly important', 'extremely important']
  prompt = f'''### HUMAN: 
Rate the given news entry on the scale from 0 to 10, where 0 is an absolutely unimportant news and 10 is extremely important influential one (such as alien invasion). Try to keep the score low unless the news is really important. Then predict GPS coordinate of the place where the news happened. Then give a short description of the news in a one or two short concise sentences (less then 150 characters). Here is an 

Formatting example:
news date: Tuesday, August 15, 2023 at 3:27:32 PM
news text: "По последним данным, из-за ракетного удара по Покровску погибли десять человек, в том числе двое спасателей

Государственная служба Украины по чрезвычайным ситуациям (ГСЧС) сообщила, что в больнице умер начальник пожарно-спасательного..."
Your response:
Importance: 3/10, (minor)
GPS: 48.284320, 37.184679
Summary: "Rocket strike in the Pokrovsk, Ukraine. Ten people died, including two rescuers."

Actual news:
news date: {format_timestamp(news_data['datetime'])}
news text: "{news_data['title']}\n{news_data['text'][:150]}"
Your response:
### RESPONSE:
Importance: '''
  if with_response:
    rate = int(news_data['importance'])
    prompt += f'''{rate}/10, ({rate_name[rate]})
GPS: {news_data['coordinates'][0]}, {news_data['coordinates'][1]}
Summary: "{news_data['summary']}"'''


  model_name = "georgesung/llama2_7b_chat_uncensored"
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  encoding = tokenizer.encode(prompt)
  num_tokens = len(encoding)
  print('New prompt has been formatted, size in tokens:', num_tokens)
  return prompt

def analyze_the_news(predictor, news_data):
  news_data['hash'] = hash_string(news_data['title'] + '\n\n' + news_data['text'])

  # Check if news with same title and text already in the DB. If so take info from there, instead of predicting it 
  news_entry = src.db_models.News.query.get(news_data['hash'])

  if news_entry:
    news_data['importance'] = news_entry.importance
    news_data['coordinates'] = news_entry.coordinates
    news_data['summary'] = news_entry.summary
  else:
    prompt = format_rate_news_entry_prompt(news_data)
    new_text = predictor.predict_from_text(prompt, temperature = 0.1, use_cache=True)
    #print(prompt + new_text)

    new_text = 'Importance: ' + new_text

    # Fine importance of the given news
    pattern = r'Importance: (\d+)/10'
    match = re.search(pattern, new_text)

    if match:
      importance = match.group(1)
      #print(f"Importance: {importance}")
    else:
      importance = None
      #print("Importance not found!")

    # Find predicted GPS coordinates of the given news
    pattern = r'GPS: ([\d\.]+), ([\d\.]+)'
    match = re.search(pattern, new_text)

    if match:
      latitude = match.group(1)
      longitude = match.group(2)
      #print(f"Coordinates: {latitude} lat, {longitude} long")
    else:
      latitude = None
      longitude = None
      #print("Coordinates not found!")

    # Find summary of the given news
    pattern = r'Summary: "([^"]*)"'
    match = re.search(pattern, new_text)

    if match:
      summary = match.group(1)
      #print(f"Summary: {summary}")
    else:
      summary = None
      #print("Summary not found!")

    news_data['importance'] = importance
    news_data['coordinates'] = [latitude, longitude]
    news_data['summary'] = summary

    return news_data


import providers.custom.meduza_news
import providers.rss.astrapress
import providers.rss.n1info
import providers.rss.theguardian

def get_news(socketio, predictor):
  print('get_news started')
  full_news_list = []

  print("Getting news out of 'Medusa' provider...")
  news_list = providers.custom.meduza_news.get(20)
  for news_data in tqdm(news_list):
    news_data = analyze_the_news(predictor, news_data)
    full_news_list.append(news_data)
    full_news_list = list(filter(lambda item: item is not None, full_news_list))
    socketio.emit("emit_news_page_news_list", full_news_list)

  print("Getting news out of RSS:'TG/Astrapress' provider...")
  news_list = providers.rss.astrapress.get()[:20]
  for news_data in tqdm(news_list):
    news_data = analyze_the_news(predictor, news_data)
    full_news_list.append(news_data)
    full_news_list = list(filter(lambda item: item is not None, full_news_list))
    socketio.emit("emit_news_page_news_list", full_news_list)

  print("Getting news out of RSS: N1 (Serbia) provider...")
  news_list = providers.rss.n1info.get()[:20]
  for news_data in tqdm(news_list):
    news_data = analyze_the_news(predictor, news_data)
    full_news_list.append(news_data)
    full_news_list = list(filter(lambda item: item is not None, full_news_list))
    socketio.emit("emit_news_page_news_list", full_news_list)

  print("Getting news out of RSS: The Guardian provider...")
  news_list = providers.rss.theguardian.get()[:100]
  for news_data in tqdm(news_list):
    news_data = analyze_the_news(predictor, news_data)
    full_news_list.append(news_data)
    full_news_list = list(filter(lambda item: item is not None, full_news_list))
    socketio.emit("emit_news_page_news_list", full_news_list)
  
  full_news_list = list(filter(lambda item: item is not None, full_news_list))
  return full_news_list


from flask import request

def init_socket_events(socketio, predictor):
  news_list = []

  @socketio.on("emit_news_page_get_news")
  def get_news():
    nonlocal news_list
    socketio.emit("emit_news_page_news_list", news_list)

  @socketio.on("emit_news_page_refresh_news")
  def refresh_news():
    nonlocal predictor, news_list
    socketio.emit("emit_news_page_start_loading")
    if predictor is None:
      predictor = llm_engine.TextPredictor(socketio)
    news_list = src.news_page.get_news(socketio, predictor)
    socketio.emit("emit_news_page_news_list", news_list)
    socketio.emit("emit_news_page_stop_loading")

  @socketio.on("emit_news_page_add_news_to_db")
  def add_news_to_db(news_data):
    nonlocal news_list
    target_news = next(n for n in news_list if n["hash"] == news_data["hash"])
    target_news.update(news_data)

    news = src.db_models.News(
      hash = target_news["hash"],
      title = target_news["title"],
      text = target_news["text"],
      summary = target_news["summary"],
      datetime = datetime.fromtimestamp(target_news["datetime"]),
      coordinates = target_news["coordinates"],
      importance = target_news["importance"]
    )
    src.db_models.db.session.add(news)
    src.db_models.db.session.commit()