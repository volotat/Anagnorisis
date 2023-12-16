# set up inference via HuggingFace
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria
import transformers
import torch
import traceback
import pickle

from peft import PeftModel, LoraConfig, get_peft_model
from transformers import GenerationConfig

import gc

cache_file = 'cache/llm_engine.pkl'
try:
  with open(cache_file, 'rb') as f:
    cache = pickle.load(f)
except FileNotFoundError:
  cache = {} #cachetools.LRUCache(maxsize=1000) 


def save_cache():
  with open(cache_file, "wb") as file:
    pickle.dump(cache, file)

class StoppingCriteriaSub(StoppingCriteria):
  def __init__(self, stops = [], encounters=1):
    super().__init__()
    self.stops = [stop.to("cuda") for stop in stops]

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
    for stop in self.stops:
      if torch.all((stop == input_ids[0][-len(stop):])).item():
        return True

    return False

class TextPredictor():
  def __init__(self, socketio) -> None:
    self.socketio = socketio

    #model = "meta-llama/Llama-2-7b-chat-hf"
    #model_name = "4bit/Llama-2-7b-chat-hf"
    model_name = "georgesung/llama2_7b_chat_uncensored"
    #model_name = "models/your_llama2"

    self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",    # I've also tried removing this line
        bnb_4bit_compute_dtype=torch.float16,
        #bnb_4bit_use_double_quant=True,    # I've also tried removing this line
    )

    self.base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,
        local_files_only=False,
        #config="models/your_llama2/adapter_config.json"
    )

    self.model = PeftModel.from_pretrained(self.base_model, 'models/your_llama2')
    #self.model = self.model.to_bettertransformer()

    '''self.pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=self.tokenizer,
        #trust_remote_code=True,
        device_map="auto",    # finds GPU
    )

    print(f"Model is on device: {self.pipeline.device}")'''

    stop_words_ids = [torch.tensor([2277, 29937]), torch.tensor([835]), ] #   "\n### ", "###"  [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze()[1:] for stop_word in stop_words]
    #print('stop_words_ids', stop_words_ids)
    self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

  # Generate responses
  def generate(self, prompt, temperature):
    self.socketio.emit("emit_llm_engine_start_processing") 

    inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

    generation_config = GenerationConfig(
      do_sample=True,
      temperature=temperature,
      top_p=0.75,
      top_k=40,
      num_beams=1,
      #early_stopping=True,
      #**kwargs,
    )

    generation_outputs = self.model.generate(
        **inputs,
        generation_config=generation_config,
        #return_dict_in_generate=True,
        #output_scores=True,
        #max_new_tokens=2048,
        max_length=512*5,
        stopping_criteria=self.stopping_criteria,
    )
    
    #output_tokens = self.tokenizer.convert_ids_to_tokens(generation_outputs[0])
    #word_token_pairs = [[word, token_id] for word, token_id in zip(output_tokens, generation_outputs[0].tolist())]
    #print('GEN OUT:', word_token_pairs)

    outputs = self.tokenizer.batch_decode(generation_outputs, skip_special_tokens=True)  

    self.socketio.emit("emit_llm_engine_end_processing")
    return outputs[0]

  def predict_from_text(self, text, temperature = 0.3, max_new_tokens = 2048, use_cache = False):
    if use_cache and (text in cache):
      return cache[text]
 
    #encoding = self.tokenizer.encode(text)
    #num_tokens = len(encoding)
    #print('Processing text. Num tokens:', num_tokens)

    response = self.generate(text, temperature)

    result = response[len(text):]
    if result[-3:] == "###": result = result[:-3]

    if use_cache:
      cache[text] = result
      save_cache()

    return result
  
  def unload_model(self):
    del self.tokenizer
    del self.base_model
    del self.model
    gc.collect()
    torch.cuda.empty_cache()