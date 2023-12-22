from datetime import datetime
import src.db_models
import datasets
import random
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel, LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer
import trl

import os
import shutil
import numpy as np

import transformers
import llm_engine

from flask import request

# Define the sliding window function
def create_fixed_size_dataset(dataset, tokens_window_size, tokenizer, stride = None):
  if stride is None: stride = tokens_window_size // 2
  # Create a list to store the fixed-size examples
  fixed_size_examples = []

  # Iterate through the original dataset
  for example in dataset:
      # Tokenize the input text using the model's tokenizer
      input_text = example["text"]
      input_tokens = tokenizer.tokenize(input_text)

      # Apply sliding window to create fixed-size examples
      for start in range(0, len(input_tokens), stride):
          end = start + tokens_window_size
          fixed_size_input_tokens = input_tokens[start:end]
          
          # Convert tokens back to text
          fixed_size_input = tokenizer.convert_tokens_to_string(fixed_size_input_tokens)
          
          # Create a new example with the fixed-size input
          new_example = {"text": fixed_size_input}  # You may need to include other fields if your dataset has them
          
          fixed_size_examples.append(new_example)

  # Create a new dataset from the fixed-size examples
  fixed_size_dataset = datasets.Dataset.from_dict({"text": [example["text"] for example in fixed_size_examples]})
  return fixed_size_dataset

def create_fixed_size_dataset_from_data_folder(data_folder, tokens_window_size, tokenizer):
  data = []

  for filename in os.listdir(data_folder):
      if filename.endswith(('.txt', '.py', '.yaml', '.md', '.html', '.js')):
          file_path = os.path.join(data_folder, filename)
          with open(file_path, 'r') as file:
              text_content = file.read()
          data.append({'path': file_path, 'text': text_content})

  # Construct the dataset
  train_dataset = datasets.Dataset.from_dict({'path': [item['path'] for item in data], 'text': [item['text'] for item in data]})

  #train_dataset = datasets.load_dataset("text", data_dir="datasets/ai_dj", split="train", sample_by="document", with_file_name=True)
  train_dataset = create_fixed_size_dataset_with_filedata(train_dataset, tokens_window_size, tokenizer)
  return train_dataset

# Define the sliding window function
def create_fixed_size_dataset_with_filedata(dataset, tokens_window_size, tokenizer, stride = None):
  tokens_window_size = tokens_window_size - 32 # keep some tokens for meta information about the file

  if stride is None: stride = tokens_window_size // 2
  # Create a list to store the fixed-size examples
  fixed_size_examples = []

  # Iterate through the original dataset
  for example in dataset:
      # Tokenize the input text using the model's tokenizer
      input_text = example["text"]
      filename = example["path"]
      #print('filename', filename, 'input_text', input_text)
      input_tokens = tokenizer.tokenize(input_text)

      # Apply sliding window to create fixed-size examples
      for start in range(0, len(input_tokens), stride):
          end = start + tokens_window_size
          fixed_size_input_tokens = input_tokens[start:end]
          
          # Convert tokens back to text
          fixed_size_input = tokenizer.convert_tokens_to_string(fixed_size_input_tokens)
          fixed_size_input = f'File: {filename} | Start token index: {start}\n{fixed_size_input}'
          # Create a new example with the fixed-size input
          new_example = {"text": fixed_size_input}  # You may need to include other fields if your dataset has them
          #print('\n\n')
          #print(new_example)
          #print('token size:', len(tokenizer.tokenize(fixed_size_input)))
          fixed_size_examples.append(new_example)

  # Create a new dataset from the fixed-size examples
  fixed_size_dataset = datasets.Dataset.from_dict({"text": [example["text"] for example in fixed_size_examples]})
  return fixed_size_dataset


def init_socket_events(socketio, predictor,cfg=None):
  #train_loss_hist = []
  #eval_loss_hist = []

  class EmitLogDataCallback(transformers.TrainerCallback):
    nonlocal socketio

    def __init__(self) -> None:
      super().__init__()

      self.train_loss_hist = []
      self.eval_loss_hist = []
        

    def on_log(self, args, state, control, logs=None, **kwargs):
      if 'loss' in logs:
        self.train_loss_hist.append(logs['loss'])
      if 'eval_loss' in logs:
        self.eval_loss_hist.append(logs['eval_loss'])

      percent = np.floor(state.global_step / state.max_steps * 100)

      data = {
        'train_loss_hist': self.train_loss_hist,
        'eval_loss_hist': self.eval_loss_hist,
        'percent': percent
      }
      socketio.emit("emit_display_loss_data", data) 

  #def emit_display_loss_data():
  #  data = {
  #    'train_loss_hist': train_loss_hist,
  #    'eval_loss_hist': eval_loss_hist,
  #    'percent': percent
  #  }
  #  socketio.emit("emit_display_loss_data", data)  

  #@socketio.on("connect")
  #def handle_connect():
  #  emit_display_loss_data()

  @socketio.on("emit_start_fine_tuning")
  def handle_emit_start_fine_tuning():
    nonlocal predictor
    if predictor is None: predictor = llm_engine.TextPredictor(socketio)

    MAX_TOKEN_SIZE = cfg.max_token_length
    num_train_epochs = cfg.num_train_epochs
    openassistant_percent = cfg.openassistant_percent


    ### Unload previous Peft model
    del predictor.model
    torch.cuda.empty_cache()

    ### Load model
    #model_name = "TinyPixel/Llama-2-7B-bf16-sharded"
    model_name = "georgesung/llama2_7b_chat_uncensored"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    '''del predictor.base_model
    torch.cuda.empty_cache()
    torch.clear_autocast_cache()
    
    predictor.base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map = "auto"
    )'''

    model = predictor.base_model
    model.config.use_cache = False

    #model = model.to_bettertransformer()

    # Load tokenizer
    tokenizer = predictor.tokenizer
    #tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'


    ### Create new dataset
    news_entries = src.db_models.News.query.all()

    shutil.rmtree('datasets/news/train')
    shutil.rmtree('datasets/news/eval')

    os.makedirs('datasets/news/train', exist_ok=True)
    os.makedirs('datasets/news/eval', exist_ok=True)

    eval_size = int(len(news_entries) * 0.1) + 1
    eval_indxs = np.random.choice(len(news_entries), eval_size, replace=False)
    for ind, news_entry in enumerate(news_entries):
      news_data = news_entry.__dict__
      news_data['datetime'] = datetime.timestamp(news_data['datetime'])
      prompt = src.news_page.format_rate_news_entry_prompt(news_data, with_response=True)

      folder = 'train'
      if ind in eval_indxs:
        folder = 'eval'

      with open(f"datasets/news/{folder}/{news_data['hash']}.txt", "w") as text_file:
        text_file.write(prompt)
        
    ### Load datasets
    openassistant_train_dataset = datasets.load_dataset('json', data_files='datasets/openassistant_best_replies_train.jsonl', split=f'train[:{openassistant_percent}%]')
    openassistant_eval_dataset = datasets.load_dataset('json', data_files='datasets/openassistant_best_replies_eval.jsonl', split=f'train[:{openassistant_percent}%]')

    openassistant_train_dataset = create_fixed_size_dataset(openassistant_train_dataset, MAX_TOKEN_SIZE, tokenizer)
    openassistant_eval_dataset = create_fixed_size_dataset(openassistant_eval_dataset, MAX_TOKEN_SIZE, tokenizer)

    print(f"OpenAssistant train size: {len(openassistant_train_dataset)}, OpenAssistant eval size: {len(openassistant_eval_dataset)}")

    aidj_train_dataset = datasets.load_dataset("text", data_dir="datasets/ai_dj", split="train", sample_by="document")
    aidj_eval_dataset = datasets.load_dataset("text", data_dir="datasets/ai_dj", split="train[:10%]", sample_by="document")

    aidj_train_dataset = create_fixed_size_dataset(aidj_train_dataset, MAX_TOKEN_SIZE, tokenizer)
    aidj_eval_dataset = create_fixed_size_dataset(aidj_eval_dataset, MAX_TOKEN_SIZE, tokenizer)

    print(f"AIDJ train size: {len(aidj_train_dataset)}, AIDJ eval size: {len(aidj_eval_dataset)}")

    memory_train_dataset = datasets.load_dataset("text", data_dir="datasets/memory", split="train", sample_by="document")
    memory_eval_dataset = datasets.load_dataset("text", data_dir="datasets/memory", split="train[:10%]", sample_by="document")

    memory_train_dataset = create_fixed_size_dataset(memory_train_dataset, MAX_TOKEN_SIZE, tokenizer)
    memory_eval_dataset = create_fixed_size_dataset(memory_eval_dataset, MAX_TOKEN_SIZE, tokenizer)

    print(f"Memory train size: {len(memory_train_dataset)}, Memory eval size: {len(memory_eval_dataset)}")

    news_train_dataset = datasets.load_dataset("text", data_dir="datasets/news/train", split="train", sample_by="document")
    news_eval_dataset = datasets.load_dataset("text", data_dir="datasets/news/eval", split="train", sample_by="document")

    news_train_dataset = create_fixed_size_dataset(news_train_dataset, MAX_TOKEN_SIZE, tokenizer)
    news_eval_dataset = create_fixed_size_dataset(news_eval_dataset, MAX_TOKEN_SIZE, tokenizer)

    print(f"News train size: {len(news_train_dataset)}, News eval size: {len(news_eval_dataset)}")

    ### Load project code as data
    project_main_folder_dataset       = create_fixed_size_dataset_from_data_folder('../Anagnorisis/', MAX_TOKEN_SIZE, tokenizer)
    project_src_folder_dataset        = create_fixed_size_dataset_from_data_folder('../Anagnorisis/src', MAX_TOKEN_SIZE, tokenizer)
    project_templates_folder_dataset  = create_fixed_size_dataset_from_data_folder('../Anagnorisis/templates', MAX_TOKEN_SIZE, tokenizer)
    project_wiki_folder_dataset       = create_fixed_size_dataset_from_data_folder('../Anagnorisis/wiki', MAX_TOKEN_SIZE, tokenizer)
    project_providers_folder_dataset  = create_fixed_size_dataset_from_data_folder('../Anagnorisis/providers', MAX_TOKEN_SIZE, tokenizer)

    project_dataset = datasets.concatenate_datasets([project_main_folder_dataset, project_src_folder_dataset, project_templates_folder_dataset, project_wiki_folder_dataset, project_providers_folder_dataset])

    # Combine all datasets
    dataset_train = datasets.concatenate_datasets([aidj_train_dataset, memory_train_dataset, news_train_dataset, openassistant_train_dataset, project_dataset]).shuffle()
    dataset_eval = datasets.concatenate_datasets([aidj_eval_dataset, memory_eval_dataset, news_eval_dataset, openassistant_eval_dataset]).shuffle()

    #dataset_train = trl.trainer.ConstantLengthDataset(tokenizer, dataset_train) 
    #dataset_eval = trl.trainer.ConstantLengthDataset(tokenizer, dataset_eval) 

    print(f"TOTAL TRAIN SIZE: {len(dataset_train)}, TOTAL EVAL SIZE: {len(dataset_eval)}")

    #data_files = {"train": train_file_name, "eval": eval_file_name}
    #dataset = load_dataset('json', data_files=data_files) # , split="train"
   
    #dataset = load_dataset("text", data_dir="datasets/memory", split="train")

    

    lora_alpha = 16 
    lora_dropout = 0.1 
    lora_r = 64

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM"
    )

    ### Load trainer
    output_dir = "./models/training_checkpoints"
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 8
    optim = "paged_adamw_32bit"
    save_steps = 10
    logging_steps = 10
    learning_rate = 2e-4
    max_grad_norm = 0.3
    
    warmup_ratio = 0.03
    lr_scheduler_type = "constant"

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=True,
        max_grad_norm=max_grad_norm,
        #max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        warmup_ratio=warmup_ratio,
        #group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        evaluation_strategy="steps",
        load_best_model_at_end = True,
        save_total_limit = 1,
        report_to = "none"
    )

    # pass everything to the model
    #max_seq_length = 512

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        peft_config=peft_config,
        dataset_text_field="text",
        #max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        callbacks=[EmitLogDataCallback(),]
        #packing=True
    )

    # We will also pre-process the model by upcasting the layer norms in float 32 for more stable training
    for name, module in trainer.model.named_modules():
      if "norm" in name:
          module = module.to(torch.float32)

    # Train the model

    # Number of retries in case of CUDA out of memory error
    max_retries = 10
    resume_from_checkpoint = False

    for retry in range(max_retries):
      try:
        # Train the model
        trainer.train(resume_from_checkpoint)
        break  # If training is successful, exit the loop
      #except torch.cuda.OutOfMemoryError as error: # Fixed in the latest Pytorch
      #if "CUDA out of memory" in str(error):
      #  print(f"Retry {retry + 1}/{max_retries} due to CUDA out of memory error.")
      #  # Optionally, you can implement additional cleanup or logging here
      #  # For example, saving checkpoints, logging relevant information, etc.
      #  trainer.resume_from_checkpoint()
      except RuntimeError as error:
        resume_from_checkpoint = True
        
        if "CUDA out of memory" in str(error):
          print(f"Retry {retry + 1}/{max_retries} due to CUDA out of memory error.")
          # Optionally, you can implement additional cleanup or logging here
          # For example, saving checkpoints, logging relevant information, etc.
          # trainer.resume_from_checkpoint()
        else:
          # Handle other runtime errors here
          print(f"Retry {retry + 1}/{max_retries} due to runtime error: {error}")
          # Optionally, you can implement additional cleanup or logging here
          # For example, saving checkpoints, logging relevant information, etc.
          # trainer.resume_from_checkpoint()

    # Evaluate the model to see if the best checkpoint is active
    eval_metric = trainer.evaluate()
    print(eval_metric)

    # Save fine-tuned model
    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
    model_to_save.save_pretrained("models/your_llama2")

    print('Fine-tuning has been completed!')

    ### Reload Peft model
    predictor.model = PeftModel.from_pretrained(predictor.base_model, 'models/your_llama2')
