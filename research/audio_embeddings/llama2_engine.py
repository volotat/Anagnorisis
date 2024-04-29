# set up inference via HuggingFace
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria, LlamaTokenizer
import transformers
import torch
import traceback
import pickle

from peft import PeftModel, LoraConfig, get_peft_model

import gc
import os
import re

from transformers import TrainingArguments
from trl import SFTTrainer
import trl



class StoppingCriteriaSub(StoppingCriteria):
  def __init__(self, stops = [], encounters=1):
    super().__init__()
    self.stops = [stop.to("cuda") for stop in stops]

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
    for stop in self.stops:
      if torch.all((stop == input_ids[0][-len(stop):])).item():
        return True

    return False

class CustomTokenizer(LlamaTokenizer):
  def tokenize(self, text, **kwargs):
    # Split the text into segments by the [music] tags
    segments = re.split(r'(\[music\].*?\[/music\])', text)

    tokens = []
    for i, segment in enumerate(segments):
      if segment.startswith('[music]') and segment.endswith('[/music]'):
        # If the segment is inside [music] tags, tokenize it on the character level
        # and preserve the [music] tags
        tokens.extend(super().tokenize('[music]', **kwargs))
        tokens.extend(list(segment[7:-8])) 
        tokens.extend(super().tokenize('[/music]', **kwargs))
      else:
        # Otherwise, use the default tokenization
        tokens.extend(super().tokenize(segment, **kwargs))

    return tokens

class TextPredictor:
  _instance = None

  def __new__(self, *args, **kwargs):
    if not self._instance:
      self._instance = super().__new__(self)
      self.tokenizer = None
      self.base_model = None
      self.model = None
      #self.load_model(self)
      
    return self._instance
    
  def __init__(self, ) -> None:
    pass

  def load_model(self, base_model_path = "./models/llama2_7b_chat_uncensored", lora_model_path = None):
    self.tokenizer = CustomTokenizer.from_pretrained(base_model_path, local_files_only=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",    # I've also tried removing this line
        bnb_4bit_compute_dtype=torch.float16,
        #bnb_4bit_use_double_quant=True,    # I've also tried removing this line
    )

    self.base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,
        local_files_only=True,
        #config="models/your_llama2/adapter_config.json"
    )

    if lora_model_path is None:
      self.model = self.base_model
    else:
      self.model = PeftModel.from_pretrained(self.base_model, lora_model_path)
    
    stop_words_ids = [torch.tensor([2277, 29937]), torch.tensor([835]), ] #   "\n### ", "###"
    #print('stop_words_ids', stop_words_ids)
    self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

  def unload_model(self):
    self.tokenizer = None
    self.base_model = None
    self.model = None
    gc.collect()
    torch.cuda.empty_cache()

  # Generate responses
  def generate(self, prompt, temperature, max_new_tokens):
    if self.tokenizer is None or self.model is None:
      self.load_model()

    inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

    if temperature == 0:
      generation_config = None
    else:
      generation_config = transformers.GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        #early_stopping=True,
        #**kwargs,
      )

    max_length = 512*5 if max_new_tokens is None else None
    generation_outputs = self.model.generate(
        **inputs,
        generation_config=generation_config,
        #return_dict_in_generate=True,
        #output_scores=True,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        stopping_criteria=self.stopping_criteria,
    )
    
    #output_tokens = self.tokenizer.convert_ids_to_tokens(generation_outputs[0])
    #word_token_pairs = [[word, token_id] for word, token_id in zip(output_tokens, generation_outputs[0].tolist())]
    #print('GEN OUT:', word_token_pairs)

    #print(generation_outputs)
    generation_outputs = generation_outputs[0].to("cpu")
    generation_outputs = generation_outputs[len(inputs["input_ids"][0]):]
    outputs = self.tokenizer.decode(generation_outputs, skip_special_tokens=True)  
    return outputs

  def predict_from_text(self, text, temperature = 0.3, max_new_tokens = 2048):
    response = self.generate(text, temperature, max_new_tokens)
    #result = response[len(text):]
    #if result[-3:] == "###": result = result[:-3]

    return response
  
  def fine_tune_model(self, dataset_train, dataset_eval, lora_model_path="./lora", num_train_epochs = 1):
    self.model.config.use_cache = False
    self.base_model.config.use_cache = False

    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.tokenizer.padding_side = 'right'

    os.makedirs(lora_model_path, exist_ok=True)

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
    output_dir = os.path.join(lora_model_path, 'training_checkpoints')
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
        evaluation_strategy="steps" if dataset_eval is not None else "no",
        load_best_model_at_end = True if dataset_eval is not None else False,
        save_total_limit = 1,
        report_to = "none",
    )

    # pass everything to the model
    #max_seq_length = 512

    trainer = SFTTrainer(
        model=self.base_model,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        peft_config=peft_config,
        dataset_text_field="text",
        #max_seq_length=max_seq_length,
        tokenizer=self.tokenizer,
        args=training_arguments,
        #callbacks=[EmitLogDataCallback(),]
        #packing=True
    )

    # We will also pre-process the model by upcasting the layer norms in float 32 for more stable training
    for name, module in trainer.model.named_modules():
      if "norm" in name: module = module.to(torch.float32)

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
        #resume_from_checkpoint = True
        
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
    print('eval_metric:', eval_metric)

    # Save fine-tuned model
    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(lora_model_path)

    print('Fine-tuning has been completed!')

    ### Reload Peft model
    self.model = PeftModel.from_pretrained(self.base_model, lora_model_path)