## Info
A small research of the best way to learn to rate music based on the user's previous ratings.  

Dataset is exported from the main code of Anagnorisis via "emit_export_audio_dataset" method.  
It contains data of the music artist, title, user score and embedding. The dataset contains 799 datapoints. 
Embeddings are generated using "./models/MERT-v1-95M" with following procedure:
1. Take 5 samples of the music from random points in time within a 5 seconds window.
2. Embed it with MERT-v1-95M model
3. Take a mean along the time dimension for each sample 
4. Take a mean of all samples producing a single 768-dimensional vector for a song at the end.

The exact code of generating embeddings at the moment of generating the dataset looking this way:
``` python
def get_time_reduced_embedding(self, outputs):
    time_reduced_mean = outputs.mean(-2)
    return time_reduced_mean

def embed_audio(self, audio_path, embedder_sampling_points = 5):
    full_waveform = self.mp3_to_waveform(audio_path)

    window_size = self.sampling_rate * self.context_window_seconds

    # Generate random start positions for each part
    start_poses = np.random.choice(len(full_waveform) - window_size, size=embedder_sampling_points)

    # Extract the waveform for each part
    part_waveforms = [full_waveform[start:start+window_size] for start in start_poses]

    # Stack the part waveforms into a batch
    batch_waveform = torch.stack(part_waveforms).to(self.device)

    # Reshape the batch waveform to be 3D
    batch_waveform = batch_waveform.view(-1, batch_waveform.shape[-1])

    # Process the batch waveform
    inputs = self.processor(batch_waveform, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)

    # Move the inputs to the GPU
    inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
    
    # Squeeze the leading dimension from the input_values tensor and its attention mask
    inputs["input_values"] = inputs["input_values"].squeeze(0)
    inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)

    # Calculate the embeddings for the batch
    with torch.no_grad():
      outputs = self.model(**inputs)

    # Get time-reduced embeddings for each part
    sampled_embeddings = self.get_time_reduced_embedding(outputs.last_hidden_state)
    mean_embedding = sampled_embeddings.mean(0)

    # Move the mean_embedding back to the CPU
    mean_embedding = mean_embedding.cpu().detach().numpy()

    return mean_embedding
```

## Metric

``` python
def calculate_metric(self, loader):
    maes = [] 
    with torch.no_grad():
      for data in loader:
        inputs, labels = data
        outputs = self.model(inputs)

        # Calculate weighted average from all predictions to get the final prediction
        predicted = torch.softmax(outputs, dim=-1) * torch.arange(0, self.rate_classes)
        predicted = predicted.sum(dim=-1)

        maes.append(torch.mean(torch.abs(predicted - labels)).item())
    return np.mean(maes)
```

## Results
For reference, metric value on test set compared with the mean value from train set: 1.1083

1. Simple fully connected NN + Full Embeddings  
1.1 Trained with CrossEntropyLoss where each score is a different class (current approach)\
Best Epoch: 29, Train Metric: 0.3227, Test Metric: 0.9615  
1.2 Trained with MAE loss (L1Loss)\
Best Epoch: 182, Train Metric: 0.1168, Test Metric: 0.8783  
1.3 Trained with MSE loss\
Best Epoch: 24, Train Metric: 0.6580, Test Metric: 1.0554  

2. Llama 2 (fine-tuned with QLora) + Truncated Quantized Embedding represented as a sting + Name of the artist and title of a song.

3. Phi-3 (fine-tuned with QLora) + Truncated Quantized Embedding represented as a sting + Name of the artist and title of a song.

## Some logs
### Llama 2 experiments

exp 2:
With [:128] embeddings, 1 epoch  
Before training metric value on test set: 4.3333  
After training metric value on test set: 1.4583  

exp 3:
With [:128] embeddings, 1 epoch + emb normalization  
Before training metric value on test set: 4.3083  
After training metric value on test set: 1.4833  

exp 4:
With [:384] embeddings, 1 epoch  
Before training metric value on test set: 4.2750  
After training: Error  

exp 5:
With [:64] embeddings, 1 epoch.  
Before training metric value on test set: 4.2833  
After training metric value on test set: 1.4833  

exp 6:
With [:128] embeddings, 4 epoch.  
Before training metric value on test set: 4.3333  
After training metric value on test set: 1.4667  

exp 7:
With only artist and title, no embeddings, 4 epoch  
Before training metric value on test set: 4.6083  
After training metric value on test set: 1.0500  

exp 8:
With only artist and title, 4 epoch. Different prompt structure.  
Before training metric value on test set: 4.4500  
After training metric value on test set: 0.9917  

exp 9:
With [:64] embeddings, 4 epoch. Different prompt structure.  
Before training metric value on test set: 4.1750  
After training metric value on test set: 1.3000  

exp 10:
With [:128] embeddings only, 4 epoch. Different prompt structure.  
Before training metric value on test set: 4.1583  
After training metric value on test set: 1.4917  

exp 11:
With nothing, 4 epoch.  
Before training metric value on test set: 4.1583  
After training metric value on test set: 1.1083   

exp 12:
With only artist and title, 10 epoch.
Before training metric value on test set: 4.2333
After training metric value on test set: 1.0000

exp 13:
With [:16] embeddings normalized, 10 epoch.  
Before training metric value on test set: 4.1667
After training metric value on test set: 0.9667

exp 14:
With [:32] embeddings normalized, 10 epoch.
Before training metric value on test set: 4.1667
After training metric value on test set: 1.0083

exp 15:
With 16-D embeddings reduced with PCA and normalized, 10 epoch. 
Before training metric value on test set: 4.1667  
After training metric value on test set: 1.2000

exp 16:
With 32-D embeddings reduced with PCA and normalized, 10 epoch. 
Before training metric value on test set: 4.1417
After training metric value on test set: 0.9667

exp 17:
With 32-D embeddings reduced with PCA and normalized, 10 epoch. At the training stage 5 different prompts are used to trained the model to recognize embeddings more easily, I hope.
Before training metric value on test set: 4.1417
After training metric value on test set: 1.3083

Metric after the training behave very weirdly. I suppose that the reason is that at the training time the evaluation calculated from the preplexity of the whole test prompts and not only the MAE score that calculated at the end. 

exp 18:
With 32-D embeddings reduced with PCA and normalized, 20 epoch. At the training stage 5 different prompts are used to trained the model to recognize embeddings more easily. No evaluation set.
Before training metric value on test set: 4.1417
1 epoch training metric value on test set: 1.3250 
2 epoch training metric value on test set: 1.4917
After training metric value on test set: 1.4750


## Conclusion
Unfortunately, using textualized embeddings with llm with the goal to produce direct data evaluation does not really work, at least on such small size of data that I currently have. So the best strategy for now is to use separate evaluation network for each datatype and find a way to produce a single network that could work with different embeddings later.