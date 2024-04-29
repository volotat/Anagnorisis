Some research into the best representation of audio embeddings (and embeddings in general) for LLMs. Might describe it more details in the future. 

### Order of running:
1. create_dataset.py - Creates dataset from a local music library. Saves data into a 'dataset.csv' file.
2. apply_PCA.py - Creates reduces embedding with PCA, lowering latent dimensionality from 768 to 32.
3. train_svm.py - Trains SVM on full and reduced embeddings and compares the results.
```
SVM results:
Full embeddings Train Accuracy: 0.74%, Test Accuracy: 0.72%;
Reduced embeddings Train Accuracy: 0.87%, Test Accuracy: 0.76%
```
4. train_simple_NN.py - Trains simple NN on full and reduced embeddings and compares the results.
```
NN results:
Full embeddings Train Accuracy: 100.00%, Test Accuracy: 84.20%;
Reduced embeddings Train Accuracy: 100.00%, Test Accuracy: 76.61%;
```
5. train_llama_2_qlora.py - Trains QLoRA for llama 2 model for text-based embedding classification. This time only on reduced embeddings.
```
Llama2 + QLoRA results:
Reduced embeddings Train Accuracy: 46.56%, Test Accuracy: 38.27%;
```