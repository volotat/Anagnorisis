Some research into the best representation of audio embeddings (and embeddings in general) for LLMs. Might describe it more details in the future. 

### Order of running:
1. create_dataset.py - Creates dataset from a local music library. Saves data into a 'dataset.csv' file.
2. apply_PCA.py - Creates reduces embedding with PCA, lowering latent dimensionality from 768 to 32.
3. train_svm.py - Trains SVM on full and reduced embeddings and compares the results.
```
SVM results:
Train Accuracy on full embeddings: 0.7355408388520971%, Test Accuracy on full embeddings: 0.7255075022065314%
Train Accuracy on reduced embeddings: 0.7355408388520971%, Test Accuracy on reduced embeddings: 0.7255075022065314%
```
4. train_simple_NN.py - Trains simple NN on full and reduced embeddings and compares the results.
```
NN results:
Best Epoch on full embeddings: 84, Train Accuracy: 99.75717439293598%, Test Accuracy: 84.11297440423654%
Best Epoch on reduced embeddings: 90, Train Accuracy: 99.93377483443709%, Test Accuracy: 83.23036187113857%
```
5. train_llama_2_qlora.py - Trains QLoRA for llama 2 model for text-based embedding classification. This time only on reduced embeddings.
```
Reduced embeddings Train Accuracy: 22.57%
Reduced embeddings Test Accuracy: 38.27%
```