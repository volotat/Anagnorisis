import pandas as pd
from sklearn.decomposition import PCA

import utils

utils.set_seed(42)

# Load the existing dataset
df = pd.read_csv("dataset.csv")

# Convert the embeddings column from string to list
df['embeddings'] = df['embeddings'].apply(lambda x: eval(x))

# Create a PCA object
pca = PCA(n_components=64)

# Fit the PCA object on the embeddings and transform the embeddings
reduced_embeddings = pca.fit_transform(df['embeddings'].tolist())
print("Reduced embeddings shape:", reduced_embeddings.shape)

# Convert the reduced embeddings to lists and add them to the dataframe
df['reduced_embeddings'] = reduced_embeddings.tolist()

# Save the dataframe to the same csv file
df.to_csv("dataset.csv", index=False)