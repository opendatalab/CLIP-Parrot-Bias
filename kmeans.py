import numpy as np
import os
import faiss

### Replace all_feats to your own CLIP features
### CLIP Model uses ViT-B-32 from OPENAI

### IncrementalPCA Training
from joblib import dump, load
from sklearn.decomposition import IncrementalPCA
transformer = IncrementalPCA(n_components=256, batch_size=200000)
for feat in all_feats:
    feats = np.load(feat)
    transformer.partial_fit(feats)
dump(transformer, 'pca.joblib') 

### IncrementalPCA Loading
transformer = load('pca.joblib')
for feats in all_feats:
    feats = transformer.transform(feats)
    feats = np.squeeze(np.float16(feats))
    np.save(path, feats)

################ Kmeans ####################
import numpy as np
import os
import faiss

### Training Kmeans
kmeans = faiss.Kmeans(256, 4000, 
    verbose=True, gpu=True, niter=300, nredo=10, seed=42,
    min_points_per_centroid=10000, max_points_per_centroid=200000)
kmeans.train(all_pca_feats)

# save centroids to file
with open("centroids.npy", 'wb') as f:
    np.save(f, kmeans.centroids)


### Eval with Kmeans
centroids = np.load("centroids.npy")
kmeans = faiss.Kmeans(256, 4000, verbose=True, gpu=True, niter=0, nredo=0, seed=42)
kmeans.train(all_eval_feats, init_centroids=centroids)
assert np.sum(kmeans.centroids - centroids) == 0, "centroids are not the same" # sanity check
cluster_distances, cluster_indices = kmeans.assign(all_eval_feats)
