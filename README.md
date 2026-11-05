# Graph-Based YouTube Video Recommendation System

A recommendation system for YouTube trending videos built on graph neural networks. It combines video metadata, channel (user) metadata, and graph structure to learn embeddings, then recommends videos by scoring users against items in that embedding space.

The original pipeline trained a **Graph Convolutional Network (GCN)**. After the preprocessing stage, the training algorithm was **changed to GraphSAGE**, which scales better because it samples a fixed number of neighbors instead of using the full normalized adjacency matrix.

## Pipeline Overview

```
Raw data (trending videos + channel features)
            |
            v
   Cleaning (dropna, drop_duplicates)
            |
            v
   Graph construction
   - Video similarity graph (same category)  -> SQLite
   - User-item graph (channel -> its videos)
            |
            v
   Feature engineering
   - Sentence embeddings (text fields)
   - Age, ratio and numeric features
   - Standard scaling, vector expansion
            |
            v
   ====== Preprocessing complete ======
            |
            v
   Training: GraphSAGE        (replaces GCN)
   - Neighbor sampling + mean aggregation
   - 2 GraphSAGE layers + linear prediction head
   - Target: Likes-to-Views ratio
            |
            v
   User and item embeddings (Z_user, Z_item)
            |
            v
   Recommendation (top-k by similarity score)
            |
            v
   YouTube search links for recommended videos
```

## Data

| File | Description |
|---|---|
| `IN_youtube_trending_data.csv` | Indian YouTube trending videos (video ID, title, tags, description, category, channel, views, likes, publish date, etc.) |
| `user_features23.csv` | Channel-level features (channel ID, about, keywords, publish date, numeric channel statistics) |

Rows with missing values and duplicates are removed from both datasets. Channels that appear in both datasets are kept, and a subset of additional videos is added to the item set, then de-duplicated by `video_id`.

## Preprocessing

### 1. Video similarity graph
Videos are connected to other videos in the same `categoryId`. For each video, the set of same-category videos (excluding itself) is its neighbor set. The graph is generated lazily and saved in batches (5,000 rows) to a SQLite database (`video_graph.db`, table `video_graph`) so it does not need to fit in memory. This graph provides the neighborhoods used by GraphSAGE.

### 2. User-item adjacency
A bipartite adjacency matrix connects each channel (user) to the videos it published. The matrix is symmetrically normalized, and the full user-item block adjacency (users + items) is assembled. This was the structure consumed by the original GCN, and it is kept for the baseline.

### 3. Feature engineering

**Item (video) features**
- `Likes_to_Views_Ratio` = likes / view_count (0 when views are 0). This is the training target.
- `video_age_days` computed from the publish date.
- Category name mapped from `categoryId` and embedded with `all-MiniLM-L6-v2`.
- Title, tags, and description embedded with `paraphrase-multilingual-MiniLM-L12-v2`.

**User (channel) features**
- `channel_age` computed from the channel's publish date.
- `about` and `keywords` text embedded with `paraphrase-multilingual-MiniLM-L12-v2`.

**Common steps**
- Identifier and raw text columns are dropped after embedding.
- Numeric columns are standardized with `StandardScaler`.
- Embedding vectors are expanded into individual numeric columns.
- `view_count` and `likes` are removed from the item features to avoid leaking the target.
- The user and item matrices are combined into one block feature matrix `X`, and intermediate artifacts are written to CSV (`Xlast.csv`, `Alast.csv`, `user_featuresfinal.csv`, `item_featuresfinal.csv`).

## Training: GraphSAGE (changed from GCN)

### What changed and why
| | GCN (original) | GraphSAGE (current) |
|---|---|---|
| Neighborhood | Full normalized adjacency matrix `A_cap` | Sampled neighbors from the video similarity graph |
| Aggregation | `A_cap @ X` (symmetric normalization) | Mean of sampled neighbor features |
| Memory | Dense `(n_users + n_items)^2` matrix | Neighbor lists only, no dense adjacency |
| Batching | Full-batch | Mini-batches (default 5,000) |
| Activation | ReLU | Leaky ReLU (slope 0.01) |
| Optimizer | Custom AdamW-style update with gradient normalization | Adam |

The dense adjacency used by the GCN grows quadratically with the number of nodes, which becomes impractical for large trending datasets. GraphSAGE aggregates a fixed-size sample of neighbors per node, so cost stays bounded and the model works on graphs stored as neighbor lists, such as the one in SQLite.

### Model (`GraphSageItem`)
The model is implemented from scratch in PyTorch with manual backpropagation.

1. **Layer 1:** mean-aggregate neighbor features (up to 5 randomly sampled neighbors per node), multiply by `W1`, apply Leaky ReLU.
2. **Layer 2:** mean-aggregate layer-1 outputs over sampled neighbors, multiply by `W2`, apply Leaky ReLU. This output is the node embedding (`get_embedding()`).
3. **Prediction head:** linear layer `W3` outputs the predicted Likes-to-Views ratio.
4. **Loss:** mean squared error against the true ratio.
5. **Optimization:** Adam (`beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`), weights initialized with Xavier uniform, mini-batch training with shuffled indices.

### Training usage

```python
X = torch.tensor(X_train.values, dtype=torch.float32)
Y = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
A = graph.copy()          # dict: video index -> set of neighbor indices

n_n_1 = int(X.shape[1])
n_n_2 = int(X.shape[1])
n_n_3 = 1
learning_rate = 0.001
n_iterations = 10

graphsageitem = GraphSageItem(A, X, Y, learning_rate, n_iterations, n_n_1, n_n_2, n_n_3)
graphsageitem.backward_propagation(batch_size=5000)
Z_item = graphsageitem.get_embedding()
```

Each epoch prints the average loss and time:

```
Epoch 1/10 | Loss: 0.000000 | Time: 0.00s
```

## Recommendation

1. Embeddings are saved as `Z_item.csv` and `Z_user.csv`.
2. Prediction scores are computed as the dot product between user and item embeddings, and the top-K items per user are retrieved.
3. For a new or test feature vector, cosine similarity against all user embeddings finds the closest users, and their top items are returned.
4. Each recommended `video_id` is turned into a YouTube search link.

Key parameters: `K_train` (items retrieved per user) and `k_top_pick` (number of similar users used).

## Requirements

```
pandas
numpy
torch
scikit-learn
sentence-transformers
polars
```

`sqlite3` is part of the Python standard library. The original code was developed in Google Colab and reads data from Google Drive, so update file paths if running elsewhere.

## Running

1. Place the two CSV files in your working directory (or update the paths).
2. Run the preprocessing steps to build the graph, features, and CSV artifacts.
3. Train the GraphSAGE model on the preprocessed features and graph.
4. Save the embeddings and run the recommendation step.

## Output Artifacts

| File | Contents |
|---|---|
| `video_graph.db` | SQLite store of the video similarity graph |
| `Xlast.csv` | Combined user-item feature matrix |
| `Alast.csv` | Normalized user-item adjacency (GCN baseline) |
| `user_featuresfinal.csv`, `item_featuresfinal.csv` | Cleaned feature tables used for lookups |
| `Z_item.csv`, `Z_user.csv` | Learned embeddings |

## Notes

- The GCN implementation is kept in the repository as a baseline for comparison with GraphSAGE.
- `GraphSageItem` trains on the video similarity graph. Make sure the user embeddings used at recommendation time come from the same embedding space as the item embeddings.
- Update file paths, dataset names, and hyperparameters to match your environment.
