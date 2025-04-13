# 📽️ User-Based Recommender System (Pure Pandas + Cosine Similarity)

This project implements a **simple user-based collaborative filtering recommender system** using only `pandas` and `scikit-learn`'s `cosine_similarity` — no specialized libraries like `Surprise`.

## 🚀 Features

- Pure Python + Pandas
- User-user similarity with Cosine Similarity
- Predicts unseen item ratings
- Lightweight and explainable
- Ideal as a base for more advanced collaborative filtering

## 🛠️ Libraries Used

- pandas
- scikit-learn (for cosine similarity)

## 📊 Example Dataset

A small sample dataset of movie ratings from users:

| user_id | item     | rating |
|---------|----------|--------|
| 1       | Movie A  | 5      |
| 1       | Movie B  | 3      |
| 2       | Movie A  | 4      |
| ...     | ...      | ...    |

## 📈 Recommendation Logic

1. Build a user-item rating matrix.
2. Compute cosine similarity between all users.
3. For a given user:
   - Find similar users.
   - Aggregate their ratings for items the user hasn’t rated.
   - Recommend the top `N` items.

## ▶️ Usage

```python
recommend_items(user_id=4)
