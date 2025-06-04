import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Input data
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 4],
    'item': ['Movie A', 'Movie B', 'Movie C', 'Movie A', 'Movie C', 'Movie B', 'Movie D', 'Movie A'],
    'rating': [5, 3, 4, 4, 2, 5, 3, 2]
}
df = pd.DataFrame(data)

# Create user-item matrix
user_item_matrix = df.pivot_table(index='user_id', columns='item', values='rating').fillna(0)

# Centered (normalized) matrix
user_means = user_item_matrix.replace(0, pd.NA).mean(axis=1)
centered_matrix = user_item_matrix.sub(user_means, axis=0).fillna(0)

# Cosine similarity
similarity = cosine_similarity(centered_matrix)
similarity_df = pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Heatmap visualization
# Heatmap of User-Item Ratings
plt.figure(figsize=(10, 6))
sns.heatmap(user_item_matrix, annot=True, cmap='YlGnBu', linewidths=0.5, fmt=".1f")

plt.title("🎬 User vs Movie Rating Matrix", fontsize=16)
plt.xlabel("Movie", fontsize=12)
plt.ylabel("User ID", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()



# Global popularity fallback
item_popularity = df.groupby('item')['rating'].mean().sort_values(ascending=False)

# Recommendation function
def recommend_items(user_id, top_n=3, min_similarity=0.1):
    if user_id not in user_item_matrix.index:
        print("🙋 User not found — recommending popular movies instead.")
        return item_popularity.head(top_n).to_dict()

    sim_users = similarity_df[user_id].drop(user_id)
    sim_users = sim_users[sim_users > min_similarity]

    scores = pd.Series(dtype=float)
    for other_user, sim in sim_users.items():
        other_ratings = centered_matrix.loc[other_user]
        scores = scores.add(other_ratings * sim, fill_value=0)

    predicted_ratings = scores + user_means[user_id]
    already_rated = user_item_matrix.loc[user_id]
    predicted_ratings = predicted_ratings[already_rated == 0]

    if predicted_ratings.empty or predicted_ratings.shape[0] < top_n:
        fallback = item_popularity[~item_popularity.index.isin(already_rated[already_rated > 0].index)]
        predicted_ratings = pd.concat([predicted_ratings, fallback]).drop_duplicates().head(top_n)

    return predicted_ratings.sort_values(ascending=False).round(2).to_dict()

# Friendly output
def print_recommendations(user_id):
    recommendations = recommend_items(user_id)
    print(f"\n🎬 Hi User {user_id}, based on your taste and similar users, you might enjoy:")
    for i, (movie, score) in enumerate(recommendations.items(), start=1):
        print(f" {i}. {movie} (predicted rating: {score}/5)")

# Example usage
print_recommendations(user_id=4)
