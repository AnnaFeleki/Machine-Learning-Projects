import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Sample user-item interaction data
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 4],
    'item': ['Movie A', 'Movie B', 'Movie C', 'Movie A', 'Movie C', 'Movie B', 'Movie D', 'Movie A'],
    'rating': [5, 3, 4, 4, 2, 5, 3, 2]
}

df = pd.DataFrame(data)

# Create user-item matrix
user_item_matrix = df.pivot_table(index='user_id', columns='item', values='rating').fillna(0)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Function to recommend items to a user
def recommend_items(user_id, num_recommendations=2):
    if user_id not in user_item_matrix.index:
        return []

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]  # Exclude self

    weighted_ratings = pd.Series(dtype=float)
    
    for other_user, sim_score in similar_users.items():
        other_ratings = user_item_matrix.loc[other_user]
        weighted_ratings = weighted_ratings.add(other_ratings * sim_score, fill_value=0)
    
    already_rated = user_item_matrix.loc[user_id]
    recommendations = weighted_ratings[already_rated == 0].sort_values(ascending=False)
    
    return recommendations.head(num_recommendations)

# Example usage
user_id = 4
print(f"Recommended items for User {user_id}:\n", recommend_items(user_id))
