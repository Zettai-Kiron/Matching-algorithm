from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Example user profiles (rows: users, columns: features)
user_profiles = np.array([
    [1, 0, 1, 0],  # User 1: likes movies, doesn't like sports
    [0, 1, 0, 1],  # User 2: likes sports, doesn't like movies
    [1, 1, 0, 0]   # User 3: likes movies and sports
])

# Compute pairwise cosine similarity
similarity_matrix = cosine_similarity(user_profiles)

# Print similarity matrix
print("Similarity Matrix:")
print(similarity_matrix)

# Find the best match for each user
for i in range(len(similarity_matrix)):
    # Exclude self-similarity
    similarities = list(enumerate(similarity_matrix[i]))
    similarities.pop(i)
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    print(f"Best match for User {i+1}: User {similarities[0][0]+1} (Similarity: {similarities[0][1]:.2f})")