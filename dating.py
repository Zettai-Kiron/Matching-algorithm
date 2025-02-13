import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Example user data
users = [
    {"name": "Alice", "age": 25, "location": [40.7128, -74.0060], "interests": ["music", "movies", "travel"]},
    {"name": "Bob", "age": 30, "location": [34.0522, -118.2437], "interests": ["sports", "movies", "food"]},
    {"name": "Charlie", "age": 28, "location": [40.7128, -74.0060], "interests": ["music", "travel", "food"]}
]

# Preprocess data
def preprocess(users):
    interest_set = set()
    for user in users:
        interest_set.update(user["interests"])
    interest_list = list(interest_set)
    
    user_vectors = []
    for user in users:
        interest_vector = [1 if interest in user["interests"] else 0 for interest in interest_list]
        user_vector = [user["age"]] + user["location"] + interest_vector
        user_vectors.append(user_vector)
    return np.array(user_vectors)

user_vectors = preprocess(users)

# Compute similarity
def compute_similarity(user_vectors):
    # Normalize age and location
    age_location = user_vectors[:, :3]
    age_location = (age_location - np.mean(age_location, axis=0)) / np.std(age_location, axis=0)
    
    # Compute Euclidean distance for age and location
    demographic_similarity = 1 / (1 + np.linalg.norm(age_location[:, np.newaxis] - age_location, axis=2))
    
    # Compute cosine similarity for interests
    interest_similarity = cosine_similarity(user_vectors[:, 3:])
    
    # Combine similarities
    total_similarity = 0.5 * demographic_similarity + 0.5 * interest_similarity
    return total_similarity

similarity_matrix = compute_similarity(user_vectors)

# Find best matches
for i in range(len(similarity_matrix)):
    similarities = list(enumerate(similarity_matrix[i]))
    similarities.pop(i)
    similarities.sort(key=lambda x: x[1], reverse=True)
    print(f"Best match for {users[i]['name']}: {users[similarities[0][0]]['name']} (Similarity: {similarities[0][1]:.2f})")