# models.py

import pandas as pd
import numpy as np

# Model 1: Collaborative Filtering (Simplified, with user ratings)
def collaborative_filtering(data, user_ratings):
    # Example logic: Incorporate user ratings to adjust recommendations
    effectiveness = np.random.uniform(0.7, 0.9)  
    efficiency = np.random.uniform(0.5, 0.7) 
    # Recommend items based on user ratings (simple example)
    recommendations = data.sample(n=5)
    
    # Sort recommendations based on user preferences
    if user_ratings is not None:
        recommendations['User Score'] = recommendations.iloc[:, 0].map(user_ratings)
        recommendations.sort_values(by='User Score', ascending=False, inplace=True)
        recommendations.drop(columns='User Score', inplace=True)
    
    return effectiveness, efficiency, recommendations

# Model 2: Content-Based Filtering (Simplified, with user ratings)
def content_based_filtering(data, user_ratings):
    # Example logic: Adjust content-based filtering with user ratings
    effectiveness = np.random.uniform(0.6, 0.85)  
    efficiency = np.random.uniform(0.6, 0.8)  
    recommendations = data.sample(n=5)
    
    # Sort recommendations based on user preferences
    if user_ratings is not None:
        recommendations['User Score'] = recommendations.iloc[:, 0].map(user_ratings)
        recommendations.sort_values(by='User Score', ascending=False, inplace=True)
        recommendations.drop(columns='User Score', inplace=True)
    
    return effectiveness, efficiency, recommendations


