#!/usr/bin/env python
# coding: utf-8

# Movie Recommandation Engine(item based):
# 
# So here, I implemented a basic movie recommandation system based on the genres, cast and the keywords.
# The data set I have used is tmdb_5000_movies.
# 
# First load all the required libraries.

# In[1]:


import numpy as np, pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load the dataset

# In[2]:


movies=pd.read_csv("tmdb_5000_movies.csv")
credits=pd.read_csv("tmdb_5000_credits.csv")


# In[3]:


movies.head()


# We can see what the movies dataset has. There are various columns. 
# The columns that I would be considering would be genres, original_title, keywords.

# In[4]:


credits.head()


# We can see what credits dataset contains. And I'm considering title movie id and cast

# In[5]:


movies.rename(columns={'id':'movie_id'},inplace=True)


# I renamed the column id as movie_id. 
# When inplace=True, the data is renamed in place, nothing is returned
# When inplace=False, it performs the operation and returns the copy of the object. 
# So according to our use we can use either of it.

# In[6]:


movies.head()


# I merged the both movies and credits on movie_id

# In[7]:


full_data=pd.merge(movies,credits,on='movie_id')
full_data.head()


# In[8]:


full_data.shape


# In[9]:


full_data.isna().any()


# In[10]:


full_data.describe()


# In[11]:


full_data.info()


# I just considered below mentioned columns for this recommandation engine

# In[42]:


mvs=full_data[['movie_id','original_title','genres','keywords','cast']]


# In[43]:


mvs.head()


# In[44]:


mvs.shape


# In[45]:


mvs.isna().any()


# Handling the missing values. Since there aren't any missing values we can let it be. if any, we can perform below command to handle
# missing values.

# In[46]:


mvs.dropna(inplace=True)


# In[47]:


mvs.shape


# Now from genre column I just want to extract genres without any id's, same for keywords and cast column. 
# 
# In the below extract_name() function, we first verify whether given input is a valid python literal or not if it is valid, we check whether the given input is a list and Extract the 'name' key from each dictionary in the list, if the dictionary contains the 'name' key and returns the empty list if there was an error or genre_list is not actaully a list.

# In[48]:


def extract_names(genre_str):
    try:
        
        genre_list = ast.literal_eval(genre_str)
        
        if isinstance(genre_list, list):
            return [item['name'] for item in genre_list if isinstance(item, dict) and 'name' in item]
    except (ValueError, SyntaxError):
        pass 
    return []


# In[49]:


mvs['genres'] = mvs['genres'].apply(extract_names)


# In[50]:


mvs.head()


# In[51]:


mvs['keywords']=mvs['keywords'].apply(extract_names)


# In[52]:


mvs.head()


# In[53]:


def extract_chars(cast_str):
    try:
        cast_list=ast.literal_eval(cast_str)
        if isinstance(cast_list,list):
            return [charc['character'] for charc in cast_list if isinstance(charc,dict) and 'character' in charc]
    except (ValueError, SyntaxError):
        pass  
    return []
            


# In[54]:


mvs['cast']=mvs['cast'].apply(extract_chars)


# In[55]:


mvs.head()


# In[61]:


get_ipython().system('pip install wordcloud')


# I used a word cloud to get a brief idea on what kind of movies are there in our dataset. We can see science fiction movies are 
# dominating others

# In[62]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
genres_column = mvs['genres']
print(genres_column)
all_genres = [genre for sublist in genres_column for genre in sublist]
genre_text = ' '.join(all_genres)


# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(genre_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[64]:


mvs['hashtags']=mvs['genres']+mvs['keywords']+mvs['cast']
mvs.head()


# In[65]:


mvs.drop(columns=['genres','keywords','cast'],inplace=True)


# In[66]:


def remove_space(L):
    New_L=[]
    for i in L:
        New_L.append(i.replace(" ",""))
    return New_L

mvs['hashtags']=mvs['hashtags'].apply(remove_space)


# In[67]:


hash_column = mvs['hashtags']

all_hash = [hash for sublist in hash_column for hash in sublist]
hash_text = ' '.join(all_hash)


# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(hash_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[68]:


mvs.head()


# In[69]:


mvs['hashtags'] = mvs['hashtags'].apply(lambda x: ' '.join(x))
mvs.head()


# In[70]:


vector=CountVectorizer()
hashtag_matrix=vector.fit_transform(mvs['hashtags'])
hashtag_matrix


# In[71]:


cosine_sim = cosine_similarity(hashtag_matrix, hashtag_matrix)
print(cosine_sim)


# In[72]:


cosine_sim.shape


# In[73]:


def recommand_movies(movietitle,cosine_sim,n):
    try:
        idx = mvs[mvs['original_title'] == movie_title].index[0]
    except IndexError:
        print(f"Error: '{movie_title}' not found in the dataset.")
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in top_sim_scores]

    return mvs.iloc[movie_indices].original_title

movie=input("Enter a movie: ")
num_recommendations=int(input("Number of recommendations you want: "))
print()
print(f'Movies similar to {movie}')
print()
recommended_movies = recommand_movies(movie, cosine_sim,num_recommendations)
print(recommended_movies)


# In[ ]:





# In[ ]:




