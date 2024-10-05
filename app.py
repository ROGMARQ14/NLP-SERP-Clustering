import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Upload data file
st.title('Keyword Clustering Based on Search Intent')
uploaded_file = st.file_uploader("Upload your keyword CSV file", type=['csv'])

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write('Sample data:', data.head())

    # Data preprocessing (lowercase keywords)
    data['Keyword'] = data['Keyword'].str.lower()
    
    # Embedding keywords and their SERP results using Sentence Transformers
    def embed_keywords_and_serps(row):
        # Combine keyword and its search result information for embedding
        text = f"{row['Keyword']} {row['Title']} {row['URL']}"
        return model.encode(text)

    data['embedding'] = data.apply(embed_keywords_and_serps, axis=1)
    embeddings = np.vstack(data['embedding'].values)
    
    # Clustering using DBSCAN (or other clustering method)
    clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(embeddings)
    data['cluster'] = clustering.labels_

    # Function to de-dupe keywords with high SERP similarity
    def deduplicate_keywords(cluster_data):
        # Calculate SERP similarity within the cluster using cosine similarity
        serp_embeddings = np.vstack(cluster_data['embedding'].values)
        similarity_matrix = cosine_similarity(serp_embeddings)
        
        # Identify duplicates (high similarity) and retain the keyword with the highest impressions
        retained_keywords = []
        for i, row in cluster_data.iterrows():
            if all(row['Keyword'] not in rk['Keyword'] for rk in retained_keywords):
                similar_indices = np.where(similarity_matrix[i] > 0.9)[0]
                similar_keywords = cluster_data.iloc[similar_indices]
                # Pick the one with the highest impression
                best_keyword = similar_keywords.loc[similar_keywords['Impr'].idxmax()]
                retained_keywords.append(best_keyword)
        return pd.DataFrame(retained_keywords)

    # Apply deduplication within clusters
    clustered_keywords = []
    for cluster in data['cluster'].unique():
        if cluster == -1:  # Noise points, skip them
            continue
        cluster_data = data[data['cluster'] == cluster]
        deduped_cluster = deduplicate_keywords(cluster_data)
        clustered_keywords.append(deduped_cluster)

    # Concatenate all clusters
    final_clusters = pd.concat(clustered_keywords, ignore_index=True)
    
    # Display the clustered results
    st.write('Clustered and deduplicated keywords:', final_clusters[['Keyword', 'Impr', 'cluster']])
    
    # Download the clustered keywords as a CSV file
    st.download_button(
        label="Download clustered keywords",
        data=final_clusters.to_csv(index=False),
        file_name='clustered_keywords.csv',
        mime='text/csv'
    )
