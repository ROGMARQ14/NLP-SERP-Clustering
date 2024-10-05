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
    
    # Embedding keywords and titles (excluding URLs)
    def embed_keywords_and_titles(row):
        # Combine keyword and title for embedding
        text = f"{row['Keyword']} {row['Title']}"
        return model.encode(text)

    # Generate embeddings for each row
    data['embedding'] = data.apply(embed_keywords_and_titles, axis=1)
    embeddings = np.vstack(data['embedding'].values)
    
    # Clustering using DBSCAN (based on cosine similarity)
    clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(embeddings)
    data['cluster'] = clustering.labels_

    # Function to deduplicate keywords within clusters based on SERP similarity
    def deduplicate_keywords(cluster_data):
        # Calculate similarity within the cluster
        serp_embeddings = np.vstack(cluster_data['embedding'].values)
        similarity_matrix = cosine_similarity(serp_embeddings)
        
        # Identify and retain keywords with the highest impressions in cases of high similarity
        retained_keywords = []
        for i, row in cluster_data.iterrows():
            if all(row['Keyword'] not in rk['Keyword'] for rk in retained_keywords):
                similar_indices = np.where(similarity_matrix[i] > 0.9)[0]
                similar_keywords = cluster_data.iloc[similar_indices]
                # Pick the one with the highest impression
                best_keyword = similar_keywords.loc[similar_keywords['Impr'].idxmax()]
                retained_keywords.append(best_keyword)
        return pd.DataFrame(retained_keywords)

    # Apply deduplication within each cluster
    clustered_keywords = []
    for cluster in data['cluster'].unique():
        if cluster == -1:  # Skip noise points
            continue
        cluster_data = data[data['cluster'] == cluster]
        deduped_cluster = deduplicate_keywords(cluster_data)
        
        # Set the cluster name to the keyword with the highest impressions
        cluster_name = deduped_cluster.loc[deduped_cluster['Impr'].idxmax(), 'Keyword']
        deduped_cluster['cluster_name'] = cluster_name
        
        clustered_keywords.append(deduped_cluster)

    # Combine all clusters into a final DataFrame
    final_clusters = pd.concat(clustered_keywords, ignore_index=True)
    
    # Display the final clustered and deduplicated keywords
    st.write('Clustered and deduplicated keywords:', final_clusters[['cluster_name', 'Keyword', 'Impr', 'Position', 'Title', 'URL']])
    
    # Allow user to download the clustered results as a CSV
    st.download_button(
        label="Download clustered keywords",
        data=final_clusters.to_csv(index=False),
        file_name='clustered_keywords.csv',
        mime='text/csv'
    )
