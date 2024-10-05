import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Upload data file
st.title('Keyword Clustering Based on Search Intent')
uploaded_file = st.file_uploader("Upload your keyword CSV file", type=['csv'])

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write('Sample data:', data.head())
    
    # Field mapping
    keyword_col = st.selectbox('Select the Keyword column:', data.columns)
    impression_col = st.selectbox('Select the Impression column:', data.columns)
    position_col = st.selectbox('Select the Position column:', data.columns)
    title_col = st.selectbox('Select the Title column:', data.columns)
    
    # Data preprocessing (lowercase keywords)
    data[keyword_col] = data[keyword_col].str.lower()
    
    # Embedding keywords and their SERP titles using Sentence Transformers
    def embed_keywords_and_titles(row):
        # Combine keyword and its search result title for embedding
        text = f"{row[keyword_col]} {row[title_col]}"
        return model.encode(text)

    data['embedding'] = data.apply(embed_keywords_and_titles, axis=1)
    embeddings = np.vstack(data['embedding'].values)
    
    # Clustering options
    clustering_method = st.selectbox('Choose a clustering method:', ['DBSCAN', 'Agglomerative', 'KMeans'])
    
    if clustering_method == 'DBSCAN':
        eps = st.slider('Select epsilon (eps) for DBSCAN:', 0.1, 1.0, 0.5)
        min_samples = st.slider('Select minimum samples for DBSCAN:', 1, 10, 2)
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
    
    elif clustering_method == 'Agglomerative':
        n_clusters = st.slider('Select number of clusters for Agglomerative Clustering:', 2, 50, 10)
        clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='cosine', linkage='average').fit(embeddings)
    
    elif clustering_method == 'KMeans':
        n_clusters = st.slider('Select number of clusters for KMeans:', 2, 50, 10)
        clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    
    # Assign cluster labels
    data['cluster'] = clustering.labels_
    
    # Function to de-dupe keywords with high SERP similarity
    def deduplicate_keywords(cluster_data):
        # Calculate SERP similarity within the cluster using cosine similarity
        serp_embeddings = np.vstack(cluster_data['embedding'].values)
        similarity_matrix = cosine_similarity(serp_embeddings)
        
        # Identify duplicates (high similarity) and retain the keyword with the highest impressions
        retained_keywords = []
        for i, row in cluster_data.iterrows():
            if all(row[keyword_col] not in rk[keyword_col] for rk in retained_keywords):
                similar_indices = np.where(similarity_matrix[i] > 0.9)[0]
                similar_keywords = cluster_data.iloc[similar_indices]
                # Pick the one with the highest impression
                best_keyword = similar_keywords.loc[similar_keywords[impression_col].idxmax()]
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
    st.write('Clustered and deduplicated keywords:', final_clusters[[keyword_col, impression_col, 'cluster']])
    
    # Visualization of embeddings
    st.write('Visualizing clusters...')
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=data['cluster'], cmap='viridis', alpha=0.6)
    plt.title('Keyword Embeddings Cluster Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter, label='Cluster')
    st.pyplot(plt)

    # Download the clustered keywords as a CSV file
    st.download_button(
        label="Download clustered keywords",
        data=final_clusters.to_csv(index=False),
        file_name='clustered_keywords.csv',
        mime='text/csv'
    )
