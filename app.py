import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from fuzzywuzzy import process
from community import community_louvain
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the model for title similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize session state for data and processing status
if 'data' not in st.session_state:
    st.session_state['data'] = None
    st.session_state['processed'] = False

# Upload data file
st.title('Keyword Clustering with LDA and SERP Overlap')
uploaded_file = st.file_uploader("Upload your keyword CSV file", type=['csv'])

if uploaded_file:
    # Load data into session state
    st.session_state['data'] = pd.read_csv(uploaded_file)
    data = st.session_state['data']
    
    # Field mapping
    keyword_col = st.selectbox('Select the Keyword column:', data.columns)
    impression_col = st.selectbox('Select the Impression column:', data.columns)
    position_col = st.selectbox('Select the Position column:', data.columns)
    url_col = st.selectbox('Select the SERP URL column:', data.columns)
    title_col = st.selectbox('Select the Title column:', data.columns)
    
    # Button to start processing
    if st.button('Run Clustering'):
        # Initialize progress bar
        progress_bar = st.progress(0)
        
        # Deduplicate exact keyword matches by keeping the one with the highest impressions
        data = data.sort_values(by=impression_col, ascending=False).drop_duplicates(subset=keyword_col, keep='first')
        
        # Update progress
        progress_bar.progress(20)
        
        # Fuzzy Matching to clean similar keywords
        unique_keywords = data[keyword_col].unique()
        keyword_mapping = {kw: process.extractOne(kw, unique_keywords)[0] for kw in unique_keywords}
        data[keyword_col] = data[keyword_col].apply(lambda x: keyword_mapping[x])
        
        # Vectorize keywords with their SERP titles
        def vectorize_serp(row):
            # Concatenate keyword with all its related SERP titles
            serp_text = row[keyword_col] + " " + " ".join(row[title_col].split(';'))  # Assuming titles are semicolon-separated
            return model.encode(serp_text)

        data['serp_vector'] = data.apply(vectorize_serp, axis=1)
        
        # Extract topics using LDA on SERP titles
        vectorizer = CountVectorizer(stop_words='english')
        title_matrix = vectorizer.fit_transform(data[title_col])
        
        lda = LatentDirichletAllocation(n_components=5, random_state=42)  # Using 5 topics as an example
        lda_topics = lda.fit_transform(title_matrix)
        
        # Add topic distributions to the DataFrame
        topic_columns = [f'topic_{i}' for i in range(lda_topics.shape[1])]
        lda_df = pd.DataFrame(lda_topics, columns=topic_columns)
        data = pd.concat([data.reset_index(drop=True), lda_df], axis=1)
        
        # Update progress
        progress_bar.progress(40)
        
        # Create a graph where nodes are keywords
        G = nx.Graph()
        
        # Add nodes
        for kw in data[keyword_col].unique():
            G.add_node(kw)
        
        # Add edges based on weighted overlap, SERP similarity, and topic similarity
        num_keywords = len(data[keyword_col].unique())
        for i, keyword1 in enumerate(data[keyword_col].unique()):
            keyword1_data = data[data[keyword_col] == keyword1]
            urls1 = keyword1_data[url_col].values.flatten()
            positions1 = keyword1_data[position_col].values.flatten()
            serp_vector1 = keyword1_data['serp_vector'].values[0]
            topic_dist1 = keyword1_data[topic_columns].values[0]
            
            for j, keyword2 in enumerate(data[keyword_col].unique()):
                if i < j:
                    keyword2_data = data[data[keyword_col] == keyword2]
                    urls2 = keyword2_data[url_col].values.flatten()
                    positions2 = keyword2_data[position_col].values.flatten()
                    serp_vector2 = keyword2_data['serp_vector'].values[0]
                    topic_dist2 = keyword2_data[topic_columns].values[0]
                    
                    # Calculate weighted overlap score
                    overlap_score = 0
                    for url, pos1 in zip(urls1, positions1):
                        if url in urls2:
                            pos2 = positions2[list(urls2).index(url)]
                            if pos1 in range(1, 4) and pos2 in range(1, 4):
                                overlap_score += 3  # High weight for positions 1-3
                            elif pos1 in range(4, 7) and pos2 in range(4, 7):
                                overlap_score += 2  # Medium weight for positions 4-6
                            elif pos1 in range(7, 11) and pos2 in range(7, 11):
                                overlap_score += 1  # Low weight for positions 7-10
                    
                    # Calculate SERP vector similarity
                    serp_similarity = np.dot(serp_vector1, serp_vector2) / (np.linalg.norm(serp_vector1) * np.linalg.norm(serp_vector2))
                    
                    # Calculate topic distribution similarity (cosine similarity)
                    topic_similarity = np.dot(topic_dist1, topic_dist2) / (np.linalg.norm(topic_dist1) * np.linalg.norm(topic_dist2))
                    
                    # Combined score: sum of weighted overlap, SERP similarity, and topic similarity
                    combined_score = overlap_score + serp_similarity + topic_similarity
                    
                    # Add edge if the combined score is positive
                    if combined_score > 0:
                        G.add_edge(keyword1, keyword2, weight=combined_score)
            
            # Update progress
            progress_bar.progress(40 + int((i / num_keywords) * 40))
        
        # Apply Louvain community detection
        if len(G.edges) > 0:  # Ensure there are edges in the graph before partitioning
            partition = community_louvain.best_partition(G, weight='weight')
            
            # Add cluster labels to the data
            data['cluster'] = data[keyword_col].map(partition)
            
            # Rename clusters based on the highest impression keyword
            cluster_names = {}
            for cluster_id in data['cluster'].unique():
                cluster_data = data[data['cluster'] == cluster_id]
                top_keyword = cluster_data.loc[cluster_data[impression_col].idxmax(), keyword_col]
                cluster_names[cluster_id] = top_keyword
                
            data['cluster_name'] = data['cluster'].map(cluster_names)
            
            # Update progress
            progress_bar.progress(100)
            
            # Store processed data in session state
            st.session_state['data'] = data
            st.session_state['processed'] = True
        else:
            st.write('No clusters formed. Input data may need to be adjusted.')
            st.session_state['processed'] = False

    # Display the results if processed
    if st.session_state['processed'] and 'cluster_name' in st.session_state['data'].columns:
        st.write('Clustered keywords:', st.session_state['data'][[keyword_col, impression_col, 'cluster_name']])
        
        # Download the clustered keywords as a CSV file
        st.download_button(
            label="Download clustered keywords",
            data=st.session_state['data'].to_csv(index=False),
            file_name='clustered_keywords.csv',
            mime='text/csv'
        )
    elif not st.session_state['processed']:
        st.write('No clusters formed or processing not yet run.')
