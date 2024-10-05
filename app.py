import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from fuzzywuzzy import process
from community import community_louvain
from sentence_transformers import SentenceTransformer

# Initialize the model for title similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize session state for data and processing status
if 'data' not in st.session_state:
    st.session_state['data'] = None
    st.session_state['processed'] = False

# Upload data file
st.title('Enhanced Keyword Clustering with Overlap Threshold and SERP Vectorization')
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
    
    # Overlap threshold input
    overlap_threshold = st.slider('Set the minimum number of overlapping URLs:', min_value=1, max_value=10, value=3)
    
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
        
        # Update progress
        progress_bar.progress(40)
        
        # Create a graph where nodes are keywords
        G = nx.Graph()
        
        # Add nodes
        for kw in data[keyword_col].unique():
            G.add_node(kw)
        
        # Add edges based on SERP URL overlap and SERP vector similarity
        num_keywords = len(data[keyword_col].unique())
        for i, keyword1 in enumerate(data[keyword_col].unique()):
            urls1 = set(data[data[keyword_col] == keyword1][url_col].values.flatten())
            serp_vector1 = data[data[keyword_col] == keyword1]['serp_vector'].values[0]
            
            for j, keyword2 in enumerate(data[keyword_col].unique()):
                if i < j:
                    urls2 = set(data[data[keyword_col] == keyword2][url_col].values.flatten())
                    serp_vector2 = data[data[keyword_col] == keyword2]['serp_vector'].values[0]
                    
                    # SERP overlap calculation with a threshold
                    overlap_urls = urls1.intersection(urls2)
                    if len(overlap_urls) >= overlap_threshold:
                        # SERP vector similarity
                        serp_similarity = np.dot(serp_vector1, serp_vector2) / (np.linalg.norm(serp_vector1) * np.linalg.norm(serp_vector2))
                        
                        # Add edge with weight based on similarity
                        G.add_edge(keyword1, keyword2, weight=serp_similarity)
            
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
            st.write('No clusters formed. Try lowering the overlap threshold.')
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
