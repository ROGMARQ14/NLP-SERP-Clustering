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
st.title('Keyword Clustering Focused on SERP Overlap')
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
    
    # SERP overlap threshold input
    overlap_threshold = st.slider('Set the minimum number of overlapping top URLs for clustering:', min_value=1, max_value=10, value=3)
    
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
        
        # Vectorize keywords with their SERP titles (optional, can be removed if focusing strictly on SERP overlap)
        data['serp_vector'] = data[title_col].apply(lambda x: model.encode(x))

        # Update progress
        progress_bar.progress(40)
        
        # Create a graph where nodes are keywords
        G = nx.Graph()
        
        # Add nodes
        for kw in data[keyword_col].unique():
            G.add_node(kw)
        
        # Add edges based strictly on SERP overlap
        num_keywords = len(data[keyword_col].unique())
        for i, keyword1 in enumerate(data[keyword_col].unique()):
            keyword1_data = data[data[keyword_col] == keyword1]
            urls1 = keyword1_data[url_col].values.flatten()
            positions1 = keyword1_data[position_col].values.flatten()
            
            for j, keyword2 in enumerate(data[keyword_col].unique()):
                if i < j:
                    keyword2_data = data[data[keyword_col] == keyword2]
                    urls2 = keyword2_data[url_col].values.flatten()
                    positions2 = keyword2_data[position_col].values.flatten()
                    
                    # Calculate overlap in top URLs
                    overlap_count = sum(1 for url in urls1 if url in urls2 and 
                                        positions1[list(urls1).index(url)] <= 5 and 
                                        positions2[list(urls2).index(url)] <= 5)
                    
                    # Add edge if the overlap count meets the threshold
                    if overlap_count >= overlap_threshold:
                        G.add_edge(keyword1, keyword2, weight=overlap_count)
            
            # Update progress
            progress_bar.progress(40 + int((i / num_keywords) * 40))
        
        # Apply Louvain community detection
        if len(G.edges) > 0:  # Ensure there are edges in the graph before partitioning
            partition = community_louvain.best_partition(G, weight='weight')
            
            # Add cluster labels to the data
            data['cluster'] = data[keyword_col].map(partition)
            
            # Rename clusters based on the keyword most relevant to others in its group
            cluster_names = {}
            for cluster_id in data['cluster'].unique():
                cluster_data = data[data['cluster'] == cluster_id]
                keywords_in_cluster = cluster_data[keyword_col].values
                
                # Calculate relevance within cluster by averaging overlap scores
                avg_similarities = []
                for keyword1 in keywords_in_cluster:
                    avg_similarity = 0
                    count = 0
                    for keyword2 in keywords_in_cluster:
                        if keyword1 != keyword2 and G.has_edge(keyword1, keyword2):
                            avg_similarity += G[keyword1][keyword2]['weight']
                            count += 1
                    avg_similarities.append((keyword1, avg_similarity / count if count > 0 else 0))
                
                # Select the keyword with the highest average similarity
                top_keyword = max(avg_similarities, key=lambda x: x[1])[0]
                cluster_names[cluster_id] = top_keyword
                
            data['cluster_name'] = data['cluster'].map(cluster_names)
            
            # Update progress
            progress_bar.progress(100)
            
            # Store processed data in session state
            st.session_state['data'] = data
            st.session_state['processed'] = True
        else:
            st.write('No clusters formed. Adjust the overlap threshold or input data.')
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
