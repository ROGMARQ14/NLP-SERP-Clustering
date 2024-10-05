import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from fuzzywuzzy import process
from community import community_louvain

# Upload data file
st.title('Keyword Clustering with SERP Overlap')
uploaded_file = st.file_uploader("Upload your keyword CSV file", type=['csv'])

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)
    
    # Field mapping
    keyword_col = st.selectbox('Select the Keyword column:', data.columns)
    url_cols = [col for col in data.columns if 'URL' in col]
    
    # Fuzzy Matching to clean keywords
    unique_keywords = data[keyword_col].unique()
    keyword_mapping = {kw: process.extractOne(kw, unique_keywords)[0] for kw in unique_keywords}
    data[keyword_col] = data[keyword_col].apply(lambda x: keyword_mapping[x])
    
    # Create a graph where nodes are keywords
    G = nx.Graph()
    
    # Add nodes
    for kw in data[keyword_col].unique():
        G.add_node(kw)
    
    # Add edges based on SERP URL overlap
    for i, keyword1 in enumerate(data[keyword_col].unique()):
        urls1 = set(data[data[keyword_col] == keyword1][url_cols].values.flatten())
        for j, keyword2 in enumerate(data[keyword_col].unique()):
            if i < j:
                urls2 = set(data[data[keyword_col] == keyword2][url_cols].values.flatten())
                overlap = len(urls1.intersection(urls2))
                if overlap > 0:
                    G.add_edge(keyword1, keyword2, weight=overlap)
    
    # Apply Louvain community detection
    partition = community_louvain.best_partition(G, weight='weight')
    
    # Add cluster labels to the data
    data['cluster'] = data[keyword_col].map(partition)
    
    # Display the results
    st.write('Clustered keywords:', data[[keyword_col, 'cluster']])
    
    # Download the clustered keywords as a CSV file
    st.download_button(
        label="Download clustered keywords",
        data=data.to_csv(index=False),
        file_name='clustered_keywords.csv',
        mime='text/csv'
    )
