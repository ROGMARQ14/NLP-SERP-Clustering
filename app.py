import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
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

    # Button to start processing
    if st.button('Run Clustering'):
        # Initialize progress bar and message
        num_unique_keywords = len(data[keyword_col].unique())
        st.session_state['processed'] = False
        progress_message = f"Keywords: {num_unique_keywords}. Duplicates removed: 0. Clusters: 0"
        progress_bar = st.progress(0)
        
        # Initial Cluster
        clusters = []
        duplicates_removed = 0
        keyword_sets_processed = 0

        # Take the first keyword's 10 results and form the initial cluster
        cluster_0 = [data.iloc[0]]
        clusters.append(cluster_0)
        first_keyword_urls = data.iloc[0][url_col]

        # Iterate over the remaining keywords to find overlaps
        for i in range(1, len(data)):
            current_keyword_urls = data.iloc[i][url_col]
            overlap_count = len(set(first_keyword_urls[:5]) & set(current_keyword_urls[:5]))  # Checking top 5

            if overlap_count > 3:
                clusters[0].append(data.iloc[i])

            # Update progress bar and message
            keyword_sets_processed += 1
            progress_bar.progress(min(int((keyword_sets_processed / num_unique_keywords) * 100), 100))
            progress_message = f"Keywords: {num_unique_keywords}. Duplicates removed: {duplicates_removed}. Clusters: {len(clusters)}"
            st.write(progress_message)

        # De-duplicate within cluster (step 4)
        unique_keywords_in_cluster = {}
        for keyword_data in clusters[0]:
            kw = keyword_data[keyword_col]
            if kw in unique_keywords_in_cluster:
                if unique_keywords_in_cluster[kw][impression_col] < keyword_data[impression_col]:
                    unique_keywords_in_cluster[kw] = keyword_data  # Keep the higher impression keyword
            else:
                unique_keywords_in_cluster[kw] = keyword_data

        clusters[0] = list(unique_keywords_in_cluster.values())

        # Remove exact matches in top 5 SERP results (step 5)
        keywords_to_remove = set()
        seen_serp_signatures = {}
        for keyword_data in clusters[0]:
            serp_signature = tuple(sorted(keyword_data[url_col][:5]))  # Top 5 SERP results in any order
            if serp_signature in seen_serp_signatures:
                if seen_serp_signatures[serp_signature][impression_col] < keyword_data[impression_col]:
                    keywords_to_remove.add(seen_serp_signatures[serp_signature][keyword_col])
                    seen_serp_signatures[serp_signature] = keyword_data
                else:
                    keywords_to_remove.add(keyword_data[keyword_col])
            else:
                seen_serp_signatures[serp_signature] = keyword_data

        # Remove keywords
        clusters[0] = [kw for kw in clusters[0] if kw[keyword_col] not in keywords_to_remove]
        duplicates_removed += len(keywords_to_remove)

        # Set cluster name using TF-IDF
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        cluster_texts = [kw[keyword_col] for kw in clusters[0]]
        tfidf_matrix = tfidf_vectorizer.fit_transform(cluster_texts)
        avg_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)
        top_word_idx = np.argmax(avg_tfidf)
        top_word = tfidf_vectorizer.get_feature_names_out()[top_word_idx]

        cluster_name = max(clusters[0], key=lambda x: top_word in x[keyword_col])[keyword_col]

        # Finalize cluster name and update the session state
        for keyword_data in clusters[0]:
            keyword_data['cluster_name'] = cluster_name

        # Final progress update
        progress_bar.progress(100)
        progress_message = f"Keywords: {num_unique_keywords}. Duplicates removed: {duplicates_removed}. Clusters: 1"
        st.write(progress_message)

        # Store processed data in session state
        st.session_state['data'] = pd.DataFrame(clusters[0])
        st.session_state['processed'] = True

    # Display the results if processed
    if st.session_state['processed']:
        st.write('Clustered keywords:', st.session_state['data'][[keyword_col, impression_col, 'cluster_name']])
        
        # Download the clustered keywords as a CSV file
        st.download_button(
            label="Download clustered keywords",
            data=st.session_state['data'].to_csv(index=False),
            file_name='clustered_keywords.csv',
            mime='text/csv'
        )
