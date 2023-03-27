#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import base64
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import os
import pickle
import seaborn as sns
import plotly.graph_objs as go

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)


tab1, tab2, tab3 = st.tabs(["Job Board", "Clustering", 'Skillz'])


path = os.path.dirname(__file__)
job_board = path+'/job_board.csv'
skill_df = path+'/skill_df.csv'
embeddings_3d = path+'/embeddings_3d.csv'
df_c = path+'/df_c.csv'

df = pd.read_csv(job_board)
df_c = pd.read_csv(df_c)
skill_df = pd.read_csv(skill_df)
embeddings_3d = pd.read_csv(embeddings_3d)
kmeans = pickle.load(open('model.pkl','rb'))
avg = pickle.load(open('avg.pkl','rb'))

# cluster with the max avg silhouette score
best_cluster_num = max(avg, key=avg.get)

# drop duplicates that evaded our previous fingerprint
df_c.drop_duplicates(['job_title', 'company'], keep='first', inplace=True)

# unique job titles
job_titles = [i for i in df_c['job_title'].unique()]

# max silhoutte score
max_score = max(avg.values())

with tab1:

    st.title('Web Scraped Job Board')

    st.markdown("""
    ### This app displays web scraped data science,analysis and engineering jobs!
    """)



    # Sidebar Area

    st.sidebar.header('User Input Features')

    categories = st.sidebar.multiselect('Job Category', df.categories.unique(), df.categories.unique())

    location = st.sidebar.multiselect('City', df.location.unique(), df.location.unique())

    level = st.sidebar.multiselect('Experience Level', df.level.unique(), df.level.unique())

    job_type = st.sidebar.multiselect('Job Type', df.job_type.unique(), df.job_type.unique())




    # Filtering data
    df_selected_jobs = df[(df.categories.isin(categories))
                          & (df.location.isin(location)) & (df.level.isin(level)) &
                          (df.job_type.isin(job_type))].copy()

    st.subheader('Display Selected Job(s)')
    st.write('Data Dimension: ' + str(df_selected_jobs.shape[0]) + ' rows and ' +
             str(df_selected_jobs.shape[1]) + ' columns.')
    st.dataframe(df_selected_jobs.iloc[:,:6])

    # file download
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="df_selected_jobs.csv">Download CSV File</a>'
        return href

    st.markdown(filedownload(df_selected_jobs), unsafe_allow_html=True)

    # pie chart
    plt.figure(figsize=(5,3))
    plt.pie(x=df.groupby('categories').count()['company'],
            explode=(0.02, 0.02, 0.02), labels= df.groupby('categories').count()['company'].index, colors = ['#429EBD', '#9FE7F5','#053F5C'])

    # draw circle
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()

    # Adding Circle in Pie chart
    fig.gca().add_artist(centre_circle)

    # Adding Title of chart
    st.header('Stake of categories in Greek market')
    st.pyplot(fig)

    # map

    st.header('The majority of the listings are in/near Athens')

    map = folium.Map(location=[37.983810, 23.727539], zoom_start=5, scrollWheelZoom=False, tiles='CartoDB positron')

    data = df_selected_jobs[['latitude','longitude']].dropna().to_numpy().tolist()

    HeatMap(data = data).add_to(map)

    st_map = st_folium(map, width=700, height=450)

with tab2:
    
    # create a trace for each cluster
    data = []
    for i in range(best_cluster_num):
        cluster = np.array(embeddings_3d)[kmeans.labels_ == i]
        trace = go.Scatter3d(x=cluster[:, 0],
                             y=cluster[:, 1],
                             z=cluster[:, 2],
                             mode='markers',
                             marker=dict(size=5),
                             name=f'Cluster {i}',
                             text=[
                                 f'Job Title: {job_titles[j]}<br>Cluster: {i}'
                                 for j in range(len(job_titles))
                                 if kmeans.labels_[j] == i
                             ],
                             hoverinfo='text')
        data.append(trace)

    # create layout
    layout = go.Layout(
        title=
        f'Jobs Clustered by Job Skills (t-SNE, 3D) with a silhouette score of {str(max_score)[:5]}',
        scene=dict(xaxis_title='TSNE-1',
                   yaxis_title='TSNE-2',
                   zaxis_title='TSNE-3'),
        margin=dict(l=0, r=0, b=0, t=40))

    # plot figure
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig)
    
    # Create a list of cluster IDs and their corresponding sizes
    cluster_sizes = skill_df.groupby('cluster').size().tolist()

    # Define the number of clusters and generate a corresponding color scale
    n_clusters = len(cluster_sizes)
    palette = sns.color_palette("Paired", n_clusters).as_hex()
    color_scale = [[i / (n_clusters - 1), color]
                   for i, color in enumerate(palette)]

    # Create a list of text labels for each marker (i.e. job titles in each cluster)
    text_labels = []
    for cluster_id in range(len(cluster_sizes)):
        jobs = skill_df.loc[skill_df['cluster'] == cluster_id, 'job_title']
        text_labels.append('<br>'.join(jobs))

    # Create the plot
    fig = go.Figure(data=[
        go.Scatter(
            x=[i + 1 for i in range(len(cluster_sizes))],
            y=[5 for i in range(len(cluster_sizes))],
            text=text_labels,
            mode='markers',
            marker=dict(
                color=cluster_sizes,
                colorscale=color_scale,
                size=cluster_sizes,
                sizemode='diameter',
                sizeref=max(cluster_sizes) / 80  # adjust the size scaling
            ))
    ])

    fig.update_layout(
        title={
            'text':
            f'Job Titles Clustered by Job Skills (t-SNE, 2D) with a silhouette score of {str(max_score)[:5]}',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Cluster Number",
        yaxis_title="",
    )

    st.plotly_chart(fig)

with tab3:
    st.title('Investigating skill overlap between jobs')           
    def skill_overlap(x,y):
        
        over = round(len([1 for i in x if i in y]) / len(x),2)

        return over
    
    # Define the job titles and corresponding lists of skills
    eng = skill_df[skill_df['job_title'].str.contains('Data Engineer')].explode(
        'skills')['skills'].value_counts().reset_index()['index'].tolist()
    sci = skill_df[skill_df['job_title'].str.contains('Data Scientist')].explode(
        'skills')['skills'].value_counts().reset_index()['index'].tolist()
    ana = skill_df[skill_df['job_title'].str.contains('Data Analyst')].explode(
        'skills')['skills'].value_counts().reset_index()['index'].tolist()

    titles = ['Data Engineer', 'Data Scientist', 'Data Analyst']
    skills = [eng, sci, ana]

    # Initialize an empty matrix to store the overlaps
    overlap_matrix = []

    # Loop through the job titles and calculate the overlaps with all other job titles
    for i, title1 in enumerate(job_titles):
        row = []
        for j, title2 in enumerate(job_titles):
            overlap = skill_overlap(job_skills[i], job_skills[j])
            row.append(overlap)
        overlap_matrix.append(row)

    # Convert the overlap matrix to a DataFrame
    mt = pd.DataFrame(overlap_matrix, columns=job_titles, index=job_titles)
    
    fig, ax = plt.subplots()
    sns.heatmap(mt, ax=ax)
    st.write(fig)
