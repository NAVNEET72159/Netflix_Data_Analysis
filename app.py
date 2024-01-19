import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Netflix Data Analysis')
st.markdown('''
This app performs simple webscraping of Netflix data (does not work for all Netflix movies)
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** [Kaggle](https://www.kaggle.com/datasets/ashfakyeafi/netflix-movies-and-shows-dataset)
''')

# Web scraping of Netflix movies
@st.cache_data(persist=True)
def load_data():
    url = "Dataset/netflix_data.csv"
    df = pd.read_csv(url)
    return df

load_data()
df = load_data()

st.dataframe(load_data(), height=500, width=2000)

# Plot different types of charts according to user selection

load_data().cast.fillna("cast unavailable", inplace=True)
load_data().country.fillna("production country unavailable", inplace=True)
load_data().director.fillna("director unavailable", inplace=True)

def generate_top_tens(list_type, graph_type):
    if list_type == 'Top 10 Actors':
        if graph_type == 'Bar Chart':
            plt.figure(figsize=(10, 10))
            cast = df[df['cast'] != 'cast unavailable'].set_index('title').cast.str.split(', ', expand=True).stack().reset_index(level=1, drop=True)
            fig = sns.countplot(y = cast, order=cast.value_counts().index[:10], palette='magma_r', saturation=.2)
            plt.title('Top 10 Actors')
            plt.xlabel('Number of Titles')
            plt.ylabel('Actors')
            st.pyplot(fig)
        elif graph_type == 'Line Chart':
            st.line_chart(load_data().cast.value_counts().head(10))
        elif graph_type == 'Area Chart':
            st.area_chart(load_data().cast.value_counts().head(10))
    elif list_type == 'Top 10 Directors':
        if graph_type == 'Bar Chart':
            st.bar_chart(load_data().director.value_counts().head(10))
        elif graph_type == 'Line Chart':
            st.line_chart(load_data().director.value_counts().head(10))
        elif graph_type == 'Area Chart':
            st.area_chart(load_data().director.value_counts().head(10))
    elif list_type == 'Top 10 Countries':
        if graph_type == 'Bar Chart':
            st.bar_chart(load_data().country.value_counts().head(10))
        elif graph_type == 'Line Chart':
            st.line_chart(load_data().country.value_counts().head(10))
        elif graph_type == 'Area Chart':
            st.area_chart(load_data().country.value_counts().head(10))
    elif list_type == 'Top 10 Genres with the Largest Number of Content Titles':
        if graph_type == 'Bar Chart':
            st.bar_chart(load_data().listed_in.value_counts().head(10))
        elif graph_type == 'Line Chart':
            st.line_chart(load_data().listed_in.value_counts().head(10))
        elif graph_type == 'Area Chart':
            st.area_chart(load_data().listed_in.value_counts().head(10))
    
# Plotting the number of movies and TV shows released over the years
st.header('Number of Movies and TV Shows Released Over the Years')
chart_type = st.selectbox('Select Chart Type', ['Bar Chart', 'Line Chart', 'Area Chart'], key='3')
if chart_type == 'Bar Chart':
    st.bar_chart(load_data().groupby('release_year').type.count())
elif chart_type == 'Line Chart':
    st.line_chart(load_data().groupby('release_year').type.count())
elif chart_type == 'Area Chart':
    st.area_chart(load_data().groupby('release_year').type.count())

# Listing the top 10 data according to user selection
st.header(f'Top 10 listing')
list_type = st.selectbox('Select Chart Type', ['Top 10 Actors', 'Top 10 Directors', 'Top 10 Countries', 'Top 10 Genres with the Largest Number of Content Titles'], key='1')
if list_type == 'Top 10 Actors':
    st.write(load_data().cast.value_counts().head(10))
elif list_type == 'Top 10 Directors':
    st.write(load_data().director.value_counts().head(10))
elif list_type == 'Top 10 Countries':
    st.write(load_data().country.value_counts().head(10))
elif list_type == 'Top 10 Genres with the Largest Number of Content Titles':
    st.write(load_data().listed_in.value_counts().head(10))

# Plotting graph of the top 10 data according to user selection
st.header(f'{list_type} graph')
graph_type = st.selectbox('Select Chart Type', ['Bar Chart', 'Line Chart', 'Area Chart'], key='2')
generate_top_tens(graph_type, list_type)

def generate_graph(graph_type_, column_1, column_2):
    if graph_type_ == 'Bar Chart':
        st.bar_chart(df(column_1)[column_2].head(10))
    elif graph_type_ == 'Line Chart':
        st.line_chart(load_data().groupby(column_1)[column_2].head(10))
    elif graph_type_ == 'Area Chart':
        st.area_chart(load_data().groupby(column_1)[column_2].head(10))


# Plotting graph by groups
st.header('Graph by Groups')
column_1 = ['cast', 'director', 'country', 'listed_in', 'release_year', 'rating', 'type', 'duration']
selection1 = st.selectbox('Select Column 1', column_1)
column_2 = ['cast', 'director', 'country', 'listed_in', 'release_year', 'rating', 'type', 'duration']
selection2 = st.selectbox('Select Column 2', column_2)
graph_type_ = st.selectbox('Select Chart Type', ['Bar Chart', 'Line Chart', 'Area Chart'])
if st.button('Generate Graph'):
    st.header(f'Graph of {selection1} and {selection2}')
    generate_graph(graph_type_, selection1, selection2)