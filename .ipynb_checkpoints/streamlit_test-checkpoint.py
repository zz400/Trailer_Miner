import time 
import sys
import random
import numpy as np
import pandas as pd

from textblob import TextBlob
from PIL import Image
import webbrowser
from bokeh.models.widgets import Div
import streamlit as st

import mojo_api

# st.write('Hello, world!')
# st.write('Hello, *World!* :sunglasses:')
# st.write(1234)
# st.write("1234")
# st.header("Header")
# st.subheader("SubHeader")
# st.text("Text")
# st.text("Text1\nText2")
# st.markdown('Streamlit is **_really_ cool**.')

# Image Link:
# link_md = "[![](image0.png)](https://streamlit.io)"
# st.markdown(link_md)

st.title("Welcome to Trailer Miner!")

st.subheader("Please input a movie title:")
user_movie = st.text_input("", value="Avengers: Endgame")

movie = mojo_api.Movie()
find_movie = movie.get_app_movie_info(user_movie)
if not find_movie:
    st.subheader("Sorry, movie '*{}*' is not found, please retry with a standard IMDB movie title.".format(user_movie))
else:
    st.subheader("Information for *{}*:".format(movie.title))
    movie_info_md = """
        - **Production company**: {}
        - **Release date**: {}
        - **MPAA**: {}
        - **Runtime**: {} min
        - **Genres**: {}
        - **Director**: {}
        - **Actors**: {}
        """.format(movie.company, movie.release_date, movie.mpaa, movie.movie_length, \
                   ", ".join(movie.genres.split(",")),\
                   ", ".join(movie.director.split(",")),\
                   ", ".join(movie.actors.split(",")))
    st.markdown(movie_info_md)
    
    st.subheader("Box office prediction:")
    bo_opening = movie.bo_opening
    bo_gross = movie.bo_gross
    bo_opening_pred_metaonly = random.uniform(50, 300)
    bo_opening_pred_overall = random.uniform(50, 300)
    if movie.released: 
        movie_bo_md = """
        - **Real opening box office**: ${:.2f} M
        - **Predicted opening box office (use metadata only)**: ${:.2f} M
        - **Predicted opening box office (use metadata + trailer)**: ${:.2f} M
        """.format(bo_opening, bo_opening_pred_metaonly, bo_opening_pred_overall)
    else:
        movie_bo_md = """
        - **Real opening box office**: not available yet.
        - **Predicted opening box office (use metadata only)**: ${:.2f} M
        - **Predicted opening box office (use metadata + trailer)**: ${:.2f} M
        """.format(bo_opening_pred_metaonly, bo_opening_pred_overall)
    st.markdown(movie_bo_md)

    

    st.subheader("")
    if st.button('Go to Trailer1'):
        url = 'https://www.streamlit.io/'
        webbrowser.open_new_tab(url)
    #     js = "window.open(url)"  # New tab or window
    #     js = "window.location.href = 'https://www.streamlit.io/'"  # Current tab
    #     html = '<img src onerror="{}">'.format(js)
    #     div = Div(text=html)
    #     st.bokeh_chart(div)   

    if st.button('Go to Trailer2'):
        url = 'https://www.streamlit.io/'
        webbrowser.open_new_tab(url)
        
# image = Image.open("./image0.png")
st.image("./image0.png", use_column_width=True)


x = st.slider('Select a value')
st.write(x, 'squared is', x * x)

progress_bar = st.progress(0)
status_text = st.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.001)

# progress_bar.empty()

chart_data = pd.DataFrame(
                np.random.randn(20, 3),
                columns=['a', 'b', 'c'])
st.line_chart(chart_data)

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
# st.button("Re-run")


file = "./data/temp/2019_27_zyYgDtY2AMY.csv"
df = pd.read_csv(file)
df['timestamp'] = df['timestamp'].apply(lambda x : x[:19])
df['datetime'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%dT%H:%M:%S")
df['Year-Month'] = df['datetime'].apply(lambda x: "{:d}-{:02d}".format(x.year, x.month))
df.sort_values('datetime')

# Sentiment extraction
df['Sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['isPos'] = df['Sentiment'].apply(lambda x: x > 0.2)
df['isNeg'] = df['Sentiment'].apply(lambda x: x < -0.2)
df_agg = df.groupby(['Year-Month']).agg({
                "datetime": "first",
                "text": "count",
                "isPos": "sum",
                "isNeg": "sum",
            })
df_agg['pos_ratio'] = df_agg['isPos'] / df_agg['text']
df_agg['neg_ratio'] = df_agg['isNeg'] / df_agg['text']

st.line_chart(df_agg.set_index("datetime")[['isPos', 'isNeg']])