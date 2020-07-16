#!/usr/bin/env python

"""
Run this script on the backend of 'Trailer Miner' web-app.

PIPELINE:
==========================================================================================
user input movie title 
        |
        |
look for movie metadata [mojo_api.Movie()]
        |
        |-------------------------------------|
        |                 if not find movie: print "Sorry... please retry ...."
        |
display movie info [display_movie_info()]
        |
        |-------------------------------------|
        |                 if movie.year < 2014: print "Sorry... Please retry with a movie after 2014...."
        |
get trailer list and update trailer_list_file [get_trailer_list(), similar with notebook 11]
        |
        |
display links to trailers [display_trailer_links()]                
        |
        |-------------------------------------|
        |                 if trailer list is empty: print "Sorry, trailers not found"
        |                                     |
        |                                  go to (*)       
        |
get, process and cache trailer comments [get_trailer_comments(), similar with notebook 12 and 13] 
        |
        |-------------------------------------|
        |                 if there's no trailer comments: print "Comments about the trailers are not available."
        |                                     |
        |                                  go to (*)
        |
plot comment sentiment [display_trailer_comments(), similar with notebook 15]
        |
        |
predict box office (*) [display_bo_prediction()]
        |
        |-------------------------------------|
        |                 if movie not in cache: build features (similar with notebook 14)
        |-------------------------------------|
        |
    fit the models
        |
    plot prediction result
==========================================================================================


TEXT EXAMPLES:
==========================================================================================
Avengers: Endgame, Terminator: dark fate
aaa (no such movie)
Titanic (before 2014)
Toy Story 4 (trailer comments disabled)
mulan (2020 movie, not in cache_df)
any movie in 2020 whose trailer info hasn't been fetched
==========================================================================================


Author: Zhengyang Zhao
Date: Jun. 13 2020.
"""

import os
import time 
import sys
import logging
import random
import numpy as np
import pandas as pd

import traceback
from apiclient.discovery import build
from textblob import TextBlob
from PIL import Image
import webbrowser
from bokeh.models.widgets import Div
import streamlit as st

import chart_studio
import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.express as px

from mojo_api import Movie
from youtube_api import get_trailer_ids, get_trialer_statistics
from youtube_api import get_comment_thread_allPages, parse_comment_thread_allPages, get_comment_thread_1page
from data_util import *
from model_util import *


def initialize_logger():
    log_file = './log/log_0.log'
    open(log_file, 'a').close()  # create log file if not existing
    
    global logger
    logger = logging.getLogger("trailer_miner")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    hdlr = logging.FileHandler(log_file)
    hdlr.setFormatter(formatter)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(hdlr) 


def random_select_youtube_api_key():
    key_file_0 = "./private_api_key/youtube_data_API_key.txt"
    api_key_0 = open(key_file_0, "r").read()
    youtube_0 = build('youtube', 'v3', developerKey=api_key_0, cache_discovery=False)
    key_file_list = "./private_api_key/youtube_data_API_key_list.txt"
    api_key_list = [k for k in open(key_file_list, "r").read().split('\n') if len(k) > 15][:]
    youtube_list = [build('youtube', 'v3', developerKey=k, cache_discovery=False) for k in api_key_list]
    return random.choice(youtube_list)


def get_trailer_list(movie, trailer_list_df):
    """
    Given the movie, find out its trailer_ids either from local cache, or using Youtube-data-api.
    """
    
    if movie.tt_id not in trailer_list_df.tt_id.unique():
        # retrieve trailers v_id using youtube-api, similar with notebook 11
        youtube = random_select_youtube_api_key()
        trailer_df = pd.DataFrame(columns=['tt_id', 'movie_title', 'release_date', \
                                   'trailer_id', 'trailer_publishTime', 'trailer_title',\
                                   'viewCount', 'commentCount', 'likeCount', 'dislikeCount', 'favoriteCount'
                                  ])
        tt_id = movie.tt_id
        movie_name = movie.title
        movie_release_date = movie.release_date
        trailer_dict, _ = get_trailer_ids(movie_name, movie_release_date, 5, youtube)
        row = 0
        for v_id in trailer_dict:
            trailer_df.loc[row, 'tt_id'] = tt_id
            trailer_df.loc[row, 'movie_title'] = movie_name
            trailer_df.loc[row, 'release_date'] = movie_release_date
            trailer_df.loc[row, 'trailer_id'] = v_id
            trailer_df.loc[row, 'trailer_publishTime'] = trailer_dict[v_id][0]
            trailer_df.loc[row, 'trailer_title'] = trailer_dict[v_id][1]
            v_stat = get_trialer_statistics(v_id, youtube)
            for item in v_stat:
                trailer_df.loc[row, item] = int(v_stat[item])
            row += 1
        
        # Remove videos with viewCount < 10k:
        trailer_df = trailer_df[trailer_df.viewCount >= 1e5]
        
        # Keep atmost 3 trailers for each movie:
        max_viewCount = trailer_df["viewCount"].max()
        trailer_df.loc[:, 'remove'] = False
        for i in trailer_df.index:
            viewCount = trailer_df.loc[i, 'viewCount']
            if 10 * viewCount < max_viewCount:
                trailer_df.loc[i, 'remove'] = True
        trailer_df = trailer_df[trailer_df.remove == False]
        trailer_df = trailer_df.sort_values(['viewCount'])
        trailer_df = trailer_df.tail(3)
        trailer_df = trailer_df.sort_values(['trailer_publishTime'])
        trailer_df = trailer_df.drop(columns=['favoriteCount', 'remove'])
        
        # Find out the trailers which disable comments
        trailer_df.loc[:, 'comment_disabled'] = False
        for i in trailer_df.index:
            video_id = trailer_df.loc[i, 'trailer_id']
            try:
                res = get_comment_thread_1page(video_id, youtube)
            except Exception:
#                 traceback.print_exc()
                trailer_df.loc[i, 'comment_disabled'] = True
        trailer_list_df = pd.concat([trailer_list_df, trailer_df])
        trailer_list_df = trailer_list_df.reset_index(drop=True)
        trailer_list_df.to_csv(trailer_list_file, index=False)
        
        trailer_list = trailer_list_df[trailer_list_df.tt_id == movie.tt_id].trailer_id.to_list() 
        logger.warning("Search trailer list on Youtube. Trailer list: {}".format(trailer_list))
        
    trailer_list = trailer_list_df[trailer_list_df.tt_id == movie.tt_id].trailer_id.to_list()    
    return trailer_list, trailer_list_df


def get_trailer_comments(video_id, trailer_list_df, trailer_comments_raw_dir, trailer_comments_dir):
    """
    Given the trailer's video_id, scrape, process and store trailer comments.
    """
    if video_id+".csv" not in os.listdir(trailer_comments_dir):
        
        # Fetch comments using youtube-api (similar with notebook 12):
        youtube = random_select_youtube_api_key()
        total_pages = get_comment_thread_allPages(video_id, youtube, save_dir=trailer_comments_raw_dir)
        parse_comment_thread_allPages(trailer_comments_raw_dir, video_id, total_pages)

        # Process comments (similar with notebook 13):
        df = pd.read_csv(os.path.join(trailer_comments_raw_dir, video_id+".csv"))
        df = df[~df.text.isnull()].reset_index(drop=True) # Remove null records.
        df.loc[:, 'sentiment_score'] = df.loc[:, 'text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        trailer_id = df.trailer_id[0]
        movie_release_date = trailer_list_df[trailer_list_df.trailer_id == trailer_id].release_date.iloc[0]
        df.loc[:, 'datetime'] = pd.to_datetime(df.loc[:, 'timestamp'], format="%Y-%m-%dT%H:%M:%S")
        release_date = datetime.datetime.strptime(movie_release_date, "%Y-%m-%d")
        model_cutoff_date = pd.Timestamp(release_date - datetime.timedelta(days=7))
        visualize_cutoff_date = pd.Timestamp(release_date + datetime.timedelta(days=90))
        df.loc[:, 'used_for_model'] = df.loc[:, 'datetime'].apply(lambda x : x < model_cutoff_date)
        df.loc[:, 'used_for_visualization'] = df.loc[:, 'datetime'].apply(lambda x : x < visualize_cutoff_date)
        df = df.sort_values('datetime').reset_index(drop=True)
        df.to_csv(os.path.join(trailer_comments_dir, video_id+".csv"), index=False)
        
        logger.warning("Fetch trailer comments on Youtube. Trailer: {}".format(video_id))


def display_movie_info(movie):
    """
    Web-app output patch 1: display the basic info of the movie.
    """
    
    st.subheader("Information for *{}*:".format(movie.title))
    movie_info_md = """
            - **Production company**: {}
            - **Release date**: {}
            - **MPAA**: {}
            - **Runtime**: {} min
            - **Genres**: {}
            - **Director**: {}
            - **Actors**: {}
            - **Budget**: ${}M
            - **IMDB score**: {}
            """.format(movie.company, movie.release_date, movie.mpaa, movie.movie_length, \
                       ", ".join(movie.genres.split(",")),\
                       ", ".join(movie.director.split(",")),\
                       ", ".join(movie.actors.split(",")),\
                       movie.budget, movie.imdb_score)
    st.markdown(movie_info_md)
    logger.info("Information for *{}*:".format(movie.title))

    
def display_trailer_links(movie, trailer_list):
    """
    Web-app output patch 2: display the links to trailers.
    """
    
    if len(trailer_list) == 0:
        st.subheader("Sorry, trailers not found for *{}*.".format(movie.title))
        logger.info("Sorry, trailers not found for *{}*.".format(movie.title))
    else:
        st.subheader("Trailers for *{}*:".format(movie.title))
        logger.info("Trailers for *{}*:".format(movie.title))
        
        # list trailers (at most three)
        n_trailer = len(trailer_list)
        trailer_hrefs = ["https://www.youtube.com/watch?v={}".format(trailer_id) for trailer_id in trailer_list]
        trailer_imgs = ["https://img.youtube.com/vi/{}/0.jpg".format(trailer_id) for trailer_id in trailer_list]
        trailer_captions = ["Trailer {}".format(i+1) for i in range(n_trailer)]
        trailer_html = "<table><tr>"
        for i in range(n_trailer):
             trailer_html += "<td> <a href={} target='_blank'><img src={} alt='Drawing' style='width: 200px;'/><figcaption>{}</figcaption> </td>"\
                .format(trailer_hrefs[i], trailer_imgs[i], trailer_captions[i])
        trailer_html += "</tr></table>"
        st.markdown(trailer_html, unsafe_allow_html=True)


def display_trailer_comments(movie, trailer_list, trailer_comments_dir):
    """
    Web-app output patch 3: plot the trailer_comment_df.
    """

    if len(trailer_list) == 0:
        st.subheader("Comments about the trailers are not available.")
        logger.info("Comments about the trailers are not available.")
    else:
        comment_df = pd.DataFrame()
        for video_id in trailer_list:
            comment_file = os.path.join(trailer_comments_dir, video_id + ".csv")
            tr_df = pd.read_csv(comment_file)
            tr_df = tr_df[tr_df["used_for_visualization"] == True]    
            comment_df = comment_df.append(tr_df, ignore_index=True)
        
        # plot comment sentiment
        st.subheader("Sentiment of trailer comments:")
        logger.info("Sentiment of trailer comments:")
        comment_fig = plot_comment_df(comment_df)
        st.plotly_chart(comment_fig, use_container_width=True)


def plot_comment_df(comment_df):
    """
    Generate a plotly figure for the trailer comments.
    """
    
    comment_df = comment_df[comment_df["used_for_visualization"] == True]    
    comment_df.loc[:, 'datetime'] = pd.to_datetime(comment_df.loc[:, 'datetime'], format="%Y-%m-%d %H:%M:%S")
    comment_df.loc[:, 'Year-Month-Day'] = comment_df.loc[:, 'datetime'].apply(lambda x: "{:d}-{:02d}-{:02d}".format(x.year, x.month, x.day))
    threshold = 0.2
    comment_df.loc[:, 'isPos'] = comment_df.loc[:, 'sentiment_score'].apply(lambda x: int(x > threshold))
    comment_df.loc[:, 'isNeg'] = comment_df.loc[:, 'sentiment_score'].apply(lambda x: int(x < -threshold))
    
    comment_df_agg = comment_df.groupby(['Year-Month-Day']).agg({
                "datetime": "first",
                "text": "count",
                "isPos": "sum",
                "isNeg": "sum",
                "used_for_model": "first"
            })
    comment_df_agg['pos_ratio'] = comment_df_agg['isPos'] / comment_df_agg['text']
    comment_df_agg['neg_ratio'] = comment_df_agg['isNeg'] / comment_df_agg['text']
    rolling_window = 7
    comment_df_agg['pos_ratio_roll'] = comment_df_agg['pos_ratio'].rolling(rolling_window).mean()
    comment_df_agg['neg_ratio_roll'] = comment_df_agg['neg_ratio'].rolling(rolling_window).mean()

    comment_df_agg = comment_df_agg.sort_values('datetime')
    comment_df_agg = comment_df_agg.set_index("datetime", drop=True)[['text', 'pos_ratio', 'neg_ratio', 'pos_ratio_roll', 'neg_ratio_roll', 'used_for_model']]

    x_start = comment_df_agg.index[0]
    x_end = comment_df_agg.index[-1]
    x_cutoff = comment_df_agg[comment_df_agg.used_for_model == True].index[-1] 
    x_open = pd.Timestamp(x_cutoff + datetime.timedelta(days=7)) # movie release date.
    y2_max = max(comment_df_agg.pos_ratio_roll.max()+0.1 , comment_df_agg.neg_ratio_roll.max()+0.1, 0.5)
    
    trace_cnt = go.Scatter(x=comment_df_agg.index, y=comment_df_agg['text'], name='Comment Count', yaxis = 'y1', line=dict(color='blue'))
    trace_pos = go.Scatter(x=comment_df_agg.index, y=comment_df_agg['pos_ratio_roll'], name='Positive Comment Ratio', yaxis = 'y2', line=dict(color='red'))
    trace_neg = go.Scatter(x=comment_df_agg.index, y=comment_df_agg['neg_ratio_roll'], name='Negative Comment Ratio', yaxis = 'y2', line=dict(color='red', dash='dash'))

    trace_shade = go.Scatter(
                    x=[x_start, x_open, x_end], 
                    y=[-0.3, 1, 1], 
                    yaxis = 'y2',
                    name="Shaded region", 
                    mode="none", 
                    fill="tozeroy", 
                    fillcolor="rgba(135, 143, 135, 0.2)",
                    line=dict(shape='hv', width=0),
                    hoverinfo="none", 
                    showlegend=False,
    )

    annode_text = "<b>Movie<br>released</b>" if x_open < x_end else ""
    trace_annode = go.Scatter(
                    x=[x_open],
                    y=[y2_max],
                    yaxis = 'y2',
                    name="movie release data",
                    mode="text",
                    text=[annode_text],
                    textfont=dict(size=16),
                    textposition="bottom right",
                    showlegend=False,
    )

    data = [trace_cnt, trace_pos, trace_neg, trace_shade, trace_annode]
    layout = go.Layout(title='',
                       margin=dict(l=10, r=10, t=10, b=10),
                       legend=dict(x=0.02, y=0.98),
                       xaxis=dict(title='Date'),
                       yaxis=dict(title='Comment Count', color='blue'),
                       yaxis2=dict(title='Comment Sentiment', color='red', overlaying='y', side='right', range=[0, y2_max])
                       )

    fig = go.Figure(data=data, layout=layout)
    return fig

                
def display_bo_prediction(movie, cache_df, trailer_list_df):
    """
    Web-app output patch 4: display box office prediction results.
    """
    if movie.tt_id not in cache_df.tt_id.unique():
        metadata_df = pd.DataFrame()
        metadata_df.loc[0, 'Title'] = movie.title
        metadata_df.loc[0, 'Year'] = movie.year
        metadata_df.loc[0, 'Rank'] = np.nan
        metadata_df.loc[0, 'tt_id'] = movie.tt_id
        metadata_df.loc[0, 'rl_id'] = movie.rl_id
        metadata_df.loc[0, 'release_date'] = movie.release_date
        metadata_df.loc[0, 'company'] = movie.company
        metadata_df.loc[0, 'mpaa'] = movie.mpaa
        metadata_df.loc[0, 'genres'] = movie.genres
        metadata_df.loc[0, 'runtime'] = movie.movie_length
        metadata_df.loc[0, 'director'] = movie.director
        metadata_df.loc[0, 'actors'] = movie.actors
        metadata_df.loc[0, 'budget'] = movie.budget
        metadata_df.loc[0, 'bo_opening'] = movie.bo_opening
        metadata_df.loc[0, 'bo_gross'] = movie.bo_gross
        metadata_df.loc[0, 'imdb_score'] = movie.imdb_score
        
        trailer_df = trailer_list_df[trailer_list_df.tt_id == movie.tt_id].reset_index(drop=True)
        df = add_trailer_features(metadata_df, trailer_df, trailer_list_df)
    else:
        df = cache_df[cache_df.tt_id == movie.tt_id]
        
    if movie.budget is None or movie.budget is np.nan:
        st.subheader("Sorry, budget of the movie is not found. Thus can't do box office prediction.")
        logger.info("Sorry, budget of the movie is not found. Thus can't do box office prediction.")
    else: 
        st.subheader("Opening box office prediction:")
        logger.info("Opening box office prediction:")
        
    df = df.reset_index(drop=True)
    X_wTrailer, y_wTrailer = build_features(df, with_Trailer=True)
    X_woTrailer, y_woTrailer = build_features(df, with_Trailer=False)
    pred_wTrailer = model_wTrailer.predict_range(X_wTrailer) * df.loc[0, 'budget']
    pred_woTrailer = model_woTrailer.predict_range(X_woTrailer) * df.loc[0, 'budget']
    true_openingBO = y_wTrailer.loc[0] * df.loc[0, 'budget']
    true_openingBO_p = true_openingBO * 1.3
    true_openingBO_m = true_openingBO * 0.7
    
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.margins(0.05)
    data = [pred_woTrailer.reshape(-1), pred_wTrailer.reshape(-1)]
    ax.boxplot(data, showmeans=False, labels=['Without Trailer Info', 'With Trailer Info'])
    # ax.set_xticklabels(labels=['Without Trailer Comments', 'With Trailer Comments'], fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("Opening Box Office ($M)", fontsize=16)
    
    if true_openingBO is not None and true_openingBO is not np.nan:
        plt.axhline(y=true_openingBO, color='b', linestyle='-')
        plt.text(1.25, true_openingBO*1.03 , 'True value', dict(size=16, color='blue'))
        plt.axhline(y=true_openingBO_p, color='b', linestyle=':')
        plt.text(1.25, true_openingBO_p*1.03 , 'True value + 30%', dict(size=16, color='blue'))
        plt.axhline(y=true_openingBO_m, color='b', linestyle=':')
        plt.text(1.25, true_openingBO_m*1.03 , 'True value - 30%', dict(size=16, color='blue'))
    st.pyplot(fig)
    
    
def process_sentiment_score(x):
    if x > 0.2:
        return 1
    if x < -0.2:
        return -1
    return 0
    
    
def add_trailer_features(metadata_df, trailer_df, trailer_list_df):
    """
    Add metadata features and trailer features to a single dataframe.
    Similor with Notebook 14.
    """
    trailer_list_df.loc[:, 'Year'] = trailer_list_df.loc[:, 'release_date'].apply(lambda x : int(x[:4]))
    year_mean_viewCount = {}
    year_mean_commentCount = {}
    for year in range(2014, 2020):
        year_mean_viewCount[year] = trailer_list_df[trailer_list_df['Year'] == year].viewCount.mean()
        year_mean_commentCount[year] = trailer_list_df[trailer_list_df['Year'] == year].commentCount.mean()
    year_mean_commentCount[2020] = year_mean_commentCount[2019]
    year_mean_viewCount[2020] = year_mean_viewCount[2019]
    year_mean_commentCount[2021] = year_mean_commentCount[2019]
    year_mean_viewCount[2021] = year_mean_viewCount[2019]
    
    movie_df = metadata_df
    tt_id = movie_df.loc[0, 'tt_id']
    movie_year = int(movie_df.loc[0, 'release_date'][:4])

    trailer_viewCount_tot = trailer_df.loc[:, 'viewCount'].sum()
    trailer_viewCount_norm = trailer_viewCount_tot / year_mean_viewCount[movie_year]
    movie_df.loc[0, 'trailer_viewCount'] = trailer_viewCount_norm

    trailer_likeCount_tot = trailer_df.loc[:, 'likeCount'].sum()
    trailer_dislikeCount_tot = trailer_df.loc[:, 'dislikeCount'].sum()
    like_dislike_ratio = trailer_likeCount_tot / trailer_dislikeCount_tot
    movie_df.loc[0, 'trailer_like_dislike_ratio'] = like_dislike_ratio

    trailer_list = trailer_df[trailer_df["comment_disabled"] == False]['trailer_id'].to_list()
    trailer_list = [f for f in trailer_list if f + ".csv" in os.listdir(trailer_comments_dir)]
    if len(trailer_list) == 0:
        return movie_df
    comment_df = pd.DataFrame()
    for tr_id in trailer_list[:]:
        tr_df = pd.read_csv(os.path.join(trailer_comments_dir, tr_id + ".csv"))
        tr_df = tr_df[tr_df["used_for_model"] == True]    # Only use comments before cut-off date.
        comment_df = comment_df.append(tr_df, ignore_index=True)
    comment_df.loc[:, 'sentiment_score_processed'] = comment_df.loc[:, 'sentiment_score'].apply(process_sentiment_score)
    agg_sentiment = comment_df.loc[:, 'sentiment_score_processed'].sum() / comment_df.shape[0]
    pos_sentiment_ratio = comment_df[comment_df['sentiment_score_processed'] == 1].shape[0] / comment_df.shape[0]
    neg_sentiment_ratio = comment_df[comment_df['sentiment_score_processed'] == -1].shape[0] / comment_df.shape[0]
    movie_df.loc[0, 'trailer_mean_sentiment'] = agg_sentiment
    movie_df.loc[0, 'trailer_commentCount'] = comment_df.shape[0] / year_mean_commentCount[movie_year]
    movie_df.loc[0, 'trailer_pos_sentiment_ratio'] = pos_sentiment_ratio
    movie_df.loc[0, 'trailer_neg_sentiment_ratio'] = neg_sentiment_ratio
    return movie_df
    
    
def build_features(df, with_Trailer=True):
    """
    Build X, y for the test movie.
    Args:
        df: a dataframe of one row (one movie).
    Return:
        X, y
    """
    columns_to_drop = ['Rank', 'Title', 'tt_id', 'rl_id', 'release_date', \
                   'company', 'mpaa', 'genres', 'director', 'actors', \
                   'bo_opening', 'bo_gross', 'imdb_score', 'budget', \
                   'trailer_viewCount',  ]
    if not with_Trailer:
        columns_to_drop += ['trailer_like_dislike_ratio', 'trailer_commentCount', 'trailer_mean_sentiment', \
                            'trailer_pos_sentiment_ratio', 'trailer_neg_sentiment_ratio']  
    
    feature_process_budget(df)
    feature_process_title(df)
    feature_process_date(df)
    feature_process_mpaa(df)
    feature_process_genre(df, top=15)
    feature_process_company(df, top=8)
    if crew_power:
        feature_process_director_power(df)
        feature_process_actor_power(df)
    else:
        feature_process_director(df)
        feature_process_actor(df)
    X = df.drop(columns_to_drop, axis=1)
    y = df['bo_opening'] / df['budget']
    return X, y
        
    
if __name__ == "__main__":
    
    # load models
    global crew_power
    crew_power = True
    if crew_power:
        model_wTrailer = joblib.load("./models/crew_power__w_trailer.model")
        model_woTrailer = joblib.load("./models/crew_power__wo_trailer.model")
    else:
        model_wTrailer = joblib.load("./models/crew_categ__w_trailer.model")
        model_woTrailer = joblib.load("./models/crew_categ__wo_trailer.model")
    
    cache_file = "./data/movie_2014-2019.csv"
    trailer_list_file = "./data/trailer_list/trailer_list_updated.csv"
    trailer_comments_dir = "./data/trailer_comments"
    trailer_comments_raw_dir = "./data/trailer_comments_raw"
    
    cache_df = pd.read_csv(cache_file)
    trailer_list_df = pd.read_csv(trailer_list_file)
    
    # prepare plotly api key
    with open('./private_api_key/plotly_API_key.txt', 'r') as f:
        s = f.read().split('\n')
    chart_studio.tools.set_credentials_file(username=s[0], api_key=s[1])
    
    initialize_logger()
    
    st.title("Welcome to Trailer Miner!")

    st.subheader("Please input the title of your interested movie (case-insensitive):")
    user_movie = st.text_input("", value="Joker")
    logger.info("=" * 30)
    logger.info("INPUT: {}".format(user_movie))
    
    movie = Movie()
    find_movie = movie.get_app_movie_info(user_movie)
    if not find_movie:
        st.subheader("Sorry, the movie '*{}*' is not found, please retry with a standard IMDB movie title.".format(user_movie))  
        st.markdown("**Examples:** ```21 Bridges```, ```Frozen II```, ```Avengers: Endgame```.")
        logger.info("Sorry, the movie '*{}*' is not found, please retry with a standard IMDB movie title.".format(user_movie))
    else:
        # Display movie info:
        display_movie_info(movie)
        if movie.year < 2014:
            st.subheader("Sorry, the movie '*{}*' is before 2014 and may not fit well with the model. Please retry with a movie after 2014.".format(user_movie))
            logger.info("Sorry, the movie '*{}*' is before 2014 and may not fit well with the model. Please retry with a movie after 2014.".format(user_movie))
        else:
            # Get trailer list of the movie:
            trailer_list, trailer_list_df = get_trailer_list(movie, trailer_list_df)
            
            # Display trailer list:
            display_trailer_links(movie, trailer_list)
            
            # Get trailer comments:
            trailer_list = trailer_list_df[(trailer_list_df.tt_id == movie.tt_id) & (trailer_list_df.comment_disabled == False)]\
                    .trailer_id.to_list() 
            for video_id in trailer_list:
                get_trailer_comments(video_id, trailer_list_df, trailer_comments_raw_dir, trailer_comments_dir)

            # Display trailer comments:
            display_trailer_comments(movie, trailer_list, trailer_comments_dir)

            # Display box office prediction:
            display_bo_prediction(movie, cache_df, trailer_list_df)

