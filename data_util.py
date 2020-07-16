#!/usr/bin/env python

"""
Utils for feature engineering. 

Author: Zhengyang Zhao
Date: Jun. 10 2020.

Title corner case: "Birdman or (The Unexpected Virtue of Ignorance)"  --> "Birdman"
"""


import os
import datetime
import holidays 
from functools import partial
import copy

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def is_sequel_movie(title):
    """
    Movie title may contains symbols: [",", "'", "?", "&", "!", ".", "/", "(", ")", "-", "\""], or ":".
    """
    
    if ":" in title:
        return True
    last_word = title.lower().split(" ")[-1]
    if last_word.isnumeric() and len(last_word) == 1:
        return True
    if last_word in ["ii", "iii", "vi"]:
        return True   # e.g.: Frozen II
    return False


def is_holiday(date):
    holidays_all = holidays.US(years=range(2014, 2021))
    holidays_exclude = ["Washington's Birthday", "Columbus Day", "Veterans Day"]
    holidays_select = [k for k in holidays_all if holidays_all[k] not in holidays_exclude]
    return date in holidays_select

def holiday_in_this_week(str_date):
    y = int(str_date[:4])
    m = int(str_date[5:7])
    d = int(str_date[-2:])
    date = datetime.date(y, m, d)
    for i in range(7):
        next_date = date + datetime.timedelta(days=i)
        if is_holiday(next_date):
            return True
    return False

def holiday_in_next_week(str_date):
    y = int(str_date[:4])
    m = int(str_date[5:7])
    d = int(str_date[-2:])
    date = datetime.date(y, m, d)
    date = datetime.date(y, m, d)
    for i in range(7, 14):
        next_date = date + datetime.timedelta(days=i)
        if is_holiday(next_date):
            return True
    return False


#################################################################
# Feature processing.
#################################################################




def feature_process_title(df):
    """
    Input feature: Title,
    Output feature: title_length, is_sequel_movie.
    """
    df.loc[:, 'title_length'] = df.loc[:, 'Title'].apply(lambda x : len(x))
    df.loc[:, 'is_sequel_movie'] = df.loc[:, 'Title'].apply(lambda x : int(is_sequel_movie(x)))
    
    
def feature_process_date(df):
    """
    Input feature: release_date,
    Output feature: Year, month, first_week_holiday, second_week_holiday
    """
    df.loc[:, 'Year'] = df.loc[:, 'release_date'].apply(lambda x : int(x[:4]))
    df.loc[:, 'month'] = df.loc[:, 'release_date'].apply(lambda x : int(x[5:7]))
    df.loc[:, 'first_week_holiday'] = df.loc[:, 'release_date'].apply(lambda x : int(holiday_in_this_week(x)))
    df.loc[:, 'second_week_holiday'] = df.loc[:, 'release_date'].apply(lambda x : int(holiday_in_next_week(x)))


def feature_process_budget(df):
    """
    Input feature: budget
    Output features:
        budget_to_mean_year (budget over the mean budget of that year).
        budget_norm (converted to 2019 USD using the CPI of each year).
            CPI data: https://fred.stlouisfed.org/series/CPIAUCNS
    """
    budget_mean = {2014: 54.4203125,
                     2015: 56.41322314049587,
                     2016: 60.24559999999999,
                     2017: 61.07619047619047,
                     2018: 56.18083333333333,
                     2019: 57.093805309734506}

#     budget_mean = {2014: 25,
#            2015: 27,
#            2016: 31,
#            2017: 40,
#            2018: 39,
#            2019: 40}
    
    cpi = {2014: 236,
           2015: 237,
           2016: 239,
           2017: 244,
           2018: 249,
           2019: 254}
    cpi[2020] = cpi[2019]
    cpi[2021] = cpi[2019]
    for i in range(df.shape[0]):
        year = df.loc[i, 'Year']
#         df.loc[i, 'budget_to_mean_year'] = df.loc[i, 'budget'] / budget_mean[year]
        df.loc[i, 'budget_norm'] = df.loc[i, 'budget'] / cpi[year] * cpi[2019]
    
    
def feature_process_mpaa(df):
    """
    Input feature: mpaa (4 types)
    Output features: one-hot encoding of mpaa
    """
    
    f_mpaa_list = "./data/feature_mpaa_list.txt"
    with open(f_mpaa_list, 'r') as f:
        f_mpaa = f.read().split('\n')[:-1]
    for f in f_mpaa:
        df.loc[:, 'mpaa_{}'.format(f)] = df.loc[:, 'mpaa'].apply(lambda x: int(x == f))
        
        
def feature_process_genre(df, top=12):
    """
    Input feature: genre (20 types)
    Output features: one-hot encoding of top-12 genre, genre number.
    """
    
    f_genre_list = "./data/feature_genre_list.txt"
    with open(f_genre_list, 'r') as f:
        f_genre = f.read().split('\n')[:-1][:top]
    for f in f_genre:
        df.loc[:, 'genres_{}'.format(f)] = df.loc[:, 'genres'].apply(lambda x: int(f in x.split(",")))
    df.loc[:, 'genres_count'] = df.loc[:, 'genres'].apply(lambda x: len(x.split(",")))
    

def feature_process_company(df, top=8):
    """
    Input feature: genre (15 companies)
    Output features: one-hot encoding of top-8 company.
    """
    
    f_company_list = "./data/feature_company_list.txt"
    with open(f_company_list, 'r') as f:
        f_company = f.read().split('\n')[:-1][:top]
    for f in f_company:
        df.loc[:, 'prod_{}'.format(f)] = df.loc[:, 'company'].apply(lambda x: int(x == f))
        
        
def feature_process_director(df):
    f_director_list = "./data/feature_director_list.txt"
    with open(f_director_list, 'r') as f:
        f_director = f.read().split('\n')[:-1]
    for f in f_director:
        df.loc[:, 'director_{}'.format(f)] = df.loc[:, 'director'].apply(lambda x: int(x == f))
        

def feature_process_actor(df):
    f_actor_list = "./data/feature_actor_list.txt"
    with open(f_actor_list, 'r') as f:
        f_actor = f.read().split('\n')[:-1]
    for f in f_actor:
        df.loc[:, 'actors_{}'.format(f)] = df.loc[:, 'actors'].apply(lambda x: int(f in x.split(",")))
        

def feature_process_director_power(df):
    dir_cnt_df = pd.read_csv("./data/directors_movie_count.csv", index_col=0)
    dir_bo_df = pd.read_csv("./data/directors_bo_power.csv", index_col=0)
    dir_imdb_df = pd.read_csv("./data/directors_imdb_power.csv", index_col=0)
    for i in df.index:
        year = df.loc[i, 'Year']
        director = df.loc[i, 'director']
        if director not in dir_cnt_df.index:
            df.loc[i, 'directors_movie_count'] = 0
            df.loc[i, 'directors_bo_power'] = np.nan
            df.loc[i, 'directors_imdb_power'] = np.nan
            continue
        dir_cnt = dir_cnt_df.loc[director, str(year)]
        dir_bo = dir_bo_df.loc[director, str(year)]
        dir_imdb = dir_imdb_df.loc[director, str(year)]
        df.loc[i, 'directors_movie_count'] = dir_cnt
        df.loc[i, 'directors_bo_power'] = dir_bo
        df.loc[i, 'directors_imdb_power'] = dir_imdb
        
        
def feature_process_actor_power(df):
    act_bo_df = pd.read_csv("./data/actors_bo_power.csv", index_col=0)
    act_imdb_df = pd.read_csv("./data/actors_imdb_power.csv", index_col=0)
    for i in df.index:
        year = df.loc[i, 'Year']
        actors = df.loc[i, 'actors'].split(',')[:]
        act_bo = []
        act_imdb = []
        for actor in actors:
            if actor not in act_bo_df.index:
                continue
            act_bo.append(act_bo_df.loc[actor, str(year)])
            act_imdb.append(act_imdb_df.loc[actor, str(year)])
        df.loc[i, 'actors_bo_power'] = np.nanmean(act_bo)
        df.loc[i, 'actors_imdb_power'] = np.nanmean(act_imdb)