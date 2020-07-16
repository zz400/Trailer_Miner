#!/usr/bin/env python

"""
Fetch the information of a Youtube video, using Youtube Data API.

Author: Zhengyang Zhao
Date: June 2 2020.

Ref: https://github.com/nikhilkumarsingh/YouTubeAPI-Examples
"""

import os
import json
from multiprocessing import Pool, cpu_count
import time
import datetime
from apiclient.discovery import build
import pandas as pd

from textblob import TextBlob

def get_trailer_ids(movie_name, movie_release_date, maxResults, youtube):
    """
    Given the movie_name and movie_release_date, search trailers of the movie.
    Using the 'search' module of Youtube Data API.
    """
    
    release_date = datetime.datetime.strptime(movie_release_date, "%Y-%m-%d")
    time_window_start = release_date - datetime.timedelta(days=500)
    time_window_end  = release_date - datetime.timedelta(days=14)
    
    part = "snippet"
    maxResults = maxResults   # acceptable values are 0 to 50
    order = "relevance"
    publishedAfter = time_window_start.strftime("%Y-%m-%dT%H:%M:%SZ")   # 1970-01-01T00:00:00Z
    publishedBefore = time_window_end.strftime("%Y-%m-%dT%H:%M:%SZ")
    q = movie_name + " trailer"
    type_ = "video"
    videoDuration = "short"  # less than 4min.
    
    trailer_res = youtube.search().list(
                    part=part,
                    maxResults=maxResults,
                    order=order,
                    publishedAfter=publishedAfter,
                    publishedBefore=publishedBefore,
                    q=q,
                    type=type_,
                    videoDuration=videoDuration
                   ).execute()
    
    trailer_dict = {}
    # Double check whether the video title contains 'trailer' (or 'teaser')
    for item in trailer_res['items']:
        video_title = item['snippet']['title']
        video_id = item['id']['videoId']
        video_publishTime = item['snippet']['publishTime']
        if ('trailer' not in video_title.lower() and 'teaser' not in video_title.lower()):
            continue
        trailer_dict[video_id] = [video_publishTime, video_title]
    
    return trailer_dict, trailer_res
    

def get_trialer_statistics(trailer_id, youtube):
    """
    Given the Youtube video_id, get the statistics of the video.
    Using the 'videos' module of Youtube Data API.
    
    Return:
        example:
        {'viewCount': '8097308',
         'likeCount': '48656',
         'dislikeCount': '1310',
         'favoriteCount': '0',
         'commentCount': '3497'}
    """
    part = "statistics"
    id_ = [trailer_id]
    res = youtube.videos().list(part=part, id=id_).execute()
    return res['items'][0]['statistics']
    
    

def get_comment_thread_1page(video_id, youtube, pageToken=None, include_replies=False):
    """
    Download the video comments in one page (1 page contains at most 100 comments).
    Check the API doc: https://developers.google.com/youtube/v3/docs/commentThreads/list
    You can also use url to fetch the json file: 
        https://www.googleapis.com/youtube/v3/commentThreads?key=xxxxxxxx&textFormat=plainText&part=snippet&videoId=qSemXDvSEZY&maxResults=100
    
    Note:
        Multiline comment: e.g. '快进下就听到\n让大家看一下我'
        Comment with emoji: e.g. '胖子出来反正你也看不到 。。。😂'
        Comment with a time pointer: e.g. '4:05 yeahh'
    """
    
    part = "snippet"
    if include_replies:
        part = "snippet,replies"
    maxResults=100
    textFormat="plainText"   # "html" or "plainText"
    #     moderationStatus = "likelySpam"
        
    comment_thread = youtube.commentThreads().list(
        part=part,
        maxResults=maxResults,
        pageToken=pageToken, # used to retreve the comments beyond maxResults.
        videoId=video_id,
        textFormat=textFormat,
    ).execute()
    return comment_thread


def get_comment_thread_allPages(video_id, youtube, save_dir="./data/trailer_comments/"):
    """
    Download all the comments about a video.
    """
    next_page_token = ""
    page = 0
    while True:
        comment_thread = get_comment_thread_1page(video_id, youtube, pageToken=next_page_token)
        with open(os.path.join(save_dir, video_id + " page_{:04d}.json".format(page)), 'w') as f:
            json.dump(comment_thread, f)
        if "nextPageToken" not in comment_thread:  # no more pages.
            break  
        next_page_token = comment_thread["nextPageToken"]
        page += 1
    return page + 1


def parse_comment_thread_1page(comment_thread):
    """
    Process the downloaded comment_thread.
    Extract ["timestamp", "text", "like_count", "reply_count", "author"] from the comment_thread.
    """
    
    comment_df = pd.DataFrame(columns=["timestamp", "text", "like_count", "reply_count", "author"])
    i = 0
    for item in comment_thread["items"]:
        comment = item["snippet"]["topLevelComment"]
        author = comment["snippet"]["authorDisplayName"]
        text = comment["snippet"]["textDisplay"]
        text = text.replace('\n',' ').replace('\r',' ')
        timestamp = comment["snippet"]["publishedAt"][:19]
        reply_count = item["snippet"]["totalReplyCount"]
        like_count = comment["snippet"]["likeCount"]
        comment_df.loc[i] = [timestamp, text, like_count, reply_count, author]
        i += 1
        # print("Comment by {}: {}".format(author, text))
        
#         if 'replies' in item.keys():
#             for reply in item['replies']['comments']:
#                 rauthor = reply['snippet']['authorDisplayName']
#                 rtext = reply["snippet"]["textDisplay"]
#             print("\n\tReply by {}: {}".format(rauthor, rtext), "\n")
    return comment_df


def parse_comment_thread_allPages(save_dir, video_id, total_pages):
    t0 = time.time()
    with Pool(cpu_count()) as p:
        comment_thread_list = [json.load(open(os.path.join(save_dir, video_id + " page_{:04d}.json".format(page)), 'r')) for page in range(total_pages)]
        df_list = p.map(parse_comment_thread_1page, comment_thread_list)
    comment_df = pd.concat(df_list)
    if comment_df.shape[0] > 0:
        comment_df.loc[:, 'trailer_id'] = video_id
        comment_df.to_csv(os.path.join(save_dir, "{}.csv".format(video_id)), index=False)
    t1 = time.time()
    print("Finished. Time: {:.1f}.".format(t1 - t0))
    return comment_df
    
    
        
