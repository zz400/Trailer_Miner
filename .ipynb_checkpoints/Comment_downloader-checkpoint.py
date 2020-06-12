#!/usr/bin/env python

"""
Fetch the comments from a Youtube video, using Youtube Data API.
Author: Zhengyang Zhao
Date: June 2 2020.

Ref: https://github.com/nikhilkumarsingh/YouTubeAPI-Examples
"""

from apiclient.discovery import build
import pandas as pd
import time

def get_comment_thread(youtube, video_id, pageToken=None, include_replies=False, textFormat="plainText"):
    """
    Download the video comments using YouTube Data API.
    Check the API doc: https://developers.google.com/youtube/v3/docs/commentThreads/list
    You can also use url to fetch the json file: 
        https://www.googleapis.com/youtube/v3/commentThreads?key=xxxxxxxx&textFormat=plainText&part=snippet&videoId=qSemXDvSEZY&maxResults=100
    
    Args:
        textFormat: "html" or "plainText".
        
    Return:
        a json file.
        
    Note:
        Multiline comment: e.g. '快进下就听到\n让大家看一下我'
        Comment with emoji: e.g. '胖子出来反正你也看不到 。。。😂'
        Comment with a time pointer: e.g. '4:05 yeahh'
    """
    
    part = "snippet"
#     moderationStatus = "likelySpam"
    if include_replies:
        part = "snippet,replies"
        
    comment_thread = youtube.commentThreads().list(
        part=part,
        maxResults=100,
        pageToken=pageToken, # used to retreve the comments beyond maxResults.
        videoId=video_id,
        textFormat=textFormat,
#         moderationStatus=moderationStatus
    ).execute()
    return comment_thread


def parse_comment_thread(comment_thread, comment_df):
    """
    Extract ["timestamp", "text", "like_count", "reply_count", "author"] from the comment_thread.
    
    Args:
        comment_df: dataframe recording all comments.
    """
    
    i = comment_df.shape[0]
    for item in comment_thread["items"]:
        comment = item["snippet"]["topLevelComment"]
        author = comment["snippet"]["authorDisplayName"]
        text = comment["snippet"]["textDisplay"]
        text = text.replace('\n',' ').replace('\r',' ')
        timestamp = comment["snippet"]["publishedAt"]
        reply_count = item["snippet"]["totalReplyCount"]
        like_count = comment["snippet"]["likeCount"]
        comment_df.loc[i] = [timestamp, text, like_count, reply_count, author]
        i += 1
        # print("Comment by {}: {}".format(author, text))
        
        if 'replies' in item.keys():
            for reply in item['replies']['comments']:
                rauthor = reply['snippet']['authorDisplayName']
                rtext = reply["snippet"]["textDisplay"]
            # print("\n\tReply by {}: {}".format(rauthor, rtext), "\n")


def download_comment_pipeline(movie_year, movie_rank, trailer_id):
    key_file = "./private_key/youtube_data_API_key.txt"
    video_id = trailer_id
    api_key = open(key_file, "r").read()
    youtube = build('youtube', 'v3', developerKey=api_key)

    next_page_token = ""
    comment_df = pd.DataFrame(columns=["timestamp", "text", "like_count", "reply_count", "author"])
    page = 0
    t0 = time.time()
    
    while True:
        comment_thread = get_comment_thread(youtube, video_id, pageToken=next_page_token)
        parse_comment_thread(comment_thread, comment_df)
        if "nextPageToken" not in comment_thread:  # no more pages.
            break  
        next_page_token = comment_thread["nextPageToken"]
        page += 1

    t1 = time.time()
    print('Finish {}-{}. {} comments fetched. Time: {:.1f} seconds. Page: {}'
          .format(movie_year, movie_rank, comment_df.shape[0], (t1 - t0), page) )
    comment_df.to_csv("./data/{}_{}_{}.csv".format(movie_year, movie_rank, trailer_id), index=False)