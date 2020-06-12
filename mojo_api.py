"""
Box Office Mojo API
Crawl movie information from www.boxofficemojo.com

Author: Zhengyang Zhao
Date: 2020-06-08

Mojo API example: https://www.youtube.com/watch?v=LcnZhJnSXTE
BeautifulSoup tutorials: 
https://www.jianshu.com/p/2b783f7914c6
https://www.dataquest.io/blog/web-scraping-beautifulsoup/

Old Mojo API's that no longer work:
https://github.com/skozilla/BoxOfficeMojo  -- Python2
https://github.com/lamlamngo/Box-Office-Mojo-API  -- 2 years ago
https://github.com/earthican/BoxOfficeMojoAPI -- 6 years ago
"""


import requests
from bs4 import BeautifulSoup
import re
from decimal import Decimal
import urllib.parse
import json
from datetime import datetime

import pandas


headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97 Safari/537.36'}
metadata_dir = "./data/movie_metadata/"
today = datetime.today().strftime('%Y-%m-%d')  # '2020-06-10'


def get_movie_list(year):
    res = requests.get("https://www.boxofficemojo.com/year/{}/".format(year), headers=headers)
    soup = BeautifulSoup(res.text, 'html5lib')
    dfs = pandas.read_html(soup.select_one('table').prettify())
    df = dfs[0]
    df = df[~df['Release'].str.contains('-release')]  # remove re-release movies.
    df = df[~df['Release'].str.contains("'s cut")]  # remove Director's Cut.
    df.loc[:, 'Year'] = year
    df = df[['Year', 'Rank', 'Release']]
    df.columns = ['Year', 'Rank', 'Title']
    return df


class Movie(object):
    def __init__(self):
        self.title = None
        self.tt_id = None  # imdb id of the movie. Also used on www.boxofficemojo.com
        self.rl_id = None  # the release id used on www.boxofficemojo.com. e.g. rl4032333313
        self.metadata = None  # json file of the movie metadata, got using omdb-api
        self.released = True
        
        self.year = None  # For training movie, from the training movie_list. For user-input movie, from omdb-api.
        self.release_date = None
        self.company = None
        self.mpaa = None  # e.g. "PG-13"
        self.genres = None
        self.movie_length = None
        self.director = None
        self.actors = None
        
        self.budget = None
        self.bo_opening = None  # first weekend box office. Note: almost all movies are released on Friday.
        self.bo_gross = None
        self.imdb_score = None

    def get_train_movie_info(self, title, year):
        self.set_title(title)
        self.set_year(year)
        find_imdb_id = self.title_to_id()
        if not find_imdb_id:
            print("imdb_id not found. title: {}, year: {}.".format(title, year))
            return
#         self.id_to_metadata()
        
        self.crawl_basic_info()
        self.crawl_boxoffice_info()
        self.crawl_cast_info()
        if self.budget == None:
            self.get_imdb_budget()

    def get_app_movie_info(self, title):
        self.set_title(title)
        find_imdb_id = self.title_to_id()
        if not find_imdb_id:
            print("movie not found. title: {}.".format(title))
            return False
#         self.id_to_metadata()
        
        self.crawl_basic_info()
        self.crawl_boxoffice_info()
        self.crawl_cast_info()
        if self.budget == None:
            self.get_imdb_budget()
        self.print_info()
        return True

    def set_title(self, title):
        self.title = title
    
    def set_year(self, year):
        self.year = year
        
    def title_to_id(self):
        title_url = urllib.parse.quote(self.title)
        imdb_url = "https://www.imdb.com/search/title/?title={}".format(title_url)
        if self.year != None:
            imdb_url = "https://www.imdb.com/search/title/?title={}&release_date={}-01-01,{}-12-31"\
                        .format(title_url, self.year - 1, self.year)
#         print(imdb_url)
        imdb_res = requests.get(imdb_url, headers=headers)
        if imdb_res.status_code != 200:
            print("Request Error. Code =", imdb_res.status_code)
            return False
        imdb_soup = BeautifulSoup(imdb_res.text, 'html5lib')
        imdb_items = imdb_soup.find_all("div", class_="lister-item-content")
        find_movie = False
        for imdb_item in imdb_items[:5]:
            imdb_title = imdb_item.h3.a.get_text()
            if imdb_title.lower() != self.title.lower():
                continue
            self.title = imdb_title
            self.tt_id = imdb_item.h3.a['href'].split('/')[-2]
            if imdb_item.div != None:
                self.imdb_score = float(imdb_item.div.div['data-value'])
            find_movie = True
            break
        return find_movie
    
    def id_to_metadata(self):
        """
        Get movie metadata directly using omdb-api.
        Note: the api-key allows 1000 queries per day.
        """
        if self.tt_id == None:
            print("tt_id is None!")
            return 
        url = "http://www.omdbapi.com/?i={}&apikey=241cf09".format(self.tt_id)
        res = requests.get(url)
        metadata_json = json.loads(res.text)
        assert(metadata_json['imdbID'] == self.tt_id)
        assert(metadata_json['Title'] == self.title)
        self.metadata = metadata_json
        with open(metadata_dir + "{}.json".format(self.tt_id), 'w') as f:
            json.dump(metadata_json, f)
        
    def crawl_basic_info(self):
        info_url = "https://www.boxofficemojo.com/title/{}/".format(self.tt_id)
        info_res = requests.get(info_url, headers=headers)
        if info_res.status_code != 200:
            print("Request Error. Code =", info_res.status_code)
            return
        info_soup = BeautifulSoup(info_res.text, 'html5lib')
        self.info_soup = info_soup
        info_table = self.info_soup.find('div', class_="a-section a-spacing-none mojo-summary-values mojo-hidden-from-mobile")
        info_entries = info_table.find_all('div')
        self.info_entries = info_entries
        
        for info_entry in self.info_entries:
            key = info_entry.span.get_text()  # 'Budget', 'Genres', etc.
            
            if key == 'Domestic Distributor':
                self.process_info_company(info_entry)
            elif key == 'Domestic Opening':
                pass  
            elif key == 'Budget':
                self.process_info_budget(info_entry)
            elif key == 'Earliest Release Date':
                pass
            elif key == 'MPAA':
                self.process_info_mpaa(info_entry)            
            elif key == 'Running Time':
                self.process_info_movieLength(info_entry)
            elif key == 'Genres':
                self.process_info_genres(info_entry) 
                
    def process_info_company(self, info_entry):
        for e in info_entry.select('span')[1]:
            self.company = str(e)
            break
            
    def process_info_budget(self, info_entry):
        money = info_entry.select('span')[1].get_text()
        self.budget = float(Decimal(re.sub(r'[^\d.]', '', money))) / 1000000.0
        
    def process_info_mpaa(self, info_entry):
        self.mpaa = info_entry.select('span')[1].get_text()
        
    def process_info_movieLength(self, info_entry):
        movie_len = info_entry.select('span')[1].get_text()
        s = movie_len.split(" ")
        if len(s) < 4:
            self.movie_length = int(s[0])
        else:
            hours = int(s[0])
            minutes = int(s[2])
            self.movie_length = 60 * hours + minutes
        
    def process_info_genres(self, info_entry):
        s = info_entry.select('span')[1].get_text()
        self.genres = re.sub("\s+", ",", s.strip())
        
    def crawl_boxoffice_info(self):
        realease_table = self.info_soup.select_one('table').select_one('tbody')
        if len(realease_table.select('tr')) != 2:
            # for a few movies which has been re-released, the domestic box office table is at another page.
            # e.g. https://www.boxofficemojo.com/title/tt0816692
            realease_table = self.crawl_boxoffice_info_helper(realease_table)
        assert(len(realease_table.select('tr')) == 2)   # rows of table 
        assert(len(realease_table.select('tr')[1].select('td')) == 4)   # columns of table 
        assert(realease_table.select('tr')[1].select('td')[0].get_text() == "Domestic")
        self.rl_id = realease_table.select('tr')[1].select('td')[0].a['href'].split('?')[0].split('/')[2]
        release_date = realease_table.select('tr')[1].select('td')[1].get_text()
        self.release_date = self.parse_date(release_date)
        self.year = int(self.release_date[:4])
        if today < self.release_date:
            self.released = False
            return
        money_opening = realease_table.select('tr')[1].select('td')[2].get_text()
        self.bo_opening = float(Decimal(re.sub(r'[^\d.]', '', money_opening))) / 1000000.0
        money_gross = realease_table.select('tr')[1].select('td')[3].get_text()
        self.bo_gross = float(Decimal(re.sub(r'[^\d.]', '', money_gross))) / 1000000.0
        
        
    def crawl_boxoffice_info_helper(self, realease_table):
        assert(realease_table.select('tr')[1].select('td')[0].a.get_text() == "Original Release")
        ref = "https://www.boxofficemojo.com" + realease_table.select('tr')[1].select('td')[0].a['href']
        temp_res = requests.get(ref, headers=headers)
        temp_soup = BeautifulSoup(temp_res.text, 'html5lib')
        temp_table = temp_soup.select_one('table').select_one('tbody')
        temp_table.select('tr')[0].extract()
        return temp_table
        
    def parse_date(self, string):
        """
        Input format: "Nov 20, 2015"
        """
        strings = string.split(" ")
        date = int(strings[1][:-1])
        year = int(strings[2])
        month_dict = {'Jan' : '01',
                     'Feb' : '02',
                     'Mar' : '03',
                     'Apr' : '04',
                     'May' : '05',
                     'Jun' : '06',
                     'Jul' : '07',
                     'Aug' : '08',
                     'Sep' : '09',
                     'Oct' : '10',
                     'Nov' : '11',
                     'Dec' : '12'}
        month = month_dict[strings[0]]
        return "{}-{}-{:02d}".format(year, month, date)
    
    def crawl_cast_info(self):
        crew_url = "https://www.boxofficemojo.com/title/{}/credits/".format(self.tt_id)
        crew_res = requests.get(crew_url, headers=headers)
        if crew_res.status_code != 200:
            print("Request Error. Code =", crew_res.status_code)
            return
        crew_soup = BeautifulSoup(crew_res.text, 'html5lib')
        crew_table = crew_soup.select('table')[0].prettify()
        crew_df = pandas.read_html(crew_table)[0]
        self.director = crew_df[crew_df['Role'] == 'Director']['Crew Member'].iloc[0]
        actor_table = crew_soup.select('table')[1].prettify()
        actor_df = pandas.read_html(actor_table)[0]
        actor1 = actor_df['Actor'].iloc[0]
        self.actors = actor1
        if actor_df.shape[0] > 1: 
            actor2 = actor_df['Actor'].iloc[1]
            self.actors += "," + actor2
        
    def crawl_boxoffice_byWeek(self):
        time_range = "weekly"
        bo_url = "https://www.boxofficemojo.com/release/{}/{}/".format(self.rl_id, time_range)
        bo_res = requests.get(bo_url, headers=headers)
        if bo_res.status_code != 200:
            print("Request Error. Code =", bo_res.status_code)
            return
        bo_soup = BeautifulSoup(bo_res.text, 'html5lib')
        bo_df = pandas.read_html(bo_soup.select_one('table').prettify())[0]
        return bo_df
    
    def get_imdb_budget(self):
        imdb_url = "https://www.imdb.com/title/{}/".format(self.tt_id)
        imdb_res = requests.get(imdb_url)
        if imdb_res.status_code != 200:
            print("Request Error. Code =", imdb_res.status_code)
            return
        imdb_soup = BeautifulSoup(imdb_res.text, 'html5lib')
        if imdb_soup.find('h4', text="Budget:") != None:
            item = imdb_soup.find('h4', text="Budget:").parent
            # print(list(item.strings))
            money = "$" + item.get_text().split('\n')[1].split('$')[-1]
            self.budget = float(Decimal(re.sub(r'[^\d.]', '', money))) / 1000000.0
        else:
            print(self.title, ": budget still not found on IMDB.")
    
    def print_info(self):
        print("title:", self.title)
        print("tt_id:", self.tt_id)
        print("rl_id:", self.rl_id)
        print("year:", self.year)
        print("release_date:", self.release_date)
        print("company:", self.company)
        print("mpaa:", self.mpaa)
        print("genres:", self.genres)
        print("movie_length:", self.movie_length)
        print("director:", self.director)
        print("actors:", self.actors)
        print("budget:", self.budget)
        print("bo_opening:", self.bo_opening)
        print("bo_gross:", self.bo_gross)
        print("imdb_score", self.imdb_score)