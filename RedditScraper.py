from turtle import title
from bs4 import BeautifulSoup, SoupStrainer
import requests
import praw
from praw.models import MoreComments
from collections import OrderedDict
import pandas as pd

class Scraper:
    def __init__(self, id, secret, agent, subreddit) -> None:
        self.reddit = praw.Reddit(client_id=id, client_secret=secret, user_agent=agent)
        self.subreddit = self.reddit.subreddit(subreddit)

    def check_flairs(self, post, barred_flairs):
        for flair in barred_flairs:
            post_flairs = vars(post)['link_flair_text']
            if type(post_flairs) == list:
                for post_flair in post_flairs:
                    if post_flair.lower() == flair.lower():
                        return False
            elif type(post_flairs) == str:
                if post_flairs.lower() == flair.lower():
                    return False
        return True

    def get_stories(self, limit, comments=False, barred_flairs=[]):
        stories = []
        num_stories = 0
        for post in list(self.subreddit.hot())[1:]:
            post_data = self.extract_data(post, comments)

            if post_data[1] is None:
                continue
            elif post.stickied:
                continue
            elif not self.check_flairs(post, barred_flairs):
                continue
            elif num_stories >= limit:
                return stories
            else:
                stories.append(post_data)
                num_stories += 1

        return stories
        
    def extract_data(self, post, comments):
        title = self.get_title(post)
        text = self.get_text_praw(post)
        if comments:
            comments = self.get_comments(post)
        return title, text, comments

    def get_title(self, post):
        return post.title
    
    def get_comments(self, post):
        comments = []
        for comment in post.comments:
            if type(comment) == MoreComments:
                continue
            comments.append(comment.body)
        return comments

    def get_text_praw(self, post):
        try:
            return post.selftext
        except AttributeError:
            return None

    def get_text_bs4(self, url):
        request = requests.get(url)
        soup = BeautifulSoup(request.text, 'html.parser')
        text = None
        for div in soup.find_all('div'):
            if div.has_attr('class'):   
                if '3xX726aBn29LDbsDtzr_6E' in "".join(div['class']):
                    text = div.get_text()
        return text

def get_nosleep_data(id, secret, agent, limit):
    scraper = Scraper(id, secret, agent, 'nosleep')
    return scraper.get_stories(limit, comments=False, barred_flairs=['series'])

def get_AITA_data(id, secret, agent, limit):
    scraper = Scraper(id, secret, agent, 'nosleep')
    return scraper.get_stories(limit, comments=False, barred_flairs=['series'])
    
if __name__ == '__main__':
    print(get_nosleep_data(id='x4xepFGg0oNAZYWp1z9qXg', secret='VJQziUlLzwHviqHU1EWhCMaCJaXBSg', agent='AITACommentary', limit=1)[0])

