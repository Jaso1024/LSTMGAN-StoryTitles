import nltk
from nltk.corpus import stopwords
from RedditScraper import get_nosleep_data
import re

class Formatter:
    def __init__(self) -> None:
        self.stopwords = set(stopwords('english'))
    
    def format_title(self, title):
        title = title.lower()
        title = [word for word in title if word not in self.stopwords]
        title = "".join(title)
        return title