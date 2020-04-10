from googlesearch import search
from newspaper import Article

"""
Returns array of the text from the top 10 articles that come up given the search term
"""



def get_articles_text_from_search_term(term):
    articles = []
    for url in search(term, start=0, stop=10, pause=1):
        try:
            a = Article(url)
            a.download()
            a.parse()
            articles.append(a.text)
        except:
            pass
            # Prevent stopping all downloads in the event of a 403 or 503 error
            
    return articles
