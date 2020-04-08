from googlesearch import search
from newspaper import Article

"""
Returns array of the text from the top 10 articles that come up given the search term
"""



def get_article_text_from_search_term(term):
    articles = []
    for url in search(term, start=0, stop=10, pause=1):
        a = Article(url)
        a.download()
        a.parse()
        articles.append(a.text)

    return articles


article = input("Input query ")

article = get_article_text_from_search_term(article)
print(article)
