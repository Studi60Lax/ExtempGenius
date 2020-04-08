
''' Using the first article for testing purposes only'''
import scrape

import clean_data

article = scrape.article[0]
print (clean_data.full_clean(article))
