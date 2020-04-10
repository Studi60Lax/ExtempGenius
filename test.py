import scrape
import clean_data
import evaluate_data

term = input("Search term\n>")
print("Scraping Articles.")
articles = scrape.get_articles_text_from_search_term(term)
print("Cleaning.")
cleaned_corpus = clean_data.full_clean(articles)

print("Generating N-Grams.\n")

print("Top 10 1-grams")
print(evaluate_data.get_top_n_grams(cleaned_corpus, 1, 10))
print("\n")

print("Top 10 2-grams")
print(evaluate_data.get_top_n_grams(cleaned_corpus, 2, 10))
print("\n")


print("Top 10 3-grams")
print(evaluate_data.get_top_n_grams(cleaned_corpus, 3, 10))
print("\n")


print("Top 10 4-grams")
print(evaluate_data.get_top_n_grams(cleaned_corpus, 4, 10))
print("\n")
