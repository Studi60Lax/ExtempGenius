import scrape
import clean_data
import evaluate_data

term = input("Search term\n>")
print("Scraping Articles.")
articles = scrape.get_articles_text_from_search_term(term)
print("Cleaning.")
cleaned_corpus = clean_data.full_clean(articles)

print("Generating N-Grams.\n")
grams4 = evaluate_data.get_top_n_grams(cleaned_corpus, 4, 10)
grams3 = evaluate_data.get_top_n_grams(cleaned_corpus, 3, 10, remove_phrases=grams4)
grams2 = evaluate_data.get_top_n_grams(cleaned_corpus, 2, 10, remove_phrases=grams4+grams3)
grams1 = evaluate_data.get_top_n_grams(cleaned_corpus, 1, 10, remove_phrases=grams4+grams3+grams2)


print("Top 10 1-grams")
print(grams1)
print("\n")

print("Top 10 2-grams")
print(grams2)
print("\n")


print("Top 10 3-grams")
print(grams3)
print("\n")


print("Top 10 4-grams")
print(grams4)
print("\n")
