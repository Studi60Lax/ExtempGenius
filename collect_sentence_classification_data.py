import scrape
import clean_data
import evaluate_data

term = input("Search term\n>")
print("Scraping Articles.")
articles = scrape.get_articles_text_from_search_term(term)
print("Cleaning.")
cleaned_corpus = clean_data.full_clean(articles)

grams1 = evaluate_data.get_top_n_grams(cleaned_corpus, 1, 4)
grams2 = evaluate_data.get_top_n_grams(cleaned_corpus, 2, 4)
grams3 = evaluate_data.get_top_n_grams(cleaned_corpus, 3, 4)
grams4 = evaluate_data.get_top_n_grams(cleaned_corpus, 4, 4)

all_grams = grams1 + grams2 + grams3 + grams4
all_grams_only_text = []

for g in all_grams:
    all_grams_only_text.append(g[0])

# Put them into their sentences
gram_sentences = {}
for g in all_grams_only_text:
    gram_sentences[g] = []
    for c in articles:
        csents = clean_data.split_into_sentences(c)
        csentsclean = clean_data.full_clean(csents)
        i = 0
        for sent in csentsclean:
            if g in sent:
                gram_sentences[g].append(csents[i])
            i += 1

inc = 0
for g in gram_sentences:
    for sent in gram_sentences[g]:
        inc += 1
        print(str(inc) + ")")
        print(sent)
        print()
        print(g)
        print()
        choice = input("Event [1]\nActor [2]\nImpact [3]\nLocation [4]\nBad Data [5]\nDon't Know [6]\n>")
        choices = ['event', 'actor', 'impact', 'location', 'ignore', 'idk']
        choice = choices[int(choice)-1]
        if choice != "idk":
            with open('./sentence_classification_data.csv','a') as file:
                file.write(choice + ',' + g + ',' + sent + '\n')
        print('\n')
        if inc % 5 == 0:
            break
