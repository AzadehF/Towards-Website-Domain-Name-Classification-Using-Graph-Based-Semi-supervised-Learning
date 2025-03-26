from collections import defaultdict, Counter
from nltk import ngrams
import pickle


def training_model(validation, test, Domain_Categories_set, min, ngrammax):

    '''For building the training model, for each category c, the frequency among domains of each n-gram
       in this category is counted. n-grams of domains are considered as sequences of characters in all
       the domain levels. For instance, in google.com, the 1-grams are {g, o, o, g, l, e, c, o, m },
       whereas the 2-grams (bi-grams) are {go, oo, og, gl, le, co, om}. For each domain, all its n-grams
       with 3 ≤ n ≤ 6 are extracted. '''

    Categories_Domains_Initial = defaultdict(list)
    for domain in Domain_Categories_set:
        Categories_Domains_Initial[Domain_Categories_set[domain]].append(domain)




    TO_REMOVE = ['-']
    categories_n_gram = defaultdict(list)
    for Category in Categories_Domains_Initial:
        for domain in Categories_Domains_Initial[Category]:
            if domain not in test and domain not in validation:
                for ch in TO_REMOVE:
                    domain = domain.replace(ch, '')
                for n in range(3, ngrammax):
                    n_grams_char = ngrams(domain, n)
                    for elem in n_grams_char:
                        ngram = ''.join(elem)
                        categories_n_gram[Category].append(ngram)

    n_gram_train = defaultdict()
    for Category in categories_n_gram:
        n_gram_train[Category] = Counter(categories_n_gram[Category])

    del_ngram = defaultdict()
    for Category in n_gram_train:
        for key in n_gram_train[Category]:
            if n_gram_train[Category][key] <= min: # n-grams with low frequencies are deleted because they resemble noise in each category
                if Category not in del_ngram:
                    del_ngram[Category] = []
                    del_ngram[Category].append(key)
                else:
                    del_ngram[Category].append(key)

    for Category in del_ngram:
        for key in del_ngram[Category]:
            del n_gram_train[Category][key]

    total_Category = defaultdict(int)
    for Category in n_gram_train:
        for key in n_gram_train[Category]:
            total_Category[Category] += n_gram_train[Category][key]

    total = 0
    for Category in n_gram_train:
        for key in n_gram_train[Category]:
            total += n_gram_train[Category][key]

    All_Category_ngram = {}
    All_Category_ngram_count = {}
    list_of_Cat_Cover_ngam = defaultdict(list)
    for Category in n_gram_train:
        for key in n_gram_train[Category]:
            if key not in All_Category_ngram:
                All_Category_ngram[key] = 1
                All_Category_ngram_count[key] = n_gram_train[Category][key]
            else:
                All_Category_ngram[key] += 1
                All_Category_ngram_count[key] += n_gram_train[Category][key]
            if Category not in list_of_Cat_Cover_ngam[key]:
                list_of_Cat_Cover_ngam[key].append(Category)

    return [n_gram_train, total_Category, All_Category_ngram_count, total, list_of_Cat_Cover_ngam]

