import sklearn
from collections import defaultdict, Counter
from nltk import ngrams
from mpmath import *
import numpy as np
from fractions import Fraction
import operator
import training_model
from heapq import nlargest
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score


def choose(n, k):

    if k > n // 2: k = n - k
    p = Fraction(1)
    for i in range(1, k + 1):
        p *= Fraction(n - i + 1, i)
    return int(p)



def NFA(domain_test, n_gram_train, total_Category, All_Category_ngram_count, total, N_gram_meaning, ngrammax):

    ''' Number of False Alarms (NFA) expresses a similarity measure between a domain name and a category.
        NFA algorithm employs the Helmholtz principle: meaningful features and notable events appear as
        significant deviations from randomness or noise. For these reasons, humans can perceive the
        significance of the characteristics mentioned above. A low value of NFA connotes a perceptually
        meaningful event. In our case to classify  the  domain,  by  assigning the  meaning  score to
        each gram  of  the  domain,  meaningfulness  structure  of  the  domainis obtained.

        Reference
        ----------
        Helen Balinsky, Alexander Balinsky, and Steven Simske. Document sentences as a small world
        n 2011 IEEE International Conference on Systems, Man, and Cybernetics, pages 2583–2588. IEEE, 2011.'''

    TO_REMOVE = ['-']

    domain = domain_test
    for ch in TO_REMOVE:
        domain = domain.replace(ch, '')

    ngram_test = []
    for n in range(3, ngrammax):
        n_grams_char = ngrams(domain, n)
        for elm in n_grams_char:
            ngram = ''.join(elm)
            ngram_test.append(ngram)


    finder_Counter = Counter(ngram_test)
    freq_sum = defaultdict(int)
    for k, v in finder_Counter.items():
        for Category_train in n_gram_train:

            if k in N_gram_meaning[Category_train]:
                freq_sum[Category_train] += v * N_gram_meaning[Category_train][k]
            else:
                if k in n_gram_train[Category_train]:
                    N_gram_meaning[Category_train][k] = log(
                        mpf(choose(All_Category_ngram_count[k],
                                   n_gram_train[Category_train][k])) / (
                                int(total / total_Category[Category_train]) ** (
                                n_gram_train[Category_train][k] - 1))) * - 1 / \
                                                        n_gram_train[Category_train][k]

                    freq_sum[Category_train] += v * N_gram_meaning[Category_train][k]
                else:
                    if k in All_Category_ngram_count:
                        N_gram_meaning[Category_train][k] = - 1 / 1 * log(

                            mpf(choose(All_Category_ngram_count[k], 0)) / (

                                    int(total / total_Category[Category_train]) ** (0 - 1)))

                        freq_sum[Category_train] += v * N_gram_meaning[Category_train][k]

    return [freq_sum, N_gram_meaning]



def NFA_Supervised(validation, test, Domain_Categories_set, ngrammax):

    ''' finding the category of each web site based on domain name using NFA measure '''


    n_gram_train, total_Category, All_Category_ngram_count, total, list_of_Cat_Cover_ngam = training_model.training_model(validation, test, Domain_Categories_set, 5, ngrammax)


    domain_Category = defaultdict(list)
    N_gram_meaning = defaultdict(list)

    for Category in n_gram_train:
        N_gram_meaning[Category] = defaultdict(list)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)



    correct = 0
    uncorrect = 0

    confusion_matrix = defaultdict(list)
    for cat in n_gram_train:
        confusion_matrix[cat] = defaultdict(int)

    Position_Error = 0
    num_test = 0
    Y_test = []
    Y_prediction = []
    for domain_test in validation:
        domain_test_wless = domain_test.replace('www.', '')
        freq_sum, N_gram_meaning = NFA(domain_test, n_gram_train, total_Category, All_Category_ngram_count, total, N_gram_meaning, ngrammax)
        Y_test.append(Domain_Categories_set[domain_test_wless])


        if sum(freq_sum.values()) != 0:
            Result_Category = max(freq_sum.items(), key=operator.itemgetter(1))[0]
            domain_Category[domain_test] = Result_Category
            pi = 1
            for j in range(0, len(freq_sum)):
                if max(freq_sum.items(), key=operator.itemgetter(1))[0] == Domain_Categories_set[domain_test_wless]:
                    break
                pi += 1
                freq_sum[max(freq_sum.items(), key=operator.itemgetter(1))[0]] = -1000

            Position_Error += (pi - 1) / 24
            num_test += 1
            if Domain_Categories_set[domain_test_wless] != Result_Category:
                Y_prediction.append(Result_Category)
                uncorrect += 1

                fn[Domain_Categories_set[domain_test_wless]] += 1
                fp[Result_Category] += 1
                confusion_matrix[Domain_Categories_set[domain_test_wless]][Result_Category] += 1

            else:
                Y_prediction.append(Result_Category)
                correct += 1
                tp[Domain_Categories_set[domain_test_wless]] += 1
                confusion_matrix[Domain_Categories_set[domain_test_wless]][
                    Domain_Categories_set[domain_test_wless]] += 1
        else:

            Y_prediction.append('prediction')

    Accuracy = metrics.accuracy_score(Y_test, Y_prediction)
    Micro_Precison = metrics.precision_score(Y_test, Y_prediction, average='micro')
    Micro_Recall = metrics.recall_score(Y_test, Y_prediction, average='micro')
    Micro_f1 = metrics.f1_score(Y_test, Y_prediction, average='micro')


    Macro_Precison = metrics.precision_score(Y_test, Y_prediction, average='macro')
    Macro_Recall = metrics.recall_score(Y_test, Y_prediction, average='macro')
    Macro_f1 = metrics.f1_score(Y_test, Y_prediction, average='macro')

    Avg_position_error = Position_Error / len(test)


    test_elem = defaultdict(int)
    for elem in Y_test:
        test_elem[elem] = 1

    test_elem2 = defaultdict(int)
    for elem in Y_prediction:
        test_elem2[elem] = 1

    Y_test_num = []
    Y_prediction_num = []
    cat_num = defaultdict(int)
    num_cat = defaultdict(int)
    i = 0
    for elem in Y_test:
        if elem not in cat_num:
            cat_num[elem] = i
            num_cat[i] = elem
            i += 1
        Y_test_num.append(cat_num[elem])
    for elem in Y_prediction:
        if elem not in cat_num:
            cat_num[elem] = i
            num_cat[i] = elem
            i += 1
        Y_prediction_num.append(cat_num[elem])
    precision1, recall1, fscore, support = score(Y_test_num, Y_prediction_num)

    prec = defaultdict(int)
    rec = defaultdict(int)
    fm = defaultdict(int)
    sup = defaultdict(int)
    for elem in range(len(precision1)):
        prec[num_cat[elem]] = precision1[elem]
        rec[num_cat[elem]] = recall1[elem]
        fm[num_cat[elem]] = fscore[elem]
        sup[num_cat[elem]] = support[elem]

    sum_ = 0
    for elem in precision1:
        sum_ += elem
    avg1 = sum_ / 24
    return [prec, rec, fm, sup, confusion_matrix, Accuracy, Avg_position_error, Macro_Precison, Macro_Recall, Macro_f1]


def Distance_Domain_NFA(all_domain, validation, test, Domain_Categories_set, ngram):
    '''Computing the distance of each domain to each category as a vector with 25 dimensions
      (number of categoriese) based on NFA '''


    n_gram_train, total_Category, All_Category_ngram_count, total, list_of_Cat_Cover_ngam = training_model.training_model(validation, test, Domain_Categories_set, 2, ngram)

    N_gram_meaning = defaultdict(list)
    for Category in n_gram_train:
        N_gram_meaning[Category] = defaultdict(list)


    Vector = []
    Number_domain = defaultdict(list)
    Domain_Number = defaultdict(int)
    num = 0
    for domain in all_domain:

        Number_domain[num] = domain
        Domain_Number[domain] = num
        num += 1
        X = []
        freq_sum, N_gram_meaning = NFA(domain, n_gram_train, total_Category, All_Category_ngram_count, total, N_gram_meaning, ngram)
        for cat in n_gram_train:
            X.append(freq_sum[cat])

        X = np.array(X)
        if np.sum(X) == 0:
            X=np.ones(len(X))
        normalized_X = X / np.sqrt(np.sum(X ** 2))
        Vector.append(normalized_X)

    Vector = np.array(Vector)
    return [Vector, Number_domain, Domain_Number]


def Similarity_based_Domain(All_domain, validation, test, Domain_Categories_set, ngram):

    ''' finding similarity between each 2 websites based on domain name using NFA '''

    Vector, Number_Domain, Domain_Number = Distance_Domain_NFA(All_domain, validation, test, Domain_Categories_set, ngram)

    similarity = sklearn.metrics.pairwise.cosine_similarity(Vector, Y=None, dense_output=True)

    print(' similarity is done ')

    return similarity, Number_Domain, Domain_Number