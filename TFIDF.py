import training_model
from collections import defaultdict, Counter
from nltk import ngrams
import operator
from heapq import nlargest
from sklearn import metrics
import numpy as np
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

from sklearn.metrics import precision_recall_fscore_support as score

def TFIDF(validation, test, Domain_Categories_set, ngrammax):

    ''' Assign the category of each web site based on domain name using TFIDF (Term Frequency–Inverse Document Frequency)
       TFIDF reflects how important an n-gram g is for a category c ∈ C to the set of all categories in the training model

        Reference
        ----------
        Stephen E Robertson and KSparck Jones. Relevance weighting of search terms  Journal of the American Society for
        Information science,27(3): 129–146, 1976. '''

    # Building the training model of n-gram of domain names
    n_gram_train, total_Category, All_Category_ngram_count, total, list_of_Cat_Cover_ngam = training_model.training_model(validation, test, Domain_Categories_set, 5, ngrammax)

    TO_REMOVE = ['-']

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    correct = 0
    uncorrect = 0
    TF_IDF = defaultdict(list)
    confusion_matrix = defaultdict(list)
    for cat in n_gram_train:
        confusion_matrix[cat] = defaultdict(int)

    num_test = 0
    Position_Error = 0



    max_grm = defaultdict(list)
    for cat in n_gram_train:
        maximum = 0
        for gm in n_gram_train[cat]:
            if n_gram_train[cat][gm] > maximum:
                maximum = n_gram_train[cat][gm]
                max_grm[cat] = gm

    Y_test = []
    Y_prediction = []

    for test_domain in validation:
        test_domain_wless = test_domain.replace('www.', '')
        Y_test.append(Domain_Categories_set[test_domain_wless])
        domain = test_domain
        for ch in TO_REMOVE:
            domain = domain.replace(ch, '')
        TF_IDF[domain] = defaultdict(int)
        ngram_test = []
        for n in range(3, ngrammax):
            ''' extract the all-gram of domain '''
            n_grams_char = ngrams(domain, n)
            for elm in n_grams_char:
                ngram = ''.join(elm)
                ngram_test.append(ngram)

        finder_Counter = Counter(ngram_test)
        freq_sum = defaultdict(int)
        for k, v in finder_Counter.items():
            for Category_train in n_gram_train:
                idf = 0
                tf = 0.5 + 0.5 * (n_gram_train[Category_train][k] / n_gram_train[Category_train][max_grm[Category_train]])
                if len(list_of_Cat_Cover_ngam[k]) != 0:
                    idf = 25 / len(list_of_Cat_Cover_ngam[k])

                TF_IDF[domain][Category_train] += v * tf * idf
                freq_sum[Category_train] += v * tf * idf
        if sum(freq_sum.values()) != 0:
            Result_Category = max(freq_sum.items(), key=operator.itemgetter(1))[0]

            pi = 1
            for j in range(0, len(freq_sum)):
                if max(freq_sum.items(), key=operator.itemgetter(1))[0] == Domain_Categories_set[test_domain_wless]:
                    break
                pi += 1
                freq_sum[max(freq_sum.items(), key=operator.itemgetter(1))[0]] = -1000

            Position_Error += (pi - 1) / 24
            num_test += 1
            if Domain_Categories_set[test_domain_wless] != Result_Category:
                uncorrect += 1
                Y_prediction.append(Result_Category)
                fn[Domain_Categories_set[test_domain_wless]] += 1

                fp[Result_Category] += 1
                confusion_matrix[Domain_Categories_set[test_domain_wless]][Result_Category] += 1
            else:

                Y_prediction.append(Result_Category)
                correct += 1
                tp[Domain_Categories_set[test_domain_wless]] += 1
                confusion_matrix[Domain_Categories_set[test_domain_wless]][
                    Domain_Categories_set[test_domain_wless]] += 1

        else:
            Y_prediction.append('prediction')



    Accuracy = metrics.accuracy_score(Y_test, Y_prediction)
    Micro_Precision = metrics.precision_score(Y_test, Y_prediction, average='micro')
    Micro_Recall = metrics.recall_score(Y_test, Y_prediction, average='micro')
    Micro_f1 = metrics.f1_score(Y_test, Y_prediction, average='micro')

    #print('acc_tfidf',Accuracy)
    Macro_Precision = metrics.precision_score(Y_test, Y_prediction, average='macro')
    Macro_Recall = metrics.recall_score(Y_test, Y_prediction, average='macro')
    Macro_f1 = metrics.f1_score(Y_test, Y_prediction, average='macro')

    avg_position_error = Position_Error / num_test


    Y_test_num = []
    Y_prediction_num = []
    cat_num = defaultdict(int)
    num_cat = defaultdict(int)
    i = 0
    for elem in Y_test:
        if elem not in cat_num:
            cat_num[elem] = i
            num_cat[i] = elem
            i+=1
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

    return [prec, rec, fm, sup,confusion_matrix, Accuracy, avg_position_error,Macro_Precision,Macro_Recall, Macro_f1]
