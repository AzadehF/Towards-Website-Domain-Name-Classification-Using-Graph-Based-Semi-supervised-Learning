from collections import defaultdict
from sklearn.model_selection import ShuffleSplit
import SVM
import TFIDF
import NFA
import Semi_Supervised_Graph as SSG
import LSTM
import k_fold_Output as KFO
import sys
from sklearn.model_selection import train_test_split
import numpy as np


# dataset containing users' web sessions
input_dataset = sys.argv[1]
# the file 'vector_domain.txt' contains vectors of domains
vector_dataset =  sys.argv[2]
# the domain-categories file is a dictionary where the domain is the key in the form 'example.com'
# the value is a category
# E.g., {'example.com': 'Category1', 'exampletwo.net': 'Category2'}
# similarweb_domain_categories = sys.argv[3]


def main():

    # reading file
    f = open(input_dataset, 'r+')
    lines = [line for line in f.readlines()]
    UserHourBag = defaultdict(list)
    LabelDomain = defaultdict(list)

    for i in range(0, len(lines)):
        if lines[i].split(' ')[0] not in UserHourBag:
            UserHourBag[lines[i].split(' ')[0]] = defaultdict(list)

        ListDomain = lines[i].split(' ')[2].replace('[','').replace(']','').split(',')
        ListLabel = lines[i].split(' ')[3].replace('[', '').replace(']', '').split(',')

        # extracting domain categories
        # if you are using the similarweb_domain_categories, comment the following for-loop
        for j in range(len(ListDomain)):
            UserHourBag[lines[i].split(' ')[0]][int(lines[i].split(' ')[1])].append(ListDomain[j])
            if ListLabel[j] != '':
                LabelDomain[ListDomain[j].replace('www.','')] = ListLabel[j]

    Domain_Categories_set = LabelDomain

    # If you are using the file similarweb_domain_categories, uncomment this line
    # Domain_Categories_set = pickle.load(open(similarweb_domain_categories, "rb"))

    i = 0
    j = 0
    train_set = []
    Set_index = defaultdict(int)
    Index_set = defaultdict(list)
    All_set_index = defaultdict(list)
    All_domain = defaultdict(int)
    cat_num = defaultdict(int)
    train_data = defaultdict(list)

    count_user_domain = defaultdict(int)

    X = []
    y = []
    for user in UserHourBag:
        for hour in range(0, 24):
            for user_domain in UserHourBag[user][hour]:
                count_user_domain[user_domain] += 1
    for user in UserHourBag:
        for hour in range(0, 24):
            for user_domain in UserHourBag[user][hour]:
                user_domain_wless = user_domain.replace('www.', '')
                if user_domain_wless in Domain_Categories_set and user_domain not in train_data:
                    train_data[user_domain] = Domain_Categories_set[user_domain_wless]
                    cat_num[Domain_Categories_set[user_domain_wless]] += 1
                    train_set.append(user_domain)
                    X.append(user_domain)
                    y.append(Domain_Categories_set[user_domain_wless])
                    Set_index[user_domain] = i
                    Index_set[i] = user_domain
                    i += 1
                if user_domain not in All_domain:
                    All_domain[user_domain] = 1
                    All_set_index[user_domain] = j
                    j += 1

     # k fold cross validation
    X_train, X_test , y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state=0 )

    k = 10
    ss = ShuffleSplit(n_splits=k, test_size=0.2, random_state=0)
    train_list_index = defaultdict(list)
    validation_list_index = defaultdict(list)
    round = 0

     #define train and validation list in 10 fold cross validation
    for train_index, validation_index in ss.split(X_train):
        train_list_index[round] = train_index
        validation_list_index[round] = validation_index
        round += 1

    Best_Precision = defaultdict(list)
    Best_Recall = defaultdict(list)
    Best_FM = defaultdict(list)
    Best_Confusion_matrix = defaultdict(list)
    Best_Accuracy = defaultdict(int)
    Best_Position_Error = defaultdict(int)
    Best_Macro_Precision = defaultdict(int)
    Best_Macro_Recall = defaultdict(int)
    Best_Macro_f1 = defaultdict(list)
    Best_support = defaultdict(list)

    for i in range(0, round):
        Best_Precision[i] = defaultdict(int)
        Best_Recall[i] = defaultdict(int)
        Best_FM[i] = defaultdict(int)
        Best_Confusion_matrix[i] = defaultdict(int)
        Best_Accuracy[i] = defaultdict(int)
        Best_Position_Error[i] = defaultdict(int)
        Best_Macro_Precision[i] = defaultdict(int)
        Best_Macro_Recall[i] = defaultdict(int)
        Best_Macro_f1[i] = defaultdict(int)
        Best_support[i] = defaultdict(int)


    Precision = [0 for i in range(round)]
    Recall = [0 for i in range(round)]
    FM = [0 for i in range(round)]
    Confusion_matrix = [0 for i in range(round)]
    Accuracy = [0 for i in range(round)]
    Position_Error = [0 for i in range(round)]
    Macro_Precision = [0 for i in range(round)]
    Macro_Recall = [0 for i in range(round)]
    Macro_f1 = [0 for i in range(round)]
    support = [0 for i in range(round)]

    n_grams = np.arange(4,10,1) # parameter range for defining maximum threshold of n-gram in TFIDF, NFA, SSDN, SSB
    threshold_DN = np.arange(0.9,0.99,0.005) # parameter range for defining epsilon of SSDN and SSB
    threshold_DS = np.arange(0.4,0.8,0.01) # parameter range for defining epsilon of SSDS and SSB


    # supervised method based on domain name with Term Frequency–Inverse Document Frequency (TFIDF)
    max = 0
    for gram in n_grams:
        avg = []
        for i in range(0, round):
            validation_list = defaultdict(int)
            for elem in validation_list_index[i]:
                validation_list[Index_set[elem]] = 1
            Precision[i], Recall[i], FM[i], support[i], Confusion_matrix[i], Accuracy[i], Position_Error[i], Macro_Precision[i], Macro_Recall[i], Macro_f1[i]= TFIDF.TFIDF(validation_list,X_test, Domain_Categories_set, 9)
            avg.append(Accuracy[i])

        if np.mean(avg) > max:
            max = np.mean(avg)
            optimal_ngram_tfidif = gram
            for i in range(0,round):
                Best_Precision[i]['TFIDF'] = Precision[i]
                Best_Recall[i]['TFIDF'] = Recall[i]
                Best_FM[i]['TFIDF'] = FM[i]
                Best_Confusion_matrix[i]['TFIDF'] = Confusion_matrix[i]
                Best_Accuracy[i]['TFIDF'] = Accuracy[i]
                Best_Position_Error[i]['TFIDF'] = Position_Error[i]
                Best_Macro_Precision[i]['TFIDF'] = Macro_Precision[i]
                Best_Macro_Recall[i]['TFIDF'] = Macro_Recall[i]
                Best_Macro_f1[i]['TFIDF'] = Macro_f1[i]
                Best_support[i]['TFIDF'] = support[i]



    # supervised method based on domain name with NFA (Number of False Alarm)
    max = 0
    for gram in n_grams:
        avg = []
        for i in range(0, round):
            validation_list = defaultdict(int)
            for elem in validation_list_index[i]:
                validation_list[Index_set[elem]] = 1
            Precision[i], Recall[i], FM[i], support[i], Confusion_matrix[i], Accuracy[i], Position_Error[i], Macro_Precision[i], \
            Macro_Recall[i], Macro_f1[i] = NFA.NFA_Supervised(validation_list, X_test ,Domain_Categories_set, 5)
            avg.append(Accuracy[i])

        if np.mean(avg) > max:
            max = np.mean(avg)
            optimal_ngram_NFA = gram
            for i in range(0, round):
                Best_Precision[i]['NFA'] = Precision[i]
                Best_Recall[i]['NFA'] = Recall[i]
                Best_FM[i]['NFA'] = FM[i]
                Best_Confusion_matrix[i]['NFA'] = Confusion_matrix[i]
                Best_Accuracy[i]['NFA'] = Accuracy[i]
                Best_Position_Error[i]['NFA'] = Position_Error[i]
                Best_Macro_Precision[i]['NFA'] = Macro_Precision[i]
                Best_Macro_Recall[i]['NFA'] = Macro_Recall[i]
                Best_Macro_f1[i]['NFA'] = Macro_f1[i]
                Best_support[i]['NFA'] = support[i]



    #supervised method based on Domain Sequence with SVM (Support Vector Machine)
    max = 0
    if 1:
        avg = []
        for i in range(0, round):
            validation_list = defaultdict(int)
            for elem in validation_list_index[i]:
                validation_list[Index_set[elem]] = 1
            Precision[i], Recall[i], FM[i], support[i], Confusion_matrix[i], Accuracy[i], Position_Error[i], Macro_Precision[i], \
            Macro_Recall[i], Macro_f1[i] = SVM.SVM(validation_list, X_test, Domain_Categories_set, All_domain, vector_dataset)
            avg.append(Accuracy[i])
        if np.mean(avg) > max:
            max = np.mean(avg)
            for i in range(0, round):
                Best_Precision[i]['SVM'] = Precision[i]
                Best_Recall[i]['SVM'] = Recall[i]
                Best_FM[i]['SVM'] = FM[i]
                Best_Confusion_matrix[i]['SVM'] = Confusion_matrix[i]
                Best_Accuracy[i]['SVM'] = Accuracy[i]
                Best_Position_Error[i]['SVM'] = Position_Error[i]
                Best_Macro_Precision[i]['SVM'] = Macro_Precision[i]
                Best_Macro_Recall[i]['SVM'] = Macro_Recall[i]
                Best_Macro_f1[i]['SVM'] = Macro_f1[i]
                Best_support[i]['SVM'] = support[i]



    # Semi Supervised method based on Domain Name (SSDN)
    max = 0
    for gram in n_grams:
        for h in threshold_DN:
            avg = []
            for i in range(0, round):
                validation_list = defaultdict(int)
                for elem in validation_list_index[i]:
                    validation_list[Index_set[elem]] = 1
                similarity, Number_Domain, Domain_Number = NFA.Similarity_based_Domain(All_domain, validation_list,
                                                                                       X_test,
                                                                                       Domain_Categories_set, gram)

                Precision[i], Recall[i], FM[i], support[i], Confusion_matrix[i], Accuracy[i], Position_Error[i], \
                Macro_Precision[i], \
                Macro_Recall[i], Macro_f1[i] = SSG.Semi_Supervised_Graph(
                    All_domain, All_set_index, validation_list, X_test, similarity, Number_Domain, Domain_Categories_set,
                    1, h, 0, domains_vectors=vector_dataset)
                avg.append(Accuracy[i])

            if np.mean(avg) > max:
                max = np.mean(avg)
                optimal_gram_SSDN = gram
                optimal_h_SSDN = h

                for i in range(0, round):
                    Best_Precision[i]['SSDN'] = Precision[i]
                    Best_Recall[i]['SSDN'] = Recall[i]
                    Best_FM[i]['SSDN'] = FM[i]
                    Best_Confusion_matrix[i]['SSDN'] = Confusion_matrix[i]
                    Best_Accuracy[i]['SSDN'] = Accuracy[i]
                    Best_Position_Error[i]['SSDN'] = Position_Error[i]
                    Best_Macro_Precision[i]['SSDN'] = Macro_Precision[i]
                    Best_Macro_Recall[i]['SSDN'] = Macro_Recall[i]
                    Best_Macro_f1[i]['SSDN'] = Macro_f1[i]
                    Best_support[i]['SSDN'] = support[i]



    # Semi Supervised method based on Domain Sequence (SSDS)
    max = 0
    for g in threshold_DS:
        avg = []
        for i in range(0, round):
            validation_list = defaultdict(int)
            for elem in validation_list_index[i]:
                validation_list[Index_set[elem]] = 1

            similarity = []
            Number_Domain = []

            Precision[i], Recall[i], FM[i], support[i], Confusion_matrix[i], Accuracy[i], Position_Error[i], \
            Macro_Precision[i], \
            Macro_Recall[i], Macro_f1[i] = SSG.Semi_Supervised_Graph(
                All_domain, All_set_index, validation_list, X_test, similarity, Number_Domain, Domain_Categories_set,
                2, 0, g, domains_vectors=vector_dataset)
            avg.append(Accuracy[i])

        if np.mean(avg) > max:
            max = np.mean(avg)
            optimal_g_SSDS = g
            for i in range(0, round):
                Best_Precision[i]['SSDS'] = Precision[i]
                Best_Recall[i]['SSDS'] = Recall[i]
                Best_FM[i]['SSDS'] = FM[i]
                Best_Confusion_matrix[i]['SSDS'] = Confusion_matrix[i]
                Best_Accuracy[i]['SSDS'] = Accuracy[i]
                Best_Position_Error[i]['SSDS'] = Position_Error[i]
                Best_Macro_Precision[i]['SSDS'] = Macro_Precision[i]
                Best_Macro_Recall[i]['SSDS'] = Macro_Recall[i]
                Best_Macro_f1[i]['SSDS'] = Macro_f1[i]
                Best_support[i]['SSDS'] = support[i]



    # Semi Supervised method based on Both domain name and domain sequence (SSB)
    max = 0
    for gram in n_grams:
        for h in threshold_DN:
            for g in threshold_DS:
                avg = []
                for i in range(0, round):
                    validation_list = defaultdict(int)
                    for elem in validation_list_index[i]:
                        validation_list[Index_set[elem]] = 1
                    similarity, Number_Domain, Domain_Number = NFA.Similarity_based_Domain(All_domain, validation_list,
                                                                                           X_test,
                                                                                           Domain_Categories_set, gram)

                    Precision[i], Recall[i], FM[i], support[i], Confusion_matrix[i], Accuracy[i], Position_Error[i], \
                    Macro_Precision[i], \
                    Macro_Recall[i], Macro_f1[i] = SSG.Semi_Supervised_Graph(
                        All_domain, All_set_index, validation_list, X_test, similarity, Number_Domain, Domain_Categories_set,
                        3, h, g, domains_vectors=vector_dataset)
                    avg.append(Accuracy[i])

                if np.mean(avg) > max:
                    max = np.mean(avg)
                    optimal_gram_SSB = gram
                    optimal_h_SSB = h
                    optimal_g_SSB = g
                    for i in range(0, round):
                        Best_Precision[i]['SSB'] = Precision[i]
                        Best_Recall[i]['SSB'] = Recall[i]
                        Best_FM[i]['SSB'] = FM[i]
                        Best_Confusion_matrix[i]['SSB'] = Confusion_matrix[i]
                        Best_Accuracy[i]['SSB'] = Accuracy[i]
                        Best_Position_Error[i]['SSB'] = Position_Error[i]
                        Best_Macro_Precision[i]['SSB'] = Macro_Precision[i]
                        Best_Macro_Recall[i]['SSB'] = Macro_Recall[i]
                        Best_Macro_f1[i]['SSB'] = Macro_f1[i]
                        Best_support[i]['SSB'] = support[i]



    for i in range(0, round):
        validation_list = defaultdict(int)
        for elem in validation_list_index[i]:
            validation_list[Index_set[elem]] = 1
        Precision[i]['LSTM'], Recall[i]['LSTM'], FM[i]['LSTM'], support[i]['LSTM'], Confusion_matrix[i]['LSTM'], Accuracy[i]['LSTM'], Position_Error[i]['LSTM'], \
        Macro_Precision[i]['LSTM'],Macro_Recall[i]['LSTM'], Macro_f1[i]['LSTM'] = LSTM.LSTM(validation_list, X_test, Domain_Categories_set)


    KFO.k_fold_Output(Best_Precision, Best_Recall, Best_FM, Best_support, Best_Confusion_matrix, Best_Accuracy,
                      Best_Position_Error,
                      Best_Macro_Precision, Best_Macro_Recall, Best_Macro_f1, k)





    # results of the best method on test data

    similarity, Number_Domain, Domain_Number = NFA.Similarity_based_Domain(All_domain, X_test, X_test,
                                                                           Domain_Categories_set, optimal_gram_SSB)
    test_Precison, test_Recall, test_FM, support, test_Confusion_matrix, test_Accuracy\
        , test_Position_Error, test_Macro_Precision, test_Macro_Recall, test_Macro_f1 = SSG.Semi_Supervised_Graph(
        All_domain, All_set_index, X_test, X_test, similarity, Number_Domain, Domain_Categories_set, 3,
        optimal_h_SSB,
        optimal_g_SSB, domains_vectors=vector_dataset)


    with open('test_result.txt','w') as file_out:
        file_out.write('test_Accuracy' + ' ' + 'test_Macro_Precision' + ' ' +
            'test_Macro_Recall' + ' ' + 'test_Macro_f1' + ' ' + 'test_Position_Error'+'\n')
        file_out.write(str(test_Accuracy) + ' ' + str(test_Macro_Precision) + ' ' + str(
                test_Macro_Recall) + ' ' + str(test_Macro_f1) + ' ' + str(test_Position_Error))




start = main()
