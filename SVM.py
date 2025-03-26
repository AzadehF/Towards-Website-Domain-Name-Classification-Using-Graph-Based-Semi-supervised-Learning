import numpy as np
from sklearn import svm
from collections import defaultdict
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score

def SVM(validation, test, Domain_Categories_set,All_domain, domains_vectors = None):


    '''     In this work, a session of sequence of visited domains by a user in one hour is considered as a training text file.
            Then Word2Vec model (fastText <=>  https://github.com/facebookresearch/fastText ) is used to represent the session
            as a vector in a vector space (domains_vectors).  It represents each domain name with a very low-dimensional vector
            with semantic meaning. Afterward, Supervised classification method i.e SVM method (Support Vector Machine) is used on
            the embedding space to assign categories to domains. '''

    
    f = open(domains_vectors, 'r+')

    lines = [line for line in f.readlines()]
    dict_ = defaultdict(list)

    for i in range(0, len(lines)):
        for j in range(1, len(lines[i].split(' ')) - 1):
            dict_[lines[i].split(' ')[0]].append(float(lines[i].split(' ')[j]))

    X = []
    Y = []
    X_test = []
    Y_test = []
    str_to_int = defaultdict(int)
    j = 0
    int_to_str = defaultdict(list)
    X_all = []
    Y_all = []
    Y_new = []
    k = 0
    test_indices = []
    class_tess = defaultdict(int)
    for elem in dict_:
        if elem in All_domain:
            elem_wless = elem.replace('www.','')

            if elem_wless in Domain_Categories_set:
                if elem not in validation and elem not in test:
                    X.append(dict_[elem])
                    if Domain_Categories_set[elem_wless] not in str_to_int:
                        str_to_int[Domain_Categories_set[elem_wless]] = j
                        int_to_str[j] = Domain_Categories_set[elem_wless]
                        j += 1
                    Y.append(str_to_int[Domain_Categories_set[elem_wless]])

                if elem in validation:
                    test_indices.append(k)
                    X_test.append(dict_[elem])
                    elem_wless = elem.replace('www.', '')

                    if Domain_Categories_set[elem_wless] not in class_tess:
                        class_tess[Domain_Categories_set[elem_wless]] = 1

                    if Domain_Categories_set[elem_wless] not in str_to_int:
                        str_to_int[Domain_Categories_set[elem_wless]] = j
                        int_to_str[j] = Domain_Categories_set[elem_wless]
                        j += 1

                    Y_test.append(str_to_int[Domain_Categories_set[elem_wless]])

    correct = 0
    uncorrect = 0
    X = np.array(X)
    Y = np.array(Y)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)



    svm_model = svm.SVC(kernel='linear').fit(X, Y)
    svm_decision_function= svm_model.decision_function(X_test)
    svm_predictions = svm_model.predict(X_test)

    Accuracy = metrics.accuracy_score(Y_test, svm_predictions)
    Micro_Precision = metrics.precision_score(Y_test, svm_predictions, average='micro')
    Micro_Recall = metrics.recall_score(Y_test, svm_predictions, average='micro')
    Micro_f1 = metrics.f1_score(Y_test, svm_predictions, average='micro')

    Macro_Precision = metrics.precision_score(Y_test, svm_predictions, average='macro')
    Macro_Recall = metrics.recall_score(Y_test, svm_predictions, average='macro')
    Macro_f1 = metrics.f1_score(Y_test, svm_predictions, average='macro')



    #print('acc_SVM: ',Accuracy)
    confusion_matrix = defaultdict(list)
    for cat in str_to_int:
        confusion_matrix[cat] = defaultdict(int)

    tp = defaultdict(int)
    fn = defaultdict(int)
    fp = defaultdict(int)

    Position_Error = 0
    for i in range(0, len(Y_test)):
        if Y_test[i] == svm_predictions[i]:
            tp[Y_test[i]] += 1
            correct += 1
            confusion_matrix[int_to_str[Y_test[i]]][int_to_str[Y_test[i]]] += 1
        else:
            fp[svm_predictions[i]] += 1
            fn[Y_test[i]] += 1
            confusion_matrix[int_to_str[Y_test[i]]][int_to_str[svm_predictions[i]]] += 1
            uncorrect += 1
        pi = 1
        for j in range(len(svm_decision_function[i])):
            if np.argmax(svm_decision_function[i]) == Y_test[i]:
                break

            pi += 1
            svm_decision_function[i][np.argmax(svm_decision_function[i])] = -1000

        Position_Error += (pi - 1) / 24

    avg_position_error = Position_Error / len(Y_test)

    precision1, recall1, fscore, support = score(Y_test, svm_predictions)

    prec = defaultdict(int)
    rec = defaultdict(int)
    fm = defaultdict(int)
    sup = defaultdict(int)
    for elem in range(len(precision1)):
        prec[int_to_str[elem]] = precision1[elem]
        rec[int_to_str[elem]] = recall1[elem]
        fm[int_to_str[elem]] = fscore[elem]
        sup[int_to_str[elem]] = support[elem]


    return [prec, rec, fm, sup, confusion_matrix, Accuracy, avg_position_error, Macro_Precision, Macro_Recall, Macro_f1]
