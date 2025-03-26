from collections import defaultdict
import pickle
import semi_supervised as SP
from collections import defaultdict
from scipy import sparse
import sklearn
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score

def Create_Graph(All_domain,Set_index, similarity, M, k, g, domains_vectors = None ):

    ''' Crete the graph G with the edges based on Domain name similarity (NFA) and domain sequence
        M is the code number of the method to weight Graph's edge '''

    G = sparse.lil_matrix((len(All_domain), len(All_domain)))
    set_domain = All_domain

    if M != 2: # if semisupervised is based on Domain Name

        G_domain = sparse.lil_matrix(similarity > k)
        G = G_domain


    if M != 1:# if semisupervised is based on domain sequence

        '''     In this work, a session of sequence of visited domains by a user in one hour is considered as a training text file.
                Then Word2Vec model (fastText <=>  https://github.com/facebookresearch/fastText ) is used to represent the session
                as a vector in a vector space (vector_domain.txt).  It represents each domain name with a very low-dimensional vector
                with semantic meaning. Afterward, cosine similarities of the vectors are computed to define the weitgh of the edges 
                of the graph '''

        f = open(domains_vectors, 'r+')
        lines = [line for line in f.readlines()]
        dict_ = defaultdict(list)
        for i in range(0, len(lines)):
            for j in range(1, len(lines[i].split(' ')) - 1):
                dict_[lines[i].split(' ')[0]].append(float(lines[i].split(' ')[j]))

        X = [0 for i in range(len(Set_index))]

        for elem in dict_:
            if elem in All_domain:
                X[Set_index[elem]] = dict_[elem]
        X = np.array(X)

        cosine_similarities = sklearn.metrics.pairwise.cosine_similarity(X, Y=None, dense_output=True)
        G_sequence = sparse.lil_matrix(cosine_similarities > g)
        G = G_sequence


    if M == 3:
        G = G_domain.__add__(G_sequence)

    #print('graph is created')

    return [G, set_domain ]

def Semi_Supervised_Graph(All_domain, Set_index, validation, test, similarity, Number_Domain, Domain_Categories_set, M, k,g, domains_vectors=None):

    ''' Assign categories to domains based on graph based semisupervised method. First the method
        of Create_Graph is called to create the graph, then the graph is given to LGC method of
        semi_supervised class to classify domains '''

    G, set_domain = Create_Graph(All_domain,Set_index, similarity, M, k,g, domains_vectors)





    Number_Label = defaultdict(list)
    Label_Number = defaultdict(int)
    j = 0
    for domain in All_domain:
        domain_wless = domain.replace('www.', '')
        if domain_wless in Domain_Categories_set and domain not in test and domain not in validation:
            if Domain_Categories_set[domain_wless] not in Label_Number:
                Label_Number[Domain_Categories_set[domain_wless]] = j
                Number_Label[j] = Domain_Categories_set[domain_wless]
                j += 1

    Categories_Domains_Initial = defaultdict(list)
    for domain in Domain_Categories_set:
        Categories_Domains_Initial[Domain_Categories_set[domain]].append(domain)
    uncorrect = 0
    correct = 0
    fn = defaultdict(int)
    tp = defaultdict(int)
    fp = defaultdict(int)

    unlabel_ = 0
    confusion_matrix = defaultdict(list)
    for cat in Categories_Domains_Initial:
        confusion_matrix[cat] = defaultdict(int)

    all_test_domian_in_set = 0
    train_data = []
    y_train_data = []
    Graph_data = []
    for domain in Set_index:#: # defining the labeled node in graph for training
        domain_wless = domain.replace('www.', '')
        if domain_wless in Domain_Categories_set and domain not in validation and domain not in test :
            train_data.append(Set_index[domain])
            y_train_data.append(Label_Number[Domain_Categories_set[domain_wless]])
        Graph_data.append(Set_index[domain])  # all node in graph


    ''' calling LGC method of graph based semi_supervised to classify domains '''
    lgc = SP.LGC(graph=G, alpha=0.001)

    x = np.array(train_data)
    y = np.array(y_train_data)

    ''' before fit'''
    lgc.fit(x, y)



    Predicted_label = lgc.predict_proba(np.array(Graph_data))  # calssify unlabeled node

    ''' Position_Error is a measure of the deviation of x correct label’s position from the top-rank in the ranking list'''
    Position_Error = 0
    num_test = 0

    Y_test = []
    Y_prediction = []
    for test_domain in validation:  # test evaluation
        test_domain_wless = test_domain.replace('www.', '')

        Y_test.append(Domain_Categories_set[test_domain_wless])

        if test_domain in set_domain:

            all_test_domian_in_set += 1
            sum__ = 0
            for i in range(len(Label_Number)):
                sum__ += Predicted_label[Set_index[test_domain]][i]
            if np.isnan(sum__) == False:  # for checking the test node that has been classified and its probality value is not non

                non_zero = 0
                for i in range(0, len(Label_Number)):
                    if Predicted_label[Set_index[test_domain]][i] != 0:
                        non_zero += 1

                Top_cat = defaultdict(int)
                Predicted_Cat = Number_Label[np.argmax(Predicted_label[Set_index[test_domain]])]

                pi = 1
                for j in range(0, min(non_zero, len(Predicted_Cat))):
                    Top_cat[Number_Label[
                        np.argmax(Predicted_label[Set_index[test_domain]])]] = 1
                    if Number_Label[np.argmax(Predicted_label[Set_index[test_domain]])] == Domain_Categories_set[
                        test_domain_wless]:
                        break
                    pi += 1
                    Predicted_label[Set_index[test_domain]][np.argmax(Predicted_label[Set_index[test_domain]])] = 0

                Position_Error += (pi - 1) / 24
                num_test += 1
                Y_prediction.append(Predicted_Cat)
                if Domain_Categories_set[test_domain_wless] != Predicted_Cat:

                    uncorrect += 1
                    fn[Domain_Categories_set[test_domain_wless]] += 1
                    fp[Number_Label[np.argmax(Predicted_label[Set_index[test_domain]])]] += 1
                    confusion_matrix[Domain_Categories_set[test_domain_wless]][Predicted_Cat] += 1

                else:
                    correct += 1
                    tp[Domain_Categories_set[test_domain_wless]] += 1
                    confusion_matrix[Domain_Categories_set[test_domain_wless]][
                        Domain_Categories_set[test_domain_wless]] += 1
            else:
                correct += 1
                #Y_prediction.append(Domain_Categories_set[test_domain_wless])
                Y_prediction.append('Predicted_Cat')
        else:
            correct += 1
            #Y_prediction.append(Domain_Categories_set[test_domain_wless])
            Y_prediction.append('Predicted_Cat')

    avg_position_error = Position_Error / num_test
    Accuracy = correct / len(test)

    Accuracy = metrics.accuracy_score(Y_test, Y_prediction)


    Micro_Precision = metrics.precision_score(Y_test, Y_prediction, average='micro')
    Micro_Recall = metrics.recall_score(Y_test, Y_prediction, average='micro')
    Micro_f1 = metrics.f1_score(Y_test, Y_prediction, average='micro')


    Macro_Precision = metrics.precision_score(Y_test, Y_prediction, average='macro')
    Macro_Recall = metrics.recall_score(Y_test, Y_prediction, average='macro')
    Macro_f1 = metrics.f1_score(Y_test, Y_prediction, average='macro')


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

    return [prec, rec, fm, sup, confusion_matrix, Accuracy, avg_position_error, Macro_Precision, Macro_Recall, Macro_f1]


