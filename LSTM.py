from __future__ import unicode_literals, print_function, division
from collections import defaultdict
import string
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score



''' In this method to assign categories to domain names a network at level of character are built and
    trained  i.e., Each domain name is considered  as a series of characters and it is given as an input
    of the LSTM (https://www.tensorflow.org/guide/keras/rnn).'''


def letterToIndex(letter):
    all_letters = string.ascii_letters + " _\-.,;'" + string.digits
    return all_letters.find(letter)

def lineToTensor(line):
    tensor = []
    for li, letter in enumerate(line):
        tensor.append(letterToIndex(letter))
    return tensor


def LSTM(validation, test, Domain_Categories_set):

    size = 0
    category_lines = defaultdict(list)
    for elem in Domain_Categories_set:
        category_lines[Domain_Categories_set[elem]].append(elem)
        size += 1


    all_categories = []
    for cat in category_lines:
        all_categories.append(cat)


    k = 0
    y_lab = defaultdict(int)
    size = 0


    test1 = []
    for elem in test:
        test1.append(elem.replace('www.',''))

    test = test1


    X_test = []
    X_train = []

    y_test = []
    y_train = []
    label_of_index = defaultdict(list)
    for elem in Domain_Categories_set:

        if elem in test:
            X_test.append(elem)
            if Domain_Categories_set[elem] not in y_lab:
                y_lab[Domain_Categories_set[elem]] = k
                label_of_index[k] = Domain_Categories_set[elem]
                k += 1
            y_test.append(y_lab[Domain_Categories_set[elem]])
        else:
            X_train.append(elem)
            if Domain_Categories_set[elem] not in y_lab:
                y_lab[Domain_Categories_set[elem]] = k
                label_of_index[k] = Domain_Categories_set[elem]
                k += 1
            y_train.append(y_lab[Domain_Categories_set[elem]])


    X1_test = []

    for elem in X_test:
        size +=1
        X1_test.append(lineToTensor(elem))

    X1_train = []

    for elem in X_train:
        size += 1
        X1_train.append(lineToTensor(elem))

    Y_train = []
    for elem in y_train:
        y1 = []
        for i in range(25):
            if elem == i:
                y1.append(1)
            else:
                y1.append(0)
        Y_train.append(y1)

    Y_test = []
    for elem in y_test:
        y1 = []
        for i in range(25):
            if elem == i:
                y1.append(1)
            else:
                y1.append(0)
        Y_test.append(y1)

    Y_test = np.array(Y_test)
    Y_train = np.array(Y_train)
    Maxlen = max([len(s) for s in X1_train])
    X1_test = pad_sequences(X1_test, Maxlen)
    X1_train = pad_sequences(X1_train, Maxlen)



    model = Sequential()
    model.add(layers.Embedding(70, 64, input_length=Maxlen))

    model.add(layers.LSTM(128))

    model.add(layers.Dense(25, activation='softmax'))

    print(model.summary())
    model.compile(
        loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=['accuracy']
    )

    history = model.fit(
        X1_train,
        Y_train,
        validation_split=0.2,
        batch_size=2098,
        epochs=65,
    )

    predicted_stock_price = model.predict(X1_test)


    confusion_matrix = defaultdict(list)
    for cat in category_lines:
        confusion_matrix[cat] = defaultdict(int)

    tp = defaultdict(int)
    fn = defaultdict(int)
    fp = defaultdict(int)


    Position_Error = 0
    correct = 0
    uncorrect = 0
    position_predicted = predicted_stock_price
    Y_true = []
    Y_prediction = []
    for i in range(len(predicted_stock_price)):
        Y_true.append[np.argmax(Y_test[i])]
        Y_prediction.append[predicted_stock_price[i]]
        if np.argmax(predicted_stock_price[i]) == np.argmax(Y_test[i]):
            tp[np.argmax(Y_test[i])] += 1
            correct += 1
            confusion_matrix[label_of_index[np.argmax(Y_test[i])]][label_of_index[np.argmax(Y_test[i])]] += 1
        else:
            fp[np.argmax(predicted_stock_price[i])] += 1
            fn[np.argmax(Y_test[i])] += 1
            confusion_matrix[label_of_index[np.argmax(Y_test[i])]][label_of_index[np.argmax(predicted_stock_price[i])]] += 1
            uncorrect += 1

        pi = 1
        for j in range(0, len(predicted_stock_price[i])):
            if np.argmax(position_predicted[i]) == np.argmax(Y_test[i]):
                break
            pi += 1
            position_predicted[i][np.argmax(position_predicted[i])] = -1000

        Position_Error += (pi - 1) / 24

    Accuracy = metrics.accuracy_score(Y_true, Y_prediction)
    Micro_Precision = metrics.precision_score(Y_true, Y_prediction, average='micro')
    Micro_Recall = metrics.recall_score(Y_true, Y_prediction, average='micro')
    Micro_f1 = metrics.f1_score(Y_true, Y_prediction, average='micro')

    Macro_Precision = metrics.precision_score(Y_true, Y_prediction, average='macro')
    Macro_Recall = metrics.recall_score(Y_true, Y_prediction, average='macro')
    Macro_f1 = metrics.f1_score(Y_true, Y_prediction, average='macro')


    avg_position_error = Position_Error/len(predicted_stock_price)

    precision1, recall1, fscore, support = score(Y_true, Y_prediction)

    prec = defaultdict(int)
    rec = defaultdict(int)
    fm = defaultdict(int)
    sup = defaultdict(int)
    for elem in range(len(precision1)):
        prec[label_of_index[elem]] = precision1[elem]
        rec[label_of_index[elem]] = recall1[elem]
        fm[label_of_index[elem]] = fscore[elem]
        sup[label_of_index[elem]] = support[elem]

    return [prec, rec, fm, sup, confusion_matrix, Accuracy, avg_position_error, Macro_Precision, Macro_Recall, Macro_f1]
