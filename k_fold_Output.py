from collections import defaultdict

def k_fold_Output(precision, Recall, FM,support, confusion_matrix, Accuracy, Position_Error, Macro_Precision, Macro_Recall, Macro_f1, k):
    '''saving the results of k-fold cross validation'''

    Sum_Precision = defaultdict(int)
    Sum_Recall = defaultdict(int)
    Sum_FM = defaultdict(int)



    count_precision = defaultdict(int)


    count = defaultdict(list)
    for elem in support:
        for method in support[elem]:
            if method not in count:
                count[method] = defaultdict(int)
            for cat in support[elem][method]:
                if support[elem][method][cat] != 0:
                    print(support[elem][method][cat])
                    count[method][cat] += 1

    for elem in precision:

        for method in precision[elem]:
            for cat in precision[elem][method]:

                if cat not in Sum_Precision:
                    Sum_Precision[cat] = defaultdict(int)
                    count_precision[cat] = defaultdict(int)
                Sum_Precision[cat][method] += precision[elem][method][cat]
                count_precision[cat][method] += 1



    count_Recall = defaultdict(int)
    for elem in Recall:
        for method in Recall[elem]:
            for cat in Recall[elem][method]:
                if cat not in Sum_Recall:
                    Sum_Recall[cat] = defaultdict(int)
                    count_Recall[cat] = defaultdict(int)
                Sum_Recall[cat][method] += Recall[elem][method][cat]
                count_Recall[cat][method] += 1

    Sum_Accuracy = defaultdict(int)
    for elem in Accuracy:
        for method in Accuracy[elem]:
            Sum_Accuracy[method] += Accuracy[elem][method]

    Sum_Position_Error = defaultdict(int)
    for elem in Position_Error:
        for method in Position_Error[elem]:
            Sum_Position_Error[method] += Position_Error[elem][method]

    count_FM = defaultdict(int)
    for elem in FM:
        for method in FM[elem]:
            for cat in FM[elem][method]:
                if cat not in Sum_FM:
                    Sum_FM[cat] = defaultdict(int)
                    count_FM[cat] = defaultdict(int)
                Sum_FM[cat][method] += FM[elem][method][cat]
                count_FM[cat][method] += 1




    with open('Precision.txt', 'w') as file_out:
        file_out.write(
            'Category' + ' ' + 'TFIDF' + ' ' + 'NFA' + ' ' + 'SVM' + ' ' + 'SSDN' + ' ' + 'SSDS' + ' ' + 'SSB' + '\n')

        for cat in count_precision:
            file_out.write(cat)
            for method in count_precision[cat]:
                if count_precision[cat][method] != 0:
                    file_out.write(' ' + str(Sum_Precision[cat][method] / count[method][cat]))
                else:
                    file_out.write(' ' + ' ')
            file_out.write('\n')


    with open('Recall.txt', 'w') as file_out:
        file_out.write(
            'Category' + ' ' + 'TFIDF' + ' ' + 'NFA' + ' ' + 'SVM' + ' ' + 'SSDN' + ' ' + 'SSDS' + ' ' + 'SSB' + '\n')

        for cat in count_Recall:
            file_out.write(cat)
            for method in count_Recall[cat]:
                if count_Recall[cat][method] != 0:
                    file_out.write(' ' + str(Sum_Recall[cat][method] / count[method][cat]))
                else:
                    file_out.write(' ' + ' ')
            file_out.write('\n')


    with open('FM.txt', 'w') as file_out:
        file_out.write(
            'Category' + ' ' + 'TFIDF' + ' ' + 'NFA' + ' ' + 'SVM' + ' ' + 'SSDN' + ' ' + 'SSDS' + ' ' + 'SSB' +  '\n')

        for cat in count_FM:
            file_out.write(cat)
            for method in count_FM[cat]:
                if count_FM[cat][method] != 0:
                    file_out.write(' ' + str(Sum_FM[cat][method] / count[method][cat]))
                else:
                    file_out.write(' ' + ' ')
            file_out.write('\n')


    with open('Accuracy.txt', 'w') as file_out:
        file_out.write(
            'round' + ' ' + 'TFIDF' + ' ' + 'NFA' + ' ' + 'SVM' + ' ' + 'SSDN' + ' ' + 'SSDS' + ' ' + 'SSB' +  '\n')
        for elem in Accuracy:
            file_out.write(str(elem))
            for method in Accuracy[elem]:
                file_out.write(' ' + str(Accuracy[elem][method]))
            file_out.write('\n')

    with open('count.txt', 'w') as file_out:
        file_out.write(
            'round' + ' ' + 'TFIDF' + ' ' + 'NFA' + ' ' + 'SVM' + ' ' + 'SSDN' + ' ' + 'SSDS' + ' ' + 'SSB' +  '\n')
        for method in count:
            file_out.write(str(elem))
            for cat in count[method]:
                file_out.write(cat+' ' + str(count[method][cat]))
                file_out.write('\n')

    with open('Macro_Precison.txt', 'w') as file_out:
        file_out.write(
            'round' + ' ' + 'TFIDF' + ' ' + 'NFA' + ' ' + 'SVM' + ' ' + 'SSDN' + ' ' + 'SSDS' + ' ' + 'SSB' + '\n')
        for elem in Macro_Precision:
            file_out.write(str(elem))
            for method in Macro_Precision[elem]:
                file_out.write(' ' + str(Macro_Precision[elem][method]))
            file_out.write('\n')

    with open('Macro_Recall.txt', 'w') as file_out:
        file_out.write(
            'round' + ' ' + 'TFIDF' + ' ' + 'NFA' + ' ' + 'SVM' + ' ' + 'SSDN' + ' ' + 'SSDS' + ' ' + 'SSB'+ '\n')
        for elem in Macro_Recall:
            file_out.write(str(elem))
            for method in Macro_Recall[elem]:
                file_out.write(' ' + str(Macro_Recall[elem][method]))
            file_out.write('\n')

    with open('Macro_f1.txt', 'w') as file_out:
        file_out.write(
            'round' + ' ' + 'TFIDF' + ' ' + 'NFA' + ' ' + 'SVM' + ' ' + 'SSDN' + ' ' + 'SSDS' + ' ' + 'SSB' + '\n')
        for elem in Macro_f1:
            file_out.write(str(elem))
            for method in Macro_f1[elem]:
                file_out.write(' ' + str(Macro_f1[elem][method]))
            file_out.write('\n')


    with open('Position_Error.txt', 'w') as file_out:
        for method in Sum_Position_Error:
            file_out.write(method + ' ' + str(Sum_Position_Error[method] / k) + '\n')



    for elem in confusion_matrix:
        with open('Confusion_TFIDF'+str(elem)+'.txt', 'w') as file_out:
            for cat1 in confusion_matrix[elem]['TFIDF']:
                file_out.write('*********\n')
                file_out.write(cat1 + '\n')
                for cat2 in confusion_matrix[elem]['TFIDF'][cat1]:
                    file_out.write(cat2 + ' ' + str(confusion_matrix[elem]['TFIDF'][cat1][cat2]) + '\n')

    for elem in confusion_matrix:
        with open('Confusion_NFA' + str(elem) + '.txt', 'w') as file_out:
            for cat1 in confusion_matrix[elem]['NFA']:
                file_out.write('*********\n')
                file_out.write(cat1 + '\n')
                for cat2 in confusion_matrix[elem]['NFA'][cat1]:
                    file_out.write(cat2 + ' ' + str(confusion_matrix[elem]['NFA'][cat1][cat2]) + '\n')

    for elem in confusion_matrix:
        with open('Confusion_SVM' + str(elem) + '.txt', 'w') as file_out:
            for cat1 in confusion_matrix[elem]['SVM']:

                file_out.write('*********\n')
                file_out.write(cat1 + '\n')
                for cat2 in confusion_matrix[elem]['SVM'][cat1]:
                    file_out.write(cat2 + ' ' + str(confusion_matrix[elem]['SVM'][cat1][cat2]) + '\n')

    for elem in confusion_matrix:
        with open('Confusion_SSDN' + str(elem) + '.txt', 'w') as file_out:
            for cat1 in confusion_matrix[elem]['SSDN']:

                file_out.write('*********\n')
                file_out.write(cat1 + '\n')
                for cat2 in confusion_matrix[elem]['SSDN'][cat1]:
                    file_out.write(cat2 + ' ' + str(confusion_matrix[elem]['SSDN'][cat1][cat2]) + '\n')

    for elem in confusion_matrix:
        with open('Confusion_SSDS' + str(elem) + '.txt', 'w') as file_out:
            for cat1 in confusion_matrix[elem]['SSDS']:

                file_out.write('*********\n')
                file_out.write(cat1 + '\n')
                for cat2 in confusion_matrix[elem]['SSDS'][cat1]:
                    file_out.write(cat2 + ' ' + str(confusion_matrix[elem]['SSDS'][cat1][cat2]) + '\n')

    for elem in confusion_matrix:
        with open('Confusion_SSB' + str(elem) + '.txt', 'w') as file_out:
            for cat1 in confusion_matrix[elem]['SSB']:

                file_out.write('*********\n')
                file_out.write(cat1 + '\n')
                for cat2 in confusion_matrix[elem]['SSB'][cat1]:
                    file_out.write(cat2 + ' ' + str(confusion_matrix[elem]['SSB'][cat1][cat2]) + '\n')

    for elem in confusion_matrix:
        with open('Confusion_LSTM' + str(elem) + '.txt', 'w') as file_out:
            for cat1 in confusion_matrix[elem]['LSTM']:
                file_out.write('*********\n')
                file_out.write(cat1 + '\n')
                for cat2 in confusion_matrix[elem]['LSTM'][cat1]:
                    file_out.write(cat2 + ' ' + str(confusion_matrix[elem]['LSTM'][cat1][cat2]) + '\n')


