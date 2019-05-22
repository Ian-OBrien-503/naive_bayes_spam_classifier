import numpy as np
import statistics as stat
from math import e, log, sqrt, pi


# this function gets all of the data into a data matrix and then splits the data into
# a test set and a training set of approximately equal size and a ratio of 40% spam to 60% not-spam
def load_data():
    # load all data into a data matrix
    data_matrix = np.loadtxt(open('spambase.data', 'rb'), delimiter=",")
    # check size of data matrix to make
    train_matrix = np.empty((2301, 58))
    test_matrix = np.empty((2300, 58))
    i = 0
    j = 0
    # split data into two test and training sets by alternating rows of the original data set 50/50 split
    for row in range(0, 4600):
            if row % 2 == 0:
                train_matrix[i] = data_matrix[row]
                i = i + 1
            else:
                test_matrix[j] = data_matrix[row]
                j = j + 1
    check_data(test_matrix, train_matrix)


# this func checks the data sets for data integrity to make sure there is approximately a 40/60 ratio in each
def check_data(test_matrix, train_matrix):
    # check ratio for test set
    spam = 0
    not_spam = 0
    for i in range(0, 2300):
        if test_matrix[i, 57] == 0:
            not_spam = not_spam + 1
        else:
            spam = spam + 1
    test_prior_spam = spam / 2300
    test_prior_not_spam = not_spam / 2300
    print("spam prior test set =", test_prior_spam)
    print("NOT spam prior test set =", test_prior_not_spam)
    # check ratio for training set
    spam = 0
    not_spam = 0
    for i in range(0, 2301):
        if train_matrix[i, 57] == 0:
            not_spam = not_spam + 1
        else:
            spam = spam + 1
    train_prior_spam = spam / 2301
    train_prior_not_spam = not_spam / 2301
    print("spam prior training set =", train_prior_spam)
    print("NOT spam prior training set =", train_prior_not_spam)
    mean_and_std_dev(test_prior_spam, test_prior_not_spam, train_matrix, test_matrix, spam, not_spam)


# compute the mean and standard deviation for each feature in the training set given each class
def mean_and_std_dev(test_prior_spam, test_prior_not_spam, train_matrix, test_matrix, spam, not_spam):
    train_spam = np.empty((spam, 58))
    train_not_spam = np.empty((not_spam, 58))
    i = 0
    j = 0
    spam_m = []
    spam_s = []
    notSpam_m = []
    notSpam_s = []

    # break training array into two subarrays that are catagorized by class
    for row in range(0, 2300):
        if train_matrix[row][57] == 0:
            train_not_spam[i] = train_matrix[row]
            i = i + 1
        else:
            train_spam[j] = train_matrix[row]
            j = j+1
    # calculate stdev, mean for both classes
    for i in range(0, 57):
        spam_mean_feature_x = stat.mean(train_spam[:, i])
        spam_m.append(spam_mean_feature_x)
        spam_std_feature_x = stat.stdev(train_spam[:, i])
        spam_s.append(spam_std_feature_x)
        not_spam_mean_feature_x = stat.mean(train_not_spam[:, i])
        notSpam_m.append(not_spam_mean_feature_x)
        not_spam_std_feature_x = stat.stdev(train_not_spam[:, i])
        notSpam_s.append(not_spam_std_feature_x)
        if spam_m[i] == 0.0:
            spam_m[i] = 0.0001
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if spam_s[i] == 0.0:
            spam_s[i] = 0.0001
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if notSpam_m[i] == 0.0:
            notSpam_m[i] = 0.0001
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if notSpam_s == 0.0:
            notSpam_s = 0.0001
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    naive_bayes(test_matrix, notSpam_s, notSpam_m, spam_s, spam_m, test_prior_spam, test_prior_not_spam)


# this is where we do the niave bayes classification
def naive_bayes(test_matrix, notSpam_s, notSpam_m, spam_s, spam_m, test_prior_spam, test_prior_not_spam):
    cond_prob_notspam = 0
    cond_prob_spam = 0
    conf_predict = []
    conf_true = test_matrix[:, 57]


    # calculate cond probability of feature given class spam
    q = 0
    r = 0
    for i in range(0, 2300):
        cond_prob_notspam = log(test_prior_not_spam)
        cond_prob_spam = log(test_prior_spam)
        for j in range(0, 56):
            x = pow(e, -1 * (((test_matrix[i][j] - spam_m[j])**2) / ((2*spam_s[j])**2)))
            x = x / (sqrt(2*pi)*spam_s[j])
            # x = log(x)
            cond_prob_spam = cond_prob_spam + x
    # calculate cond probability of feature given class NOTSPAM
            z = pow(e, -1 * ( ((test_matrix[i][j] - notSpam_m[j])**2) / ((2*notSpam_s[j])**2) ))
            z = z / (sqrt(2*pi)*spam_s[j])
            # z = log(z)
            cond_prob_notspam = cond_prob_notspam + z

        if cond_prob_notspam > cond_prob_spam:
            q = q + 1
            conf_predict.append(0)
        else:
            r = r + 1
            conf_predict.append(1)
    print(q, "NOT SPAM PREDICTIONS")
    print(r, "SPAM PREDICTIONS")

    # build confusion matrix and do some calculations
    p = 0
    n = 0
    # build confusion matrix
    for i in range(0, 2300):
        if conf_true[i] == 1:
            p = p + 1
    n = (2300 - p)

    tn = 0
    tp = 0
    fp = 0
    fn = 0
    for i in range(0, 2300):
            if conf_predict[i] == conf_true[i] and conf_predict[i] == 1 and conf_true[i] == 1:
                tp = tp + 1
            if conf_predict[i] == conf_true[i] and conf_predict[i] == 0 and conf_true[i] == 0:
                tn = tn + 1
            if conf_predict[i] != conf_true[i] and conf_predict[i] == 0 and conf_true[i] == 1:
                fn = fn + 1
            if conf_predict[i] != conf_true[i] and conf_predict[i] == 1 and conf_true[i] == 0:
                fp = fp + 1

    # these calculations are in accordance with pg. 23 of marsland text
    accuracy = (tp + fp) / (tp + fp + tn + fn)
    print("accuracy of model ~", accuracy)
    precision = tp / (p + fn)
    print("precision of model ~", precision)
    recall = tp / (tp + fn)
    print("recall of model ~", recall)

    conf_matrix = np.empty((2,2))
    conf_matrix[0][0] = tp
    conf_matrix[1][0] = fp
    conf_matrix[0][1] = fn
    conf_matrix[1][1] = tn

    print("~~CONFUSION MATRIX~~")
    print(conf_matrix)








# main program
load_data()
