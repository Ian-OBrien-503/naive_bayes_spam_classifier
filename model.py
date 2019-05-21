import numpy as np
import statistics as stat


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
    test_prior_spam = spam / 2301
    test_prior_not_spam = not_spam /2301
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
    train_prior_spam = spam / 2300
    train_prior_not_spam = not_spam / 2300
    print("spam prior test set =", train_prior_spam)
    print("NOT spam prior test set =", train_prior_not_spam)
    mean_and_std_dev(train_matrix)


# compute the mean and standard deviation for each feature in the training set given each class
def mean_and_std_dev(train_matrix):
    min = 0.0001

    # first column will be MEAN and second column will be standard deviation
    train_class_spam = np.empty(57, 2)
    train_class_not_spam = np.empty(57, 2)

# main program
load_data()
