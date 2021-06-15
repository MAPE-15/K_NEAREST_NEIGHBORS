# K NEAREST NEIGHBORS ALGORITHM WITHOUT USING SKLEARN !!!

# FINISHED !!!


import numpy as np
import math
import random
from collections import Counter

# using my module for reading and customizing datasets using input
from DATASET_CUSTOMIZING.DATASET_READING_CUSTOMIZING_INPUT import make_dataset


def k_nearest_neighbors(dataset, k=3, test_size=0.2):

    # this is the class column, the Y, the label, must be the last column in the dataset
    Y_col_name = list(dataset.columns)[-1]

    # make empty lists of knn_dict, train_Set, test_set, dicts will have keys as class names, and values as all the values it has besides from that class name
    # knn_dict --> will have all values
    # train_set --> will have values only for training
    # test_set --> will have values for testing
    knn_dict = {}
    train_set = {}
    test_set = {}

    # unique classes will be a list of unique classes in Y column in label column
    unique_classes = list(dataset[Y_col_name].unique())

    # for each class for every dictionary create an empty list so far
    for class_name in unique_classes:
        knn_dict[class_name] = []
        train_set[class_name] = []
        test_set[class_name] = []

    # as.type(float) will convert all values to floats
    # .values --> will take all values from each column in dataset
    # .tolist() --> will make a list of lists, and each list will be like each row in dataset with all values
    full_data = dataset.astype(float).values.tolist()

    # iterate through each row, data
    for data in full_data:
        # data[-1] --> the last value, last column is label Y column so that will be a key in our dictionary and append that data besides that class name
        knn_dict[data[-1]].append(data[:-1])

    # iterate through each key in dictionary
    for class_name in knn_dict:
        # shuffle all for key
        random.shuffle(knn_dict[class_name])

    # iterate through each class name, each key in dict
    for class_name in knn_dict:
        # train set will have all the values from start to the size of test_size (f.e. test_size --> 0.2, meaning that test set will have from 0 percent up to 80 percent of all data)
        # test_set will have all the rest of the values (test_size --> 0.2, will have all from 80 percent to 100 percent, last 20 percent

        # calculate the amount percentage / 100 (test_size --> 0.2) multiplied by the length of all data
        train_set[class_name] = knn_dict[class_name][:-int(math.ceil(test_size * len(knn_dict[class_name])))]
        test_set[class_name] = knn_dict[class_name][-int(math.ceil(test_size * len(knn_dict[class_name]))):]


    # this will count how many points the train set has
    how_many_points_in_test_set = 0
    for class_name in knn_dict:
        for _ in test_set[class_name]:
            how_many_points_in_test_set += 1

    # this will count how many points the test set has
    how_many_points_in_train_set = 0
    for class_name in knn_dict:
        for _ in train_set[class_name]:
            how_many_points_in_train_set += 1

    # make a list of lists yet empty, the amount of lists in list is the amount of test points (because each test point is calculated with all train points)
    distances = list(eval(how_many_points_in_test_set * '[], '))


    # this is for calculation between each test point and all training data, each testing point must be calculated with every test point
    i = 0
    # iterate through each class group in knn dict
    for class_group_test in knn_dict:
        # iterate through each test data for each class group
        for test_data in test_set[class_group_test]:

            # iterate through each class group in knn dict
            for class_group_train in knn_dict:
                # iterate through each train data for each class group
                for train_data in train_set[class_group_train]:

                    # calculate the distance between all train data and each test point
                    distance = np.sqrt(np.sum((np.array(train_data) - np.array(test_data)) ** 2))

                    # if the calculations has been occured for all data with one point in test set, move to another test point
                    if len(distances[i]) == how_many_points_in_train_set:
                        i += 1
                    # append every distance with the group it has in training data
                    distances[i].append([distance, class_group_train])


    # an empty k closest disatances
    k_closest_distances = []
    # go thorugh each distances which has been calculated for each test point
    for distances_all_train_all_test in distances:
        # sort those distances and then take the k first smallest distances
        k_closest_distances.append(sorted(distances_all_train_all_test)[:k])


    # iterate through each and every distance and take the last item, the last item is class name
    votes = [[closest_distance[-1] for closest_distance in closest_distances] for closest_distances in k_closest_distances]
    # Counter will give a count for every unique class in the given list, it will give dictionary and key is the name of the class and the value is the count of that class
    # .most_common() will give only those counts which has the highest count, it will give a tuple in the list, 1st element is the class name, and 2nd element is the count
    # that's why [0][0], we are taking the class name
    vote_results = [Counter(vote_list).most_common(1)[0][0] for vote_list in votes]


    # a list which will contain the testing points and also the class name as the last element
    test_full_data = []
    for test_class, test_points in test_set.items():

        for test_point in test_points:
            test_full_data.append(test_point + [test_class])


    # for calculating accuracy correct/total
    correct = 0
    total = 0
    # iterate through each test point which has last element the name of the class, and through the vote results lists which has all predicitons
    for test_points, prediction in zip(test_full_data, vote_results):
        # take the group name
        group = test_points[-1]

        # if that group is equal to the prediction add 1 to correct
        if group == prediction:
            correct += 1

        # always cont total number of test points
        total += 1

    # calculate the accuracy in percentage, and then print it out
    accuracy = np.round(correct / total * 100, 2)

    print('')
    print('Accuracy:', str(accuracy) + '%')



while True:
    # check for any input error
    try:
        # read or make a dataset
        df = make_dataset()

        print('')
        print('THESE ARE YOUR COLUMNS NAMES:', list(df.columns))
        print('')
        # ask for column(s) which will be all X values, which will be our features
        ask_IVs = input('Type all column names in the dataset, you want for X (Independent variables / features) (split with comma + space): ').split(', ')

        # if that column that was input does not occur in the dataset, raise an error
        for col_X in ask_IVs:
            if col_X not in list(df.columns):
                raise Exception

        print('')
        # ask for one single column which will be our label Y (DV)
        ask_DV = input('Type column name in the dataset you want for Y (Dependent variable / label): ').split(', ')

        # if there will be more than one column for label input, raise an error
        if len(ask_DV) != 1:
            raise Exception

        # and again if that label column does not occur in the dataset, raise an error
        for col_y in ask_DV:
            if col_y not in list(df.columns):
                raise Exception

        # make a shorten dataset, where last is the label column and the rest features are in order like in input
        df = df[ask_IVs + ask_DV]


    except Exception:
        print('')
        print('Oops, at least one the columns you have inout does not appear in the dataset, try again !!!')


    # if everything seems to be working and no error has been raised, break the loop
    else:
        break



while True:
    # check for any input error
    try:
        print('')
        # ask for k nearest/closest neighbors/points
        k = int(input('Type the k, the number of closest/nearest neighbors to account: '))
        # ask for testing size
        test_size = float(input('Type the test size of the dataset, f.e. when test_size = 0.1, 10% of the data will be for testing and the rest 90% will be for training (IN RANGE 0 - 1): '))

        if (test_size >= 0) and (test_size <= 1):
            pass
        else:
            raise ValueError

    except ValueError:
        print('')
        print('Oops, something went wrong with your input, k must be an integer number and test_size must be a float number between 0 and 1, try again !!!')

    else:
        break


# make an accuracy check
k_nearest_neighbors(df, k=k, test_size=test_size)
