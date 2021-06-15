# K NEAREST NEIGHBORS ALGORITHM WITH SCIKIT LEARN !!!

# FINISHED !!!


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# custom made, made it myself, just for plot analysis and customizing dataset, can be found in github plot analysis and dataset customizing
from DATASET_CUSTOMIZING.DATASET_READING_CUSTOMIZING_INPUT import make_dataset
from PLOT_ANALYSIS.PLOT_ANALYSIS_INPUT import make_analysis

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

# show all columns in the dataframe while printing it
pd.set_option('display.max_columns', None)


def kNN_algorithm(df):

    # make the plot analysis
    make_analysis(df)

    print('')
    print(160 * '-')
    print('')


    while True:

        # check for any input error
        try:
            print(df)

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


        except Exception:
            print('')
            print('Oops, at least one the columns you have inout does not appear in the dataset, try again !!!')


        # if everything seems to be working and no error has been raised, break the loop
        else:
            break



    # change those values, those columns into numpy arrays, X is 2D and Y is 1D
    X = np.array(df[ask_IVs])
    Y = np.array(df[''.join(ask_DV)])



    while True:
        # check for any exceptions
        try:
            # ask for cross validation n-fold value for n (an integer)
            print('')
            cv = int(input('''Type n, what n-fold cross validation you wanna use (5-old, 10-fold)
Generally 5-10 or 10-fold cross validation is preferred.
As k gets larger, the difference in size between the training set and the resampling subsets gets smaller.
The smaller it gets, the bias of the technique becomes smaller.
Type cross-validation (cv) value: '''))

            # make an classifier with all deafults
            knn = KNeighborsClassifier()
            # make param grid with n neighbors values from 2 to 10
            param_grid = {'n_neighbors': np.arange(2, 10)}
            # make a GridSearchCV with classifier, param_grid and with input cv
            knn_gscv = GridSearchCV(knn, param_grid, cv=cv)
            # fit the data
            knn_gscv.fit(X, Y)
            # knn_gscv.best_params_ gives a dictionary a key the same like in param gra ('n_neighbors') and the value is the best k for specified cross validation
            best_k = knn_gscv.best_params_['n_neighbors']

        except ValueError:
            print('')
            print('Oops, wrong input, cross-validation (cv) value must be an integer, try again !!!')

        except Exception:
            print('')
            print('Oops, something went wrong, try again !!!')

        # if no error occurs
        else:
            break



    while True:
        # check for any exceptions
        try:
            # ask for k value, the recommended will be the one with best parameters and best accuracy
            print('')
            k = int(input('''Type the value for k 
Recommended (the best from 2-10) k --> ''' + str(best_k) + ''' !!!
Type k: '''))
            break

        except ValueError:
            print('')
            print('Oops, wrong input, k must be an integer, try again !!!')


    # final model which will make predictions has n_neighbors specified by the user
    knn_final = KNeighborsClassifier(n_neighbors=k)
    # fit train the model with the data
    knn_final.fit(X, Y)

    # make a prediction array
    y_pred = knn_final.predict(X)
    # this will give the accuracy of the model
    accuracy = knn_final.score(X, Y)
    # make a confusion matrix y_true, y_pred and the labels are the name of classes, the unique values in Y column
    con_matrix = confusion_matrix(Y, y_pred, labels=list(df[''.join(ask_DV)].unique()))

    # print out the accuracy round to 4 decimals and the confusion matrix
    print('')
    print('!!! Accuracy of the Model:', np.round(accuracy, 4))

    print('')
    print('!!! Confusion Matrix of the Model, also a Plot !!!')
    print(con_matrix)

    # make a confusion matrix plot
    plot_confusion_matrix(knn_final, X, Y, cmap=plt.cm.Greens)
    plt.suptitle('                      \nCONFUSION MATRIX PLOT')
    plt.show()

    # !!! ------------------------------------------ PREDICTION MAKING AS Y------------------------------------------------ !!!
    def predict_new_data():

        while True:

            print('')
            # ask the user if wants to make prediction for new data that will be input
            ask_new_data = input('Do you wanna make some predictions on new data, which you wanna type here? Yes/No: ').upper()

            if ask_new_data == 'YES':

                # make an empty array which will include all new data for X
                new_data_array = np.array([])

                # ask for each IV, feature that is in the model for the new data
                for independent_variable in ask_IVs:
                    ask_new_data_IV = input('New Data (split with comma + space) | ' + independent_variable + ' : ').split(', ')

                    # if the data can't be converted into floats, raise an error
                    try:
                        for new_data in ask_new_data_IV:
                            _ = float(new_data)

                    except ValueError:
                        print('')
                        print('Oops something went wrong with the new data you have input, they must be numeric and well separated, try again !!!')
                        predict_new_data()

                    # make a numpy array of it and in float dtype
                    ask_new_data_IV_array = np.array(ask_new_data_IV, dtype=float)

                    # append that array with new data into the empty numpy array
                    new_data_array = np.append(new_data_array, ask_new_data_IV_array)


                try:
                    # but all data are all in one row, the array is not sorted and not in 2D
                    # so to reshape we have to first reshape that each row will be for each feature
                    # and we want it to be that each column will be for each feature soo make a transpose
                    new_data_array = new_data_array.reshape((len(ask_IVs), len(ask_new_data_IV_array))).transpose()

                except ValueError:
                    print('')
                    print('Oops something went wrong, each independent variable group must have the same number of number, try again !!!')
                    predict_new_data()


                # an numpy 1d array of predictions of new feature sets
                predictions = knn_final.predict(new_data_array)

                # make a horizontal stack with new data feature_sets and with their predictions
                new_data_array_with_predictions = np.hstack([new_data_array, predictions.reshape((len(predictions), 1))])

                # make a dataframe out of that array, last element is the prediction
                new_data_pred_df = pd.DataFrame(new_data_array_with_predictions, columns=ask_IVs + ['Pred (' + ''.join(ask_DV) + ')'])

                print('')
                print('')
                print('NEW OBSERVATIONS WITH THEIR PREDICTIONS')
                print('')
                print(new_data_pred_df.to_string(index=False))
                print('')

                # this is the prediction probabilities matrix, will tell how confident it is by it's expectation, prediction
                prediction_probabilities = knn_final.predict_proba(new_data_array)
                print('')
                print('THIS IS THE PREDICTION PROBABILITIES MATRIX (k =', str(k) + ')')
                print(prediction_probabilities)


                # ask if wants to try again !!!
                print('')
                ask_again = input('Do you wanna try predicting new values again? Yes/No: ').upper()

                if ask_again == 'YES':
                    print('OK, starting again.')

                elif ask_again == 'NO':
                    print('OK, no more predicting new values.')
                    break

                else:
                    print('Oops, something went wrong with your input (Yes/No), try again !!!')


            elif ask_new_data == 'NO':
                print('')
                print('OK, there will be no prediction making.')
                break

            else:
                print('')
                print('Oops, something went wrong with your input (Yes/No), try again !!!')

    predict_new_data()
    # !!! ------------------------------------------ PREDICTION MAKING AS Y------------------------------------------------ !!!



# # EXAMPLE DATASET !!!
df = pd.read_csv('breast-cancer-wisconsin.data', sep=',')
df.replace('?', -99_999, inplace=True)
df.drop(['id'], axis=1, inplace=True)


# read or modify the dataset
# df = make_dataset()

kNN_algorithm(df)
