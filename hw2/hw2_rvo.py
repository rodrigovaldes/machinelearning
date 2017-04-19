# Rodrigo Valdes Ortiz
# Machine Learning for Public Policy
# Spring 2017
# HW 2

## PLEASE SEE THE PDF FILE WITH THE OTHER ANSWERS
## PLEASE SEE THE PDF FILE WITH THE OTHER ANSWERS
## PLEASE SEE THE PDF FILE WITH THE OTHER ANSWERS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

'''
----------------------------------------------------
------------------- TASK 1 -------------------------
Load the data from a traditional csv file.
----------------------------------------------------
'''


def modify_dictionary_of_column_names(long_names, allowed_len = 15):
    '''
    Generates a dictionary to reduce the lengh of the name of the
    columns. Auxiliary function to create_new_names_columns()

    Inputs:
        long_names = dictionary of long names, like {old_name_1: old_name_1, ...}
        allowed_len = integer that indicates maximum len of the name of the
            column without promp an alert to change the name

    Output:
        small_names_dic = dictionary, {old_name_1: new_name_1, ...}
    '''
    small_names_dic = {}
    for element in long_names:

        if len(element) <= allowed_len:
            small_names_dic[element] = element
        else:
            small_names_dic[element] = input(element + "  New name for this?  ")

    return small_names_dic


def create_new_names_columns(name_columns, allowed_len = 15):
    '''
    Returns a dictionary to tranform the name of the columns from long
    to short. It asks the user if he want to modify names when there are many
    changes to do.

    Inputs:
        name_columns = list of original names of the columns
        allowed_len = integer that indicates maximum len of the name of the
            column without promp an alert to change the name

    Output:
        small_names_dic = dictionary, {old_name_1: new_name_1, ...}
            or {old_name_1: old_name_1, ...}
    '''

    long_names = {}
    for element in name_columns:
        long_names[element] = element

    if len(long_names) > 5:
        print("You will modify ", len(long_names), " names of columns. Do you really want to do this?")
        boolean = input("y/n  ")
        if boolean == "y":
            small_names_dic = modify_dictionary_of_column_names(long_names, allowed_len)
        elif boolean == "n":
            return long_names
        else:
            return "Try again"
    else:
        small_names_dic = modify_dictionary_of_column_names(long_names, allowed_len)

    return small_names_dic

def load_data(NAME_FILE):
    '''
    Returns a dataframe with small names of the columns

    Inputs:
        NAME_FILE = string, name of the file, like "data.csv"

    Output:
        data = pandas dataframe with smaller (equal) name of the columns
    '''

    data = pd.read_csv(NAME_FILE)
    name_columns = list(data.columns)
    new_name_columns = create_new_names_columns(name_columns)
    if new_name_columns != name_columns:
        data.rename(columns=new_name_columns, inplace=True)

    return data

'''
----------------------------------------------------
------------------- TASK 2 -------------------------
Explore Data
----------------------------------------------------
'''

def bar_graph(title, x_locations, x_names, x_ocurrences, name_y, name_file):
    '''
    Saves a file named name_file, which is a bar graph

    Inputs:
        title = string
        x_locations = locations of the information of x, like [0,1,2,3,4]
        x_names = names to be show in the x categories
        x_ocurrences = ocurreces of each category
        name_y = name for y axis
        name_file = name of the image to save

    Outputs:
        None
    '''

    width = 0.35       # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x_locations, x_ocurrences, width, color='r')
    ax.set_ylabel(name_y)
    ax.set_title(title)
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_names, rotation = 90)
    fig.savefig(name_file, bbox_inches='tight')
    plt.show()
    plt.close()

    return print("Bar graph done")

# Example code to run the function above
x_locations = [0,1,2,3]
x_names = ["one", "two", "three", "four"]
x_ocurrences = [10, 20, 15, 17]
name_y = "Example Y"
title = "Ocurrences of Numbers"
name_file = "test_image.png"

## Histograms

def plot_histogram(df, name_column, x_label, name_file, y_label = "Frequency"):
    df[name_column].hist()
    title = "Histogram of " + str(name_column)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(name_file, bbox_inches='tight')
    plt.show()
    plt.close()

    return print("Histogram done")




'''
----------------------------------------------------
------------------- TASK 3 -------------------------
Pre - Process Data
----------------------------------------------------
'''

def describe_data(df):
    '''
    Auxiliary function to undertand the general characteristics of
    the data

    Imput:
        df = pandas dataframe
    Output:
        None
    '''

    print(" ")
    print("*********************************")
    print("Description of the data")
    print("*********************************")
    print(df.describe())
    print(" ")

    print("*********************************")
    print("General Information")
    print("*********************************")
    print(df.info())
    print(" ")

    print("*********************************")
    print("About Misisng Values")
    print("*********************************")
    print(df.isnull().sum())

    print(" ")
    return "Done"

def fill_na(df):
    '''
    Fills NANs

    Input: df
    Output: df
    '''

    # Fill with the mean
    df = df.fillna(df.mean())

    return df

'''
----------------------------------------------------
------------------- TASK 4 -------------------------
Generate Features/Predictors
----------------------------------------------------
'''

def aux_discretize_1(df, column, number_buckets = 5):
    '''
    Returns the list of cutoffs for the categories

    Inputs:
        df = pandas dataframe
        column = name of the var to discretize
        number_buckets = desire number of categories
    Output:
        list_breaking points = list for cutoffs of the categories
    '''
    division = 1 / number_buckets

    initial_point = 0
    list_divisors = [0]
    for i in range(number_buckets):
        initial_point += division
        list_divisors.append(initial_point)

    list_breaking_points = (number_buckets + 1) * [None]
    for i in range(number_buckets + 1):
        list_breaking_points[i] = df[column].quantile(list_divisors[i])

    return list_breaking_points


def aux_discretize_2(row, list_breaking_points):
    '''
    Returns the category for one row

    Inputs:
        row = original value in one row
        list_breaking_points = list generated with aux_discretize_1
    Outputs:
        rv = category of the row
    '''
    number_buckets = len(list_breaking_points) - 1

    bucket = 0
    rv = 0
    for i in range(number_buckets):
        bucket += 1
        if row > list_breaking_points[i] and row < list_breaking_points[i+1]:
            rv = i + 1
            break

    return rv


def make_discrete(df, name_column, name_new_column, number_buckets = 5):
    '''
    Dicretize a continuos variable

    Inputs:
        df = pandas dataframe
        name_column = name of the column to discretize
    Output
        df = pandas dataframe with the new discretize var
    '''
    list_breaking_points = aux_discretize_1(df, name_column, number_buckets)

    df[name_new_column] = df[name_column].apply(lambda row: aux_discretize_2(
        row, list_breaking_points))

    return df


def aux_cat_to_dummy(row, threshold):
    '''
    Returns one if the row is at least the threshold,
    zero in other case

    Input:
        row = value
        threshold = turning point
    Output:
        rv = dummy value, one or zero
    '''

    if row >= threshold:
        rv = 1
    elif row < threshold:
        rv = 0

    return rv


def categorical_to_dummy(df, name_column, name_new_column, threshold):
    '''
    Inputs:
        df = dataframe
        name_column = name of the categorical column
        name_new_column = name of the categorial column transformed
        threshold = turnign point
    Outputs:
        df = dataframe
    '''
    list_categories = list(df[name_column].unique())
    list_categories.sort()

    df[name_new_column] = df[name_column].apply(lambda row: aux_cat_to_dummy(row, threshold))

    return df

'''
----------------------------------------------------
------- LOAD DATA WITH PREVIOUS FUNCTIONS  ---------
Run files
----------------------------------------------------
'''
NAME_FILE = "credit-data.csv" # Modify this for new data

df = pd.read_csv(NAME_FILE)
df.columns = ['PersonID',
 'serious_dlq',
 'revolving',
 'age',
 'zipcode',
 '30_59_days',
 'DebtRatio',
 'MonthlyIncome',
 'open_credits',
 '90_late',
 'real_state',
 '60_89_days',
 'dependents']

df = fill_na(df)

plot_histogram(df, "age", "Ages", "age.png", y_label = "Frequency")
plot_histogram(df, "DebtRatio", "Debt Ratio", "debt.png", y_label = "Frequency")
plot_histogram(df, "open_credits", "Open Credits", "open_credits.png", y_label = "Frequency")

# Summary stats by relevanrt variable

d_serious = df.groupby(["serious_dlq"]).mean().reset_index()


'''
----------------------------------------------------
------------------- TASK 5 -------------------------
Build Classifier
----------------------------------------------------
'''

# create dataframes with an intercept column and dummy variables for
# occupation and occupation_husb
# ciattion: http://nbviewer.jupyter.org/gist/justmarkham/6d5c061ca5aee67c4316471f8c2ae976

# Open credits > 8

# Exercise with ithe variable
# df = categorical_to_dummy(df, "open_credits", "CreditsDummy", 8)

y, X = dmatrices('serious_dlq ~  age + MonthlyIncome + dependents', df, return_type="dataframe")

y = np.ravel(y)

model = LogisticRegression()
model = model.fit(X, y)

'''
----------------------------------------------------
------------------- TASK 6 -------------------------
Evaluate Classifier
# Many ideas from:
# ciattion: http://nbviewer.jupyter.org/gist/justmarkham/6d5c061ca5aee67c4316471f8c2ae976
----------------------------------------------------
'''
# Accuracy
model.score(X, y)

# what percentage is the mean?
y.mean()

# examine the coefficients
X.columns
np.transpose(model.coef_)

# Training ans testing data
# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)

# predict class labels for the test set
predicted = model2.predict(X_test)
print(predicted)

# generate class probabilities
probs = model2.predict_proba(X_test)
print(probs)

# generate evaluation metrics
print(metrics.accuracy_score(y_test, predicted))
print(metrics.roc_auc_score(y_test, probs[:, 1]))

print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))

# Cross Validation 10 times
# evaluate the model using 10-fold cross-validation
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print(scores)
print(scores.mean())

## PLEASE SEE THE PDF FILE WITH THE OTHER ANSWERS
## PLEASE SEE THE PDF FILE WITH THE OTHER ANSWERS
## PLEASE SEE THE PDF FILE WITH THE OTHER ANSWERS






