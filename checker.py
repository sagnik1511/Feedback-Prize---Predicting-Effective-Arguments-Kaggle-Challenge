"""
Problem setter only needs to change the gen_score function as per requirements.
Also in verify_submission function you must assign the value of ID i.e name of ID or index column.
If anyone wishes to add suggestions to the candidates in DS question type then it can be done by 
changing the print statement as displayed at the last line in Verify_submission function.
"""
import csv
import json
import sys
from sklearn import metrics
import pandas as pd
import numpy as np
​
data =  json.loads(input()) #json file containing the paths of testcase and user submission is loaded
testcase = data['input_testcase_file_path'] #Path for the testcase file is assigned to this variable
user_submission = data['submission_file_path'] #Path for the submission file is assigned to this variable
​
​
def gen_score(actual, predicted):
    """
    Formula to calcucate score is to be defined in this function.
    Ideally score should be normalised between 0 and 100.
    Example: score = metrics.f1_score(actual,predicted)
    """
    score = 100 * max(0, metrics.r2_score(actual, predicted))
    return score
​
def verify_submission():
    """
    This function checks whether the candidate's submission has:
    1. '.csv' as the file type
    2. equal number of rows as of the testcase
    3. same columns as of the testcase
    4. has predictions for all the id's
​
    This functions checks the above mentioned points. If all the checks are passed then the truth values 
    and the predictions are passed into the gen_score function to generate the score. If any one of the 
    checks re failed then this function raises an exception error which is then displayed to the candidate
    in the interface.
​
    You are required to add the name of the id column in this function while creating a checker.
    Example: ID = 'Q_Id'
​
    If you want to show some hints to the candidate then that can be done using this function as well.
    To do so you must replace the existing print statement with print(json.dumps({'score':score,'messages':['Pass']}))
    Here instead of 'Pass' you can add your instruction(s)
    """
    ID = "timestamp" #Write the name of Index/Id column here. For example ID.
    # check whether the candidate's submission ends with '.csv' or not
    if not user_submission.endswith('.csv'):
        raise Exception ('Please upload a csv file containing predictions in the format given in the sample submission file')
​
    # Loads data
    fp_testcase = pd.read_csv(testcase)
    fp_submission = pd.read_csv(user_submission)
    
    # Candidate's submission file must have same number of rows as of the testcase
    num_rows = fp_testcase.shape[0] 
    if fp_submission.shape[0] != num_rows:
        raise Exception('File does not contain the correct number of rows')
    
    # Candidate's submission file must have same column names as of the testcase
    if list(fp_submission.columns) != list(fp_testcase.columns):
        raise Exception ('File does not contain correct headers, please check sample_submission file')
    
    # Set ID column as index for both testcase and user's submission
    fp_testcase.set_index(ID, inplace=True)
    fp_submission.set_index(ID, inplace=True)
    
    # extracting the label column(s) present in the testcase (Extracting target columns)
    label_cols = list(set(fp_testcase.columns))
    
    # find the unique ids that are common between the testcase and the user's submission
    intersection = set(fp_submission.index).intersection(set(fp_testcase.index))
    
    # find the unique ids that present in testcase but not in the user submission file
    # This is checked as the candidate's submission must have all the predictions for the ids provided in the test data
    not_in_test = set(fp_testcase.index) - set(fp_submission.index)
    if len(not_in_test) == 0:
        pass
    else:
        key_not_found = list(not_in_test)[0]
        raise Exception('File does not contain prediction for {0}'.format(key_not_found))
        
    # Select the predictions that are common between the testcase and user's submission
    fp_testcase = fp_testcase.loc[pd.Index(intersection)]
    fp_submission = fp_submission.loc[pd.Index(intersection)]
    
    # Select the truth values
    actual_values = fp_testcase[label_cols] 
    # Select the predictions
    predicted_values = fp_submission[label_cols]
    # Calculate score based on actual and predicted values
    score = gen_score(actual_values, predicted_values)
    print("{0:.6f}".format(score))
    #print(json.dumps({'score':score,'messages':['Pass']})) # If any hints or messages needs to be displayes they should be in message section
​
try:
    verify_submission()
except Exception as e:
    sys.exit(e.args[0])