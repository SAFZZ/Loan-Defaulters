# Loan-Defaulters
Creating a machine learning model to find loan defaulters.

#imporing packages

from datetime import datetime
import numpy as np
import pandas as pd
import xgboost
import re
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

#reading dataset
dfTest = pd.read_csv('test_indessa.csv')
dfTrain = pd.read_csv('train_indessa.csv')

dfTrain = dfTrain[['member_id', 'loan_amnt', 'funded_amnt', 'addr_state', 'funded_amnt_inv', 'sub_grade', 'term', 'desc', 'emp_length', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc','total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'last_week_pay', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim', 'loan_status']]
dfTest = dfTest[['member_id', 'loan_amnt', 'funded_amnt', 'addr_state', 'funded_amnt_inv', 'sub_grade', 'term', 'desc','emp_length', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'last_week_pay', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']]

#transforming data

def get_last_week_pay(raw) :
    try :
        return int(re.sub("\D", "", raw))
    except :
        return -9999
    
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
def clean_text(raw_text):
    cleantext = np.nan
    if type(raw_text) == str :
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, ' ', raw_text)
        cleantext = cleantext.replace('>', '')
        cleantext = ' '.join(cleantext.split())
        
        stop_words = set(stopwords.words("english"))
        words = word_tokenize(cleantext)
        
        filtered_sentence = []

        for w in words:
            if w not in stop_words:
                filtered_sentence.append(w)
        return len(filtered_sentence)
    
    else :
        return 0 


dfTrain['term'] = dfTrain['term'].apply(lambda x : int(re.sub("\D", "", x)))
dfTrain['last_week_pay'] = dfTrain['last_week_pay'].apply(get_last_week_pay)
dfTrain['desc'] = dfTrain['desc'].apply(clean_text)
dfTrain['emp_length'].replace('n/a', '0', inplace=True)
dfTrain['emp_length'].replace(to_replace='\+ years', value='', regex=True, inplace=True)
dfTrain['emp_length'].replace(to_replace=' years', value='', regex=True, inplace=True)
dfTrain['emp_length'].replace(to_replace='< 1 year', value='0', regex=True, inplace=True)
dfTrain['emp_length'].replace(to_replace=' year', value='', regex=True, inplace=True)
dfTest['term'] = dfTest['term'].apply(lambda x : int(re.sub("\D", "", x)))
dfTest['last_week_pay'] = dfTest['last_week_pay'].apply(get_last_week_pay)
dfTest['desc'] = dfTest['desc'].apply(clean_text)
dfTest['emp_length'].replace(to_replace='\+ years', value='', regex=True, inplace=True)
dfTest['emp_length'].replace(to_replace=' years', value='', regex=True, inplace=True)
dfTest['emp_length'].replace(to_replace='< 1 year', value='0', regex=True, inplace=True)
dfTest['emp_length'].replace(to_replace=' year', value='', regex=True, inplace=True)

dfTrain['term'] = pd.to_numeric(dfTrain['term'], errors='coerce')
dfTest['term'] = pd.to_numeric(dfTest['term'], errors='coerce')

dfTrain['last_week_pay'] = pd.to_numeric(dfTrain['last_week_pay'], errors='coerce')
dfTest['last_week_pay'] = pd.to_numeric(dfTest['last_week_pay'], errors='coerce')

dfTrain['emp_length'] = pd.to_numeric(dfTrain['emp_length'], errors='coerce')
dfTest['emp_length'] = pd.to_numeric(dfTest['emp_length'], errors='coerce')

dfTrain['sub_grade'] = pd.to_numeric(dfTrain['sub_grade'], errors='coerce')
dfTest['sub_grade'] = pd.to_numeric(dfTest['sub_grade'], errors='coerce')

#feature engineering

test_member_id = pd.DataFrame(dfTest['member_id'])

train_target = pd.DataFrame(dfTrain['loan_status'])

selected_cols = ['member_id', 'emp_length', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'int_rate', 'annual_inc', 'dti', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'revol_bal', 'revol_util', 'total_acc', 'total_rec_int', 'total_rec_late_fee', 'mths_since_last_major_derog', 'last_week_pay', 'tot_cur_bal', 'total_rev_hi_lim', 'tot_coll_amt', 'recoveries', 'collection_recovery_fee', 'term', 'acc_now_delinq', 'collections_12_mths_ex_med']
finalTrain = dfTrain[selected_cols]
finalTest = dfTest[selected_cols]

finalTrain['loan_to_income'] = finalTrain['annual_inc']/finalTrain['funded_amnt_inv']
finalTest['loan_to_income'] = finalTest['annual_inc']/finalTest['funded_amnt_inv']
finalTrain['bad_state'] = finalTrain['acc_now_delinq'] + (finalTrain['total_rec_late_fee']/finalTrain['funded_amnt_inv']) + (finalTrain['recoveries']/finalTrain['funded_amnt_inv']) + (finalTrain['collection_recovery_fee']/finalTrain['funded_amnt_inv']) + (finalTrain['collections_12_mths_ex_med']/finalTrain['funded_amnt_inv'])
finalTest['bad_state'] = finalTest['acc_now_delinq'] + (finalTest['total_rec_late_fee']/finalTest['funded_amnt_inv']) + (finalTest['recoveries']/finalTest['funded_amnt_inv']) + (finalTest['collection_recovery_fee']/finalTest['funded_amnt_inv']) + (finalTrain['collections_12_mths_ex_med']/finalTest['funded_amnt_inv'])

finalTrain.loc[finalTrain['bad_state'] > 0, 'bad_state'] = 1
finalTest.loc[finalTest['bad_state'] > 0, 'bad_state'] = 1

finalTrain['avl_lines'] = finalTrain['total_acc'] - finalTrain['open_acc']
finalTest['avl_lines'] = finalTest['total_acc'] - finalTest['open_acc']


finalTrain['int_paid'] = finalTrain['total_rec_int'] + finalTrain['total_rec_late_fee']
finalTest['int_paid'] = finalTest['total_rec_int'] + finalTest['total_rec_late_fee']

finalTrain['emi_paid_progress_perc'] = ((finalTrain['last_week_pay']/(finalTrain['term']/12*52+1))*100)
finalTest['emi_paid_progress_perc'] = ((finalTest['last_week_pay']/(finalTest['term']/12*52+1))*100)

finalTrain['total_repayment_progress'] = ((finalTrain['last_week_pay']/(finalTrain['term']/12*52+1))*100) + ((finalTrain['recoveries']/finalTrain['funded_amnt_inv']) * 100)
finalTest['total_repayment_progress'] = ((finalTest['last_week_pay']/(finalTest['term']/12*52+1))*100) + ((finalTest['recoveries']/finalTest['funded_amnt_inv']) * 100)


null= finalTrain.isnull().sum().sort_values(ascending=False)
total =finalTrain.shape[0]
percent_missing= (finalTrain.isnull().sum()/total).sort_values(ascending=False)

missing_data= pd.concat([null, percent_missing], axis=1, keys=['Total missing', 'Percent missing'])

missing_data.reset_index(inplace=True)
missing_data= missing_data.rename(columns= { "index": " column name"})
 
print ("Null Values in each column:\n", missing_data.sort_values(by ='Total missing', ascending = False))

null= finalTest.isnull().sum().sort_values(ascending=False)
total =finalTest.shape[0]
percent_missing= (finalTest.isnull().sum()/total).sort_values(ascending=False)

missing_data= pd.concat([null, percent_missing], axis=1, keys=['Total missing', 'Percent missing'])

missing_data.reset_index(inplace=True)
missing_data= missing_data.rename(columns= { "index": " column name"})
 
print ("Null Values in each column:\n", missing_data.sort_values(by ='Total missing', ascending = False))
def fill_nulls(value):
    cols_fill = ['mths_since_last_record','mths_since_last_major_derog',
                 'mths_since_last_delinq','total_rev_hi_lim','tot_cur_bal',
                 'tot_coll_amt','emp_length','revol_util','collections_12_mths_ex_med',
                 'open_acc','total_acc','acc_now_delinq','avl_lines','loan_to_income',
                 'annual_inc','bad_state','total_repayment_progress']
    
    if value == -9999:
        for col in cols_fill:
            finalTest.loc[finalTest[col].isnull(), col] = -9999
    else : 
        for col in cols_fill:
            finalTest.loc[finalTest[col].isnull(), col] = finalTrain[col].median()
            
fill_nulls(0)
def fill_nulls(value):
    cols_fill = ['mths_since_last_record','mths_since_last_major_derog',
                 'mths_since_last_delinq','total_rev_hi_lim','tot_cur_bal',
                 'tot_coll_amt','emp_length','revol_util','collections_12_mths_ex_med',
                 'open_acc','total_acc','acc_now_delinq','avl_lines','loan_to_income',
                 'annual_inc','bad_state','total_repayment_progress']
    if value == -9999:
        for col in cols_fill:
            finalTrain.loc[finalTrain[col].isnull(), col] = -9999
    else : 
        for col in cols_fill:
            finalTrain.loc[finalTrain[col].isnull(), col] = finalTrain[col].median()
            
fill_nulls(0)
finalTrain = finalTrain.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_trainScaled = scaler.fit_transform(finalTrain)

X_train, X_test, y_train, y_test = train_test_split(np.array(finalTrain), np.array(train_target), test_size=0.30)
eval_set=[(X_test, y_test)]

clf = xgboost.sklearn.XGBClassifier(
    objective="binary:logistic", 
    learning_rate=0.09, 
    seed=9616, 
    max_depth=30, 
    gamma=10, 
    n_estimators=500)

clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc", eval_set=eval_set, verbose=True)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(np.array(y_test).flatten(), y_pred)

submission_file_name = "Loan_Defaulter_submission"
print("Accuracy: %.10f%%" % (accuracy * 100.0))
submission_file_name = submission_file_name + ("_Accuracy_%.6f" % (accuracy * 100)) + '_'

accuracy_per_roc_auc = roc_auc_score(np.array(y_test).flatten(), y_pred)
print("ROC-AUC: %.10f%%" % (accuracy_per_roc_auc * 100))
submission_file_name = submission_file_name + ("_ROC-AUC_%.6f" % (accuracy_per_roc_auc * 100))

final_pred = pd.DataFrame(clf.predict_proba(np.array(finalTest)))
dfSub = pd.concat([test_member_id, final_pred.iloc[:, 1:2]], axis=1)
dfSub.rename(columns={1:'loan_status'}, inplace=True)
dfSub.to_csv((('%s.csv') % (submission_file_name)), index=False)

##OUTPUT:
![image](https://user-images.githubusercontent.com/75234912/221400119-81102ec8-832b-4495-992b-632924d45e31.png)

##RESULT:
Thus,a model to find loan defaulters has been successfully created.
