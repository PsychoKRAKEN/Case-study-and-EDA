# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_excel("loan.xlsx")

data.head()


#Data Filtering
data.isnull().sum()

#Percentage of Null Data Values
round(data.isnull().sum()/len(data.index),2)*100

#Lets Remove data values with more than 90% Null values
null_col=data.columns[100*(data.isnull().sum()/len(data.index))>90]
null_col
data=data.drop(null_col,axis=1)
data.shape

#Analyis data of columns with 32% and 64% Null Values
data.loc[:,['desc','mths_since_last_delinq']].head()

#We can see they are not of any use so now drop these columns too
data=data.drop(['desc','mths_since_last_delinq'],axis=1)

#Now check Null values in rows 
data.isnull().sum(axis=1)
#Everything looks good

#lest remove columns that are of no use 
data=data.drop(['title', 'url', 'zip_code'],axis=1)

x=data.drop([
  "delinq_2yrs",
  "earliest_cr_line",
  "inq_last_6mths",
  "open_acc",
  "pub_rec",
  "revol_bal",
  "revol_util",
  "total_acc",
  "out_prncp",
  "out_prncp_inv",
  "total_pymnt",
  "total_pymnt_inv",
  "total_rec_prncp",
  "total_rec_int",
  "total_rec_late_fee",
  "recoveries",
  "collection_recovery_fee",
  "last_pymnt_d",
  "last_pymnt_amnt",
  "last_credit_pull_d",
  "application_type"],axis=1)

x.info()

#first correct the data type of different columns
x.info()
# remove % from int rate and convert to numeric
x['int_rate']=x['int_rate']*100

#lets Analyis loan Status
x['loan_status']=x['loan_status'].astype('category')
x['loan_status'].value_counts()

#We cannot analyis Current status data so first we remove them and convert other two in 0/1 form
x=x[x['loan_status']!='Current']
x['loan_status'] = x['loan_status'].apply(lambda i: 0 if i=='Fully Paid' else 1)
x['loan_status']=x['loan_status'].astype('int')
x['loan_status'].value_counts()

#Anaysis on graph
# default rate
round(np.mean(x['loan_status']), 2)

# lets define a function to plot loan_status across categorical variables
def plot_cat(cat_var):
    sns.barplot(x=cat_var, y='loan_status', data=x)
    plt.show()
    
#Realtion Between grade and loan_status
plot_cat('grade')

# term: 60 months loans default more than 36 months loans
plot_cat('term')

# sub-grade: as expected - A1 is better than A2 better than A3 and so on 
plt.figure(figsize=(16, 6))
plot_cat('sub_grade')

# home ownership: not a great discriminator
plot_cat('home_ownership')

# verification_status: surprisingly, verified loans default more than not verifiedb
plot_cat('verification_status')

# purpose: small business loans defualt the most, then renewable energy and education
plt.figure(figsize=(16,6))
sns.barplot(x='purpose', y='loan_status', data=x)
plt.xticks(rotation=15)
plt.show()


# let's also observe the distribution of loans across years
# extracting month and year from issue_date
x['month'] = x['issue_d'].apply(lambda x: x.month)
x['year'] = x['issue_d'].apply(lambda x: x.year)
# lets compare the default rates across years
# the default rate had suddenly increased in 2011, inspite of reducing from 2008 till 2010
plot_cat('year')

# comparing default rates across months: not much variation across months
plt.figure(figsize=(16, 6))
plot_cat('month')

#Let's now analyse how the default rate varies across continuous variables.
# loan amount: the median loan amount is around 10,000
sns.distplot(x['loan_amnt'])
plt.show()

#The easiest way to analyse how default rates vary across continous variables is to bin the variables into discrete categories.
#Let's bin the loan amount variable into small, medium, high, very high.
# binning loan amount
def loan_amount(n):
    if n < 5000:
        return 'low'
    elif n >=5000 and n < 15000:
        return 'medium'
    elif n >= 15000 and n < 25000:
        return 'high'
    else:
        return 'very high'
        
x['loan_amnt'] = x['loan_amnt'].apply(lambda x: loan_amount(x))
# let's compare the default rates across loan amount type
# higher the loan amount, higher the default rate
plot_cat('loan_amnt')

# let's also convert funded amount invested to bins
x['funded_amnt_inv'] = x['funded_amnt_inv'].apply(lambda x: loan_amount(x))

# funded amount invested
plot_cat('funded_amnt_inv')

# lets also convert interest rate to low, medium, high
# binning loan amount
def int_rate(n):
    if n <= 10:
        return 'low'
    elif n > 10 and n <=15:
        return 'medium'
    else:
        return 'high'
    
    
x['int_rate'] = x['int_rate'].apply(lambda x: int_rate(x))

# comparing default rates across rates of interest
# high interest rates default more, as expected
plot_cat('int_rate')

# debt to income ratio
def dti(n):
    if n <= 10:
        return 'low'
    elif n > 10 and n <=20:
        return 'medium'
    else:
        return 'high'
    

x['dti'] = x['dti'].apply(lambda x: dti(x))

# comparing default rates across debt to income ratio
# high dti translates into higher default rates, as expected
plot_cat('dti')

# installment
def installment(n):
    if n <= 200:
        return 'low'
    elif n > 200 and n <=400:
        return 'medium'
    elif n > 400 and n <=600:
        return 'high'
    else:
        return 'very high'
    
x['installment'] = x['installment'].apply(lambda x: installment(x))

# comparing default rates across installment
# the higher the installment amount, the higher the default rate
plot_cat('installment')

# annual income
def annual_income(n):
    if n <= 50000:
        return 'low'
    elif n > 50000 and n <=100000:
        return 'medium'
    elif n > 100000 and n <=150000:
        return 'high'
    else:
        return 'very high'

x['annual_inc'] = x['annual_inc'].apply(lambda x: annual_income(x))

# annual income and default rate
# lower the annual income, higher the default rate
plot_cat('annual_inc')

# State and default rate
# Most defaulting borrowers’ are from NEVADA
plt.figure(figsize=(16,6))
plot_cat('addr_state')
plt.xticks(rotation=90)
plt.show()

x.info()
#Correlation Matrix
cor = x.corr()

# figure size
plt.figure(figsize=(20,8))

# heatmap
sns.heatmap(cor, cmap="YlGnBu", annot=True)
plt.show()