# %%
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
#%matplotlib inline 

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools

# %%
bank_data= pd.read_csv('https://raw.githubusercontent.com/IndreBZ/bank/716b7cd2a11db98dc4175a8ccce927869caca041/bank2.csv')
print("The first 5 rows of the bank data") 
bank_data.head()



# %%
bank_data.shape

# %%
bank_data.dtypes

# %%
##find missing values
print('Find if there are missed values in data set')
missing_data = bank_data.isnull()
missing_data.head(5)
## no missing values

# %%
##DATA MANIPULATION TASK
#1. select random subsample of data set
print('random subsample of data set')
sample = bank_data.sample(frac=0.5, replace=False, random_state=1)
print('50% sample of data set')
print(sample.shape)


# %%
#2. filter desired rows using simple and more complex conditions;
df = bank_data
print("how many clients aged(30-50) subscribed a term deposit?")
age_df = df[(df["age"] > 30) & (df["age"] < 50) & (df["y"] == "yes")]
print(age_df.shape)
print("how many clients aged(30-50) didn't subscribe a term deposit?")
age_df1 = df[(df["age"] > 30) & (df["age"] < 50) & (df["y"] == "no")]
print(age_df1.shape)
print("group by job and term deposit and calculate sum of balance. Filter jobs where balance > average")
job_result_grouped = df.groupby(["y", "job"]).agg({"balance": "sum"})
balance =job_result_grouped ["balance"].mean()
print("Balance average",balance)
job_result = job_result_grouped[(job_result_grouped["balance"] > balance)]
print(job_result)
    

# %%
#3. drop unnecessary variables, rename some variables
#unnecessary(guess) variables could be related with the last contact of the current campaign
df.drop(["contact", "day","month","duration"], axis=1, inplace=True)
print('rename some variables')
df.rename(columns={'campaign':'num_of_contact'}, inplace=True)
df.rename(columns={'pdays':'num_of_days'}, inplace=True)
df.head()

# %%
#4. calculate summarizing statistics (for full sample and by categorical variables as well)
#print(bank_data.dtypes)
print(bank_data.describe())##original dataset
print(bank_data.info())


# %%
#summarizing statistics only for not categorical variables
df.describe()## updated datasetprint (classification_report(yyy_test, yyyhat))
## the best model evaluation: reason add variable duration which has bigger correlation coefficient

# %%

#5. create new variables using simple transformation and custom functions
print('change categorical columns to numeric')
df['y'].replace(['no', 'yes'],[0, 1], inplace=True)
df['job'].replace(["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"],[0,1,2,3,4,5,6,7,8,9,10,11], inplace=True)
df['marital'].replace(["married","divorced","single"],[0, 1,2], inplace=True)
df['education'].replace(["unknown","secondary","primary","tertiary"],[0,1,2,3], inplace=True)
df['default'].replace(['no', 'yes'],[0, 1], inplace=True)
df['housing'].replace(['no', 'yes'],[0, 1], inplace=True)
df['loan'].replace(['no', 'yes'],[0, 1], inplace=True)
df['poutcome'].replace(["unknown","other","failure","success"],[0,1,2,3], inplace=True)

# %%
print("Change column data type")
#look into updated data types and summarizing statistics for new dataset df
print(df.dtypes)
print(df.describe())

# %%
#6. order data set by several variables.
#df['job'].value_counts()

sorted_balance = df.sort_values(by=['balance'], ascending=True)
print(sorted_balance.head(3))
sorted_age = df.sort_values(by=['age'], ascending=False)
print(sorted_age.head(3)) 
sorted_job_educational = df.sort_values(by=['job','education'], ascending=True)
print(sorted_job_educational.head(5))

# %%
##DATA VISUALISATION TASK
## try to find relations between variables
df.corr()
## correlations are very small

# %%
##Pie
bank_data_yes =bank_data[(bank_data["y"] == "yes")]
#bank_data.head()
df_job = bank_data_yes.groupby(['job'])['y'].count().reset_index()
#df_job
label =list(df_job['job'])
label
#df_job['y']
#Pie on jobs 
fig,ax=plt.subplots()
ax.pie(df_job['y'],autopct='%1.1f%%', pctdistance=1.2) #using explode to highlight the lowest 
ax.set_aspect('equal')  # Ensure pie is drawn as a circle
plt.title('Job names when result is yes')
ax.legend(df_job['job'],bbox_to_anchor=(1, 0, 0.5, 1))#, include legend, if you donot want to pass the labels
plt.show()





# %%
df_age = df[['age']]
df_age.plot(kind='box', figsize=(10, 6))
plt.title('Variable of age distribution in bank data dataset')
plt.show()

# %%
sns.catplot(x = 'y', y = 'balance', data = df)

# %%
##BOX PLOT
new_df =df[['age']]
new_df.plot(kind='box', figsize=(10, 6))
plt.title('Variable of age distribution in bank data dataset')

plt.show()

# %%
#MODELLING TASK
#logistic regression
df.head()
#the first model
#define x(dependent variable) and y(independent variable) of dataset
x = np.asarray(df[['age','job','marital','education','default','balance','housing','loan','num_of_contact','num_of_days','previous','poutcome']])
print('x',x[0:5])
y = np.asarray(df['y'])
print('y',y[0:5])

# %%
#normalize the dataset
x = preprocessing.StandardScaler().fit(x).transform(x)
x[0:5]

# %%
##Define train and test sets
#from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.3, random_state=5)## worse result with test_size=0.2
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)

# %%
#Define logistic regression model
LR = LogisticRegression(C=0.01, solver='liblinear').fit(x_train,y_train)
LR

# %%
#normalize the dataset
#from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x)
x[0:5]

# %%
## MODELLING TASK
#logistic regression
df.head()
#the first model
#define X of datasert
x = np.asarray(df[['age','job','marital','education','default','balance','housing','loan','num_of_contact','num_of_days','previous','poutcome']])
print('x',x[0:5])
y = np.asarray(df['y'])
print('y',y[0:5])

# %%
#jaccard index: for accuracy evaluation. we can define jaccard as the size of the intersection divided by the size of the union of the two label sets. If the entire set of predicted labels for a sample strictly matches with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.
jaccard_score(y_test, yhat,pos_label=0)

# %%

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))

# %%
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['y=1','y=0'],normalize= False,  title='Confusion matrix')

# %%
print (classification_report(y_test, yhat))
##model predict y = 0 94% correct but only  19% for y=1

# %%
###the second model
xx = np.asarray(df[['age','job','education','marital','default','housing','balance','loan']])
yy = np.asarray(df['y'])
xx = preprocessing.StandardScaler().fit(xx).transform(xx)
xx_train, xx_test, yy_train, yy_test = train_test_split( xx, yy, test_size=0.2, random_state=5)
LR = LogisticRegression(C=0.01, solver='liblinear').fit(xx_train,yy_train)
yyhat = LR.predict(xx_test)
yyhat_prob = LR.predict_proba(xx_test)
jaccard_score(yy_test, yyhat,pos_label=0)

# %%
#jaccard index
jaccard_score(y_test, yhat,pos_label=0)


# %%
print (classification_report(yy_test, yyhat))
## results are much worse than the first model

# %%
cnf_matrix = confusion_matrix(yy_test, yyhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['y=1','y=0'],normalize= False,  title='Confusion matrix')
## in almost all cases model predict Y = 0, not good

# %%
##The third model
df_LR_new = bank_data
print('change categorical columns to numeric')
df_LR_new['y'].replace(['no', 'yes'],[0, 1], inplace=True)
df_LR_new['job'].replace(["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"],[0,1,2,3,4,5,6,7,8,9,10,11], inplace=True)
df_LR_new['marital'].replace(["married","divorced","single"],[0, 1,2], inplace=True)
df_LR_new['education'].replace(["unknown","secondary","primary","tertiary"],[0,1,2,3], inplace=True)
df_LR_new['default'].replace(['no', 'yes'],[0, 1], inplace=True)
df_LR_new['housing'].replace(['no', 'yes'],[0, 1], inplace=True)
df_LR_new['loan'].replace(['no', 'yes'],[0, 1], inplace=True)
df_LR_new['contact'].replace(['unknown','telephone','cellular'],[0, 1,2], inplace=True)
df_LR_new['month'].replace(['jan', 'feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],[0, 1,2,3,4,5,6,7,8,9,10,11], inplace=True)
df_LR_new['poutcome'].replace(["unknown","other","failure","success"],[0,1,2,3], inplace=True)
print("Change column data type")
df_LR_new.dtypes

# %%
df_LR_new.corr()

# %%
xxx = np.asarray(df_LR_new[['age','job','education','marital','default','housing','balance','loan','contact','duration']])## added duration which was deleted in the df dataset
yyy = np.asarray(df_LR_new['y'])
xxx = preprocessing.StandardScaler().fit(xxx).transform(xxx)
xxx_train, xxx_test, yyy_train, yyy_test = train_test_split( xxx, yyy, test_size=0.2, random_state=5)
LR = LogisticRegression(C=0.01, solver='liblinear').fit(xxx_train,yyy_train)
yyyhat = LR.predict(xxx_test)
yyyhat_prob = LR.predict_proba(xxx_test)
jaccard_score(yyy_test, yyyhat,pos_label=0)

# %%
cnf_matrix = confusion_matrix(yyy_test, yyyhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['y=1','y=0'],normalize= False,  title='Confusion matrix')

# %%

df_LR_new.corr()


