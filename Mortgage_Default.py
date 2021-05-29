import pandas as pd
import numpy as np
import math
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

print(datetime.now())

np.random.seed(123)

dfMortgageOriginal = pd.read_csv('mortgage_default.csv')

dfMortgageOriginal = resample(dfMortgageOriginal,
                        replace=False,
                        n_samples=50000,
                        random_state=42)

#######################################
# Data Cleasing / Wrangling
#######################################
# In order to simplify the model, it will be dismiss all variables which provides
# median and mode data about the building and keep just the average values
print("Removing MEDIAN and MODE variables")
print(dfMortgageOriginal.columns[58:86])

dfMortgage = dfMortgageOriginal.drop(columns = dfMortgageOriginal.columns[58:86], axis=1)

print("Column Kept (a)")
print(dfMortgage.columns)

# Drop features which the meaning is not clear in the domain context
# Variables related to Home Credit Form filling
dropCols = []
for i in range(2,22):
    colName = "FLAG_DOCUMENT_" + str(i)
    dropCols.append(colName)

print("Removing Form Filling Variables")
print(dropCols)
dfMortgage.drop(columns=dropCols, axis=1, inplace=True)
print("Column Kept (b)")
print(dfMortgage.columns)

# Credit scores can be given by 3 diferent sources
# All credit score amount is normalized between 0 and 1
# Get an average of credit score if the customer has more than one credit source
# Drop rows where there is none credit score
dfMortgage.dropna(axis=0, subset=['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3'], inplace = True, how="all")

dfMortgage['AVG_CREDIT_SCORE'] = dfMortgage.apply(
                    lambda x : np.nanmean([ x['EXT_SOURCE_1'],x['EXT_SOURCE_2'],x['EXT_SOURCE_3'] ]), axis=1)

print(dfMortgage['AVG_CREDIT_SCORE'] )
dropCols = [ 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
dfMortgage.drop(columns=dropCols, axis=1, inplace=True)

print("Column Kept (c)")
print(dfMortgage.columns)

# After the first time the Linear Model regression analysis ran
# It was identified that all P-Values from the variables tended to Zero
# A research on StackOverflow showed a tip about what was happening, as below
#
# R/GLM and statsmodels.GLM have different ways of handling "perfect separation"
# (which is what is happening when fitted probabilities are 0 or 1).
# In Statsmodels, a fitted probability of 0 or 1 creates Inf values on the logit scale,
# which propagates through all the other calculations, generally giving NaN values for everything.
# There are many ways of dealing with perfect separation.
# One option is to manually drop variables until the situation resolves.
# There are also some automated approaches. Statsmodels has elastic net penalized logistic regression
# (using fit_regularized instead of fit). But this will give you point estimates without standard errors.
# The statsmodels master has conditional logistic regression. I don't think Statsmodels has Firth's method.
#
# It was possible to come up to the conclusion that the model used is too complexy and have many variables and should
# Somehow been simplified, by eliminating several columns / features
# Based on the Domain knowledge it was eliminated those variables which are likely not to impact in the outcome
# Also for some variables that seems to be redundant and highly correlated, just one of them were kept
# For instance the dataset had The Avg Rating of Clients where the customer lives and also for the City where the customer lives
# In this case, it was kept just the last one

dropCols = [ 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL','REGION_RATING_CLIENT',
'WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START','LIVE_REGION_NOT_WORK_REGION','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION',
'LIVE_REGION_NOT_WORK_REGION','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','AMT_REQ_CREDIT_BUREAU_YEAR']

dfMortgage.drop(columns=dropCols, axis=1, inplace=True)

dfMortgage['AMT_REQ_CREDIT_BUREAU_QRT_TOTAL'] = dfMortgage.apply(lambda x : x['AMT_REQ_CREDIT_BUREAU_HOUR']+x['AMT_REQ_CREDIT_BUREAU_DAY']+
                                                x['AMT_REQ_CREDIT_BUREAU_WEEK']+x['AMT_REQ_CREDIT_BUREAU_MON']+x['AMT_REQ_CREDIT_BUREAU_QRT'],
                                                axis = 1)
dropCols = ['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK',
                'AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT']
dfMortgage.drop(columns=dropCols, axis=1, inplace=True)

# The GLM function continuously struggled to work properly on a dataset over 100,000 rows
# As approach to make feasible to GLM runs for the entire dataset
# It was decided to eliminate previously all columns which p-value > .9
# Since they were completely irrelavant for determining if a client is get default or not
# The variables to be removed before being converted to Dummies were : NAME_EDUCATION_TYPE, CODE_GENDER, NAME_FAMILY_STATUS
# NAME_EDUCATION_TYPE_Academic degree                -1.3275   817.8400  -0.0016 0.9987  -1604.2645  1601.6095
# NAME_EDUCATION_TYPE_Higher education               -1.0479   817.8398  -0.0013 0.9990  -1603.9845  1601.8887
# NAME_EDUCATION_TYPE_Incomplete higher              -0.9002   817.8398  -0.0011 0.9991  -1603.8368  1602.0363
# NAME_EDUCATION_TYPE_Lower secondary                -0.5536   817.8398  -0.0007 0.9995  -1603.4902  1602.3829
# NAME_EDUCATION_TYPE_Secondary / secondary special  -0.6566   817.8398  -0.0008 0.9994  -1603.5932  1602.2800
# CODE_GENDER_F                                       4.8386  5724.8786   0.0008 0.9993 -11215.7173 11225.3945
# CODE_GENDER_M                                       5.1230  5724.8786   0.0009 0.9993 -11215.4328 11225.6789
# CODE_GENDER_XNA                                   -14.4475 15538.9562  -0.0009 0.9993 -30470.2419 30441.3470
# NAME_FAMILY_STATUS_Civil marriage                  -0.8078   817.8398  -0.0010 0.9992  -1603.7443  1602.1288
# NAME_FAMILY_STATUS_Married                         -1.0031   817.8398  -0.0012 0.9990  -1603.9397  1601.9334
# NAME_FAMILY_STATUS_Separated                       -0.7674   817.8398  -0.0009 0.9993  -1603.7039  1602.1692
# NAME_FAMILY_STATUS_Single / not married            -0.8544   817.8398  -0.0010 0.9992  -1603.7909  1602.0822
# NAME_FAMILY_STATUS_Widow                           -1.0532   817.8398  -0.0013 0.9990  -1603.9898  1601.8834

dropCols = ['NAME_EDUCATION_TYPE', 'CODE_GENDER', 'NAME_FAMILY_STATUS']
dfMortgage.drop(columns=dropCols, axis=1, inplace=True)

print("Column Kept (d)")
print(dfMortgage.columns)

# dfMortgage.to_csv("validation.csv")
# exit(0)

# Remove the columns which have more than 20% of values as null
# These columns will not be used cause there are so many missing values for that
# So they are not good predictors
MAX_COLUMN_NULL_RATE = 0.2
def removeColumnsTooNull(dfIn):
    colsNull = dfIn.isnull().sum() / len(dfIn)
    # print(colsNull)
    dropCols = []
    for index, value in colsNull.items():
        if value > MAX_COLUMN_NULL_RATE:
            print("Deleted column {}, {:.2f}% of nulls > {:.2f}%".format(index, value *100, MAX_COLUMN_NULL_RATE *100))
            dropCols.append( index )
    if dropCols:
        dfIn =  dfIn.drop(dropCols, axis=1)
    return dfIn

dfMortgage = removeColumnsTooNull(dfMortgage)

# Drop rows which have missing values
print("Rows before drop nulls")
print(dfMortgage.shape[0])

dfMortgage = dfMortgage.dropna()

print("Rows after drop nulls")
print(dfMortgage.shape[0])

print("Column Kept (e)")
print(dfMortgage.columns)

# Examining the correlation matrix in order to eliminate redundant variables highly correlated and
# to simplify the model
corr = dfMortgage.corr()
# corr.to_csv("correlation.csv")

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(corr, annot=True,cmap='coolwarm',fmt='.2g')
plt.savefig('Correlation')
# plt.show()

# By running the correlation matrix was possible to remove very correlated columns
# AMT_CREDIT and AMT_GOODS_PRICE - 98.7%
# CNT_FAM_MEMBERS and CNT_CHILDREN - 88.1%

dropCols = ['AMT_GOODS_PRICE','CNT_CHILDREN']
dfMortgage.drop(columns=dropCols, axis=1, inplace=True)

# Reducing the number of Independent variables after the first time I ran Logistic Regression Analysis
# Since the model struggled to deal with so many variables
# Based on the domain knowledge eliminate variables which are likely to not interfere in the Default status
dfMortgage.drop(columns=['ORGANIZATION_TYPE', 'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE', 'NAME_TYPE_SUITE'], axis = 1, inplace=True)

print("Column Kept (e)")
print(dfMortgage.columns)

print("Shape of the dataset after simplification - Rows and Columsn removal")
print(dfMortgage.shape)

# Unemployed customers will have 0 days as employed
dfMortgage['DAYS_EMPLOYED'] = dfMortgage['DAYS_EMPLOYED'].apply(lambda x : 0 if x > 0 else x )

# Creating a flag to Employed to give more weight to the fact the customer has or not an income source
dfMortgage['FLAG_EMPLOYED'] = dfMortgage['DAYS_EMPLOYED'].apply(lambda x : 0 if x == 0 else 1 )

# Transform Days variables as positive
dfMortgage[['DAYS_EMPLOYED', 'DAYS_BIRTH','DAYS_LAST_PHONE_CHANGE']] = dfMortgage[
                    ['DAYS_EMPLOYED','DAYS_BIRTH','DAYS_LAST_PHONE_CHANGE']].apply(lambda x : x * -1 )

# Count different values in categorical variables
def classifyColumnsDummy(dfIn, columns):
    dfCountUnique = dfIn[columns].nunique()
    dummyTwo = []
    dummMany = []
    for index, value in dfCountUnique.items():
        if value > 2:
            dummMany.append( index )
        else:
            dummyTwo.append( index )
    return dummyTwo, dummMany

dummyColumns = ['FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_CONTRACT_TYPE']

dummyTwo, dummyMany = classifyColumnsDummy(dfMortgage,dummyColumns)

print("Dummy Variables with Just 2 different Values")
print(dummyTwo)
print("Dummy Variables with more than 2 different Values")
print(dummyMany)

dfMortgage = pd.get_dummies(dfMortgage, columns=dummyTwo,drop_first=True)
dfMortgage = pd.get_dummies(dfMortgage, columns=dummyMany,drop_first=False)

print(dfMortgage.shape)

# print(dfMortgage.head())
# dfMortgage.to_csv("validation.csv")
# exit(0)

# Drop customer ID variable that is not used in the model
dfMortgage.drop(['SK_ID_CURR'], axis = 1, inplace=True)

# The dataset is highly unbalanced with about 10 times Non Default (0) than default (1)
# This is leading to all predictions becoming as Non default
# The unbalancing must be treated on the training dataset
from sklearn.utils import resample

dfMortgage_0 = dfMortgage[dfMortgage['TARGET']==0]
dfMortgage_1 = dfMortgage[dfMortgage['TARGET']==1]

# Upsample minority class
print(dfMortgage.shape)
print(dfMortgage_0.shape[0])
print(dfMortgage_1.shape[0])
print("Starting Resample")

size_0 = dfMortgage_0.shape[0]
size_1 = dfMortgage_1.shape[0]

final_size = size_0 if size_1 * 3 > size_0 else size_1 * 3

dfMortgage_0 = resample(dfMortgage_0,
                        replace=False,     # sample without replacement
                        n_samples=final_size,
                        random_state=42) # reproducible results

dfMortgage_1 = resample(dfMortgage_1,
                        replace=True,     # sample with replacement
                        n_samples=final_size,
                        random_state=42) # reproducible results

# Combine resampled majority class with minority class
dfMortgage_Balanced = pd.concat([dfMortgage_0, dfMortgage_1])

# Display new class counts
print(dfMortgage_Balanced['TARGET'].value_counts())

#########################################
# Obtain X e y from the dataset
#########################################
X_dataset = dfMortgage_Balanced.drop(['TARGET'], axis=1)
y_dataset = dfMortgage_Balanced['TARGET']

# Setting what are the scalable columns for future any purpose
scalableCols = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','DAYS_EMPLOYED', 'DAYS_BIRTH','DAYS_LAST_PHONE_CHANGE',
                'REGION_POPULATION_RELATIVE','CNT_FAM_MEMBERS','REGION_RATING_CLIENT_W_CITY',
                'OBS_30_CNT_SOCIAL_CIRCLE',	'DEF_30_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_QRT_TOTAL',
                'DAYS_LAST_PHONE_CHANGE','AVG_CREDIT_SCORE'
]

# X_dataset.to_csv("validation.csv")
# exit(0)

###########################################################
# Implementing Logistic Regression Analysis over a dataset
############################################################
def LogisticRegressionAnalysis(X_dataset, y_dataset, scalableCols, verbose=True):
    import statsmodels.api as sm

    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    scaledCols = [col for col in scalableCols if col in X_dataset.columns ]
    X_scaled = X_dataset.copy()
    X_scaled.loc[:,scaledCols] = scaler.fit_transform(X_scaled.loc[:,scaledCols])

    print("-I- Begining Logistic Binomial Model Fitting")
    print(datetime.now())
    # Get Summary
    logit_model = sm.GLM(y_dataset.to_numpy(), X_dataset, family=sm.families.Binomial())
    result=logit_model.fit(fit_intercept=True)
    print(datetime.now())
    print("-I- Ending Logistic Binomial Model Fitting")
    if verbose:
        print("\nSummary of Logistic Binomial Regression Analysis (GLM function) for the Balanced Dataset")
        print(result.summary2())

    # odds ratios and 95% CI
    params = result.params
    conf = result.conf_int()
    conf['Odds Ratio'] = params
    conf.columns = ['2.5%','97.5%','Odds Ratio']
    if verbose:
        print("\nOdds Ratio (using GLM function) for the Full Dataset")
        print(np.exp(conf).round(6))

###########################################################
# General Routine to Print a specific data from all columns in the dataset
###########################################################
def listFeaturesData(X,feature_data,name_data):
    data_columns = X.columns
    print(data_columns)
    features_table = pd.DataFrame(data=zip(data_columns,feature_data),columns=['Feature',name_data])
    print("\n")
    print(features_table)
    print("\n")

##################################################################
# General routine to reduce Features for any Classification Model
##################################################################
# Comment : Features the are statistically significant :
# It means that at least in 95% of the times we do a prediction, these feature influences the result
# On the other hand, statiscally non significant features influences the resuly by chance
# The P-Value of a feature statistically significant should be less (< 0.05), since the Hypotesis is that the variable is NOT statistically significant
def reduceFeaturesByModel(model, model_name, X, verbose=True):
    print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Features reduction using model {}\n".format(model_name))

    # List the variables that are and not are statistically significant
    # False = insignificant feature; True = significant fearture
    columns_significance = model.get_support()

    if verbose:
        listFeaturesData(X,columns_significance,"Statistically Significant")
        print(columns_significance)

    data_columns = X.columns
    non_significant_columns = []
    significant_columns = []
    for i in range(len(columns_significance)):
        if columns_significance[i] == False:
            non_significant_columns.append(data_columns[i])
        else:
            significant_columns.append(data_columns[i])
    if verbose:
        print("\nColumns to be eliminated using {}".format(model_name))
        print(non_significant_columns)

    num_columns_kept = len(columns_significance)-len(non_significant_columns)
    if verbose:
        print("\nNumber of Columnns to be KEPT after eliminating features using {} : {}".format(model_name,num_columns_kept))
        print("Number of Columnns to be REMOVED using {} : {}".format(model_name,len(non_significant_columns)))
    return significant_columns

def reduceFeaturesbyLogisticaBinomial(X, y, model_name, verbose=True, print_GLM_summary = True):
    import statsmodels.api as sm

    print("-I- Begining Logistic Binomial Model Fitting")
    print(datetime.now())
    model_logistic_binomial = sm.GLM(y.to_numpy(), X, family=sm.families.Binomial())
    result=model_logistic_binomial.fit(fit_intercept=True)
    print(datetime.now())
    print("-I- Ending Logistic Binomial Model Fitting")
    summary = result.summary2()
    print("\nSummary of Logistic Binomial Regression Analysis (GLM function) for the Full Dataset")
    if verbose and print_GLM_summary:
        print(summary)
    # Comment : Features the are statistically significant :
    # It means that at least in 95% of the times we do a prediction, these feature influences the result
    # On the other hand, statiscally non significant features influences the resuly by chance
    # The P-Value of a feature statistically significant should be less (< 0.05), since the Hypotesis is that the variable is NOT statistically significant
    # By Using the Summary, all variables with P-Value larger than 0.05 should be eliminate

    # List the features / columns that are not statistically significant
    data_columns = X.columns
    non_significant_columns = []
    significant_columns = []
    for i in range(len(data_columns)):
        if result.pvalues[i] > 0.05:
            non_significant_columns.append(data_columns[i])
        else:
            significant_columns.append(data_columns[i])

    if verbose:
        print("\nColumns to be eliminated using {}".format(model_name))
        print(non_significant_columns)

    num_columns_kept = len(data_columns)-len(non_significant_columns)
    if verbose:
        print("\nNumber of Columnns to be KEPT after eliminating features using {} : {}".format(model_name,num_columns_kept))
        print("Number of Columnns to be REMOVED using {} : {}".format(model_name,len(non_significant_columns)))

    return significant_columns

def evaluateLinearRegressionDataset(X_dataset, y_dataset):
    from sklearn.linear_model import LogisticRegression

    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size=0.4, random_state=42)
    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(X_train, y_train)
    accuracy = logreg.score(X_test, y_test)
    return accuracy

def reduceFeatures(X_original, y_original, scalableCols, verbose=True):
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LogisticRegression

    # For numeric variables will be applied transformation

    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    scalledCols = [col for col in scalableCols if col in X_original.columns ]
    X_scaled = X_original.copy()
    X_scaled.loc[:,scalledCols] = scaler.fit_transform(X_scaled.loc[:,scalledCols])

    # The dataframe that will store the evaluation results
    datasets_models = pd.DataFrame(columns = ['Model_Name','Columns_Kept','Model_Dataset_Accuracy'])

    #### Performing Feature Selection

    #### OPTION 1 - Using Logistic Regression
    model_logistic_regression = SelectFromModel(estimator=LogisticRegression(max_iter=5000)).fit(X_scaled, y_original)
    columns_kept = reduceFeaturesByModel(model_logistic_regression,
                                                    "Logistic Regression", X_scaled, verbose=verbose)
    datasets_models = datasets_models.append({'Model_Name' : "Logistic Regression",
                                                'Columns_Kept' : columns_kept, 'Num_Columns_Kept' : len(columns_kept)},
                                                ignore_index=True)
    # List the abs(coefficient) value for feature significance
    # The fatures above the abs(thersholds) are not statistically significant
    # It means that it cannot be guaranteed that this variables will impact the prediction in at least 95% of the cases
    if verbose:
        print("\nLogistic Threshold used to remove statistically significant feature :")
        # Cutoff abs(coefficient) value for feature significance
        print(model_logistic_regression.threshold_)
        # List the coeficients of each one of the variables used to predict
        print("\nLogistic Coeficient for each feature:")
        listFeaturesData(X_scaled,model_logistic_regression.estimator_.coef_[0], "Feature Coeficient")

    # Logistic Coeficient Interpretation
    # 1) Sign of the Coeficient
    # A positive sign means that, all else being equal, it is more likely to have positive outcome
    # A negative sign means that, all else being equal, it is less likely to have positive outcome
    # 2) Magnitude
    # If everything is a very similar magnitude, a larger pos/neg coefficient means larger effect, all things being equal.
    # A feature with 0.2 Coeficient impacts in the positive outcome twice as another feature with Coeficient equal 0.1
    # However, if your data isn't normalized, the magnitude of the coefficients don't mean anything (without context).
    # For instance you could get different coefficients by changing the units of measure to be larger or smaller.
    # So keep in mind that logistic regression coefficients are in fact odds ratios,
    # and you need to transform them to probabilities to get something more directly interpretable.
    # By calculating the Coeficient Threshold as X, it means that the Features with impact less than X will be elimiated by the model

    #### Performing Feature Selection
    #### OPTION 2 - Using Linear Support Vector Classification
    from sklearn.svm import LinearSVC
    from sklearn.feature_selection import SelectFromModel

    lsvc = LinearSVC(C=0.02, penalty="l1", dual=False).fit(X_scaled, y_original)
    model_LSVC = SelectFromModel(lsvc, prefit=True)
    model_name = "LinearSVC"
    columns_kept = reduceFeaturesByModel(model_LSVC, model_name, X_scaled, verbose=verbose)
    datasets_models = datasets_models.append({'Model_Name' : model_name,
                               'Columns_Kept' : columns_kept, 'Num_Columns_Kept' : len(columns_kept) },ignore_index=True)

    #### Performing Feature Selection
    #### OPTION 3 - Using SelectKBest and Mutual Info Classification
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import SelectKBest

    # # As parameter for KBest we will use half of the features
    model_KBest_mutual = SelectKBest(score_func=mutual_info_classif,k=int(len(X_scaled.columns)/2))
    results = model_KBest_mutual.fit(X_scaled,y_original)
    model_name = "SelectKBest / Mutual Info Classification"
    columns_kept = reduceFeaturesByModel(model_KBest_mutual, model_name , X_scaled, verbose=verbose)
    datasets_models = datasets_models.append({'Model_Name' : model_name,
                               'Columns_Kept' : columns_kept, 'Num_Columns_Kept' : len(columns_kept) },ignore_index=True)

    if verbose:
        print("\nKBest scores for each feature:")
        listFeaturesData(X_scaled,results.scores_, "Feature Score")
    # score_func=mutual_info_classif Interpretation
    # The mutual information (MI) between two random variables or random vectors
    # measures the “amount of information”, i.e. the “loss of uncertainty” that one can
    # bring to the knowledge of the other, and vice versa.

    #### Performing Feature Selection
    #### OPTION 4 - RUN THE MODEL WITH ALL FEATURES AND SELECT ONLY SIGNIFICANT ONES
    model_name = "Logistic Binomial Model"
    columns_kept = reduceFeaturesbyLogisticaBinomial(X_scaled, y_original, model_name, print_GLM_summary = True)
    datasets_models = datasets_models.append({'Model_Name' : model_name,
                                 'Columns_Kept' : columns_kept, 'Num_Columns_Kept' : len(columns_kept) },ignore_index=True)

    #############################################################
    # Evaluate the best Logistic Regression performance dataset
    #############################################################
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    from sklearn.model_selection import train_test_split

    for i, datasets_models_row in datasets_models.iterrows():
        accuracy = evaluateLinearRegressionDataset(X_scaled.loc[:,datasets_models_row['Columns_Kept']],
                                                        y_original)
        datasets_models.loc[i,'Model_Dataset_Accuracy'] = accuracy

    if verbose:
        print("\nAccuracy of Linear Regression Classifiers used to Reduce Features from Dataset:")
        print(datasets_models[['Model_Name','Num_Columns_Kept','Model_Dataset_Accuracy']])
    datasets_models = datasets_models.sort_values(by=['Model_Dataset_Accuracy'], ascending=False).reset_index()

    if verbose:
        print("\nModel used to reduce Dataset features : {}".format(datasets_models.iloc[0]['Model_Name']))

    return datasets_models.iloc[0]['Columns_Kept'], datasets_models.iloc[0]['Model_Name']

LogisticRegressionAnalysis(X_dataset, y_dataset, scalableCols)

# print(X_dataset.head(20))
# print(y_dataset.head(20))
# print(X_dataset.columns)

columns_kept, model_name = reduceFeatures(X_dataset,y_dataset,scalableCols)
X_dataset = X_dataset.loc[:,columns_kept]

print("\nColumns kept in the dataset after Features reduction using {}:".format(model_name))
print(X_dataset.columns)
print("\nDataset shape after Features reduction using {}:".format(model_name))
print(X_dataset.shape)

# print(X_dataset.head(20))
# print(y_dataset.head(20))

#######################################################
# Show the Confusion Matrix HeatMap
#######################################################
def showConfusionMatrixGraph(confusion_matrix, y_test):
    # Graph Confusion Matrix
    import seaborn as sns

    df_cm = pd.DataFrame(confusion_matrix, columns=np.unique(y_test), index = np.unique(y_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (5,5))
    sns.set(font_scale=1.2)
    sns.heatmap(df_cm, annot=True,annot_kws={"size": 12}, cbar=False, square=True, fmt="d", cmap="Reds")
    # sns.heatmap(df_cm , annot=True,annot_kws={"size": 12}, cbar=False, vmax=500, square=True, fmt="d", cmap="Reds")
    plt.savefig('Confusion')
    # plt.show()

#######################################################
# Show the ROC Graph
#######################################################
def show_ROC_Graph(X_test, y_test, model):
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt

    logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('ROC')
    # plt.show()

################################
# Starting Training / Testing
################################

# Removing outliers from the dataset
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(X_dataset)
mask = yhat != -1
X_dataset = X_dataset[mask]
y_dataset = y_dataset[mask]

# Removing outliers from the training set - Option2
# from sklearn.covariance import EllipticEnvelope
# ee = EllipticEnvelope(contamination=0.1)
# yhat = ee.fit_predict(X_dataset)
# mask = yhat != -1
# X_dataset = X_dataset[mask]
# y_dataset = y_dataset[mask]

# For numeric variables will be applied transformation
# RobustScaler was the best result one to this dataset / model
# RobustScaler is suitable to reduce the impact if outliers on the predictions
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
# scaler = MinMaxScaler(feature_range=(-1,1))
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = RobustScaler() # Better so far

scalledCols = [col for col in scalableCols if col in X_dataset.columns ]
X_dataset.loc[:,scalledCols] = scaler.fit_transform(X_dataset.loc[:,scalledCols])

X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size=0.4, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, log_loss

classifiers_names = [
    "Logistic Regression", # 0
    "Nearest Neighbors", # 1
    "Linear SVM", # 2
    # "RBF SVM", # 3 # NOT feasible with 50,000 rows
    # "MPL Neural Net", # 4 # NOT feasible with 50,000 rows
    "Decision Tree", # 5
    "Naive Bayes (GaussianNB)", # 6
    "Random Forest", # 7
    "Bagging Classifier", # 8
    "AdaBoost", # 9
    "XGBoost", # 10
    # "Gaussian Process", # 11 # NOT feasible with 50,000 rows
    # "QDA" # 12 # Performed really poorly with 50,000 rows
]

classifiers = [
    LogisticRegression(),
    # LogisticRegression(class_weight = 'balanced'),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability=True),
    # SVC(gamma=2, C=1, probability=True),
    # MLPClassifier(alpha=0.001, solver='lbfgs', learning_rate='adaptive', max_iter=1000), # Neuro Net
    DecisionTreeClassifier(max_depth=5),
    GaussianNB(),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    BaggingClassifier(),
    AdaBoostClassifier(),
    XGBClassifier(),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    # QuadraticDiscriminantAnalysis()
]

# For test purposes
# i = 0
# classifiers =  [ classifiers[i] ]
# classifiers_names = [ classifiers_names[i] ]

scores = []

print("\nStarting Classifiers Evaluation Pipeline")
for classifier in classifiers:
    print("\n-----------------------------------------------------------------------------------")
    print("-I- Begining Classifier")
    # print(datetime.now())
    print("Training using Classifier {}".format(classifier))
    pipe = Pipeline(steps=[
                      ('classifier', classifier)])
    pipe.fit(X_train, y_train)
    X_train.to_csv("validation.csv")
    # print(X_train.head(20))
    # print(y_train.head(20))
    score = pipe.score(X_test, y_test)
    scores.append(score)
    print("model score: %.3f" % score)
    # print(datetime.now())
    print("-E- Ending Classifier")
    print("\n-----------------------------------------------------------------------------------")

#end of pipeline
scores_df = pd.DataFrame(zip(classifiers_names,classifiers, scores), columns=['Classifier_Name', 'Classifier','AccuracyScore'])
scores_df.sort_values(by=['AccuracyScore'], ascending=False, inplace=True)
print("")
print(scores_df)

# Select the Model with best accuracy
model_name = scores_df.iloc[0]['Classifier_Name']
print("\nSelected Model is {}".format(model_name))
model_final = scores_df.iloc[0]['Classifier']

# Make predictions on Test dataset
print(model_final)
y_predicted = model_final.predict(X_test)

# Confusion MATRIX
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_predicted)

print("\nConfusion Matrix of Testing using {}".format(model_name))
print(confusion_matrix)

showConfusionMatrixGraph(confusion_matrix,y_test)

from sklearn.metrics import classification_report
print("\nClassification Report of Training using {}".format(model_name))
print(classification_report(y_test, y_predicted))

show_ROC_Graph(X_test, y_test, model_final)

prob = model_final.predict_proba(X_test)

print(type(X_test))
X_test.insert(0,'y_pred',0)
X_test.loc[:,['y_pred']] =  y_predicted
X_test.insert(1,'pred_prob', 0.0)
X_test.loc[:,['pred_prob']] = prob[:,1]
X_test.loc[:,scalledCols] = scaler.inverse_transform(X_test.loc[:,scalledCols])

X_test.to_csv("test_result.csv")

# Out of Sample Predictions

import statistics

# print(columns_kept)

out_sample_original_case =  {'AMT_INCOME_TOTAL' : [180000],
                    'AMT_CREDIT' : [540000],
                    'AMT_ANNUITY' : [27000],
                    'REGION_POPULATION_RELATIVE': [0.02461],
                    'DAYS_BIRTH': [15326],
                    'DAYS_EMPLOYED' : [1038],
                    'FLAG_EMP_PHONE' : [1],
                    'REGION_RATING_CLIENT_W_CITY':[2],
                    'DAYS_LAST_PHONE_CHANGE' : [429],
                    'AVG_CREDIT_SCORE' : [statistics.mean([0.372110259,0.50648424])],
                    'FLAG_OWN_CAR_Y': [0]
                     }

out_sample_cases = pd.DataFrame.from_dict(out_sample_original_case)

new_case = out_sample_cases.iloc[0]
new_case['DAYS_EMPLOYED'] = 10000
out_sample_cases = out_sample_cases.append(new_case, ignore_index=True)

new_case = out_sample_cases.iloc[0]
new_case['AVG_CREDIT_SCORE'] = 0.9
out_sample_cases = out_sample_cases.append(new_case, ignore_index=True)

# pd.set_option('display.max_columns', 11)
# print(out_sample_cases)

out_sample_cases.loc[:,scalledCols] = scaler.transform(out_sample_cases.loc[:,scalledCols])
case_predicted = model_final.predict(out_sample_cases)
case_probs = model_final.predict_proba(out_sample_cases)
out_sample_cases['Default Predicted'] = case_predicted
# To be fixed below !!!!
out_sample_cases['Prob'] = case_probs[:,1]

print(out_sample_cases)

out_sample_cases.loc[:,scalledCols] = scaler.inverse_transform(out_sample_cases.loc[:,scalledCols])
out_sample_cases.to_csv("out_of_sample_predictions.csv")

print(datetime.now())
