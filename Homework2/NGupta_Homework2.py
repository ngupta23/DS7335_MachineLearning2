# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 05:36:01 2019
@author: Nikhil Gupta
"""

# Basic
import numpy as np
import matplotlib.pyplot as plt

# Preprocessing and Dimensionality Reduction
from sklearn.preprocessing import OneHotEncoder

# Train/Test Split
from sklearn.model_selection import train_test_split

# Model(s)
from sklearn.ensemble import RandomForestClassifier

# Grid Search
from sklearn.model_selection import GridSearchCV

# Should details be printed?
verbose = 1

# Read the data
data = np.genfromtxt('claim.sample.csv',
                     delimiter=",",
                     dtype=None,
                     encoding=None,
                     names=True,
                     autostrip=True)


###################
#### Problem 1 ####
###################

"""
1. J-codes are procedure codes that start with the letter 'J'.
     A. Find the number of claim lines that have J-codes.
     B. How much was paid for J-codes to providers for 'in network' claims?
     C. What are the top five J-codes based on the payment to providers?
"""


## Problem 1A
# First entry is " for some reason so we are
# checking for J in the second entry (index = 1)

ans1a = sum(np.char.find(data['ProcedureCode'], "J") == 1)
print("1A >> The number of claim lines that start with J-codes = {}\n".format(ans1a))


## Problem 1B
data_subset = data[(data['InOutOfNetwork'] == '"I"') &
                   (np.char.find(data['ProcedureCode'], "J") == 1)]
ans1b = round(sum(data_subset['ProviderPaymentAmount']), 2)
print("1B >> Amount paid for J-codes to providers for 'In Network' claims = ${}\n".format(ans1b))


## Problem 1C
data_subset = data[np.char.find(data['ProcedureCode'], "J") == 1]

# https://stackoverflow.com/questions/50950231/group-by-with-numpy-mean
sum_dict = {}
for i in np.unique(data_subset['ProcedureCode']):
    tmp = data_subset[np.where(data_subset['ProcedureCode'] == i)]
    sum_dict[i] = np.sum(tmp['ProviderPaymentAmount'])

ans1c_all = sorted(sum_dict.items(),
                   key=lambda x: x[1],
                   reverse=True)
topx = 5

print("1C >> Top {} J Codes based on payment to providers: \n".format(topx))
for entry in ans1c_all[0:topx]:
    print("J-Code: {}, Payment: ${}".format(entry[0], round(entry[1],2)))

print("\n", "-"*150, "\n")

###################
#### Problem 2 ####
###################

"""
For the following exercises, determine the number of providers that were paid for at least one J-code. Use the J-code claims for these providers to complete the following exercises.
    A. Create a scatter plot that displays the number of unpaid claims (lines where the Provider.Payment.Amount field is equal to zero) for each provider versus the number of paid claims.
    B. What insights can you suggest from the graph?
    C. Based on the graph, is the behavior of any of the providers concerning? Explain.
"""

## Data Prep
# Those that got paid at least 1  J Code claim
provider_subset = data[(np.char.find(data['ProcedureCode'], "J") == 1) &
                       (data['ProviderPaymentAmount'] > 0)]
unique_providers_subset = list(set(provider_subset['ProviderID']))

if (verbose >= 2):
    print("Unique Providers >> {}".format(unique_providers_subset))
    
data_subset_final = data[(np.isin(data['ProviderID'],
                                  unique_providers_subset)) &
                         (np.char.find(data['ProcedureCode'], "J") == 1)]

## Problem 2A

print ("2A >>")
# Step 1: Get required data ----
paid_unpaid_array = np.empty(shape=(0, 3))

for provider in np.unique(data_subset_final['ProviderID']):
    tmp = data_subset_final[np.where(data_subset_final['ProviderID'] == provider)]
    paid_unpaid_array = np.append(paid_unpaid_array,
                                  [provider,
                                   np.sum(tmp['ProviderPaymentAmount'] > 0),
                                   np.sum(tmp['ProviderPaymentAmount'] == 0)]
                                  )
# Comes out flat, so need to reshape
paid_unpaid_array = paid_unpaid_array.reshape(len(unique_providers_subset),3)

if (verbose > 2):
    print(paid_unpaid_array)

xs = paid_unpaid_array[:,1].astype(int)
ys = paid_unpaid_array[:,2].astype(int)

# Step 2: Plot the data ----
fig1, ax = plt.subplots(figsize = (10,6), nrows=1, ncols=1)

# https://stackoverflow.com/questions/12236566/setting-different-color-for-each-series-in-scatter-plot-on-matplotlib
import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
for i, (y, c) in enumerate(zip(ys, colors)):
    ax.scatter(xs[i], ys[i], color=c)
    ax.text(xs[i]+1, ys[i]+1, paid_unpaid_array[i,0].astype(str), fontsize=12)

ax.set_title("Unpaid Claims vs. Paid Claims")
ax.set_xlabel('Paid Claims')
ax.set_ylabel('Unpaid Claims')
plt.show()

## Problem 2B
print("\n2B >> INSIGHTS: As the number of paid claims increases, the number of unpaid claims increase as well\n")

## Problem 2C
print("2C >> Based on this graph, the behavior of a couple of providers seems concerning.",
      "These providers (FA0001387001 and FA0001389001) have very few paid claims and an unusual number of unpaid claims.",
      "Based on other providers, this seems to be unusually high number of unpaid claims for the given number of paid claims.",
      "This might warrant further investigation.\n")

print("\n", "-"*150, "\n")

###################
#### Problem 3 ####
###################

"""
3. Consider all claim lines with a J-code.
     A. What percentage of J-code claim lines were unpaid?
     B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.
     C. How accurate is your model at predicting unpaid claims?
     D. What data attributes are predominately influencing the rate of non-payment?
"""

## Data Prep
data_subset = data[(np.char.find(data['ProcedureCode'], "J") == 1)]

## Problem 3A
percent_unpaid = round(sum(data_subset['ProviderPaymentAmount'] == 0) / data_subset.shape[0] * 100, 2)
print("3A >> Percentage of unpaid J-code claims = {}%\n".format(percent_unpaid))

## Problem 3B

# Remove unnecessary columns 

# >> Index column V1, since these should not be included in the machine learning algorithm
# >> Claim Number, since future claim numbers will not be the same as what was seen in the past
# >> Claim.Line.Number. This should be removed since the line item number is irrelevant and 
#    the entries could as well be swapped within a claim. Rather this should be converted into 
#    a new feature which captures the number of line items in a claim since this could have a 
#    correlation to the outcome. This has been left as an exercide for the future.
# >> Denial Code Reason: We would not know this field in production (since we are predicting if the claim will be paid or unpaid)
# >> Member ID: There could be new members in the future in which case we wont be able to predict if we include member ID
#    If we were only looking to see how current members influence non patment, then we can keep this field in the feature set.
# >> Claim Current Status: This wont be known in the future at the time of prediction
#    and hence should be removed

# >> Label: Create from ProviderPaymentAmount and remove that column from features

data_JCodes = data[np.char.find(data['ProcedureCode'], "J") == 1]

y = data_JCodes['ProviderPaymentAmount'] > 0

# https://stackoverflow.com/questions/15575878/how-do-you-remove-a-column-from-a-structured-numpy-array
def remove_field_name(a, cols_to_delete):
    names = list(a.dtype.names)
    cols_to_delete = list(cols_to_delete)
    
    tokeep = np.setdiff1d(names, cols_to_delete, assume_unique = True)
    return(a[tokeep])

X = remove_field_name(data_JCodes, ["V1",
                                    "ClaimNumber",
                                    "DenialReasonCode",
                                    "ClaimCurrentStatus",
                                    "MemberID",
                                    "ProviderPaymentAmount"
                                    ])

# Separate Categorical and Numeric Features
cat_features = []
numeric_features = []

for name in X.dtype.names:
    if (X.dtype.fields[name][0].kind in ['U', 'S','a']):
        cat_features.append(name)
        
    if (X.dtype.fields[name][0].kind in ['i', 'u','f']):
        numeric_features.append(name)

print ("3B >>")
print("\nPRECHECKS ----\n")
# Check to make sure there is no mistake
print("Checking to make sure there is no mistake in the categorization of numeric and categorical features")
print("Categorical Features to be used in the model: ", str(cat_features))
print("Numeric Features to be used in the model: ", str(numeric_features), "\n")

cat_array = np.array(data_JCodes[cat_features].tolist())
numeric_array = np.array(data_JCodes[numeric_features].tolist())

# Run the OneHotEncoder
ohe = OneHotEncoder(sparse=False)
cat_array_ohe = ohe.fit_transform(cat_array)

# Get all categorical feature names after One Hot Encoding
cat_feat_names_ohe = []
for name in ohe.get_feature_names():
    first_split = name.split('_')
    index = first_split[0].split('x')[1]
    pre = cat_features[int(index)]
    cat_feat_names_ohe.append(pre + "_" + first_split[1])
    
# Get all feature names after One Hot Encoding
# Needed later for Feature Importance
all_feat_names = numeric_features + cat_feat_names_ohe

# Concatenate Numeric and Categorical Features
X = np.concatenate((numeric_array, cat_array_ohe), axis=1)

print("After One Hot Encoding, we have " + str(X.shape[0]) + 
      " observations and " + str(X.shape[1]) + " columns.\n")

# Split into Train and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=101)

rf = RandomForestClassifier(random_state=101, n_jobs=-1) 
parameters = {'n_estimators': [50, 100],
              'max_features': ['auto'],
              'max_depth': [4, 8],
              'criterion':['gini']}

print("Performing Grid Search to find best model parameters...")
model = GridSearchCV(rf, parameters, cv=5, n_jobs=-1, verbose=1) 
model.fit(X=X_train, y=y_train)

# Results
print("\nBest Model: {}\n".format(model.best_params_))


# Explanation of modeling approach
print("For this example. I choose to use the Random Forest Classifier. Random Forest was chosen since ",
      "there were a lot of missing values in the data and Random Forest can deal with missing data effectively. ",
      "Many other classification algorithms would have required us to impute the data which could have led to issues.\n")

print("One thing to be careful of is the fact that we have a lot of features after one hot encoding which could lead to overfitting.",
      "Next, I will check the model accuracy across train and test set to make sure that the model is not overfitting.\n")

## Problem 3C

# Checking best model accuracy
print("3C >>")
print("Best Model Score (Dev/Holdout score averaged across the folds)  : {:.3}".format(model.best_score_))
from sklearn.metrics import accuracy_score
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
print("Best Model Score (Full Training dataset, no cross-validation)   : {:.3}".format(accuracy_score(pred_train, y_train)))
print("Best Model Score (Test dataset)                                 : {:.3}".format(accuracy_score(pred_test, y_test)))

print("Since the test error is close to the train error, we can be sure that the model is not overfitting.\n")

## Problem 3D

# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

importances = model.best_estimator_.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.best_estimator_.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

print ("3D >>")

# Print the feature ranking
print("\nFeature ranking:")

# Showing only top 10 features
num_feat_to_plot = 10
top_x_features = []
print("The following attributes predominantly influence the rate of non-payment")
for f in range(num_feat_to_plot):  
    print("{}. {} ({})".format(f + 1, all_feat_names[indices[f]], round(importances[indices[f]],4)))
    top_x_features.append(all_feat_names[indices[f]])

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(num_feat_to_plot), importances[indices][0:num_feat_to_plot],
        color="r", yerr=std[indices][0:num_feat_to_plot], align="center")
plt.xticks(range(num_feat_to_plot), top_x_features, rotation='vertical')
plt.xlim([-1, num_feat_to_plot])
plt.show()
