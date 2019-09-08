# adapt this to run

# 1. write a function to take a list or dictionary of clfs and hypers ie use logistic regression, each with 3 different sets of hyper parameters for each
# 2. expand to include larger number of classifiers and hyperparmater settings
# 3. find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Collaborate to get things
# 7. Investigate grid search function


# TODO -----
# Change to accept regresssion models as well (needs passing of scorer)

# Datasets
from sklearn import datasets

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Train/Test Split
from sklearn.model_selection import train_test_split

# All my code can be found in this local library
from GridSearchCV import GridSearchCV

if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    n_folds = 5

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=101)

    # Support for:
    # (1) Multiple model types and
    # (2) multiple independent grids for each model
    grids = {'LR': {'model': LogisticRegression(solver='lbfgs',
                                             multi_class='auto'),
                    'param_grid': [{'C': [1, 10],
                                    'max_iter': [500, 1000],
                                    'penalty': ['l2']}]},
             'SVM': {'model': SVC(),
                     'param_grid': [{'C': [0.01, 0.1, 1, 10],
                                     'kernel': ['linear']},
                                    {'C': [0.01, 0.1, 1, 10],
                                     'gamma': [0.0001, 0.001, 0.01, 1],
                                     'kernel': ['rbf']}]},
             'RF': {'model': RandomForestClassifier(random_state=101, n_jobs=-1),
                    'param_grid': {'n_estimators': [50, 100],
                                   'max_features': ['auto', 'sqrt'],
                                   'max_depth': [4, 8],
                                   'criterion':['gini', 'entropy']}}
           }

    cv_obj = GridSearchCV(X_train, y_train, grids=grids, n_splits=5,
                          random_state = 101, verbose=1)
    cv_obj.train()
    cv_obj.visualize_all_grids(verbose=1)
    
    # Results
    print("Best Model: {}\n".format(cv_obj.best_model))
    print("Best Model Score (Dev/Holdout score averaged across the folds)  : {:.3}".format(cv_obj.best_score))
        
    # Checking best model accuracy
    from sklearn.metrics import accuracy_score
    pred_train = cv_obj.best_model.predict(X_train)
    pred_test = cv_obj.best_model.predict(X_test)
    print("Best Model Score (Full Training dataset, no cross-validation)   : {:.3}".format(accuracy_score(pred_train, y_train)))
    print("Best Model Score (Test dataset)                                 : {:.3}".format(accuracy_score(pred_test, y_test)))

