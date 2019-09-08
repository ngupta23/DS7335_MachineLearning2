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

# Base Libraries
import numpy as np
import itertools
from pprint import pprint

# Datasets
from sklearn import datasets

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Resamples
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# Metrics
from sklearn.metrics import accuracy_score

# Plotting
import matplotlib.pyplot as plt

# Misc
from sklearn.base import clone

def run(arModel, data, clf_hyper={}):
    X, y, n_folds = data

    if arModel._estimator_type == 'classifier':
        kf = StratifiedKFold(n_splits=n_folds)
    else:
        kf = KFold(n_splits=n_folds)

    ret = {}

    for idx, (train_index, test_index) in enumerate(kf.split(X, y)):
        model = clone(arModel)
        model.set_params(**clf_hyper)
        model.fit(X[train_index], y[train_index])
        pred = model.predict(X[test_index])
        score = accuracy_score(y[test_index], pred)

        resample_name = 'fold' + str(idx)
        ret[resample_name] = {'model': model,
                              'train_index': train_index,
                              'test_index': test_index,
                              'score': score}
    return ret


# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def format_results_single_grid(arResults,
                               arModelName,
                               arGridNumber=0,
                               verbose=1):
    """
    arResults: Results from Grid Search
    arModelName: Name of the model to plot results for
                 Name should match one of the names defined in the Grid Search
    arGridNumber: Grid Index for the named model. Index starts from 0 (default)
    verbose: 0 (no print statements)
             1 (final results printed)
             2 (detailed results)
    """

    params_all = []  # List all all hyperparameters (each entry is a dictionary)
    scores_all = []  # List of lists; inner lisr consists of individual fold scores for all hyperparameters
    means = []  # List of mean scores for all hyperparameters
    stds = []  # List of std of scores for all hyperparameters

    for hyper_param_id in arResults[arModelName][arGridNumber].keys():
        params = arResults[arModelName][arGridNumber][hyper_param_id]['params']
        if (verbose >= 2):
            print (params)
        
        scores = []  # Scores for the folds for a single set of hyperparameters
        for fold in arResults[arModelName][arGridNumber][hyper_param_id]['results'].keys():
            score = arResults[arModelName][arGridNumber][hyper_param_id]['results'][fold]['score']
            if (verbose >= 2):    
                print("    " + fold + " Score: " + str(score))
            scores.append(score)
        
        # Compute mean and std for this set of hyperparameters
        mean_score = np.mean(scores)  # Mean Score for a single set of hyperparameters
        std_score = np.std(scores)  # Std of Score for a single set of hyperparameters
        
        # Append to global lists
        params_all.append(params)
        scores_all.append(scores)
        means.append(mean_score)
        stds.append(std_score)
    
        if (verbose >= 2):    
            print("Mean Score: {}".format(mean_score))
            print("Std Score: {}".format(std_score))
            
    return (params_all, scores_all, means, stds)

def visualize_single_grid(arResults, arModelName, arGridNumber=0, arFigsize=(20,4), verbose=0):
    """
    arResults: Results from Grid Search
    arModelName: Name of the model to plot results for
                 Name should match one of the names defined in the Grid Search
    arGridNumber: Grid Index for the named model. Index starts from 0 (default)
    verbose: 0 (no print statements)
             1 (final results printed)
             2 (detailed results)
    """
    
    params_all, scores_all, means, stds = format_results_single_grid(
            arResults, arModelName, arGridNumber, verbose)
    
    fig1, axes = plt.subplots(nrows=1, ncols=2, figsize=arFigsize)
    # Box Plot
    axes[0].boxplot(scores_all)
    axes[0].set_title("Model Resample Scores Comparison | Model '{}', Grid {}".format(arModelName, arGridNumber))
    axes[0].set_xlabel('Model Number')
    axes[0].set_ylabel('Score')
    
    # Scatter Plot
    axes[1].scatter(means, stds, alpha=0.5)
    axes[1].set_title("Bias / Variance Analysis | Model '{}', Grid {}".format(arModelName, arGridNumber))
    axes[1].set_xlabel('Mean Resample Scores per Model')
    axes[1].set_ylabel('Std of Resample Scores per Model')
    
    if (verbose >= 1):
        print()
        print("-"*50)
        print("Model '{}', Grid {}".format(arModelName, arGridNumber))
        print("-"*50)
        for index, (mean, std, params) in enumerate(zip(means, stds, params_all)):
            # print("Model %d >>    Score: %0.3f (+/-%0.03f) | Parameters: %r" % (index+1, mean, std * 2, params))
            print("Model {:3} >>  Score: {:>5.3f} (+/-{:>5.3f} 95% CI) | Parameters: {}".format(index+1, mean, std * 2, params))
            

def visualize_all_grids(arResults, arFigsize=(20,4), verbose=1):
    for modelName in arResults.keys():
        for gridNumber in arResults[modelName].keys():
            visualize_single_grid(results, modelName, gridNumber, arFigsize, verbose)

if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    n_folds = 5
    data = (X, y, n_folds)
    
    
    # Support for:
    # (1) Multiple model types and
    # (2) multiple independent grids for each model
    gs = {'LR': {'model': LogisticRegression(solver='lbfgs', multi_class='auto'),
                 'param_grid': [{'C': [1, 10],
                                 'max_iter': [500, 1000],
                                 'penalty': ['l2']}]},
          'SVM': {'model': SVC(),
                  'param_grid': [{'C': [0.01, 0.1, 1, 10],
                                  'kernel': ['linear']},
                                 {'C': [0.01, 0.1, 1, 10],
                                  'gamma': [0.0001, 0.001, 0.01, 1],
                                  'kernel': ['rbf']}]
                  }
          }
    
    verbose = 0  # 0 for no prints, 1 for hypers and score, 2 for detailed
    results = {}
    
    # results{<model_name>,<grid_number>,<hyper_params_id>,'params'}
    #  - holds actual dictionary of parameters
    #  - {'C': 1, 'max_iter': 100, 'penalty': 'l2'}
    # results{<model_name>,<grid_number>,<hyper_params_id>,'results',<fold index>}
    #  - holds results of the folds (resamples)
    
    # Example usage:  results['LR'][0][0]['params']
    # Example usage:  results['LR'][0][0]['results'][0]
    
    for name in gs.keys():
        model = gs[name]['model']
        lst_of_grids = gs[name]['param_grid']
        temp1 = {}
        for i in range(len(lst_of_grids)):
            ind_gs = lst_of_grids[i]
            kwargs_list = list(product_dict(**ind_gs))
            temp2 = {}
            for j in range(len(kwargs_list)):
                if (verbose >= 1):
                    print(kwargs_list[j])
                temp3 = run(model, data, clf_hyper=kwargs_list[j])
                if (verbose >= 1):
                    for fold in temp3.keys():
                        print("{} Score: {}".format(fold, temp3[fold]['score']))
                temp2.update({j: {'params': kwargs_list[j], 'results': temp3}})
            temp1.update({i: temp2})
    
        results.update({name: temp1})
        if (verbose >= 2):
            pprint(results)
    
    visualize_all_grids(results)
