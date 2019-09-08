# Base Libraries
import numpy as np
import itertools
from pprint import pprint

# Resamples
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# Metrics
from sklearn.metrics import accuracy_score

# Plotting
import matplotlib.pyplot as plt

# Misc
from sklearn.base import clone

class GridSearchCV:
    
    def __init__(self, X, y, grids, n_splits=1, maximize=True,
                 random_state=None, verbose=0):
        """
        Arguments:
            X: Training Features
            y: Training Labels
            grids: Support for:
                    (1) Multiple model types and
                    (2) multiple independent grids for each model
                   param_grid can be a single dictionary (e.g. RF below) or a 
                   list of dictionaries (e.g. LR below). If it is a list, then
                   each dictionary in the list is an independent grid search
                    
                   Example
                   gs = {'LR': {'model': LogisticRegression(solver='lbfgs',
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
                         'RF': {'model': RandomForestClassifier(random_state=101),
                                'param_grid': {'n_estimators': [200, 500],
                                               'max_features': ['auto', 'sqrt', 'log2'],
                                               'max_depth': [4, 8],
                                               'criterion':['gini', 'entropy']}}
                       }
            n_splits: Number of CV folds to use.
            maximize: If True, pick the best model based on maximum value of 
                      the scoring metric, else pick based on the minimum value
            random_state: seed value used internally
            verbose = 0  # 0 for no prints, 1 for hypers, 2 for scores, 3 for detailed (debug)

        Outputs:
            results{<model_name>,<grid_number>,<hyper_params_id>,'params'}
              - holds actual dictionary of parameters
              - {'C': 1, 'max_iter': 100, 'penalty': 'l2'}
            results{<model_name>,<grid_number>,<hyper_params_id>,'results',<fold index>}
              - holds results of the folds (resamples)
    
            Example usage:  results['LR'][0][0]['params']
            Example usage:  results['LR'][0][0]['results'][0]
        """
        self.X = X
        self.y = y
        self.grids = grids
        self.n_splits = n_splits
        self.maximize = maximize
        self.random_state = random_state
        self.verbose = verbose
        self.results = {}
        self.best_model = None
        if self.maximize is True:
            self.best_score = -1000000000
        elif self.maximize is False:
            self.best_score = 1000000000
        else:
            raise TypeError("Argument 'maximize' must be a boolean")


    def train(self, verbose=None):
        loVerbose = self.__set_verbosity(verbose)
        
        for name in self.grids.keys():
            model = self.grids[name]['model']
            grids = self.grids[name]['param_grid']
            lst_of_grids = []
            if type(grids) == dict:
                lst_of_grids.append(grids)
            elif type(grids) == list:
                lst_of_grids = grids
            else:
                raise Exception("grids must be either a list of dictionaries or a single dictionary")
            
            temp1 = {}
            for i in range(len(lst_of_grids)):
                ind_gs = lst_of_grids[i]
                kwargs_list = list(self.__product_dict(**ind_gs))
                temp2 = {}
                for j in range(len(kwargs_list)):
                    if (loVerbose >= 1):
                        print("Training Model: {} | ".format(name) + str(kwargs_list[j]))
                    temp3 = self.run(model, clf_hyper=kwargs_list[j])
                    
                    # Compute Mean Model Score
                    loScore = 0
                    for fold in temp3.keys():
                        if (loVerbose >= 2):
                            print("    {} Score: {}".format(fold, temp3[fold]['score']))
                        
                        loScore = loScore + temp3[fold]['score']
                    loScore = loScore/len(temp3.keys())  # Divide by the number of folds
                    
                    # Update best model
                    if self.maximize is True:
                        if loScore > self.best_score:
                            self.best_model = model.set_params(**kwargs_list[j])
                            self.best_score = loScore
                    elif self.maximize is False:
                        if loScore < self.best_score:
                            self.best_model = model.set_params(**kwargs_list[j])
                            self.best_score = loScore
                    
                    temp2.update({j: {'params': kwargs_list[j], 'results': temp3}})
                temp1.update({i: temp2})
            self.results.update({name: temp1})
        
        # Retraining the best model
        if (loVerbose >= 1):
            print ("Retraining the best model with the full training dataset...\n")
        self.best_model.fit(self.X, self.y)
        
        if (loVerbose >= 3):
            pprint(self.results)


    def run(self, arModel, clf_hyper={}):
        """
        Trains a single model from the Grid Search
        """
        if arModel._estimator_type == 'classifier':
            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        else:
            kf = KFold(n_splits=self.n_splits, random_state=self.random_state)
    
        ret = {}
    
        for idx, (train_index, test_index) in enumerate(kf.split(self.X, self.y)):
            model = clone(arModel)
            model.set_params(**clf_hyper)
            model.fit(self.X[train_index], self.y[train_index])
            pred = model.predict(self.X[test_index])
            score = accuracy_score(self.y[test_index], pred)
    
            resample_name = 'fold' + str(idx)
            ret[resample_name] = {'model': model,
                                  'train_index': train_index,
                                  'test_index': test_index,
                                  'score': score}
        return ret


    def format_results_single_grid(self,
                                   arModelName,
                                   arGridNumber=0,
                                   verbose=None):
        """
        arModelName: Name of the model to plot results for
                     Name should match one of the names defined in the Grid Search
        arGridNumber: Grid Index for the named model. Index starts from 0 (default)
        verbose: 0 (no print statements)
                 1 (final results printed)
                 2 (detailed results)
        """
        loVerbose = self.__set_verbosity(verbose)

        params_all = []  # List all all hyperparameters (each entry is a dictionary)
        scores_all = []  # List of lists; inner lisr consists of individual fold scores for all hyperparameters
        means = []  # List of mean scores for all hyperparameters
        stds = []  # List of std of scores for all hyperparameters
    
        for hyper_param_id in self.results[arModelName][arGridNumber].keys():
            params = self.results[arModelName][arGridNumber][hyper_param_id]['params']
            if (loVerbose >= 2):
                print(params)
    
            scores = []  # Scores for the folds for a single set of hyperparameters
            for fold in self.results[arModelName][arGridNumber][hyper_param_id]['results'].keys():
                score = self.results[arModelName][arGridNumber][hyper_param_id]['results'][fold]['score']
                if (loVerbose >= 2):    
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
    
            if (loVerbose >= 2):
                print("Mean Score: {}".format(mean_score))
                print("Std Score: {}".format(std_score))
    
        return (params_all, scores_all, means, stds)


    def visualize_single_grid(self,
                              arModelName,
                              arGridNumber=0,
                              arFigsize=(20, 4),
                              verbose=None):
        """
        arModelName: Name of the model to plot results for
                     Name should match one of the names defined in the Grid Search
        arGridNumber: Grid Index for the named model. Index starts from 0 (default)
        verbose: 0 (no print statements)
                 1 (final results printed)
                 2 (detailed results)
        """
        loVerbose = self.__set_verbosity(verbose)

        params_all, scores_all, means, stds = self.format_results_single_grid(
                arModelName, arGridNumber, verbose)
    
        fig1, axes = plt.subplots(nrows=1, ncols=2, figsize=arFigsize)
        # Box Plot
        axes[0].boxplot(scores_all)
        axes[0].set_title(
                "Model Resample Scores Comparison | Model '{}', Grid {}".format(
                        arModelName, arGridNumber))
        axes[0].set_xlabel('Model Number')
        axes[0].set_ylabel('Score')
    
        # Scatter Plot
        axes[1].scatter(means, stds, alpha=0.5)
        axes[1].set_title(
                "Bias / Variance Analysis | Model '{}', Grid {}".format(
                        arModelName, arGridNumber))
        axes[1].set_xlabel('Mean Resample Scores per Model')
        axes[1].set_ylabel('Std of Resample Scores per Model')
    
        if (loVerbose >= 1):
            print("-"*50)
            print("Model '{}', Grid {}".format(arModelName, arGridNumber))
            print("-"*50)
            for index, (mean, std, params) in enumerate(zip(means, stds, params_all)):
                print("Model {:3} >>  Score: {:>5.3f} (+/-{:>5.3f} 95% CI) | Parameters: {}".format(index+1, mean, std * 2, params))
            print()
        
    
    
    def visualize_all_grids(self, arFigsize=(20, 4), verbose=None):
        loVerbose = self.__set_verbosity(verbose)
        
        for modelName in self.results.keys():
            for gridNumber in self.results[modelName].keys():
                self.visualize_single_grid(modelName,
                                           gridNumber,
                                           arFigsize,
                                           loVerbose)
                
    
    # https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    def __product_dict(self, **kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))
            
    def __set_verbosity(self, arVerbose):
        if arVerbose is None:
            rvVerbose = self.verbose
        else:
            rvVerbose = arVerbose
        return(rvVerbose)


