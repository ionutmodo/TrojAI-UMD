import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from skopt import BayesSearchCV, gp_minimize
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings("ignore")


def read_features(p_path):
    """
        This method returns the features from a metadata file and its location is specified by the method parameter
        The features from the csv are denoted by columns whose name end in "mean_diff" and "std_diff"
        This method discards other columns and returns numpy array with features (columns ending in "mean_diff" and "std_diff") and labels (clean/backdoored)
    """
    csv_file = pd.read_csv(p_path)
    initial_columns = csv_file.columns
    col_model_label = csv_file['model_label'].copy(deep=True)
    for c in initial_columns:
        if not c.endswith('mean_diff') and not c.endswith('std_diff'):
            del csv_file[c]
    features = csv_file.values
    labels = np.array([int(col_model_label.iloc[i] == 'backdoor') for i in range(len(csv_file))])
    return abs(features), labels


def get_base_classifier():
    """
        This method returns a classifier whose hyperparameters were chosen by HyperParameterOptimization.
        Run again hyper_parameter_optimization and put the optimal parameters into SVC constructor (or use other classifier)
    """
    # return svm.SVC(C=10.5, kernel='rbf', gamma='scale', probability=True)
    return LogisticRegression(C=2.05)


def hyper_parameter_optimization(file):
    """
        This method performs hyper-parameter optimization.
        Draws C from an uniform distribution in interval [0.001, 20]
    """
    X, y = read_features(file)  # clean data is automatically added

    search_space = [
        Real(0.001, 20, 'uniform', name='C'),

        # Real(0.00001, 100.0, 'uniform', name='gamma'),
        Categorical(['scale'], name='gamma'),

        Categorical(['rbf'], name='kernel')  # other kernels: linear, poly, rbf, sigmoid
    ]

    @use_named_args(search_space)
    def evaluate_model(**params):
        """
            This method estimates a hyper-parameter configuration for SVC using RepeatedStratifiedKFold
            Finally, computes the negative-log-loss (CrossEntropy) whose value is negative
        """
        # enable probabilistic version of SVM
        params['probability'] = True # add this here because I don't know the skopt.space for boolean values
        model = svm.SVC()
        model.set_params(**params)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        score = cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring='neg_log_loss')
        estimate = np.mean(score)
        return -estimate # use minus because the estimate will be negative

    result = gp_minimize(evaluate_model, search_space)
    print('Best Score: %.3f' % result.fun)
    print('Best Parameters: %s' % result.x)


def training_data_cross_validation(file):
    n_splits = 10
    n_repeats = 3
    X, y = read_features(file)

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
    scores_roc = cross_val_score(get_base_classifier(), X, y, scoring='roc_auc', cv=cv, n_jobs=-1, error_score='raise')

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
    scores_xent = cross_val_score(get_base_classifier(), X, y, scoring='neg_log_loss', cv=cv, n_jobs=-1, error_score='raise')
    print(f'ROC: mean={np.mean(scores_roc):.3f} std={np.std(scores_roc):.3f}')
    print(f' CE: mean={-np.mean(scores_xent):.3f} std=({np.std(scores_xent):.3f})')
    print()


def main():
    umd_metadata_file = 'round3-train-dataset_square-30-random-rgb-filters.csv'
    # hyper_parameter_optimization(umd_metadata_file)
    training_data_cross_validation(umd_metadata_file)


if __name__ == '__main__':
    main()
