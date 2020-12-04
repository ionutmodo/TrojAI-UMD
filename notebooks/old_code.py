def HPO():
    # path_csv = r'confusion-reports\ics_svm\round2-train-dataset\round2-train-dataset_square-25-filters_all-classes_gray.csv'
    path_csv = r'confusion-reports\ics_svm\round3-train-dataset\round3-train-dataset_square-30-filters_all-classes_gray.csv'

    trigger_type_aux_str = None
    if 'confusion-matrix' in path_csv:
        print('Approach: confusion matrix and original CNN')
        X, y = read_features_confusion_matrix(path_csv, trigger_type_aux_str)
    else:
        print('Approach: confusion distribution and SDNs')
        X, y = read_features(path_csv, trigger_type_aux_str) # clean data is automatically added
    print(X.shape, y.shape)

    search_space = list()
    ## LogisticRegression params
    search_space.append(Real(0.001, 100.0, 'log-uniform', name='C'))

    ## SVM params
    # search_space.append(Real(0.00001, 100.0, 'log-uniform', name='C'))
    # # search_space.append(Integer(1, 5, name='degree'))
    # # search_space.append(Real(0.00001, 100.0, 'log-uniform', name='gamma'))
    # search_space.append(Categorical(['scale'], name='gamma'))
    # # search_space.append(Categorical(['rbf'], name='kernel'))
    # search_space.append(Categorical(['rbf'], name='kernel')) # linear, poly, rbf, sigmoid

    @use_named_args(search_space)
    def evaluate_model(**params):
        model = LogisticRegression()
        model.set_params(**params)
    #     params['probability'] = True
    #     model = svm.SVC()
    #     model.set_params(**params)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        result = cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring='neg_log_loss')
        estimate = np.mean(result)
        return abs(estimate)

    result = gp_minimize(evaluate_model, search_space)
    print('Best Score: %.3f' % (result.fun))
    print('Best Parameters: %s' % (result.x))