import os
import ast
import umap
import pickle
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly
import plotly.graph_objs as go
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.metrics import BinaryCrossentropy


def read_features(p_path, trigger_type_aux_str=None, arch=None, data='diffs', label_type='binary', append_arch=False, arch_one_hot=False):
    report = pd.read_csv(p_path)
    report = filter_df(report, trigger_type_aux_str, arch)
    check = None
    if data == 'diffs':
        check = 'end'
        check_str_1 = 'mean_diff'
        check_str_2 = 'std_diff'
        print('Using diffs')
    elif data == 'raw':
        check = 'end'
        check_str_1 = 'mean'
        check_str_2 = 'std'
        print('Using mean & stds')
    elif data == 'h' or data == 'kl' or data == 'hkl':
        check = 'start'
        check_str_1 = 'h_'
        check_str_2 = 'kl_'
        print(f'Using {data}')
    else:
        print(f'Invalid data type: {data}')

    initial_columns = report.columns
    col_model_label = report['model_label'].copy(deep=True)
    col_arch_code = report['architecture_code'].copy(deep=True)

    # consider moving here the if statements related to label
    if label_type.startswith('backdoor'): # choose backdoor_code (for round 3) or backdoor_code_0/1
        col_back_code = report[label_type].copy(deep=True)

    if label_type == 'binary':
        labels = np.array([int(col_model_label.iloc[i] == 'backdoor') for i in range(len(report))])
    elif label_type in ['backdoor_code_0', 'backdoor_code_1']:
        labels = np.array(report[label_type])
    elif label_type == 'backdoor_code_2':
        labels = [ast.literal_eval(c) for c in report[label_type]]
    else:
        print('Invalid value for label_type: should be binary or backdoor')

    for c in initial_columns:
        if check == 'end':
            if not c.endswith(check_str_1) and not c.endswith(check_str_2):
                del report[c]
        elif check == 'start':
            if data == 'hkl':
                if not c.startswith(check_str_1) and not c.startswith(check_str_2):
                    del report[c]
            elif data == 'h':
                if not c.startswith('h'):
                    del report[c]
            elif data == 'kl':
                if not c.startswith('kl'):
                    del report[c]
    features = report.values
    if append_arch:
        if arch_one_hot:
            col_arch = OneHotEncoder(sparse=False).fit_transform(col_arch_code.values.reshape(-1, 1))
        else:
            col_arch = col_arch_code.values.reshape(-1, 1)
        features = np.hstack((features, col_arch))
    return features, labels


def filter_df(df, trigger_type_aux_str=None, arch=None):
    # if trigger_type_aux_str is not None and arch is not None:
    #     print(f'selecting only {arch}s-{trigger_type_aux_str}s')
    #     indexes = []
    #     for i in range(len(df)):
    #         tta = df['trigger_type_aux'].iloc[i]
    #         a = df['model_architecture'].iloc[i]
    #         cond_tta = (trigger_type_aux_str in tta.lower()) or (tta.lower() == 'none')
    #         cond_arch = a.startswith(arch)
    #         if cond_tta and cond_arch:
    #             indexes.append(i)
    #     df = df.iloc[indexes]
    if arch is not None:
        print(f'selecting only {arch}s')
        indexes = []
        for i in range(len(df)):
            col = df['model_architecture'].iloc[i]
            if arch in col:
                indexes.append(i)
        df = df.iloc[indexes]
    if trigger_type_aux_str is not None:
        print(f'selecting only {trigger_type_aux_str}s')
        indexes = []
        for i in range(len(df)):
            col = df['trigger_type_aux'].iloc[i]
            if (trigger_type_aux_str in col.lower()) or (col.lower() == 'none'):
                indexes.append(i)
        df = df.iloc[indexes]
    return df


def get_trigger_type_aux_value(trigger_type, polygon_side_count, instagram_filter_type):
    if trigger_type == 'instagram':
        instagram_filter_type = instagram_filter_type.replace('FilterXForm', '').lower()
        return f'instagram-{instagram_filter_type}'
    else:
        if trigger_type == 'polygon':
            return f'{trigger_type}-{polygon_side_count}'
        else:
            return trigger_type.lower()


def encode_architecture(model_architecture):
    arch_codes = ['densenet', 'googlenet', 'inception', 'mobilenet', 'resnet', 'shufflenet', 'squeezenet', 'vgg']
    for index, arch in enumerate(arch_codes):
        if arch in model_architecture:
            return index
    return None


def encode_backdoor(trigger_type_aux):
    code = None
    if trigger_type_aux == 'none':
        code = 0
    elif 'polygon' in trigger_type_aux:
        code = 1
    elif 'gotham' in trigger_type_aux:
        code = 2
    elif 'kelvin' in trigger_type_aux:
        code = 3
    elif 'lomo' in trigger_type_aux:
        code = 4
    elif 'nashville' in trigger_type_aux:
        code = 5
    elif 'toaster' in trigger_type_aux:
        code = 6
    return code


def get_predicted_label(model, image, device):
    output = model(image.to(device))
    softmax = nn.functional.softmax(out[0].cpu(), dim=0)
    pred_label = out.max(1)[1].item()
    return pred_label


def save_obj(obj, folder, name):
    if name is None:
        name = 'model'
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, f'{name}.pkl'), 'wb') as handle:
        pickle.dump(obj, handle)


def load_obj(filename):
    if not os.path.isfile(filename):
        print('Pickle {} does not exist.'.format(filename))
        return None
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def keras_save(model, folder, name=None):
    if name is None:
        name = 'model'
    if not os.path.isdir(folder):
        os.makedirs(folder)
    model_json = model.to_json()
    with open(os.path.join(folder, f'{name}.json'), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(folder, f'{name}.h5'))


def keras_load(folder, model_name=None):
    if model_name is None:
        model_name = 'model'
    json_file = open(os.path.join(folder, f'{model_name}.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(folder, f'{model_name}.h5'))
    return loaded_model


def evaluate_classifier(clf, train_x, train_y, test_x, test_y):
    # clf = svm.SVC(C=11, kernel='rbf', gamma='scale', probability=True)
    #     clf = LogisticRegression(C=2)
    #     print('clf=', str(clf))
    clf.fit(train_x, train_y)
    y_score = clf.predict(test_x)
    y_pred = clf.predict_proba(test_x)
    #     print(y_score[:5].tolist())
    #     print(y_pred[:5,:].tolist())
    #     print()
    roc_auc = roc_auc_score(y_true=test_y, y_score=y_score)
    cross_entropy = log_loss(y_true=test_y, y_pred=y_pred)
    return roc_auc, cross_entropy


def get_base_classifier():
    # return svm.SVC(C=11, kernel='rbf', gamma='scale', probability=True)
    return LogisticRegression()


#     return RandomForestClassifier(n_estimators=500)


def scatter(df, x, y):
    clean = df[df['ground_truth'] == 0]
    backdoored = df[df['ground_truth'] == 1]
    plt.figure(figsize=(10, 6)).patch.set_color('white')

    #     x_clean = (clean[x] - clean[x].mean()) / clean[x].std()
    #     y_clean = (clean[y] - clean[y].mean()) / clean[y].std()

    #     x_back = (backdoored[x] - backdoored[x].mean()) / backdoored[x].std()
    #     y_back = (backdoored[y] - backdoored[y].mean()) / backdoored[y].std()

    #     plt.scatter(x_clean, y_clean, label='clean')
    #     plt.scatter(x_back, y_back, label='backdoored', c='red')
    plt.scatter(clean[x], clean[y], label='clean')
    plt.scatter(backdoored[x], backdoored[y], label='backdoored', c='orange')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.grid()


def overlay_two_histograms(hist_first_values, hist_second_values, first_label, second_label, xlabel, title=''):
    plt.figure(figsize=(16, 10)).patch.set_color('white')
    plt.hist([hist_first_values, hist_second_values], bins=50, label=[f'{first_label} (black)', f'{second_label} (dotted)'])
    plt.axvline(np.mean(hist_first_values), color='k', linestyle='-', linewidth=3)
    plt.axvline(np.mean(hist_second_values), color='b', linestyle='--', linewidth=3)
    plt.xlabel(xlabel)
    plt.ylabel('Number of Instances')
    plt.title(title)
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()


def read_pickle(file):
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    return obj


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    plt.figure(figsize=(16, 10)).patch.set_color('white')
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
    plt.grid()


def plot2D(data2plot, x, y, labels, title=''):
    data = pd.DataFrame()
    data[x] = data2plot[:, 0]
    data[y] = data2plot[:, 1]

    plt.figure(figsize=(16, 10)).patch.set_color('white')
    sns.scatterplot(x=x,
                    y=y,
                    hue=labels,
                    data=data,
                    s=50,
                    legend="full")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid()
    plt.title(title)
    plt.show()


def plot3D(data2plot, x, y, z, labels):
    data = pd.DataFrame()
    data[x] = data2plot[:, 0]
    data[y] = data2plot[:, 1]
    data[z] = data2plot[:, 2]

    target_names = ['clean', 'backdoored']
    if isinstance(labels, np.ndarray):
        target_values = list(set(labels))
    else:
        target_values = list(set(labels.values.squeeze()))
    target_colors = ['blue', 'orange']

    plotly.offline.init_notebook_mode()
    traces = []
    for value, color, name in zip(target_values, target_colors, target_names):
        if isinstance(labels, np.ndarray):
            indicesToKeep = (labels == value).squeeze()
        else:
            indicesToKeep = (labels == value).values.squeeze()
        trace = go.Scatter3d(x=data.loc[indicesToKeep, x],
                             y=data.loc[indicesToKeep, y],
                             z=data.loc[indicesToKeep, z],
                             mode='markers',
                             marker={'color': color, 'symbol': 'circle', 'size': 5},
                             name=name)
        traces.append(trace)
    plot_figure = go.Figure(data=traces,
                            layout=go.Layout(autosize=True,
                                             title='PCA plot',
                                             margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
                                             scene=dict(xaxis_title=x,
                                                        yaxis_title=y,
                                                        zaxis_title=z)))
    plotly.offline.iplot(plot_figure)
