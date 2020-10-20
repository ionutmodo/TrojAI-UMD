from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import plotly
import plotly.graph_objs as go

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
    plt.figure(figsize=(16,10)).patch.set_color('white')
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

    plt.figure(figsize=(16,10)).patch.set_color('white')
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
                             marker={'color':color, 'symbol':'circle', 'size':5},
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