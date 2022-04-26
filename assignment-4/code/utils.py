import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(q2=True):
    cities = ['BOS', 'NYC', 'DC', 'MIA', 'CHI', 'SEA', 'SF', 'LA', 'DEN']

    d = {cities[i]: i for i in range(len(cities))}
    dinv = {i:cities[i] for i in range(len(cities))}

    arr = [[0 for _ in range(9)] for _ in range(9)]

    arr[d['BOS']][d['NYC']] = 206
    arr[d['BOS']][d['DC']] = 429
    arr[d['BOS']][d['MIA']] = 1504
    arr[d['BOS']][d['CHI']] = 963
    arr[d['BOS']][d['SEA']] = 2976
    arr[d['BOS']][d['SF']] = 3095
    arr[d['BOS']][d['LA']] = 2979
    arr[d['BOS']][d['DEN']] = 1949

    arr[d['NYC']][d['DC']] = 233
    arr[d['NYC']][d['MIA']] = 1308
    arr[d['NYC']][d['CHI']] = 802
    arr[d['NYC']][d['SEA']] = 2815
    arr[d['NYC']][d['SF']] = 2934
    arr[d['NYC']][d['LA']] = 2786
    arr[d['NYC']][d['DEN']] = 1771

    arr[d['DC']][d['MIA']] = 1075
    arr[d['DC']][d['CHI']] = 671
    arr[d['DC']][d['SEA']] = 2684
    arr[d['DC']][d['SF']] = 2799
    arr[d['DC']][d['LA']] = 2631
    arr[d['DC']][d['DEN']] = 1616

    arr[d['MIA']][d['CHI']] = 1329
    arr[d['MIA']][d['SEA']] = 3273
    arr[d['MIA']][d['SF']] = 3053
    arr[d['MIA']][d['LA']] = 2687
    arr[d['MIA']][d['DEN']] = 2037

    arr[d['CHI']][d['SEA']] = 2013
    arr[d['CHI']][d['SF']] = 2142
    arr[d['CHI']][d['LA']] = 2054
    arr[d['CHI']][d['DEN']] = 996

    arr[d['SEA']][d['SF']] = 808
    arr[d['SEA']][d['LA']] = 1131
    arr[d['SEA']][d['DEN']] = 1307

    arr[d['SF']][d['LA']] = 379
    arr[d['SF']][d['DEN']] = 1235

    arr[d['LA']][d['DEN']] = 1059

    arr = np.matrix(arr)

    for j in range(9):
        for i in range(j+1, 9):
            arr[i, j] = arr[j, i]
    
    return cities, d, dinv, arr

def norm(a):
    return np.linalg.norm(a, ord=2)

def obj(arr, x):
    val = 0

    for i in range(9):
        for j in range(9):
            val += np.square(norm(x[i] - x[j]) - arr[i, j])

    return val

def gradient(arr, x, i):
    coef = 0
    val = np.zeros(2,)

    for j in range(9):
        if j == i:
            continue
        
        coef = (norm(x[i] - x[j]) -arr[i, j]) / norm(x[i]-x[j])

        val += coef * (x[i]-x[j])
    
    val *= 4

    return val

def update(x, i, grad, lr=.001):
    x[i] -= lr * grad

    return

def plot(X, Y, k_means_clusters=None, k_means_centers=None, method=None, n_neighbors=None, save_loc=None):
    if k_means_clusters is not None:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,6))

        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y, ax=ax[0])
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=k_means_clusters, ax=ax[1])

        if k_means_centers is not None:
            sns.scatterplot(x=k_means_centers[:, 0], y=k_means_centers[:, 1], color='red', alpha=1, ax=ax[1], label=f'{method} Centers')

        ax[0].set_title('Desired Clusters')
        
        if method is not None:
            if n_neighbors is not None: 
                ax[1].set_title(f'Obtained Clusters using {method} with r={n_neighbors}')
            else:
                ax[1].set_title(f'Obtained Clusters using {method}')
        else:
            ax[1].set_title(f'Obtained Clusters')

        ax[0].legend(loc="upper right")
        ax[1].legend(loc="upper right")

        if save_loc is not None:
            fig.savefig(save_loc)

        return
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y)

    if save_loc is not None:
        plt.savefig(save_loc)

