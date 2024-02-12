import random
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# from: https://github.com/niteshsukhwani/MLSMOTE/blob/master/mlsmote.py
def _get_tail_label(df):
    """
    Give tail label colums of the given target dataframe
    
    args
    df: pandas.DataFrame, target label df whose tail label has to identified
    
    return
    tail_label: list, a list containing column name of all the tail label
    """
    columns = df.columns
    n = len(columns)
    irpl = np.zeros(n)
    for column in range(n):
        irpl[column] = df[columns[column]].value_counts()[1]
    irpl = max(irpl)/irpl
    mir = np.average(irpl)
    tail_label = []
    for i in range(n):
        if irpl[i] > mir:
            tail_label.append(columns[i])
    return tail_label


def _get_index(df):
  """
  give the index of all tail_label rows
  args
  df: pandas.DataFrame, target label df from which index for tail label has to identified
    
  return
  index: list, a list containing index number of all the tail label
  """
  tail_labels = _get_tail_label(df)
  index = set()
  for tail_label in tail_labels:
    sub_index = set(df[df[tail_label] == 1].index)
    index = index.union(sub_index)
  return list(index)


def _get_minority_samples(X, y):
    """
    Give minority dataframe containing all the tail labels
    
    args
    X: pandas.DataFrame, the feature vector dataframe
    y: pandas.DataFrame, the target vector dataframe
    
    return
    X_sub: pandas.DataFrame, the feature vector minority dataframe
    y_sub: pandas.DataFrame, the target vector minority dataframe
    """
    index = _get_index(y)
    X_sub = X[X.index.isin(index)].reset_index(drop = True)
    y_sub = y[y.index.isin(index)].reset_index(drop = True)
    return X_sub, y_sub


def _nearest_neighbour(X):
    """
    Give index of 5 nearest neighbor of all the instance
    
    args
    X: np.array, array whose nearest neighbor has to find
    
    return
    indices: list of list, index of 5 NN of each element in X
    """
    nbs = NearestNeighbors(n_neighbors=5,metric='euclidean', algorithm='kd_tree').fit(X)
    euclidean, indices = nbs.kneighbors(X)
    return indices


def _MLSMOTE(X, y, n_sample):
    """
    Give the augmented data using MLSMOTE algorithm
    
    args
    X: pandas.DataFrame, input vector DataFrame
    y: pandas.DataFrame, feature vector dataframe
    n_sample: int, number of newly generated sample
    
    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
    """
    indices2 = _nearest_neighbour(X)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0, n-1)
        neighbour = random.choice(indices2[reference,1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis = 0, skipna = True)
        target[i] = np.array([1 if val>2 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference, :] - X.loc[neighbour, :]
        new_X[i] = np.array(X.loc[reference, :] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    new_X = pd.concat([X, new_X], axis=0)
    target = pd.concat([y, target], axis=0)
    return new_X, target


# function called from outside
def oversample(X, Y):
    X_shape_old = X.shape
    class_dist = [y/Y.shape[0] for y in Y.sum(axis=0)]
    print('checking for minority classes in train split...')
    X_sub, Y_sub = _get_minority_samples(pd.DataFrame(X), pd.DataFrame(Y))

    if np.shape(X_sub)[0] > 0: # only oversample training set if minority samples are found
        print('minority classes found.')
        print('oversampling...')
        try:
            X_res, Y_res = _MLSMOTE(X_sub, Y_sub, round(X.shape[0]/5))       
            X = np.concatenate((X, X_res.to_numpy())) # append augmented samples
            Y = np.concatenate((Y, Y_res.to_numpy())) # to original dataframes
            print('oversampled.')
            class_dist_os = [y/Y.shape[0] for y in Y.sum(axis=0)]
            print('CLASS DISTRIBUTION:')
            print(f'Before MLSMOTE: {X_shape_old}, {class_dist}')
            print(f'After MLSMOTE: {X.shape}, {class_dist_os}')
        except ValueError:
            print('could not oversample because n_samples < n_neighbors in some classes')
    else:
        print('no minority classes.')
    return X, Y