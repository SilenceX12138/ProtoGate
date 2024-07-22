import argparse
import datetime
import math
import os
import random
import re
from ast import literal_eval, parse
from typing import DefaultDict

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
from joblib import dump, load
from pytorch_lightning import loggers as pl_loggers
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             mean_squared_error, precision_score, recall_score,
                             roc_auc_score)
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample, shuffle
from sklearn_extra.cluster import KMedoids
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm


def seed_everything(args):
    seed = args.seed_global

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # set `deterministic` to True only when debugging
    if 'dev' in args.tags:
        torch.backends.cudnn.deterministic = True

    return seed


def pearson_distance(x, y):
    return 1 - pearsonr(x, y)[0]


def spearman_distance(x, y):
    return 1 - spearmanr(x, y)[0]


def cosine_similarity(x, y):
    return 1 - cosine_distance(x, y)


def get_PCA_projected_data(X, y, n_components, test_size, split_seed, PCA_seed, whiten=False):
    """
    Run PCA of a split of size (1-`test_size`) of X

    Args:
    - X: data
    - n_components (int): number of PCA components
    - train_test_seed (int): seed for spliting into train/test

    Returns:
    - X_train_pca, X_test_pca: PCA reduced data
    - explained_variance_ratio (float)
    """
    scaler = StandardScaler()

    X_train, X_test, _, _ = train_test_split(X,
                                             y,
                                             stratify=y,
                                             test_size=test_size,
                                             random_state=split_seed)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=n_components, whiten=whiten, random_state=PCA_seed)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    explained_variance_ratio = float(np.sum(pca.explained_variance_ratio_))

    return X_train_pca, X_test_pca, explained_variance_ratio