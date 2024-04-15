# Databricks notebook source
# DBTITLE 1,KYC Risk Engine Training Set Up
# MAGIC %md 
# MAGIC This notebook sets up the environment for the KYC onboarding risk engine training repo
# MAGIC

# COMMAND ----------

# MAGIC %pip install minepy

# COMMAND ----------

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from minepy import MINE


def get_iv_class(y: np.ndarray, x: np.ndarray, x_name: str = None, uniq=10, bins=5):
    """

    :param y:
    :param x:
    :param x_name:
    :param uniq:
    :param bins:
    :return:
    """
    if bool(x_name):
        pass
    else:
        x_name = 'x'

    # Stack arrays with responses and predictions
    arr = np.vstack((y, x)).T

    # Convert to data frame
    in_df = pd.DataFrame(arr, columns=['y', x_name])

    # checking the datatype of the input variable
    if in_df[x_name].dtype == 'O':
        in_df[x_name] = in_df[x_name].apply(np.float32)

    # impute missing values with median else do nothing
    if in_df[x_name].isnull().sum() > 0:
        med = in_df[x_name].median()
        in_df[x_name] = in_df[x_name].apply(lambda x: med if pd.isnull(x) == True else x)

    n_uniq = in_df[x_name].nunique()
    # binning X variable separately for indicator and continuous values
    if n_uniq <= uniq:
        in_df['bins'] = in_df[x_name]
    else:
        in_df['bins'] = pd.qcut(in_df[x_name], bins, labels=False, duplicates='drop')

    # if variable has only 1 bin then return 0 IV
    if in_df['bins'].nunique() <= 1:
        return [x_name, 0.0]

    else:
        in_df.sort_values(by=['bins'], inplace=True)
        wwoe = []
        for i, val in enumerate(in_df['y'].unique()):

            # calculating total events and non events
            resp = float(in_df[in_df['y'] == val].shape[0])
            non_resp = float(in_df[in_df['y'] != val].shape[0])

            # calculating bin level distribution of events and non events
            resp_bin = in_df[in_df['y'] == val].groupby(['bins'])['y'].apply(lambda x: x.count() / resp)
            non_resp_bin = in_df[in_df['y'] != val].groupby(['bins'])['y'].apply(lambda x: x.count() / non_resp)

            # calculating differnce in bin level distribution of events and non events
            if n_uniq <= uniq:
                distribution_diff = np.array(non_resp_bin - resp_bin).reshape(n_uniq, )
            else:
                distribution_diff = np.array(non_resp_bin - resp_bin).reshape(in_df['bins'].nunique(), )

            # calculating WOE and IV
            if n_uniq <= uniq:
                woe_bin = np.array(np.log(non_resp_bin / resp_bin)).reshape(n_uniq, )
            else:
                woe_bin = np.array(np.log(non_resp_bin / resp_bin)).reshape(in_df['bins'].nunique(), )

            wwoe.append(distribution_diff * woe_bin)

        return [x_name, round(np.array(wwoe).max(axis=0).sum(), 3)]


def get_iv_reg(y: np.ndarray, x: np.ndarray, x_name: str = None, uniq=10, bins=5):
    """

    :param y:
    :param x:
    :param x_name:
    :param uniq:
    :param bins:
    :return:
    """
    if bool(x_name):
        pass
    else:
        x_name = 'x'

    # Stack arrays with responses and predictions
    arr = np.vstack((y, x)).T

    # Convert to data frame
    in_df = pd.DataFrame(arr, columns=['y', x_name])

    # checking the datatype of the input variable
    if in_df[x_name].dtype == 'O':
        in_df[x_name] = in_df[x_name].apply(np.float32)

    # impute missing values with median else do nothing
    if in_df[x_name].isnull().sum() > 0:
        med = in_df[x_name].median()
        in_df[x_name] = in_df[x_name].apply(lambda x: med if pd.isnull(x) == True else x)

    # calculating sum of dep var
    dep_sum = float(in_df['y'].values.sum())
    dep_count = float(in_df['y'].count())

    n_uniq = in_df[x_name].nunique()
    # binning X variable separately for indicator and continuous values
    if n_uniq <= uniq:
        in_df['bins'] = in_df[x_name]
    else:
        in_df['bins'] = pd.qcut(in_df[x_name], bins, labels=False, duplicates='drop')

    # if variable has only 1 bin then return 0 IV
    if in_df['bins'].nunique() <= 1:
        return [x_name, 0.0]

    else:
        # calculating bin level distribution of bin sum and total records
        resp_bin = in_df.groupby(['bins'])['y'].apply(lambda x: x.sum() / dep_sum)
        pop_bin = in_df.groupby(['bins'])['y'].apply(lambda x: (x.count()) / dep_count)

        # calculating differnce in bin level distribution of events and non events
        if n_uniq <= uniq:
            distribution_diff = np.array(pop_bin - resp_bin).reshape(1, n_uniq)
        else:
            distribution_diff = np.array(pop_bin - resp_bin).reshape(1, in_df['bins'].nunique())

        # calculating WOE and IV
        if n_uniq <= uniq:
            woe_bin = np.array(np.log(pop_bin / resp_bin)).reshape(n_uniq, 1)
        else:
            woe_bin = np.array(np.log(pop_bin / resp_bin)).reshape(in_df['bins'].nunique(), 1)
        iv = np.dot(distribution_diff, woe_bin)[0, 0]

        return [x_name, round(iv, 3)]


def iv_group(iv):
    """

    :param iv:
    :return:
    """
    if iv >= 0.5 and np.isinf(iv) == False:
        return 'suspecious'
    elif 0.5 > iv >= 0.3:
        return 'strong'
    elif 0.3 > iv >= 0.1:
        return 'medium'
    elif 0.1 > iv >= 0.02:
        return 'weak'
    else:
        return 'useless'


def mic_score(y: np.ndarray, x: np.ndarray, x_name: str = None):
    """

    :param y:
    :param x:
    :param x_name:
    :return: list containing variable name and mic score
    """

    if bool(x_name):
        pass
    else:
        x_name = 'x'

    # Stack arrays with responses and predictions
    arr = np.vstack((y, x)).T

    # Convert to data frame
    in_df = pd.DataFrame(arr, columns=['y', x_name])

    # Take Sample of 50000
    if in_df.shape[0] > 50000:
        in_df = in_df.sample(50000).reset_index(drop=True)

    mine = MINE(alpha=0.6, c=15, est="mic_approx")
    mine.compute_score(in_df['y'], in_df[x_name])
    return [x_name, np.around(mine.mic(), decimals=4)]


def get_spearmanr_sq(y: np.ndarray, x: np.ndarray, x_name: str = None) -> list:
    """

    :param y:
    :param x:
    :param x_name:
    :return:
    """
    if bool(x_name):
        pass
    else:
        x_name = 'x'

    # Stack arrays with responses and predictions
    arr = np.vstack((y, x)).T

    # Compute rank correlation
    r2_rank = np.square(spearmanr(arr)[0])

    return [x_name, np.around(r2_rank, 3)]


# COMMAND ----------

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from time import time


def explained_variance(df: pd.DataFrame, varlist: list, sel_varlist: list) -> dict:
    """
        Attributes
        ----------
        df:
            input dataframe
        varlist: 
            variable list passed as an input to the algorithm
        sel_varlist: 
            selected variables from the algorithm

        Returns
        -------
        A dictionary
    """
    superset = list(set(varlist + sel_varlist))
    corr_mat_df = pd.DataFrame(np.corrcoef(df[superset].to_numpy(dtype=np.float32), rowvar=False),
                               index=superset, columns=superset)

    keep_index = np.where(np.nan_to_num(np.diagonal(corr_mat_df)) != 0)[0].tolist()
    corr_mat_df = corr_mat_df.iloc[keep_index, keep_index]
    to_drop = list(set(varlist) - set(corr_mat_df.columns.tolist()))
    corr_mat_abs = corr_mat_df.abs()
    upper = corr_mat_abs.where(np.triu(np.ones(corr_mat_abs.shape), k=1).astype(bool))
    to_drop = [var for var in upper.columns if any(upper[var] > 1)] + to_drop
    to_keep = [var for var in corr_mat_df.columns if var not in to_drop]
    sel_varlist = [var for var in sel_varlist if var in to_keep]

    inv = np.linalg.pinv(corr_mat_df.loc[sel_varlist, sel_varlist])
    var_exp_n = np.trace(np.dot(np.dot(corr_mat_df.loc[:, sel_varlist], inv), corr_mat_df.loc[sel_varlist, :]))
    var_exp_d = float(corr_mat_df.shape[0])
    var_exp = var_exp_n / var_exp_d

    return {'exp_exp': np.around(var_exp_n, decimals=3),
            'total_var': np.around(var_exp_d, decimals=3),
            'ratio': np.around(var_exp, decimals=3),
            'vars_dropped': to_drop}


def variance_inflation(df: pd.DataFrame, sel_varlist: list, verbose: int = 0, time_taken: int = 0) -> pd.DataFrame:
    """
        Attributes
        ----------
        df:
            input dataframe
        sel_varlist: 
            selected variables from the algorithm

        Returns
        -------
        A DataFrame containing VIF values
    """
    start = time()
    if verbose > 0:
        print('No. of features = ' + str(len(sel_varlist)))
    corr_mat = pd.DataFrame(np.corrcoef(df[sel_varlist].to_numpy(dtype=np.float32), rowvar=False),
                            index=sel_varlist, columns=sel_varlist)

    vif_list = []
    for i, var in enumerate(sel_varlist):
        x_list = list(set(sel_varlist) - {var})
        x_list_inv = np.linalg.pinv(corr_mat.loc[x_list, x_list])
        r2 = np.trace(np.dot(np.dot(corr_mat.loc[[var], x_list], x_list_inv), corr_mat.loc[x_list, [var]]))
        if r2 == 1.00:
            vif = round((1 / (1 - 0.999)), 2)
        else:
            vif = round((1 / (1 - r2)), 2)
        vif_list.append([var, vif])

    if time_taken == 1:
        print()
        print("Runtime is %s minutes " % round((time() - start) / 60.0, 2))

    return pd.DataFrame(vif_list, columns=['feature', 'VIF']).sort_values(by='VIF', ascending=False,
                                                                          inplace=False)


def _inner_loop_unsup(var, sel_feat, x, feature_list2, inv) -> list:
    sel_feat_temp = list(set([var] + sel_feat))
    x22 = x[feature_list2, :][:, feature_list2]
    x11_inv = inv(x[sel_feat_temp, :][:, sel_feat_temp])
    x21 = x[feature_list2, :][:, sel_feat_temp]
    x12 = x[sel_feat_temp, :][:, feature_list2]

    return [var, np.trace(x22 - np.dot(x21, np.dot(x11_inv, x12)))]


def _inner_loop_sup(var, sel_feat, x, dep_var_list, inv) -> list:
    sel_feat_temp = list(set([var] + sel_feat))
    x22 = x[dep_var_list, :][:, dep_var_list]
    x11_inv = inv(x[sel_feat_temp, :][:, sel_feat_temp])
    x21 = x[dep_var_list, :][:, sel_feat_temp]
    x12 = x[sel_feat_temp, :][:, dep_var_list]

    return [var, np.trace(x22 - np.dot(x21, np.dot(x11_inv, x12)))]


# COMMAND ----------

# -*- coding: utf-8 -*-

from typing import NoReturn
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from collections import Counter
from joblib import Parallel, delayed
import multiprocessing
from time import time
try:
    import SharedArray
except ModuleNotFoundError:
    pass


class RecursivePCA:
    """class for PCA Recursion.

        Attributes
        ----------
        maxeigen:
            2nd EigenValue threshold in each cluster (based on cluster Principal Components Analysis).
            Higher value will result in less clusters and lower value will result in more clusters

        report_method:
            - 'correlation': Within each cluster select the feature that has the maximum correlation with others
            - 'centroid': Within each cluster select the feature that is closest to the centroid of the cluster
            - 'r-square_ratio': minimize (1-r_squre)/(1-r_square_nearest_cluster)
            - 'closest_to_PCA': feature closest to first component of PCA of the cluster
    """

    def __init__(self, maxeigen: float = 0.7, report_method: str = 'r-square_ratio', random_state: int = 123):

        self.maxeigen = maxeigen
        self.report_method = report_method
        self.seed = random_state
        self.use_x_list = []
        self.stand_scale = False
        self.corr_mat = pd.DataFrame()
        self.df = pd.DataFrame()
        self.heir_list = None
        self.flat_list = None

    def fit(self, df, use_x_list: list = None, exclude_x_list: list = None, is_normalized: bool = False) -> NoReturn:
        """
        Attributes
        ----------
        df:
            Input DataFrame
        use_x_list:
            features to use for analysis (if empty then use all columns in data)
        exclude_x_list:
            features to exclude from analysis (if empty then use all columns in data)
        is_normalized:
            whether the input is already Standardized or not (True/False)
        """

        self.stand_scale = is_normalized

        if len(use_x_list) == 0 and len(exclude_x_list) == 0:
            self.use_x_list = df.columns.tolist()

        elif len(use_x_list) > 0:
            self.use_x_list = use_x_list

        else:
            self.use_x_list = [var for var in df.columns.tolist() if var not in exclude_x_list]

        # Computing correlation matrix
        self.corr_mat = pd.DataFrame(np.corrcoef(df[self.use_x_list].to_numpy(dtype=np.float32), rowvar=False),
                                     index=self.use_x_list, columns=self.use_x_list)
        keep_index = np.where(np.nan_to_num(np.diagonal(self.corr_mat)) != 0)[0].tolist()
        self.corr_mat = self.corr_mat.iloc[keep_index, keep_index]
        self.use_x_list = self.corr_mat.columns.tolist()

        if not self.stand_scale:

            print("Standardizing data")
            X = df[self.use_x_list].to_numpy(dtype=np.float32)
            X = (X - X.mean(axis=0)) / X.std(axis=0)
            X = X.astype(np.float32)
            self.df = pd.DataFrame(X, columns=self.use_x_list)

        else:

            print("Data is already standardized")
            X = df[self.use_x_list].to_numpy(dtype=np.float32)
            self.df = pd.DataFrame(X, columns=self.use_x_list)

        print("Recursive PCA Hierarchical Clustering")
        print("Number of features = " + str(len(self.df.columns)))
        try:
            self.heir_list = self._pca_rec(self.corr_mat, self.maxeigen, False)
        except RecursionError:
            self.heir_list = self._pca_rec(self.corr_mat, self.maxeigen, True)
        self.flat_list = self._flatten(self.heir_list)
        print("Number of clusters = " + str(len(self.flat_list)))

    def report(self) -> pd.DataFrame:
        """
        Returns
        -------
        DataFrame of Cluster Summary
        """
        all_feats = []
        clus_nums = []
        report = pd.DataFrame()
        for i, clus in enumerate(self.flat_list):
            all_feats += clus
            clus_nums += list(np.zeros(len(clus), dtype=int) + i)
        C = Counter(clus_nums)
        a = pd.DataFrame()
        a['feat_name'] = all_feats
        a['cluster'] = clus_nums
        clus_list = C.keys()
        selected_feats = []

        if self.report_method == 'r-square_ratio':
            all_clus_PCA = []
            for clus in clus_list:
                clus_feats = list(a[a['cluster'] == clus]['feat_name'])
                pca = PCA(n_components=1)
                trans = pca.fit_transform(self.df[clus_feats])
                all_clus_PCA.append(trans[:, 0])
            inter_clus_d = squareform(pdist(np.array(all_clus_PCA)))
            nearest_neighbours = np.argmin(inter_clus_d + np.max(inter_clus_d) * np.identity(len(clus_list)), axis=1)
        report_list = []
        for clus_num, clus in enumerate(clus_list):
            clus_feats = list(a[a['cluster'] == clus]['feat_name'])
            if len(clus_feats) == 0:
                continue
            else:
                if self.report_method == 'correlation':
                    cols = ['cluster', 'feature', 'sum_cross_corr']
                    corre = self.df[clus_feats].corr().fillna(-1)
                    sum_cross_corr = np.sum(abs((np.array(corre) - np.identity(corre.shape[0]))), axis=1)
                    for i, feat in enumerate(clus_feats):
                        report_list.append([clus_num, feat, sum_cross_corr[i]])
                    selected_feat = clus_feats[np.argmax(sum_cross_corr)]
                    selected_feats.append(selected_feat)
                    report = pd.DataFrame(report_list, columns=cols)

                elif self.report_method == 'r-square_ratio':
                    cols = 'Cluster feature R2_Own Next_Closest R2_NC R2_Ratio'.split()
                    r2_ratio_list = []
                    for feat in clus_feats:
                        lrg = LinearRegression()
                        lrg.fit(self.df[feat].values.reshape(self.df[feat].shape[0], 1),
                                all_clus_PCA[clus_num].reshape(all_clus_PCA[clus_num].shape[0], 1))
                        r2 = lrg.score(self.df[feat].values.reshape(self.df[feat].shape[0], 1),
                                       all_clus_PCA[clus_num].reshape(all_clus_PCA[clus_num].shape[0], 1))

                        lrg = LinearRegression()
                        lrg.fit(self.df[feat].values.reshape(self.df[feat].shape[0], 1),
                                all_clus_PCA[nearest_neighbours[clus_num]].reshape(all_clus_PCA[clus_num].shape[0], 1))
                        r2_nearestClus = lrg.score(self.df[feat].values.reshape(self.df[feat].shape[0], 1),
                                                   all_clus_PCA[nearest_neighbours[clus_num]].reshape(
                                                       all_clus_PCA[clus_num].shape[0], 1))
                        if r2_nearestClus == 1:
                            r2_ratio = round((1 - r2) / (1 - 0.99999999999), 2)
                        else:
                            r2_ratio = round((1 - r2) / (1 - r2_nearestClus), 2)
                        r2_ratio_list.append(r2_ratio)
                        report_list.append(
                            [clus_num, feat, round(r2, 2), nearest_neighbours[clus_num], round(r2_nearestClus, 2),
                             r2_ratio])
                    report = pd.DataFrame(report_list, columns=cols)
                    selected_feat = clus_feats[np.argmin(r2_ratio_list)]
                    selected_feats.append(selected_feat)
                elif self.report_method == 'centroid':
                    cols = ['cluster', 'feature', 'Distance_to_centroid']
                    centroid = np.array(self.df[clus_feats].mean(axis=1))
                    dists = []
                    for f in clus_feats:
                        distance = np.linalg.norm(np.array(df1[f]) - centroid)
                        dists.append(distance)
                        report_list.append([clus_num, f, distance])
                    selected_feat = clus_feats[np.argmin(dists)]
                    selected_feats.append(selected_feat)
                    report = pd.DataFrame(report_list, columns=cols)
                elif self.report_method == 'closest_to_PCA':
                    cols = ['cluster', 'feature', 'Distance_to_PCA']
                    pca = PCA()
                    trans = pca.fit_transform(self.df[clus_feats])
                    dists = []
                    for feat in clus_feats:
                        distance = np.linalg.norm(np.array(df1[feat]) - trans[:, 0])
                        dists.append(distance)
                        report_list.append([clus_num, feat, distance])
                    selected_feat = clus_feats[np.argmin(dists)]
                    selected_feats.append(selected_feat)
                    report = pd.DataFrame(report_list, columns=cols)
        del self.df
        return report

    def _pca_rec(self, df: pd.DataFrame, maxeigen: float, include_thres: bool = False):

        if include_thres:

            if df.shape[1] <= 2:

                return list(df.columns)

            else:

                u, s, v = np.linalg.svd(df.values)

                if s[1] >= maxeigen:

                    clusters = np.abs(u[:, :2]).argmax(axis=1)

                    vars1 = [df.columns[i] for i, x in enumerate(clusters) if x == 0]
                    vars2 = [df.columns[j] for j, x in enumerate(clusters) if x == 1]

                    df1 = df.loc[vars1, vars1]
                    df2 = df.loc[vars2, vars2]

                    return [self._pca_rec(df1, maxeigen, include_thres), self._pca_rec(df2, maxeigen, include_thres)]

                else:

                    return list(df.columns)

        else:

            if df.shape[1] < 2:

                return list(df.columns)

            else:

                u, s, v = np.linalg.svd(df.values)

                if s[1] > maxeigen:

                    clusters = np.abs(u[:, :2]).argmax(axis=1)

                    vars1 = [df.columns[i] for i, x in enumerate(clusters) if x == 0]
                    vars2 = [df.columns[j] for j, x in enumerate(clusters) if x == 1]

                    df1 = df.loc[vars1, vars1]
                    df2 = df.loc[vars2, vars2]

                    return [self._pca_rec(df1, maxeigen, include_thres), self._pca_rec(df2, maxeigen, include_thres)]

                else:

                    return list(df.columns)

    def _flatten(self, s: list) -> list:

        if len(s) == 0:
            return s

        if type(s[0]) == str:
            return s

        elif type(s[0]) == list:

            if type(s[0][0]) == str and type(s[1][0]) == str:
                return s

            elif type(s[0][0]) == str and type(s[1][0]) == list:
                return [s[0]] + self._flatten(s[1])

            elif type(s[0][0]) == list and type(s[1][0]) == str:
                return self._flatten(s[0]) + [s[1]]

            else:
                return self._flatten(s[0]) + self._flatten(s[1])


class VariancePreservation:
    """class for Variance Preservation.

        Attributes
        ----------
        varexp:
            maximum variance to be explained
        maxeffects:
            maximum features to select
        tol:
            minimum gain in variance explained (R-Square) at each forward selection step
        method:
            - 'pearsonr': Pearson Correlation matrix analysis
            - 'spearmanr': Spearman rank Correlation matrix analysis
            - 'cov': Covariance matrix analysis
            - 'sscp': Sum of Squares and Cross Products matrix analysis
        inv_method:
            - classic: standard inverse computation (can handle imperfect multicollinearity)
            - pseudo: inverse computation based on SVD (can handle perfect multicollinearity)
        corr_cutoff:
            if abs value of 2 features is correlated > cutoff then 1 of them will be dropped from the analysis
            (give this cutoff < 1 for inv_method = 'classic' inverse computation)
        verbose:
            verbosity 0 or 1
        n_jobs:
            number of parallel jobs to run
    """

    def __init__(self, varexp: float = 0.95, maxeffects: int = 200, tol: float = 0.001, method: str = 'pearsonr',
                 inv_method: str = 'pseudo', corr_cutoff: float = 1, verbose: int = 0, n_jobs: int = 1):

        self.varexp = varexp
        self.maxeffects = maxeffects
        self.tol = tol
        self.method = method
        self.inv_method = inv_method
        self.corr_cutoff = corr_cutoff
        self.verbose = verbose
        self.in_df = pd.DataFrame()
        self.selection = None
        self.target_col = None
        self.target_type = None
        self.use_x_list = []
        self.exclude_x_list = []
        self.inv = None
        self.dep_var_list = []
        self.data = pd.DataFrame()
        self.corr_mat = pd.DataFrame()
        self.drop_list = []
        self.output = pd.DataFrame()
        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = n_jobs
        rng = np.random.RandomState()
        self.rand_int = rng.randint(100000000)

    def fit(self, df: pd.DataFrame, selection: str = 'Supervised', target_col: str = 'target',
            target_type: str = 'Class', use_x_list: list = None, exclude_x_list: list = None) -> NoReturn:
        """
        Attributes
        ----------
        df:
            python dataframe
        selection:
            UnSupervised/Supervised
        target_col:
            Variable to be explaned in case of Supervised selection, else give ''
        target_type:
            Binary/Class/Continuous
        use_x_list:
            consider only this list of variables for analysis
        exclude_x_list:
            features to exclude from the analysis like ['ip','event_date']
        """
        start = time()
        self.in_df = df
        self.selection = selection
        self.target_col = target_col
        self.target_type = target_type
        self.use_x_list = use_x_list
        self.exclude_x_list = exclude_x_list

        self.data = self._get_data(self.in_df, self.use_x_list, self.exclude_x_list, self.target_col)

        if self.verbose == 1:
            print("Preprocessing")

        if self.target_type in ['Binary', 'Class'] and self.selection == 'Supervised':

            if self.verbose == 1:
                print('Transforming target variable for discrim analysis')
            self.data, self.dep_var_list = self._if_class(self.data, self.target_col)

        if self.target_type == 'Continuous' and self.selection == 'Supervised':
            self.data = self.data
            self.dep_var_list = [self.target_col]

        if self.selection == 'UnSupervised':

            if self.verbose == 1:
                print("Removing the target variable from data")
            self.data = self._drop_target(self.data, self.target_col)

            if self.verbose == 1:
                print("# features for analysis = " + str(self.data.shape[1]))
            self.dep_var_list = []

        if self.verbose == 1:
            print("Computing " + self.method + " matrix")

        self.corr_mat, self.drop_list = self._corr_matrix(self.data, self.dep_var_list, self.method, self.corr_cutoff)

        if self.verbose == 1:
            print("Shape of " + str(self.method) + " matrix is " + str(self.corr_mat.shape))

        if len(self.drop_list) > 0:

            if self.verbose == 1:
                print("Features dropped from the analysis = " + str(len(self.drop_list)))

            for i in self.drop_list:

                if self.verbose == 1:
                    print(i)
        else:

            if self.verbose == 1:
                print("No Features dropped from the analysis")

        self.data = pd.DataFrame()
        self.inv = self._inverse(self.inv_method)

        if self.selection == 'UnSupervised':
            if self.verbose == 1:
                print("UnSupervised Forward Feature Selection")
            self.output = self._UnSupervised(self.corr_mat)
        if self.selection == 'Supervised':
            if self.verbose == 1:
                print("Supervised Forward Feature Selection")
            self.output = self._Supervised(self.corr_mat, self.dep_var_list)
        if self.verbose == 1:
            print("Runtime is %s minutes " % round((time() - start) / 60.0, 2))

    def report(self):
        """
        Returns
        -------
        Dataframe containing a list of features sorted from most important to least important and a correlation dataframe
        """
        return self.output, self.corr_mat

    def _get_data(self, pydf: pd.DataFrame, use_cols_list: list, exc_cols_list: list, target_col: str) -> pd.DataFrame:

        colnames = pydf.columns.tolist()
        if len(use_cols_list) > 0:
            feature_names = list(set(use_cols_list))
            if len(target_col) > 0:
                feature_names = list(set(feature_names + [target_col]))
        else:
            feature_names = list(set([col for col in colnames if col not in exc_cols_list]))
        feature_names = list(set([col for col in feature_names if col not in exc_cols_list]))
        pydf = pydf[feature_names]
        if self.verbose == 1:
            print("# Columns " + str(len(feature_names)))

        return pydf

    @staticmethod
    def _drop_target(pydf: pd.DataFrame, target: str) -> pd.DataFrame:

        if len(target) > 0:
            return pydf.drop(target, axis=1)
        else:
            return pydf

    def _if_class(self, pydf: pd.DataFrame, dep_var: str):
        num_cat = pydf[dep_var].nunique()
        dep_var_list = [dep_var + '_' + str(i) for i in range(num_cat)]
        total = float(pydf[dep_var].count())
        if self.verbose == 1:
            print("# Obs = " + str(total))
        for i, val in enumerate(pydf[dep_var].unique()):
            events = float(pydf[pydf[dep_var] == val].shape[0])
            if self.verbose == 1:
                print("Event Rate for " + str(val) + " is " + str(np.around((events / total), decimals=3)))
            value_e_e = (1 / np.sqrt(total)) * (np.sqrt(total / events) - np.sqrt(events / total))
            value_e_ne = (-1 / np.sqrt(total)) * (np.sqrt(events / total))
            pydf.loc[:, dep_var + '_' + str(i)] = pydf[dep_var].apply(lambda x: value_e_e if x == val else value_e_ne)

        del pydf[dep_var]

        return pydf, dep_var_list

    @staticmethod
    def _corr_matrix(pydf: pd.DataFrame, dep_list: list, method: str, cutoff: float):

        corr_mat_df = pd.DataFrame()
        to_drop = []

        feature_names = pydf.columns.tolist()
        feature_names_count = len(feature_names)

        if method == 'cov':
            corr_mat_df = pd.DataFrame(np.cov(pydf[feature_names].to_numpy(dtype=np.float32), rowvar=False),
                                       index=feature_names,
                                       columns=feature_names)
            to_drop = []

        if method == 'sscp':
            pydf_a = pydf.to_numpy(dtype=np.float32)
            corr_mat_df = pd.DataFrame(np.dot(pydf_a.T, pydf_a) / pydf_a.shape[0], index=feature_names,
                                       columns=feature_names)
            to_drop = []

        if method == 'pearsonr':
            corr_mat_df = pd.DataFrame(np.corrcoef(pydf[feature_names].to_numpy(dtype=np.float32), rowvar=False),
                                       index=feature_names, columns=feature_names)
            keep_index = np.where(np.nan_to_num(np.diagonal(corr_mat_df)) != 0)[0].tolist()
            corr_mat_df = corr_mat_df.iloc[keep_index, keep_index].copy()
            to_drop = list(set(feature_names) - set(corr_mat_df.columns.tolist()))
            corr_mat_abs = corr_mat_df.abs()
            upper = corr_mat_abs.where(np.triu(np.ones(corr_mat_abs.shape), k=1).astype(bool))
            to_drop = [var for var in upper.columns if any(upper[var] > cutoff)] + to_drop
            to_drop = [var for var in to_drop if var not in dep_list]
            to_keep = [var for var in corr_mat_df.columns if var not in to_drop]
            corr_mat_df = corr_mat_df.loc[to_keep, to_keep].copy()

        if method == 'spearmanr':

            if feature_names_count <= 1:
                rho = np.array([[1.0]])

            elif feature_names_count == 2:
                rho = pydf[feature_names].corr('spearman').values

            else:
                rho = np.array(spearmanr(pydf[feature_names].to_numpy(dtype=np.float32))[0], ndmin=2)
                if len(rho) != feature_names_count:
                    rho = pydf[feature_names].corr('spearman').values

            corr_mat_df = pd.DataFrame(rho, index=feature_names, columns=feature_names)
            keep_index = np.where(np.nan_to_num(np.diagonal(corr_mat_df)) != 0)[0].tolist()
            corr_mat_df = corr_mat_df.iloc[keep_index, keep_index]
            to_drop = list(set(feature_names) - set(corr_mat_df.columns.tolist()))
            corr_mat_abs = corr_mat_df.abs()
            upper = corr_mat_abs.where(np.triu(np.ones(corr_mat_abs.shape), k=1).astype(bool))
            to_drop = [var for var in upper.columns if any(upper[var] > cutoff)] + to_drop
            to_drop = [var for var in to_drop if var not in dep_list]
            to_keep = [var for var in corr_mat_df.columns if var not in to_drop]
            corr_mat_df = corr_mat_df.loc[to_keep, to_keep]

        return corr_mat_df, to_drop

    # def _shared_array(self, df=pd.DataFrame):
    #
    #     try:
    #         SharedArray.delete('shared_array_' + str(self.rand_int))
    #     except:
    #         pass
    #
    #     shared_array = SharedArray.create('shared_array_' + str(self.rand_int), df.shape, dtype=float)
    #
    #     for i, row in enumerate(df.values):
    #         shared_array[i, :] = row
    #
    #     return shared_array

    @staticmethod
    def _inverse(inv_type: str):

        if inv_type == 'pseudo':
            inv = np.linalg.pinv
        else:
            inv = np.linalg.inv
        return inv

    def _UnSupervised(self, corr_mat_df: pd.DataFrame) -> pd.DataFrame:

        feature_list = corr_mat_df.columns.tolist()
        feature_list_idx = range(len(feature_list))
        # X = self._shared_array(corr_mat_df)
        X = corr_mat_df.to_numpy(dtype=np.float32)

        tot_var = np.trace(X)
        sel_feat_idx = []
        Var_Score = []

        for run, idx in enumerate(feature_list_idx):
            feature_list2_idx = list(set(feature_list_idx) - set(sel_feat_idx))

            t_score = [Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(_inner_loop_unsup)(var_idx, sel_feat_idx, X, feature_list2_idx, self.inv) for var_idx in
                feature_list2_idx)]

            score_df = pd.DataFrame(t_score[0], columns=['feature_idx', 'score'])
            sel_feat_score = score_df.score.min()
            sel_feat_idx.append(score_df[score_df.score == sel_feat_score].feature_idx.tolist()[0])
            Var_Score.append((tot_var - sel_feat_score) / float(tot_var))
            if self.verbose == 1:
                print('\r',
                      'Explained Variance ' + str(round(Var_Score[-1], 3)) + '/' + str(self.varexp) + ' with ' + str(
                          len(sel_feat_idx)) + ' features out of ' + str(len(feature_list)), end='')
            if run > 0:
                if Var_Score[run] >= self.varexp or len(sel_feat_idx) >= self.maxeffects or (
                        Var_Score[run] - Var_Score[run - 1]) <= self.tol:
                    if Var_Score[run] >= self.varexp:
                        if self.verbose == 1:
                            print("varexp_cutoff reached")
                    if len(sel_feat_idx) >= self.maxeffects:
                        if self.verbose == 1:
                            print("maxeffects_cutoff reached")
                    if (Var_Score[run] - Var_Score[run - 1]) < self.tol:
                        if self.verbose == 1:
                            print("Marginal variance explained is less than tol")
                    break
        sel_feat = [feature_list[var_idx] for var_idx in sel_feat_idx]
        report = pd.DataFrame(list(zip(sel_feat, np.around(Var_Score, decimals=4))), columns=['feature', 'var_exp'])

        # try:
        #     SharedArray.delete('shared_array_' + str(self.rand_int))
        # except:
        #     pass

        return report

    def _Supervised(self, corr_mat_df: pd.DataFrame, dep_var_list: list) -> pd.DataFrame:

        feature_list = corr_mat_df.columns.tolist()
        feature_list_idx = range(len(feature_list))
        # X = self._shared_array(corr_mat_df)
        X = corr_mat_df.to_numpy(dtype=np.float32)

        dep_var_list_idx = [feature_list.index(var) for var in feature_list if var in dep_var_list]
        x_feature_list_idx = list(set(feature_list_idx) - set(dep_var_list_idx))

        tot_var = np.trace(X[dep_var_list_idx, :][:, dep_var_list_idx])

        sel_feat_idx = []
        Var_Score = []

        for run, idx in enumerate(x_feature_list_idx):
            x_feature_list2_idx = list(set(x_feature_list_idx) - set(sel_feat_idx))

            t_score = [Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(_inner_loop_sup)(var, sel_feat_idx, X, dep_var_list_idx, self.inv) for var in
                x_feature_list2_idx)]

            score_df = pd.DataFrame(t_score[0], columns=['feature_idx', 'score'])
            sel_feat_score = score_df.score.min()
            sel_feat_idx.append(score_df[score_df.score == sel_feat_score].feature_idx.tolist()[0])
            Var_Score.append((tot_var - sel_feat_score) / float(tot_var))
            if self.verbose == 1:
                print('\r' + 'Explained Variance ' + str(round(Var_Score[-1], 3)) + '/' + str(
                    self.varexp) + ' with ' + str(len(sel_feat_idx)) + ' features out of ' + str(
                    len(x_feature_list_idx)), end='')
            if run > 0:
                if Var_Score[run] >= self.varexp or len(sel_feat_idx) >= self.maxeffects or (
                        Var_Score[run] - Var_Score[run - 1]) <= self.tol:
                    if Var_Score[run] >= self.varexp:
                        if self.verbose == 1:
                            print("varexp_cutoff reached")
                    if len(sel_feat_idx) >= self.maxeffects:
                        if self.verbose == 1:
                            print("maxeffects_cutoff reached")
                    if (Var_Score[run] - Var_Score[run - 1]) < self.tol:
                        if self.verbose == 1:
                            print("Marginal variance explained is less than tol")
                    break
        sel_feat = [feature_list[var_idx] for var_idx in sel_feat_idx]
        report = pd.DataFrame(list(zip(sel_feat, np.around(Var_Score, decimals=4))), columns=['feature', 'var_exp'])

        # try:
        #     SharedArray.delete('shared_array_' + str(self.rand_int))
        # except:
        #     pass

        return report


# COMMAND ----------

def variance_inflation(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df: input dataframe with selected variables
    :return: VIF report as pandas dataframe
    """
    start = time()
    sel_varlist = df.columns.tolist()
    print('No. of features = ' + str(len(sel_varlist)))
    x = df.values
    x = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)
    corr_mat = pd.DataFrame(np.dot(x.T, x) / x.shape[0], index=sel_varlist, columns=sel_varlist)
    vif_list = []
    for i, var in enumerate(sel_varlist):
        x_list = list(set(sel_varlist) - set([var]))
        x_list_inv = np.linalg.pinv(corr_mat.loc[x_list, x_list])
        r2 = np.trace(np.dot(np.dot(corr_mat.loc[[var], x_list], x_list_inv), corr_mat.loc[x_list, [var]]))
        if r2 == 1.00:
            vif = round((1 / (1 - 0.999)), 2)
        else:
            vif = round((1 / (1 - r2)), 2)
        vif_list.append([var, vif])
    print("Runtime is %s minutes " % round((time() - start) / 60.0, 2))
    return pd.DataFrame(vif_list, columns=['feature', 'vif']).sort_values(by='vif', ascending=False,inplace=False)
    