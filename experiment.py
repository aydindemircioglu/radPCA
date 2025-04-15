from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_classif
from sklearn.decomposition import KernelPCA, PCA, FastICA, FactorAnalysis, NMF, TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
#from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import MiniBatchDictionaryLearning

import os
import sys
import pickle
import glob

from featureselection import *
from SRP import *
import umap

import hashlib
from functools import partial
from joblib import dump, Parallel, delayed
import numpy as np
import pandas as pd
import random
import time

import radMLBench

import optuna
import warnings
from optuna.exceptions import ExperimentalWarning
optuna.logging.set_verbosity(optuna.logging.FATAL)


n_outer_cv = 5
n_inner_cv = 10
num_repeats = 10
selection_cache = {}
cache_to = "memory"

search_space = {
    'fs_method': ["Bhattacharyya", "ANOVA", "LASSO", "ET", "Kendall", "MRMRe", "tTest", "RFE_LogReg", "Boruta",\
            "UMAP", "KernelPCA", "PCA", "ICA", "FA", "NMF", "SRP", "TruncatedSVD", "MiniBatchDict", "None"],
    'N': [2**k for k in range(0,6)],
    'clf_method': ["RBFSVM", "RandomForest", "LogisticRegression", "NaiveBayes"],
    'RF_n_estimators': [10,25,50,100,250,500,1000],
    'C_LR': [2**k for k in range(-7,7,2)],
    'C_SVM': [2**k for k in range(-7,7,2)]
}


def get_md5_checksum(X):
    md5 = hashlib.md5()
    md5.update(X.tobytes())
    return md5.hexdigest()


def cached_select_features_disk(X_train, y_train, fs_method, N, dataset, repeat, do_dump=False):
    cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)

    checksum = get_md5_checksum(X_train)
    cache_key = f"{checksum}_{N}_{fs_method}_{dataset}_{repeat}"
    cache_path = os.path.join(cache_dir, cache_key)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as cache_file:
            Z = list(pickle.load(cache_file))
            # Set computation time to zero since it's read from cache
            Z[-1] = 0
            return Z

    start_time = time.time()
    X_train_selected, fsel = select_features(X_train, y_train, fs_method, N)
    end_time = time.time()
    total_fs_time = end_time - start_time

    with open(cache_path, "wb") as cache_file:
        pickle.dump((X_train_selected, fsel, total_fs_time), cache_file)

    return X_train_selected, fsel, total_fs_time



def cached_select_features_memory(X_train, y_train, fs_method, N, dataset, repeat, do_dump = False):
    global selection_cache
    checksum = get_md5_checksum(X_train)
    cache_key = (checksum, fs_method, N)

    if cache_key in selection_cache[f"{fs_method}_{dataset}_{repeat}"]:
        # here we did not had to compute anything, so we nil the time
        Z = list(selection_cache[f"{fs_method}_{dataset}_{repeat}"][cache_key])
        Z[-1] = 0
        return Z

    start_time = time.time()
    X_train_selected, fsel = select_features(X_train, y_train, fs_method, N)
    end_time = time.time()
    total_fs_time = end_time - start_time

    # real computation has total_fs_time
    selection_cache[f"{fs_method}_{dataset}_{repeat}"][cache_key] = X_train_selected, fsel, total_fs_time
    return X_train_selected, fsel, total_fs_time



def select_features(X, y, fs_method, N):
    if fs_method == "LASSO":
        clf_fs = LogisticRegression(penalty='l1', max_iter=100, solver='liblinear', C=1, random_state=42)
        fsel = SelectFromModel(clf_fs, prefit=False, max_features=N, threshold=-np.inf)
    elif fs_method == "RFE_LogReg":
        clf_fs = LogisticRegression(penalty='l2', max_iter=100, solver='liblinear', C=1, random_state=42)
        fsel = RFE(clf_fs, n_features_to_select=N, step = 0.1)
    # elif fs_method == "SFS-LR":
    #     clf_fs = LogisticRegression(penalty='l2', max_iter=100, solver='liblinear', C=1, random_state=42)
    #     fsel = SequentialFeatureSelector(clf_fs, n_features_to_select=N, direction='forward', scoring='roc_auc', cv=5)
    elif fs_method == "MiniBatchDict":
        fsel = MiniBatchDictionaryLearning(n_components=N, random_state=42)
    elif fs_method == "ANOVA":
        fsel = SelectKBest(f_classif, k=N)
    elif fs_method == "Bhattacharyya":
        fsel = SelectKBest(bhattacharyya_score_fct, k=N)
    elif fs_method == "MRMRe":
        mrmre_score_fct = partial(mrmre_score, nFeatures = N)
        fsel = SelectKBest(mrmre_score_fct, k = N)
    elif fs_method == "MRMR":
        mrmr_score_fct = partial(mrmr_score, nFeatures = N)
        fsel = SelectKBest(mrmr_score_fct, k = N)
    elif fs_method == "ET":
        clf_fs = ExtraTreesClassifier(random_state=42)
        fsel = SelectFromModel(clf_fs, prefit=False, max_features=N, threshold=-np.inf)
    elif fs_method == "Boruta":
        fsel = SelectKBest(boruta_fct, k = N)
    elif fs_method == "UMAP":
        fsel = umap.UMAP(n_components=N, n_jobs=1, random_state=42)
    elif fs_method == "KernelPCA":
        fsel = KernelPCA(n_components=N, kernel='rbf')
    elif fs_method == "SPCA":
        fsel = SPCA(k=N)
    elif fs_method == "SRP":
        fsel = SRP(k=N)
    elif fs_method == "KSRP":
        fsel = KSRP(k=N)
    elif fs_method == "tTest":
        fsel = SelectKBest(score_func=ttest_score, k=N)
    elif fs_method == "NMF":
        fsel = NMF(n_components=N)
    elif fs_method == "PCA":
        fsel = PCA(n_components=N)
    elif fs_method == "ICA":
        fsel = FastICA(n_components=N, random_state=42)
    elif fs_method == "FA":
        fsel = FactorAnalysis(n_components=N, random_state=42)
    elif fs_method == "Kendall":
        fsel = SelectKBest(score_func=kendall_score_func, k=N)
    elif fs_method == "TruncatedSVD":
        fsel = TruncatedSVD(n_components=N)
    elif fs_method == "None":
        fsel = FunctionTransformer(lambda X: X, validate=True)

    if fs_method in ["LASSO", "ANOVA", "Pearson", "Bhattacharyya", "ET", "Kendall", "tTest", "SRP", "KSRP", "SPCA", "MRMRe", "None", "RFE_LogReg", "Boruta"]:
        X_selected = fsel.fit_transform(X, y)
        return X_selected, fsel
    elif fs_method in ["PCA", "ICA", "KernelPCA", "FA", "UMAP", "NMF", "LDA", "TruncatedSVD", "MiniBatchDict"]:
        try:
            X_selected = fsel.fit_transform(X)
        except:
            print (X.shape)
            print (X)
            print ("N=", N)
            raise Exception ("WTF")
        return X_selected, fsel

    raise Exception (f"Unknown method {fs_method}")



def getClassifier(best_params):
    if best_params['clf_method'] == "LogisticRegression":
        clf = LogisticRegression(max_iter=500, solver='liblinear', C=best_params['C_LR'])
    elif best_params['clf_method'] == "NaiveBayes":
        clf = GaussianNB()
    elif best_params['clf_method'] == "RandomForest":
        clf = RandomForestClassifier(n_estimators=best_params['RF_n_estimators'])
    elif best_params['clf_method'] == "RBFSVM":
        clf = SVC(kernel="rbf", C=best_params['C_SVM'], gamma='auto', probability=True)
    return clf



def inner_objective(trial, X_train_outer, y_train_outer, fs_method, dataset, repeat, num_inner_splits=5):
    if fs_method != "None":
        N = trial.suggest_categorical("N", search_space['N'])
    else:
        # fake to avoid error
        N = trial.suggest_categorical("N", [0])
    clf_method = trial.suggest_categorical("clf_method", search_space['clf_method'])

    if clf_method == "LogisticRegression":
        C_LR = trial.suggest_categorical("C_LR", search_space['C_LR'])
        clf = LogisticRegression(max_iter=500, solver='liblinear', C=C_LR)
    elif clf_method == "NaiveBayes":
        clf = GaussianNB()
    elif clf_method == "RandomForest":
        RF_n_estimators = trial.suggest_categorical("RF_n_estimators", search_space['RF_n_estimators'])
        clf = RandomForestClassifier(n_estimators=RF_n_estimators)
    elif clf_method == "RBFSVM":
        C_SVM = trial.suggest_categorical("C_SVM", search_space['C_SVM'])
        clf = SVC(kernel="rbf", C=C_SVM, gamma='auto', probability=True)

    inner_cv = RepeatedStratifiedKFold(n_splits=num_inner_splits, n_repeats=1, random_state=42)
    y_probs = []
    y_gt = []
    fs_times = []

    for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer, y_train_outer):
        X_inner_train = X_train_outer[inner_train_idx]
        X_inner_val = X_train_outer[inner_val_idx]
        y_inner_train = y_train_outer[inner_train_idx]
        y_inner_val = y_train_outer[inner_val_idx]

        X_inner_train_selected, fsel, fs_time = cached_select_features(X_inner_train, y_inner_train, fs_method, N, dataset, repeat)
        X_inner_val_selected = fsel.transform(X_inner_val)
        fs_times.append(fs_time)

        clf.fit(X_inner_train_selected, y_inner_train)
        y_prob = clf.predict_proba(X_inner_val_selected)[:, 1]

        if np.any(np.isnan(y_prob)):
            is_constant = np.all(X_inner_train_selected == X_inner_train_selected[0])
            if is_constant:
                # this can happen now, if the data is too small, happens with
                # Bhattacharyya. in that case we replace the probs randomly
                random_probs = np.random.random(size=len(y_prob))
                y_prob = random_probs / np.sum(random_probs)
            else:
                # this should never happen
                print("\n\n\nNaN values found in y_gt_flat with non-constant features")
                print(f"Feature selection method: {fs_method}")
                print("Selected features:")
                print(X_inner_train_selected)
                print("Training labels:")
                print(y_inner_train)
                raise Exception("NaN values detected in non-constant features")

        y_probs.append(y_prob)
        y_gt.append(y_inner_val)

    y_prob_flat = [p for y in y_probs for p in y]
    y_true_flat = [gt for y in y_gt for gt in y]
    cv_auc = roc_auc_score(y_true_flat, y_prob_flat)

    trial.set_user_attr("fs_time", str(fs_times))
    trial.set_user_attr("y_prob_int", y_prob_flat)
    trial.set_user_attr("y_true_int", y_true_flat)
    trial.set_user_attr("fs_method", fs_method)
    trial.set_user_attr("auc_int", cv_auc)

    return cv_auc



def evaluate_feature_selection_method(fs_method, dataset, X, y, outer_train_idx, outer_test_idx, repeat):
    X_train_outer = X[outer_train_idx]
    X_test_outer = X[outer_test_idx]
    y_train_outer = y[outer_train_idx]
    y_test_outer = y[outer_test_idx]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study = optuna.create_study(sampler=optuna.samplers.BruteForceSampler(), direction="maximize")
        study.optimize(lambda trial: inner_objective(trial, X_train_outer, y_train_outer, fs_method, dataset, repeat, n_inner_cv))
        best_params = study.best_params
        auc_cv_int = study.best_value
        # df = study.trials_dataframe()

    clf = getClassifier(best_params)
    X_train_outer_selected, fsel, fs_time = cached_select_features(X_train_outer, y_train_outer, fs_method, best_params['N'], dataset, repeat)
    clf.fit(X_train_outer_selected, y_train_outer)

    X_test_selected = fsel.transform(X_test_outer)
    y_prob = clf.predict_proba(X_test_selected)[:, 1]
    auc_fold = roc_auc_score(y_test_outer, y_prob)

    return {
        'y_prob': y_prob,
        'y_true': y_test_outer,
        'y_prob_int': study.best_trial.user_attrs.get("y_prob_int"),
        'y_true_int': study.best_trial.user_attrs.get("y_true_int"),
        'auc_int': auc_cv_int,
        'auc_fold': auc_fold,
        'best_params': best_params,
        'fs_time': fs_time
    }



def nested_cv_optimization(fs_method, dataset, repeat):
    os.makedirs("./results", exist_ok=True)

    # init cache
    global selection_cache
    cache_key = f"{fs_method}_{dataset}_{repeat}"

    output_path = f"./results/cv_{dataset}_{repeat}_{fs_method}.pkl"
    if os.path.exists(output_path):
        return None

    print(f"Starting nested CV for {fs_method} on dataset {dataset} with repeat {repeat}")
    X, y = radMLBench.loadData(dataset, return_X_y=True, local_cache_dir="./datasets")
    outer_cv = RepeatedStratifiedKFold(n_splits=n_outer_cv, n_repeats=1, random_state=repeat)

    if fs_method in ["NMF", "LDA"]:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.copy()

    fold_results = []
    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X_scaled, y)):
        selection_cache[cache_key] = {}
        result = evaluate_feature_selection_method(fs_method, dataset, X_scaled, y, outer_train_idx, outer_test_idx, repeat)
        fold_results.append(result)

        # cache cannot be useful when there is another outer test/train (very unlikely so)
        del selection_cache[cache_key]

    # clear cache
    if cache_to == "disk":
        files = glob.glob(f"./cache/*_{fs_method}_{dataset}_{repeat}")
        for file in files:
            os.remove(file)

    y_true_all = np.concatenate([r['y_true'] for r in fold_results])
    y_prob_all = np.concatenate([r['y_prob'] for r in fold_results])
    auc = roc_auc_score(y_true_all, y_prob_all)

    results = {
        'dataset': dataset,
        'repeat': repeat,
        'fs_method': fs_method,
        'auc': auc,
        'fold_results': fold_results,
        'total_fs_time': sum(r['fs_time'] for r in fold_results)
    }

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    return results



if __name__ == '__main__':
    datasets = radMLBench.listDatasets("nInstances")

    if cache_to == "memory":
        cached_select_features = cached_select_features_memory
    else:
        cached_select_features = cached_select_features_disk

    # 128gb is not enough if SRP? or MRMRe? are all running at the same time
    experiments = [(fs_method, dataset, repeat)
        for repeat in range(num_repeats)
        for dataset in datasets
        for fs_method in search_space['fs_method']]
    random.shuffle(experiments)

    results = Parallel(n_jobs=30)(
        delayed(nested_cv_optimization)(fs_method, dataset, repeat)
            for (fs_method, dataset, repeat) in experiments
    )

#
