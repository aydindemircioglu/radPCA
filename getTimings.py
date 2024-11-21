import time
import radMLBench
import os
import pickle
from joblib import Parallel, delayed, parallel_backend
from threadpoolctl import threadpool_limits

from experiment import *


os.environ["OMP_NUM_THREADS"] = "1"  # Controls OpenMP threads (used by many libraries like numpy, sklearn, etc.)
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Controls OpenBLAS threads (used by numpy, scipy)
os.environ["MKL_NUM_THREADS"] = "1"  # Controls Intel MKL threads (used by numpy, scipy)
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Controls macOS Accelerate threads
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # Controls NumExpr threads



search_space = {
    'fs_method': ["Bhattacharyya", "ANOVA", "LASSO", "ET", "Kendall", "MRMRe", "tTest",\
            "UMAP", "KernelPCA", "PCA", "ICA", "FA", "NMF", "SRP", "None"],
    'N': [2**k for k in range(0,6)],
}



def getColor(method):
    import seaborn as sns
    colors = sns.color_palette("tab10")
    method_index = hash(method) % len(colors)
    return colors[method_index]



def process_dataset(dataset):
    print(f"Timing for dataset {dataset}")
    X, y = radMLBench.loadData(dataset, return_X_y=True, local_cache_dir="./datasets")
    dataset_timings = {fs_method: {N: [] for N in search_space['N']} for fs_method in search_space['fs_method']}

    for fs_method in search_space['fs_method']:
        for N in search_space['N']:
            if fs_method in ["NMF"]:
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X.copy()

            start_time = time.time()
            with threadpool_limits(limits=1, user_api="blas"):
                X_train_selected, fsel = select_features(X_scaled, y, fs_method, N)
            end_time = time.time()
            total_fs_time = end_time - start_time
            dataset_timings[fs_method][N].append(total_fs_time)

    return dataset, dataset_timings



if __name__ == '__main__':
    datasets = radMLBench.listDatasets("nInstances")
    results = Parallel(n_jobs=25)(delayed(process_dataset)(dataset) for dataset in datasets)

    with open("./paper/timings_pre.pkl", 'wb') as f:
        pickle.dump(results, f)

    # Combine results
    timings = {fs_method: {N: [] for N in search_space['N']} for fs_method in search_space['fs_method']}
    for dataset, dataset_timings in results:
        for fs_method, N_times in dataset_timings.items():
            for N, times in N_times.items():
                timings[fs_method][N].extend(times)

    with open("./paper/timings.pkl", 'wb') as f:
        pickle.dump(timings, f)


#
