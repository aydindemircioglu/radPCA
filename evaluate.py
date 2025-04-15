from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
from scipy.stats import wilcoxon, linregress
from sklearn.metrics import f1_score

from matplotlib import gridspec

import pickle
import cv2
from glob import glob

from joblib import dump, load
import numpy as np
import pandas as pd
import seaborn as sns

from PIL import Image
from PIL import ImageDraw, ImageFont

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


import radMLBench
from utils import *


n_outer_cv = 5


def readResults ():
    array = np.array
    results = []
    for dataset in radMLBench.listDatasets():
        for z in glob(f"./results/cv_{dataset}_*.pkl"):
            with open(z, "rb") as file:
                df = pickle.load(file)
            # for now..
            repeat = int(z.split("/")[-1].split("_")[2])

            row = {"FSMethod": df["fs_method"]}
            row["Dataset"] = dataset
            row["Repeat"] = repeat
            row["FSTime"] = df["total_fs_time"]
            assert (len(df["fold_results"]) == n_outer_cv) # outer CV
            # outer test folds
            row["y_prob"] = np.concatenate([df["fold_results"][j]["y_prob"] for j in range(n_outer_cv)])
            row["y_true"] = np.concatenate([df["fold_results"][j]["y_true"] for j in range(n_outer_cv)])
            row["Params"] = np.array([df["fold_results"][j]["best_params"] for j in range(n_outer_cv)])

            # the outer AUC is exactly this
            #auc = roc_auc_score(row["y_true"], row["y_prob"])
            row["AUC"] = df["auc"] # this is pooled

            thresholds = np.linspace(0.01, 0.99, 99)
            #f1s = [f1_score(row["y_true"], row["y_prob"] >= t) for t in thresholds]
            f1s = [f1_score(row["y_true"], row["y_prob"] >= t, zero_division=0) for t in thresholds]
            best_f1 = np.max(f1s)
            row["F1"] = best_f1


            # also extract inner CV at least, but currently not used
            tmp = pd.DataFrame(df["fold_results"])
            row["AUC_Inner"] = tmp["auc_int"].values # this is pooled, but internally
            row["AUC_Outer"] = tmp["auc_fold"].values # this is unpooled
            results.append(row)

    results = pd.DataFrame(results).reset_index(drop = True)

    # assert that everything is complete
    for z in results["FSMethod"].unique():
        for d in results["Dataset"].unique():
            subdf = results.query("FSMethod == @z and Dataset == @d")
            assert(len(subdf) == 12)

    return results


def getBestTable (results, metric = "AUC"):
    bestTable = []
    for dataset in results['Dataset'].unique():
        crow = {"Dataset": dataset}
        for proj in allMethods:
            tmpdf = results.query("Dataset == @dataset and FSMethod == @proj")
            bestP = tmpdf.sort_values([metric]).iloc[-1][metric]
            crow[proj] = bestP
        bestTable.append(crow)
    bestTable = pd.DataFrame(bestTable).sort_values(["Dataset"])
    return bestTable


def getRankingTable (results):
    rTable = {}
    for m in ["AUC", "F1"]:
        tableM = getBestTable (results, m)
        tableM = tableM.drop(["Dataset"], axis = 1).T

        tR = tableM.rank(axis = 0, ascending = False)
        tR = tR.mean(axis = 1)
        tR = pd.DataFrame(tR).round(1)
        tR.columns = [f"Mean rank ({m})"]
        tR = tR.sort_values([f"Mean rank ({m})"])

        tA = tableM.mean(axis=1)
        tA = tA.loc[tR.index]
        rTable[m] = tR.copy()
        rTable[m][f"Mean {m}"] = tA.round(3)

        tM = tableM.mean(axis=1)
        tM = tM.loc[tR.index]
        tM = tM - tM["None"]
        tM = tM.round(3)
        rTable[m][f"Mean gain in {m}"] = tM

        # how often the method performed best
        tC = tableM.rank(axis = 0, ascending = False)
        tX = tC.min(axis = 0)
        tB = np.sum(tC == tX, axis = 1)
        tB = tB.loc[tR.index]
        rTable[m][f"Best-performing datasets count ({m})"] = tB

        tX = tableM - tableM.loc["None"]
        tX = tX.max(axis = 1)
        tX = tX.round(3)
        rTable[m][f"Maximum gain in {m}"] = tX
        rTable[m].index = getNames(rTable[m].index)

    #rTable["F1"] = rTable["F1"].loc[rTable["AUC"].index]

    drawArray(rTable["F1"], aspect = 0.6, fsize = (10,7), \
        cmap = [("-", 6.8, (6.8+13.3)/2, 13.3), \
                ("+", 0.710, (0.710+0.747)/2, 0.747), \
                ("+", -0.025, 0.0, 0.025), \
                ("+", 0, 4, 8),
                ("g", 0, 0.08, 0.16)],
                fName = "FigRanking_F1", paper = True, DPI = 500)

    drawArray(rTable["AUC"], aspect = 0.6, fsize = (10,7), \
        cmap = [("-", 7.9, (7.9+12.6)/2, 12.6), \
                ("+", 0.65, 0.68, 0.71), \
                ("+", -0.05, 0.0, 0.05), \
                ("+", 0, 3.5, 7),\
                ("g", 0, 0.125, 0.25)],
                fName = "FigRanking_AUC", paper = True, DPI = 500)

    rTable["AUC"].to_csv("./paper/ranking_AUC.csv")
    rTable["F1"].to_csv("./paper/ranking_F1.csv")
    return rTable["AUC"].index



def getResults():
    try:
        results = load("./paper/results_trial.dump")
    except:
        # recompute everything
        print("Recomputing AUCs")
        results = readResults ()
        _ = dump(results, "./paper/results_trial.dump")

    return results



def testRanks (results, ranking):
    bestTable = getBestTable (results)
    fMat = bestTable.copy()
    fMat.index = fMat["Dataset"]
    fMat = fMat.drop(["Dataset"], axis = 1).copy()
    fMat.columns = getNames(fMat.columns)
    fMat = fMat[ranking]

    # freedman does not need pre-ranking in scipy
    print (f"Friedman test: {friedmanchisquare(*[fMat[d] for d in fMat.columns])[1]:.3f}")
    scMat = posthoc_nemenyi_friedman(fMat)

    cell_widths = [0.20]*scMat.shape[0]
    scMat.index = getNames(scMat.index)
    scMat.columns = getNames(scMat.columns)

    strMat = scMat.round(3)
    drawArray2(scMat, strMat, fsize = (7,10), cell_widths = cell_widths, hofs = 0.45, vofs = 0.99,\
            colmaps = [("+", 0.0001, 0.05, 1.0)]*scMat.shape[0], ffac = 0.77, DPI = 160,
                    fName = "FigPostHoc")



def getBenefitMatrix (results, ranking):
    df = getBestTable (results)
    K = df.drop(["Dataset"], axis = 1).mean()
    K = K.sort_values()
    K.index = getNames(K.index)
    df.columns = getNames(df.columns)

    comparison_matrix = pd.DataFrame(0, index=ranking, columns=ranking[::-1], dtype=float)
    gain_matrix =  pd.DataFrame(0, index=ranking, columns=ranking[::-1], dtype=float)

    for m in K.index:
        for n in K.index:
            if m != n:
                wins = np.sum(df[m] < df[n])
                losses = np.sum(df[m] > df[n])
                score = -(wins - losses)
                comparison_matrix.loc[n,m] = float(score)
                gain_matrix.loc[m,n] = float(np.mean(df[n]-df[m]))
            else:
                comparison_matrix.loc[m,m] = None
                gain_matrix.loc[m,n] = None

    scMat = comparison_matrix.fillna(0).astype(int)
    strMat = gain_matrix.round(3).astype(str).replace('nan', '')
    strMat = strMat.astype(str).replace('nan', '')
    strMat = scMat.astype(str) + '\n(' + strMat + ')'
    for k in strMat.index:
        strMat.loc[k,k] = None
    cell_widths = [0.25]*scMat.shape[0]

    scMat.index = getNames(scMat.index)
    scMat.columns = getNames(scMat.columns)

    strMat.index = getNames(strMat.index)
    strMat.columns = getNames(strMat.columns)

    drawArray2(scMat, strMat, fsize = (7,10), cell_widths = cell_widths, hofs = 0.45, vofs = 0.99,\
            colmaps = [("+", -25, 0, 25)]*scMat.shape[0], ffac = 0.77, DPI = 150,
                    fName = "FigBenefit")



def getTimings():
    with open('paper/timings.pkl', 'rb') as file:
        timings = pickle.load(file)
    data = []
    N_values = [1, 2, 4, 8, 16, 32]
    global sMethods
    global pMethods
    sMethods = sorted(sMethods)
    pMethods = sorted(pMethods)

    for method_group, methods in [("sMethods", sMethods), ("pMethods", pMethods)]:
        for method in methods:
            for N in N_values:
                times = timings.get(method, {}).get(N, [0])
                for time in times:
                    data.append({
                        'Method': method,
                        'N': N,
                        'Time': time,
                        'Method_Group': method_group
                    })

    df = pd.DataFrame(data)

    sns.set(style="white")
    plt.rc('text', usetex=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"]})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'''
        \usepackage{mathtools}
        \usepackage{helvet}
        \renewcommand{\familydefault}{\sfdefault}        '''
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=400)
    greencol = np.array([0.502, 0.729, 0.419])

    palette = {
        1: tuple(greencol * 1.3),
        2: tuple(greencol * 1.2),
        4: tuple(greencol * 1.1),
        8: tuple(greencol),
        16: tuple(greencol * 0.9),
        32: tuple(greencol * 0.8),
    }


    fig = plt.figure(figsize=(14, 12), dpi=400)
    gs = gridspec.GridSpec(8, 1, height_ratios=[1, 0.1, 3, 0.5, 1, 0.1, 3, 0.5])  # sMethods, s-break, pMethods, p-break

    # sMethods broken axis
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[2], sharex=ax0)
    plt.setp(ax0.get_xticklabels(), visible=False)

    # pMethods broken axis
    ax2 = plt.subplot(gs[4])
    ax3 = plt.subplot(gs[6], sharex=ax2)
    plt.setp(ax2.get_xticklabels(), visible=False)

    pos_ax1 = ax1.get_position()
    pos_ax2 = ax2.get_position()

    low_ylim_S = (0, 3)
    high_ylim_S = (62, 80)

    low_ylim_P = (0, 1)
    high_ylim_P = (1, 13)

    # Plot sMethods data
    sns.barplot(
        x='Method', y='Time', hue='N',
        data=df[df['Method_Group'] == 'sMethods'],
        ax=ax0, dodge=True, palette=palette, errorbar=None
    )
    sns.barplot(
        x='Method', y='Time', hue='N',
        data=df[df['Method_Group'] == 'sMethods'],
        ax=ax1, dodge=True, palette=palette, errorbar=None
    )

    # Plot pMethods data
    sns.barplot(
        x='Method', y='Time', hue='N',
        data=df[df['Method_Group'] == 'pMethods'],
        ax=ax2, dodge=True, palette=palette, errorbar=None
    )
    sns.barplot(
        x='Method', y='Time', hue='N',
        data=df[df['Method_Group'] == 'pMethods'],
        ax=ax3, dodge=True, palette=palette, errorbar=None
    )

    ax0.set_ylim(high_ylim_S)
    ax1.set_ylim(low_ylim_S)
    ax2.set_ylim(high_ylim_P)
    ax3.set_ylim(low_ylim_P)

    ax1.get_legend().remove()
    ax3.get_legend().remove()
    ax0.get_legend().remove()
    ax2.get_legend().remove()

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels,
               title="Dimension",
               title_fontsize=20, fontsize=18,
               loc="upper left",
               fancybox=False,
               facecolor='white',
               bbox_to_anchor=(0.07, 0.98),
               frameon=True)

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels,
               title="Dimension",
               title_fontsize=20, fontsize=18,
               fancybox=False,
               facecolor='white',
               loc="upper left",
               bbox_to_anchor=(0.07, 0.50),
               frameon=True)

    # Break markers
    kwargs = dict(marker=[(-1, -1), (1, 1)], markersize=12,
                  linestyle='none', color='k', mec='k', mew=1, clip_on=False)

    for ax_upper, ax_lower in [(ax0, ax1), (ax2, ax3)]:
        ax_upper.plot([0, 1], [0, 0], transform=ax_upper.transAxes, **kwargs)
        ax_lower.plot([0, 1], [1, 1], transform=ax_lower.transAxes, **kwargs)

    for ax in [ax0, ax1, ax2, ax3]:
        ax.tick_params(axis='y', labelsize=18)
        ax.set_xlabel('')
        ax.set_yticklabels(ax.get_yticks(), fontname='Arial', fontsize=18)

    ax1.set_ylabel('     Time (seconds)', fontsize=22, fontname='Arial')
    ax3.set_ylabel('     Time (seconds)', fontsize=22, fontname='Arial')
    ax0.set_ylabel('', fontsize=22, fontname='Arial')
    ax2.set_ylabel('', fontsize=22, fontname='Arial')

    ax1.set_xticks(np.arange(len(sMethods)))
    ax1.set_xticklabels(getNames(sMethods), fontsize=16, fontname='Arial')
    ax3.set_xticks(np.arange(len(pMethods)))
    ax3.set_xticklabels(getNames(pMethods), fontsize=16, fontname='Arial')

    for ax in [ax0, ax1, ax2, ax3]:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    plt.savefig('./paper/FigTimings.png', bbox_inches='tight')
    plt.close()



def getBestMethods(results):
    diffs = []
    for dataset in results['Dataset'].unique():
        # for each dataset we have 10 repeats, each repeat one AUC, but these
        # are different models, meaning have different N and different classifiers.
        tmpdf = results.query("Dataset == @dataset")
        tmpP = []
        for f in pMethods:
            tmpf = tmpdf.query("FSMethod == @f")
            tmpP.append({"FSMethod": f, "AUC":np.mean(tmpf["AUC"]), "AUC_Std": np.std(tmpf["AUC"])})
        tmpP = pd.DataFrame(tmpP)
        bestP = tmpP.sort_values(["AUC"]).iloc[-1]

        tmpS = []
        for f in sMethods:
            tmpf = tmpdf.query("FSMethod == @f")
            tmpS.append({"FSMethod": f, "AUC":np.mean(tmpf["AUC"]), "AUC_Std": np.std(tmpf["AUC"])})
        tmpS = pd.DataFrame(tmpS)
        bestS = tmpS.sort_values(["AUC"]).iloc[-1]

        # best None,
        tmpN = tmpdf.query("FSMethod == 'None'")
        tmpN = [{"FSMethod": 'None', "AUC": np.mean(tmpN["AUC"]), "AUC_Std": np.std(tmpN["AUC"])}]
        tmpN = pd.DataFrame(tmpN)
        bestN = tmpN.sort_values(["AUC"]).iloc[-1]


        diffs.append({"Dataset": dataset, \
            "NoneAUC": bestN["AUC"],\
            "NoneAUCStr": f'{bestN["AUC"]:.3f} +/- {bestN["AUC_Std"]:.3f}',\
            "DiffAUC_Proj_to_None": bestP["AUC"]-bestN["AUC"],\
            "ProjAUC": bestP["AUC"],\
            "ProjAUCStr": f'{bestP["AUC"]:.3f} +/- {bestP["AUC_Std"]:.3f}',\
            "Projection": bestP["FSMethod"],\
            "ProjAUC": bestP["AUC"],\
            "ProjAUCStr": f'{bestP["AUC"]:.3f} +/- {bestP["AUC_Std"]:.3f}',\
            "Selection": bestS["FSMethod"],\
            "SelAUC": bestS["AUC"],\
            "SelAUCStr": f'{bestS["AUC"]:.3f} +/- {bestS["AUC_Std"]:.3f}',\
            "DiffAUCStr": f'{bestP["AUC"]-bestS["AUC"]:.3f}',\
            "DiffAUC": bestP["AUC"]-bestS["AUC"]})
    diffs = pd.DataFrame(diffs).sort_values(["DiffAUC"])

    print (diffs["Projection"].value_counts())
    print(pd.DataFrame(diffs)["Selection"].value_counts())

    print ("MEAN DIFF Sel to Proj", np.mean(pd.DataFrame(diffs)["DiffAUC"]))
    stat, p_value = wilcoxon(diffs['ProjAUC'], diffs['SelAUC'])
    print (f"p-value: {p_value}")

    # this is now None to the BEST of proj, so indeed this can be then better than just NMF
    # but this is of no real interest to us.
    print ("MEAN DIFF Proj to None", np.mean(pd.DataFrame(diffs)["DiffAUC_Proj_to_None"]))
    stat, p_value = wilcoxon(diffs['ProjAUC']*0, diffs['DiffAUC_Proj_to_None'])
    print (f"p-value: {p_value}")

    tmpdiff = diffs.copy()#drop(["DiffAUC"], axis = 1)
    tmpdiff.to_excel("./paper/TableDiffs.xlsx")

    return diffs



def createDatasetTable():
    tbl = []
    for dataset in radMLBench.listDatasets():
        m = radMLBench.getMetaData(dataset)
        _, y = radMLBench.loadData(dataset, return_X_y=True)
        tbl.append({"Dataset": dataset, "Modality": m["modality"], "Outcome": m["outcome"],
            "Instances": m['nInstances'], "Positive Instances": np.sum(y == 1),
            "Negative Instances": np.sum ( y== 0),
            "Features": m["nFeatures"], "Dimensionality": m["Dimensionality"], "Balance": m["ClassBalance"]})
    tbl = pd.DataFrame(tbl)
    tbl
    tbl.to_excel("./paper/TableDatasets.xlsx")



def bootstrap_regression(x, y, n_bootstrap=1000):
    bootstrap_slopes = []
    bootstrap_intercepts = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(range(len(x)), size=len(x), replace=True)
        x_sample = x[indices]
        y_sample = y[indices]
        slope, intercept, *_ = linregress(x_sample, y_sample)
        bootstrap_slopes.append(slope)
        bootstrap_intercepts.append(intercept)
    return np.array(bootstrap_slopes), np.array(bootstrap_intercepts)



def testRelations(diffs, results, DPI = 300):
    for r in range(len(diffs)):
        dataset = diffs.iloc[r]["Dataset"]
        diffs.at[r, "nFeatures"] = radMLBench.getMetaData(dataset)["nFeatures"]
        diffs.at[r, "nInstances"] = radMLBench.getMetaData(dataset)["nInstances"]
        diffs.at[r, "D"] = radMLBench.getMetaData(dataset)["Dimensionality"]

    diffs = diffs.sort_values(["DiffAUC"])
    nFeatures_values = diffs["nFeatures"].values
    nInstances_values = diffs["nInstances"].values
    dimensionality_values = diffs["D"].values
    diff_values = diffs["DiffAUC"].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi = DPI)
    axis_fontsize = 16
    title_fontsize = 15

    # never understand this shit
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "text.usetex": False,
    })

    plt.subplots_adjust(wspace=4.7)
    print (diff_values)
    # Plot 1: Number of Features
    slope, intercept, r_value, p_value, std_err = linregress(diff_values, nFeatures_values)
    slopes, intercepts = bootstrap_regression(diff_values, nFeatures_values)
    y_pred_mean = slope * diff_values + intercept
    y_pred_bootstrap = [s * diff_values + b for s, b in zip(slopes, intercepts)]
    y_pred_lower = np.percentile(y_pred_bootstrap, 2.5, axis=0)
    y_pred_upper = np.percentile(y_pred_bootstrap, 97.5, axis=0)

    axes[0].scatter(diff_values, nFeatures_values, s=15, color='black')
    axes[0].plot(diff_values, y_pred_mean, color='black')
    axes[0].fill_between(diff_values, y_pred_lower, y_pred_upper, color='grey', alpha=0.3)
    axes[0].text(0.05, 0.95, f"$R^2 = {r_value**2:.2f}$\n(p = {p_value:.2g})", transform=axes[0].transAxes, fontsize=15, verticalalignment='top')
    axes[0].set_ylabel("Number of Features", fontsize=axis_fontsize)
    axes[0].set_xlabel("Difference (in AUC)", fontsize=axis_fontsize)
    axes[0].set_xticks([-0.1, -0.05, 0, 0.05])
    axes[0].set_xticklabels(axes[0].get_xticks().round(2), fontsize=14, fontname='Arial')
    axes[0].set_yticklabels(axes[0].get_yticks().astype(np.int32), fontsize=14, fontname='Arial')
    axes[0].text(-0.17, -0.08, "(a)", transform=axes[0].transAxes, fontsize=20, verticalalignment='top', horizontalalignment='left')

    # Plot 2: Number of Instances
    slope, intercept, r_value, p_value, std_err = linregress(diff_values, nInstances_values)
    slopes, intercepts = bootstrap_regression(diff_values, nInstances_values)
    y_pred_mean = slope * diff_values + intercept
    y_pred_bootstrap = [s * diff_values + b for s, b in zip(slopes, intercepts)]
    y_pred_lower = np.percentile(y_pred_bootstrap, 2.5, axis=0)
    y_pred_upper = np.percentile(y_pred_bootstrap, 97.5, axis=0)

    axes[1].scatter(diff_values, nInstances_values, s=15, color='black')
    axes[1].plot(diff_values, y_pred_mean, color='black')
    axes[1].fill_between(diff_values, y_pred_lower, y_pred_upper, color='grey', alpha=0.3)
    axes[1].text(0.05, 0.95, f"$R^2 = {r_value**2:.2f}$\n(p = {p_value:.2g})", transform=axes[1].transAxes, fontsize=15, verticalalignment='top')
    axes[1].set_ylabel("Number of Instances", fontsize=axis_fontsize)
    axes[1].set_xlabel("Difference (in AUC)", fontsize=axis_fontsize)
    axes[1].set_xticks([-0.1, -0.05, 0, 0.05])
    axes[1].set_xticklabels(axes[1].get_xticks().round(2), fontsize=14, fontname='Arial')
    axes[1].set_yticklabels(axes[1].get_yticks().astype(np.int32), fontsize=14, fontname='Arial')
    axes[1].text(-0.17, -0.08, "(b)", transform=axes[1].transAxes, fontsize=20, verticalalignment='top', horizontalalignment='left')

    # Plot 3: Dimensionality (D)
    slope, intercept, r_value, p_value, std_err = linregress(diff_values, dimensionality_values)
    slopes, intercepts = bootstrap_regression(diff_values, dimensionality_values)
    y_pred_mean = slope * diff_values + intercept
    y_pred_bootstrap = [s * diff_values + b for s, b in zip(slopes, intercepts)]
    y_pred_lower = np.percentile(y_pred_bootstrap, 2.5, axis=0)
    y_pred_upper = np.percentile(y_pred_bootstrap, 97.5, axis=0)

    axes[2].scatter(diff_values, dimensionality_values, s=15, color='black')
    axes[2].plot(diff_values, y_pred_mean, color='black')
    axes[2].fill_between(diff_values, y_pred_lower, y_pred_upper, color='grey', alpha=0.3)
    axes[2].text(0.05, 0.95, f"$R^2 = {r_value**2:.2f}$\n(p = {p_value:.2g})", transform=axes[2].transAxes, fontsize=15, verticalalignment='top')
    axes[2].set_ylabel("Dimensionality", fontsize=axis_fontsize)
    axes[2].set_xlabel("Difference (in AUC)", fontsize=axis_fontsize)
    axes[2].set_xticks([-0.1, -0.05, 0, 0.05])
    axes[2].set_xticklabels(axes[2].get_xticks().round(2), fontsize=14, fontname='Arial')
    axes[2].set_yticklabels(axes[2].get_yticks().astype(np.int32), fontsize=14, fontname='Arial')
    axes[2].text(-0.17, -0.08, "(c)", transform=axes[2].transAxes, fontsize=20, verticalalignment='top', horizontalalignment='left')

    plt.tight_layout()
    plt.savefig("./paper/FigRelation.png")



def addText (finalImage, text = '', org = (0,0), fontFace = '', fontSize = 12, color = (255,255,255)):
     # Convert the image to RGB (OpenCV uses BGR)
     #tmpImg = cv2.cvtColor(finalImage, cv2.COLOR_BGR2RGB)
     tmpImg = finalImage
     pil_im = Image.fromarray(tmpImg)
     draw = ImageDraw.Draw(pil_im)
     font = ImageFont.truetype(fontFace + ".ttf", fontSize)
     draw.text(org, text, font=font, fill = color)
     #tmpImg = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
     tmpImg = np.array(pil_im)
     return (tmpImg.copy())




def addBorder (img, pos, thickness):
    if pos == "H":
        img = np.hstack([255*np.ones(( img.shape[0],int(img.shape[1]*thickness), 3), dtype = np.uint8),img])
    if pos == "V":
        img = np.vstack([255*np.ones(( int(img.shape[0]*thickness), img.shape[1], 3), dtype = np.uint8),img])
    return img



def addBlackBorder(img, pixel):
    return cv2.copyMakeBorder(img, pixel, pixel, pixel, pixel, cv2.BORDER_CONSTANT, value=(0, 0, 0))



def joinRankingFigures():
    fontFace = "Arial"

    imA = cv2.imread("./paper/FigRanking_AUC.png")
    imA = addText(imA, "(a)", (40, 40), fontFace, 148, color=(0, 0, 0))

    imB = cv2.imread("./paper/FigRanking_F1.png")
    imB = cv2.resize(imB, (imA.shape[1], imA.shape[0]), interpolation=cv2.INTER_LINEAR)
    imB = addText(imB, "(b)", (40, 40), fontFace, 148, color=(0, 0, 0))

    # imA = addBlackBorder(imA, 10)
    # imB = addBlackBorder(imB, 10)
    #imB = addText(imB, "b", (40, 40), fontFace, 112, color=(0, 0, 0))

    imB = addBorder(imB, "H", 0.10)
    imgU = np.hstack([imA, imB])

    cv2.imwrite("./paper/FigRanking.png", imgU)



if __name__ == '__main__':
    results = getResults()
    createDatasetTable()

    # just ensure we have all methods
    assert set(results["FSMethod"]) == set(allMethods)

    # main table
    ranking = getRankingTable(results)
    joinRankingFigures()

    getBenefitMatrix (results, ranking)
    testRanks (results, ranking)
    diffs = getBestMethods(results)
    testRelations (diffs, results)

    getTimings()


#
