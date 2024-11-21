import statsmodels.stats.api as sms

import re
import math

import numpy as np
import seaborn as sns

import matplotlib.colors as colors
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt

from PIL import Image, ImageChops


# to use them properly, we define these globally
pMethods = ["UMAP", "KernelPCA", "PCA", "ICA", "FA", "NMF", "SRP"]
sMethods = ["Bhattacharyya", "ANOVA", "LASSO", "ET", "Kendall", "MRMRe", "tTest"]
nMethods = ["None"] #  to avoid error when comparing p vs s
allMethods = pMethods.copy()
allMethods.extend(sMethods)
allMethods.extend(nMethods)


def getColor (name):
    # assume pMethod/sMethods are in place
    color = "black"
    if name in pMethods:
        color = "blue"
    if name in sMethods:
        color = "green"
    if name == "t-Score":
        color = "green"
    if name == "kPCA":
        color = 'blue'
    return color




def getCI(arr):
    mean = np.mean(arr)
    confidence_interval = sms.DescrStatsW(arr).tconfint_mean()
    mean_rounded = round(mean, 2)
    ci_lower_rounded = round(confidence_interval[0], 2)
    ci_upper_rounded = round(confidence_interval[1], 2)
    ci_str = f"{ci_lower_rounded} - {ci_upper_rounded}"
    return f"{mean_rounded} ({ci_str})"


def trim_white_borders(file_path):
    img = Image.open(file_path)
    img_data = np.array(img)
    top = 0
    while np.all(img_data[top] == img_data[top, 0]): top += 1
    bottom = img_data.shape[0] - 1
    while np.all(img_data[bottom] == img_data[bottom, 0]): bottom -= 1
    left = 0
    while np.all(img_data[:, left] == img_data[0, left]): left += 1
    right = img_data.shape[1] - 1
    while np.all(img_data[:, right] == img_data[0, right]): right -= 1
    img.crop((left, top, right+1, bottom+1)).save(file_path)



def extract_main_value(value):
    match = re.match(r'([+-]?[0-9]*\.?[0-9]+)', value)
    if match:
        return float(match.group(1))
    return None


def getNames(mlnames):
    oz = []
    for z in mlnames:
        if z == "tTest":
            oz.append("t-Score")
        elif z == "KernelPCA":
            oz.append("kPCA")
        else:
            oz.append(z)
    return oz



def getPal (cmap):
    if cmap == "g":
        diverging_pal = sns.diverging_palette(20, 120, s=60, l=70, as_cmap=False)
        green_color_hex = to_hex(diverging_pal[-1]) # #80ba6b
        pal = sns.light_palette(green_color_hex, reverse=False, as_cmap=True)
    elif cmap == "o":
        pal = sns.light_palette("#ff4433", reverse=False, as_cmap=True)
    elif cmap == "+":
        pal = sns.diverging_palette(20, 120, s=60, l=70, as_cmap=True)
    elif cmap == "-":
        pal  = sns.diverging_palette(120, 20, s=60, l=70, as_cmap=True)
    else:
        pal = sns.light_palette("#ffffff", reverse=False, as_cmap=True)
    return pal



def drawArray2 (scMat, strMat, colmaps = None, cell_widths = None, clipRound = True, fsize = (9,7), vofs = 0.8, hofs = None, aspect = None, DPI = 220, fName = None, ffac = 1.0):
    plt.rc('text', usetex=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"]})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'''
        \usepackage{mathtools}
        \usepackage{helvet}
        \renewcommand{\familydefault}{\sfdefault}        '''

    # Create the Matplotlib figure
    fig, ax = plt.subplots(figsize = fsize, dpi = DPI, constrained_layout=False)

    cmaps = {}
    for j, column in enumerate(scMat.columns):
        cm, vmin, vcenter, vmax = colmaps[j]
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        cmap = getPal(cm)
        cmaps[column] = cmap, norm

    table = plt.table(cellText=scMat.values, colLabels=scMat.columns, rowLabels=scMat.index, cellLoc='center', loc='center', colColours=['#f5f5f5']*len(scMat.columns))

    # Plot the DataFrame as a table with colored cells
    for key, cell in table._cells.items():
        if key[0] == 0:
            cell.set_text_props(weight='bold')
        cell.set_fontsize(12)
        try:
            column_name = scMat.columns[key[1]]
            cmap, norm = cmaps[column_name]
            cell.set_facecolor(cmap(norm(float(cell.get_text().get_text()))))
            cell.set_width(cell_widths[key[1]])
        except:
            pass

    for key, cell in table._cells.items():
        if key[1] == -1:
            cell.set_text_props(ha='right')

        cell.set_fontsize(22)
        try:
            cell.set_width(cell_widths[key[1]])
        except:
            pass
        if key[0] == 0:
            cell.set_height(0.5)  # Set the cell height
            cell.set_text_props(rotation=45, ha='left', va = 'top')  # Rotate and align column names

        else:
            cell.set_height(0.125)  # Set the cell height

    # Remove the x and y ticks
    ax.axis('off')

    # Hide grid lines for index and column names
    table.auto_set_font_size(False)
    table.set_fontsize(22)
    for key, cell in table._cells.items():
        if key[0] == 0 or key[1] == -1:
            cell.set_edgecolor('white')

    for i in range(strMat.shape[0]):
        for j in range(strMat.shape[1]):
            table.get_celld()[(i+1, j)].get_text().set_text(strMat.iloc[i,j])

    # delete columns, we add them in a moment
    for j in range(strMat.shape[1]):
        table.get_celld()[(0, j)].get_text().set_text("")


    ofs = hofs+cell_widths[0]/2
    cp_widths = [ofs*ffac]
    for k in range(1,len(cell_widths)):
        ofs += cell_widths[k-1]/2
        ofs += cell_widths[k]/2
        cp_widths.append(ofs*ffac)


    # fig.canvas.draw()  # Ensure the canvas is rendered
    # for key, cell in table._cells.items():
    #     if key[0] == 0:  # Header row
    #         column_name = scMat.columns[key[1]]
    #         cell_bbox = cell.get_window_extent(fig.canvas.get_renderer())
    #         ax_bbox = ax.get_window_extent(fig.canvas.get_renderer())
    #         # Calculate the relative x position based on the cell's center
    #         x_rel = (cell_bbox.x0 + cell_bbox.x1) / 2 / ax_bbox.width + hofs
    #         bbox = ax.get_position() # Get the axes' bounding box
    #         vofs_dynamic = bbox.ymax + vofs
    #         ax.annotate(column_name,
    #                     xytext=(x_rel, vofs_dynamic),
    #                     textcoords='figure fraction',
    #                     xy=(0, 0),
    #                     fontsize=22, rotation=45)

    for j, (key, cell) in enumerate(table._cells.items()):
        if j == 0:
            cofs = cell.get_x()
        if key[0] == 0:
            column_name = scMat.columns[key[1]]
            cell.set_facecolor('white')
            bbox = ax.get_position()  # Get the axes' bounding box
            vofs_dynamic = bbox.ymax + vofs
            ax.annotate(column_name,
                        xytext=(cp_widths[key[1]], vofs_dynamic),
                        textcoords='figure fraction',
                        xy=(0, 0),
                        fontsize=22, rotation=45)
    plt.tight_layout()

    if fName is not None:
        fig.savefig(f"./paper/{fName}.png", facecolor = 'w', bbox_inches='tight')
        # tight_layout does not work whyever sometimes.
        trim_white_borders(f"./paper/{fName}.png")



def drawArray (table3, cmap = None, clipRound = True, fsize = (9,7), aspect = None, DPI = 400, fontsize = None, fName = None, paper = False):
    def colorticks(event=None):
        locs, labels = plt.xticks()
        for k in range(len(labels)):
            labels[k].set_color(getColor(labels[k]._text))

        locs, labels = plt.yticks()
        for k in range(len(labels)):
            labels[k].set_color(getColor(labels[k]._text))


    #table3 = tO.copy()
    table3 = table3.copy()
    if clipRound == True:
        for k in table3.index:
            for l in table3.columns:
                if str(table3.loc[k,l])[-2:] == ".0":
                    table3.loc[k,l] = str(int(table3.loc[k,l]))
    # display graphically
    scMat = table3.copy()
    strMat = table3.copy()
    strMat = strMat.astype( dtype = "str")
    # replace nans in strMat
    strMat = strMat.replace("nan", "")

    if 1 == 1:
        plt.rc('text', usetex=True)
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"]})
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'''
            \usepackage{mathtools}
            \usepackage{helvet}
            \renewcommand{\familydefault}{\sfdefault}        '''

        fig, ax = plt.subplots(figsize = fsize, dpi = DPI)
        sns.set(style='white')
        #ax = sns.heatmap(scMat, annot = cMat, cmap = "Blues", fmt = '', annot_kws={"fontsize":21}, linewidth = 2.0, linecolor = "black")
        dx = np.asarray(scMat, dtype = np.float64)

        if len(cmap) > 1:
            for j, (cm, vmin, vcenter, vmax) in enumerate(cmap):
                pal = getPal(cm)
                m = np.ones_like(dx)
                m[:,j] = 0
                Adx = np.ma.masked_array(dx, m)
                tnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                ax.imshow(Adx, cmap=pal, norm = tnorm, interpolation='nearest', aspect = aspect)
                #cba = plt.colorbar(pa,shrink=0.25)
        else:
            if cmap[0][0] == "*":
                for j in range(scMat.shape[1]):
                    pal = getPal("o")
                    m = np.ones_like(dx)
                    m[:,j] = 0
                    Adx = np.ma.masked_array(dx, m)
                    vmin = np.min(scMat.values[:,j])
                    vmax = np.max(scMat.values[:,j])
                    vcenter = (vmin + vmax)/2
                    tnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                    ax.imshow(Adx, cmap=pal, norm = tnorm, interpolation='nearest', aspect = aspect)
                    #cba = plt.colorbar(pa,shrink=0.25)
            else:
                cm, vmin, vcenter, vmax = cmap[0]
                pal = getPal(cm)
                tnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                ax.imshow(dx, cmap=pal, norm = tnorm, interpolation='nearest', aspect = aspect)

        # Major ticks
        mh, mw = scMat.shape
        ax.set_xticks(np.arange(0, mw, 1))
        ax.set_yticks(np.arange(0, mh, 1))

        # Minor ticks
        ax.set_xticks(np.arange(-.5, mw, 1), minor=True)
        ax.set_yticks(np.arange(-.5, mh, 1), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

        for i, c in enumerate(scMat.index):
            for j, f in enumerate(scMat.keys()):
                ax.text(j, i, strMat.at[c, f],    ha="center", va="center", color="k", fontsize = fontsize)
        plt.tight_layout()
        ax.xaxis.set_ticks_position('top') # the rest is the same
        ax.set_xticklabels(scMat.keys(), rotation = 45, ha = "left", fontsize = fontsize)
        ax.set_yticklabels(scMat.index, rotation = 0, ha = "right", fontsize = fontsize)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_tick_params ( labelsize= fontsize)
        colorticks()

    if fName is not None:
        fig.savefig(f"./paper/{fName}.png", facecolor = 'w', bbox_inches='tight')
        trim_white_borders(f"./paper/{fName}.png")


#
