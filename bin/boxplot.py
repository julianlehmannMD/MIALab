import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import listdir
import csv
import seaborn as sns


def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


def main():
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    #  in a boxplot

    # Read in data file
    T1_results = pd.read_csv('mia-result/results_all/Normal/T1_results.csv', sep=';')
    T1_and_T2_results = pd.read_csv('mia-result/results_all/Normal/T1_and_T2_results.csv', sep=';')
    standard_7_results = pd.read_csv('mia-result/results_all/Normal/7_standard_results.csv', sep=';')
    # T1_and_gradients_results = pd.read_csv('mia-result/results_all/Normal/T1_and_Gradients_results.csv', sep=';')

    # Assign feature number to new data set
    data1 = pd.DataFrame(T1_results).assign(Features='T1 intensity')
    data2 = pd.DataFrame(T1_and_T2_results).assign(Features='T1 and T2 intensities')
    # data3 = pd.DataFrame(T1_and_gradients_results).assign(Features='T1 intensity and T1&T2 gradients')
    data4 = pd.DataFrame(standard_7_results).assign(Features='7 standard')

    # Concat data sets for boxplot
    cdf = pd.concat([data1, data2, data4])
    # ax = sns.boxplot(x="LABEL", y="HDRFDST", hue="Features", data=cdf)
    # plt.show()

    _1 = pd.read_csv('mia-result/results_all/Summary/1.csv', sep=';')
    _2 = pd.read_csv('mia-result/results_all/Summary/2.csv', sep=';')
    _3 = pd.read_csv('mia-result/results_all/Summary/3.csv', sep=';')
    _4 = pd.read_csv('mia-result/results_all/Summary/4.csv', sep=';')
    _5 = pd.read_csv('mia-result/results_all/Summary/5.csv', sep=';')
    _6 = pd.read_csv('mia-result/results_all/Summary/6.csv', sep=';')

    # Evaluate Dice and Housedorf
    results_dict = {1: _1, 2: _2, 3: _3, 4: _4, 5: _5, 6: _6}
    dict_dice_multiplied_by_std_labels = {'1': {}, '2': {}, '3': {}, '4': {}, '5': {}, '6': {}}
    dict_dice_mean = {'1': {}, '2': {}, '3': {}, '4': {}, '5': {}, '6': {}}
    dict_dice_std = {'1': {}, '2': {}, '3': {}, '4': {}, '5': {}, '6': {}}

    for i in results_dict:
        data = results_dict[i]
        summary_metric = data['METRIC']
        summary_label = data['LABEL']
        summary_statistic = data['STATISTIC']
        summary_value = data['VALUE']

        dice_idx = np.asarray(summary_metric[:] == 'DICE')
        mean_idx = np.asarray(summary_statistic[:] == 'MEAN')
        std_idx = np.asarray(summary_statistic[:] == 'STD')

        amygdala_idx = np.asarray(summary_label[:] == 'Amygdala')
        mean = np.asarray(summary_value[np.logical_and(amygdala_idx, np.logical_and(dice_idx, mean_idx))])
        std = np.asarray(summary_value[np.logical_and(amygdala_idx, np.logical_and(dice_idx, std_idx))])
        if mean == 0:
            std = np.nan
        amygdala_dice_std_times_mean = (1 - mean) * std
        amygdala_dice_mean = mean
        amygdala_dice_std = std

        greyMatter_idx = np.asarray(summary_label[:] == 'GreyMatter')
        mean = np.asarray(summary_value[np.logical_and(greyMatter_idx, np.logical_and(dice_idx, mean_idx))])
        std = np.asarray(summary_value[np.logical_and(greyMatter_idx, np.logical_and(dice_idx, std_idx))])
        if mean == 0:
            std = np.nan
        greyMatter_dice_std_times_mean = (1 - mean) * std
        greyMatter_dice_mean = mean
        greyMatter_dice_std = std

        hippocampus_idx = np.asarray(summary_label[:] == 'Hippocampus')
        mean = np.asarray(summary_value[np.logical_and(hippocampus_idx, np.logical_and(dice_idx, mean_idx))])
        std = np.asarray(summary_value[np.logical_and(hippocampus_idx, np.logical_and(dice_idx, std_idx))])
        if mean == 0:
            std = np.nan
        hippocampus_dice_std_times_mean = (1 - mean) * std
        hippocampus_dice_mean = mean
        hippocampus_dice_std = std

        thalamus_idx = np.asarray(summary_label[:] == 'Thalamus')
        mean = np.asarray(summary_value[np.logical_and(thalamus_idx, np.logical_and(dice_idx, mean_idx))])
        std = np.asarray(summary_value[np.logical_and(thalamus_idx, np.logical_and(dice_idx, std_idx))])
        if mean == 0:
            std = np.nan
        thalamus_dice_std_times_mean = (1 - mean) * std
        thalamus_dice_mean = mean
        thalamus_dice_std = std

        whiteMatter_idx = np.asarray(summary_label[:] == 'WhiteMatter')
        mean = np.asarray(summary_value[np.logical_and(whiteMatter_idx, np.logical_and(dice_idx, mean_idx))])
        std = np.asarray(summary_value[np.logical_and(whiteMatter_idx, np.logical_and(dice_idx, std_idx))])
        if mean == 0:
            std = np.nan
        whiteMatter_dice_std_times_mean = (1 - mean) * std
        whiteMatter_dice_mean = mean
        whiteMatter_dice_std = std

        if i == 1:
            dict_ = '1'
        elif i == 2:
            dict_ = '2'
        elif i == 3:
            dict_ = '3'
        elif i == 4:
            dict_ = '4'
        elif i == 5:
            dict_ = '5'
        elif i == 6:
            dict_ = '6'
        dict_dice_multiplied_by_std_labels[dict_] = {'Amygdala': amygdala_dice_std_times_mean.item(),
                                                     'GreyMatter': greyMatter_dice_std_times_mean.item(),
                                                     'Hippocampus': hippocampus_dice_std_times_mean.item(),
                                                     'Thalamus': thalamus_dice_std_times_mean.item(),
                                                     'WhiteMatter': whiteMatter_dice_std_times_mean.item()}

        dict_dice_mean[dict_] = {'Amygdala': amygdala_dice_mean.item(),
                                 'GreyMatter': greyMatter_dice_mean.item(),
                                 'Hippocampus': hippocampus_dice_mean.item(),
                                 'Thalamus': thalamus_dice_mean.item(),
                                 'WhiteMatter': whiteMatter_dice_mean.item()}

        dict_dice_std[dict_] = {'Amygdala': amygdala_dice_std.item(),
                                'GreyMatter': greyMatter_dice_std.item(),
                                'Hippocampus': hippocampus_dice_std.item(),
                                'Thalamus': thalamus_dice_std.item(),
                                'WhiteMatter': whiteMatter_dice_std.item()}

    # plot results
    plt.plot(range(len(dict_dice_multiplied_by_std_labels['1'])),
             list(dict_dice_multiplied_by_std_labels['1'].values()), '--o', label='1')
    plt.xticks(range(len(dict_dice_multiplied_by_std_labels['1'])),
               list(dict_dice_multiplied_by_std_labels['1'].keys()))
    plt.plot(range(len(dict_dice_multiplied_by_std_labels['2'])),
             list(dict_dice_multiplied_by_std_labels['2'].values()), '--o', label='2')
    plt.plot(range(len(dict_dice_multiplied_by_std_labels['3'])),
             list(dict_dice_multiplied_by_std_labels['3'].values()), '--o', label='3')
    plt.plot(range(len(dict_dice_multiplied_by_std_labels['4'])),
             list(dict_dice_multiplied_by_std_labels['4'].values()), '--o', label='4')
    plt.plot(range(len(dict_dice_multiplied_by_std_labels['5'])),
             list(dict_dice_multiplied_by_std_labels['5'].values()), '--o', label='5')
    plt.plot(range(len(dict_dice_multiplied_by_std_labels['6'])),
             list(dict_dice_multiplied_by_std_labels['6'].values()), '--o', label='6')
    plt.ylabel(('(1- Mean Dice)* STD Dice'))
    plt.legend()
    plt.show()

    plt.plot(range(len(dict_dice_mean['1'])),
             list(dict_dice_mean['1'].values()), '--o', label='1')
    plt.xticks(range(len(dict_dice_mean['1'])),
               list(dict_dice_mean['1'].keys()))
    plt.plot(range(len(dict_dice_mean['2'])),
             list(dict_dice_mean['2'].values()), '--o', label='2')
    plt.plot(range(len(dict_dice_mean['3'])),
             list(dict_dice_mean['3'].values()), '--o', label='3')
    plt.plot(range(len(dict_dice_mean['4'])),
             list(dict_dice_mean['4'].values()), '--o', label='4')
    plt.plot(range(len(dict_dice_mean['5'])),
             list(dict_dice_mean['5'].values()), '--o', label='5')
    plt.plot(range(len(dict_dice_mean['6'])),
             list(dict_dice_mean['6'].values()), '--o', label='6')
    plt.ylabel(('Mean Dice'))
    plt.legend()
    plt.show()

    plt.plot(range(len(dict_dice_std['1'])),
             list(dict_dice_std['1'].values()), '--o', label='1')
    plt.xticks(range(len(dict_dice_std['1'])),
               list(dict_dice_std['1'].keys()))
    plt.plot(range(len(dict_dice_std['2'])),
             list(dict_dice_std['2'].values()), '--o', label='2')
    plt.plot(range(len(dict_dice_std['3'])),
             list(dict_dice_std['3'].values()), '--o', label='3')
    plt.plot(range(len(dict_dice_std['4'])),
             list(dict_dice_std['4'].values()), '--o', label='4')
    plt.plot(range(len(dict_dice_std['5'])),
             list(dict_dice_std['5'].values()), '--o', label='5')
    plt.plot(range(len(dict_dice_std['6'])),
             list(dict_dice_std['6'].values()), '--o', label='6')
    plt.ylabel(('STD Dice'))
    plt.legend()
    plt.show()
if __name__ == '__main__':
    main()
