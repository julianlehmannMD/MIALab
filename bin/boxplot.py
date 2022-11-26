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

    T1_results_summary = pd.read_csv('mia-result/results_all/Summary/T1_results_summary.csv', sep=';')
    T1_and_T2_results_summary = pd.read_csv('mia-result/results_all/Summary/T1_and_T2_results_summary.csv', sep=';')
    standard_7_results_summary = pd.read_csv('mia-result/results_all/Summary/7_standard_results_summary.csv', sep=';')

    # Evaluate Dice and Housedorf
    results_dict = {1: T1_results_summary, 2: T1_and_T2_results_summary, 3: standard_7_results_summary}
    dict_dice_multiplied_by_std_labels = {'T1': {}, '7_standard': {}, 'T1_and_T2': {}}

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
        amygdala_dice_std_times_mean = (1 - np.asarray(
            summary_value[np.logical_and(amygdala_idx, np.logical_and(dice_idx, mean_idx))])) * np.asarray(
            summary_value[np.logical_and(amygdala_idx, np.logical_and(dice_idx, std_idx))])

        greyMatter_idx = np.asarray(summary_label[:] == 'GreyMatter')
        greyMatter_dice_std_times_mean = (1 - np.asarray(
            summary_value[np.logical_and(greyMatter_idx, np.logical_and(dice_idx, mean_idx))])) * np.asarray(
            summary_value[np.logical_and(greyMatter_idx, np.logical_and(dice_idx, std_idx))])

        hippocampus_idx = np.asarray(summary_label[:] == 'Hippocampus')
        hippocampus_dice_std_times_mean = (1 - np.asarray(
            summary_value[np.logical_and(hippocampus_idx, np.logical_and(dice_idx, mean_idx))])) * np.asarray(
            summary_value[np.logical_and(hippocampus_idx, np.logical_and(dice_idx, std_idx))])

        thalamus_idx = np.asarray(summary_label[:] == 'Thalamus')
        thalamus_dice_std_times_mean = (1 - np.asarray(
            summary_value[np.logical_and(thalamus_idx, np.logical_and(dice_idx, mean_idx))])) * np.asarray(
            summary_value[np.logical_and(thalamus_idx, np.logical_and(dice_idx, std_idx))])

        whiteMatter_idx = np.asarray(summary_label[:] == 'WhiteMatter')
        whiteMatter_dice_std_times_mean = (1 - np.asarray(
            summary_value[np.logical_and(whiteMatter_idx, np.logical_and(dice_idx, mean_idx))])) * np.asarray(
            summary_value[np.logical_and(whiteMatter_idx, np.logical_and(dice_idx, std_idx))])

        if i == 1:
            dict_ = 'T1'
        elif i == 2:
            dict_ = 'T1_and_T2'
        elif i == 3:
            dict_ = '7_standard'

        dict_dice_multiplied_by_std_labels[dict_] = {'Amygdala': amygdala_dice_std_times_mean.item(),
                                                     'GreyMatter': greyMatter_dice_std_times_mean.item(),
                                                     'Hippocampus': hippocampus_dice_std_times_mean.item(),
                                                     'Thalamus': thalamus_dice_std_times_mean.item(),
                                                     'WhiteMatter': whiteMatter_dice_std_times_mean.item()}

    # plot results
    plt.plot(range(len(dict_dice_multiplied_by_std_labels['T1'])),
             list(dict_dice_multiplied_by_std_labels['T1'].values()), label='T1')
    plt.xticks(range(len(dict_dice_multiplied_by_std_labels['T1'])),
               list(dict_dice_multiplied_by_std_labels['T1'].keys()))
    plt.plot(range(len(dict_dice_multiplied_by_std_labels['T1_and_T2'])),
             list(dict_dice_multiplied_by_std_labels['T1_and_T2'].values()), label='T1 and T2')
    plt.plot(range(len(dict_dice_multiplied_by_std_labels['7_standard'])),
             list(dict_dice_multiplied_by_std_labels['7_standard'].values()), label='7 standard')
    plt.ylabel(('(1- Mean Dice)* STD Dice'))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
