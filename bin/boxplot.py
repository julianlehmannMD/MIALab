import time

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

    _0 = pd.read_csv('mia-result/results_all/Summary/0.csv', sep=';')
    _1 = pd.read_csv('mia-result/results_all/Summary/1.csv', sep=';')
    _2 = pd.read_csv('mia-result/results_all/Summary/2.csv', sep=';')
    _3 = pd.read_csv('mia-result/results_all/Summary/3.csv', sep=';')
    _4 = pd.read_csv('mia-result/results_all/Summary/4.csv', sep=';')
    _5 = pd.read_csv('mia-result/results_all/Summary/5.csv', sep=';')
    _6 = pd.read_csv('mia-result/results_all/Summary/6.csv', sep=';')
    _7 = pd.read_csv('mia-result/results_all/Summary/7.csv', sep=';')
    _8 = pd.read_csv('mia-result/results_all/Summary/8.csv', sep=';')

    # Evaluate Dice and Housedorf
    results_dict = {0: _0, 1: _1, 2: _2, 3: _3, 4: _4, 5: _5, 6: _6, 7: _7, 8: _8}
    dict_dice_multiplied_by_std_labels = {'0': {},'1': {}, '2': {}, '3': {}, '4': {}, '5': {}, '6': {}, '7': {}, '8': {}}
    dict_dice_mean = {'0': {},'1': {}, '2': {}, '3': {}, '4': {}, '5': {}, '6': {}, '7': {}, '8': {}}
    dict_dice_std = {'0': {},'1': {}, '2': {}, '3': {}, '4': {}, '5': {}, '6': {}, '7': {}, '8': {}}

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
        amygdala_dice_std = std
        if mean <= 0.01:
            std = np.nan
        amygdala_dice_std_times_mean = (1 - mean) * std
        amygdala_dice_mean = mean

        greyMatter_idx = np.asarray(summary_label[:] == 'GreyMatter')
        mean = np.asarray(summary_value[np.logical_and(greyMatter_idx, np.logical_and(dice_idx, mean_idx))])
        std = np.asarray(summary_value[np.logical_and(greyMatter_idx, np.logical_and(dice_idx, std_idx))])
        greyMatter_dice_std = std
        if mean <= 0.01:
            std = np.nan
        greyMatter_dice_std_times_mean = (1 - mean) * std
        greyMatter_dice_mean = mean

        hippocampus_idx = np.asarray(summary_label[:] == 'Hippocampus')
        mean = np.asarray(summary_value[np.logical_and(hippocampus_idx, np.logical_and(dice_idx, mean_idx))])
        std = np.asarray(summary_value[np.logical_and(hippocampus_idx, np.logical_and(dice_idx, std_idx))])
        hippocampus_dice_std = std
        if mean <= 0.01:
            std = np.nan
        hippocampus_dice_std_times_mean = (1 - mean) * std
        hippocampus_dice_mean = mean

        thalamus_idx = np.asarray(summary_label[:] == 'Thalamus')
        mean = np.asarray(summary_value[np.logical_and(thalamus_idx, np.logical_and(dice_idx, mean_idx))])
        std = np.asarray(summary_value[np.logical_and(thalamus_idx, np.logical_and(dice_idx, std_idx))])
        thalamus_dice_std = std
        if mean <= 0.01:
            std = np.nan
        thalamus_dice_std_times_mean = (1 - mean) * std
        thalamus_dice_mean = mean

        whiteMatter_idx = np.asarray(summary_label[:] == 'WhiteMatter')
        mean = np.asarray(summary_value[np.logical_and(whiteMatter_idx, np.logical_and(dice_idx, mean_idx))])
        std = np.asarray(summary_value[np.logical_and(whiteMatter_idx, np.logical_and(dice_idx, std_idx))])
        whiteMatter_dice_std = std
        if mean <= 0.01:
            std = np.nan
        whiteMatter_dice_std_times_mean = (1 - mean) * std
        whiteMatter_dice_mean = mean



        if i == 0:
            dict_ = '0'
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
        elif i == 7:
            dict_ = '7'
        elif i == 8:
            dict_ = '8'
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



    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    dict_labels = {1: 't1w_intensity_feature + t2w_laplacian_feature',
                   2: 't1w_intensity_feature + t2w_laplacian_feature + t1w_laplacian_feature',
                   3: 't1w_intensity_feature + t2w_laplacian_feature + t1w_laplacian_feature + t2w_intensity_feature',
                   4: 't1w_intensity_feature + t2w_laplacian_feature + t1w_laplacian_feature + t2w_intensity_feature + coordinates_feature',
                   5: 't1w_intensity_feature + t2w_laplacian_feature + t1w_laplacian_feature + t2w_intensity_feature + coordinates_feature + t1w_gradient_intensity_feature',
                   6: 't1w_intensity_feature + t2w_laplacian_feature + t1w_laplacian_feature + t2w_intensity_feature + coordinates_feature + t1w_gradient_intensity_feature + t2w_gradient_intensity_feature',
                   7: 't1w_intensity_feature + t2w_laplacian_feature + t1w_laplacian_feature + t2w_intensity_feature + coordinates_feature + t1w_gradient_intensity_feature + t2w_gradient_intensity_feature + t1w_sobel_feature',
                   8: 't1w_intensity_feature + t2w_laplacian_feature + t1w_laplacian_feature + t2w_intensity_feature + coordinates_feature + t1w_gradient_intensity_feature + t2w_gradient_intensity_feature + t1w_sobel_feature + t2w_sobel_feature'}

    ax1.plot(range(len(dict_dice_multiplied_by_std_labels['0'])),
             list(dict_dice_multiplied_by_std_labels['0'].values()), '--o', label='0')
    ax1.plot(range(len(dict_dice_multiplied_by_std_labels['1'])),
             list(dict_dice_multiplied_by_std_labels['1'].values()), '--o', label='1')
    ax1.set_xticks(range(len(dict_dice_multiplied_by_std_labels['1'])),
                   list(dict_dice_multiplied_by_std_labels['1'].keys()))
    ax1.plot(range(len(dict_dice_multiplied_by_std_labels['2'])),
             list(dict_dice_multiplied_by_std_labels['2'].values()), '--o', label='2')
    ax1.plot(range(len(dict_dice_multiplied_by_std_labels['3'])),
             list(dict_dice_multiplied_by_std_labels['3'].values()), '--o', label='3')
    ax1.plot(range(len(dict_dice_multiplied_by_std_labels['4'])),
             list(dict_dice_multiplied_by_std_labels['4'].values()), '--o', label='4')
    ax1.plot(range(len(dict_dice_multiplied_by_std_labels['5'])),
             list(dict_dice_multiplied_by_std_labels['5'].values()), '--o', label='5')
    ax1.plot(range(len(dict_dice_multiplied_by_std_labels['6'])),
             list(dict_dice_multiplied_by_std_labels['6'].values()), '--o', label='6')
    ax1.plot(range(len(dict_dice_multiplied_by_std_labels['7'])),
             list(dict_dice_multiplied_by_std_labels['7'].values()), '--o', label='7')
    ax1.plot(range(len(dict_dice_multiplied_by_std_labels['8'])),
             list(dict_dice_multiplied_by_std_labels['8'].values()), '--o', label='8')
    ax1.set_ylabel(('(1- Mean Dice)* STD Dice'))
    ax1.legend()
    # plt.show()

    ax2.plot(range(len(dict_dice_mean['0'])),
             list(dict_dice_mean['0'].values()), '--o', label='0')
    ax2.plot(range(len(dict_dice_mean['1'])),
             list(dict_dice_mean['1'].values()), '--o', label='1')
    ax2.set_xticks(range(len(dict_dice_mean['1'])),
                   list(dict_dice_mean['1'].keys()))
    ax2.plot(range(len(dict_dice_mean['2'])),
             list(dict_dice_mean['2'].values()), '--o', label='2')
    ax2.plot(range(len(dict_dice_mean['3'])),
             list(dict_dice_mean['3'].values()), '--o', label='3')
    ax2.plot(range(len(dict_dice_mean['4'])),
             list(dict_dice_mean['4'].values()), '--o', label='4')
    ax2.plot(range(len(dict_dice_mean['5'])),
             list(dict_dice_mean['5'].values()), '--o', label='5')
    ax2.plot(range(len(dict_dice_mean['6'])),
             list(dict_dice_mean['6'].values()), '--o', label='6')
    ax2.plot(range(len(dict_dice_mean['7'])),
             list(dict_dice_mean['7'].values()), '--o', label='7')
    ax2.plot(range(len(dict_dice_mean['8'])),
             list(dict_dice_mean['8'].values()), '--o', label='8')
    ax2.set_ylabel(('Mean Dice'))
    ax2.legend()
    # plt.show()

    ax3.plot(range(len(dict_dice_std['0'])),
             list(dict_dice_std['0'].values()), '--o', label='0')
    ax3.plot(range(len(dict_dice_std['1'])),
             list(dict_dice_std['1'].values()), '--o', label='1')
    ax3.set_xticks(range(len(dict_dice_std['1'])),
                   list(dict_dice_std['1'].keys()))
    ax3.plot(range(len(dict_dice_std['2'])),
             list(dict_dice_std['2'].values()), '--o', label='2')
    ax3.plot(range(len(dict_dice_std['3'])),
             list(dict_dice_std['3'].values()), '--o', label='3')
    ax3.plot(range(len(dict_dice_std['4'])),
             list(dict_dice_std['4'].values()), '--o', label='4')
    ax3.plot(range(len(dict_dice_std['5'])),
             list(dict_dice_std['5'].values()), '--o', label='5')
    ax3.plot(range(len(dict_dice_std['6'])),
             list(dict_dice_std['6'].values()), '--o', label='6')
    ax3.plot(range(len(dict_dice_std['7'])),
             list(dict_dice_std['7'].values()), '--o', label='7')
    ax3.plot(range(len(dict_dice_std['8'])),
             list(dict_dice_std['8'].values()), '--o', label='8')
    ax3.set_ylabel(('STD Dice'))
    ax3.legend()
  #  plt.show()


    xaxis = ['t2w_laplacian', 't1w_laplacian', 't2w_intensity', 'coordinates','t1w_gradient_intensity','t2w_gradient_intensity','t1w_sobel','t2w_sobel']
    fig, byLabelNumber = plt.subplots()
    byLabelNumber.plot(xaxis,
             [dict_dice_multiplied_by_std_labels["0"]["Amygdala"],dict_dice_multiplied_by_std_labels["1"]["Amygdala"], dict_dice_multiplied_by_std_labels["2"]["Amygdala"],
              dict_dice_multiplied_by_std_labels["3"]["Amygdala"], dict_dice_multiplied_by_std_labels["4"]["Amygdala"],
              dict_dice_multiplied_by_std_labels["5"]["Amygdala"], dict_dice_multiplied_by_std_labels["6"]["Amygdala"],
              dict_dice_multiplied_by_std_labels["7"]["Amygdala"], dict_dice_multiplied_by_std_labels["8"]["Amygdala"]],'o' ,label="Amygdala")
    byLabelNumber.plot(xaxis, [dict_dice_multiplied_by_std_labels["0"]["GreyMatter"],dict_dice_multiplied_by_std_labels["1"]["GreyMatter"],
                                        dict_dice_multiplied_by_std_labels["2"]["GreyMatter"],
                                        dict_dice_multiplied_by_std_labels["3"]["GreyMatter"],
                                        dict_dice_multiplied_by_std_labels["4"]["GreyMatter"],
                                        dict_dice_multiplied_by_std_labels["5"]["GreyMatter"],
                                        dict_dice_multiplied_by_std_labels["6"]["GreyMatter"],
                                        dict_dice_multiplied_by_std_labels["7"]["GreyMatter"],
                                        dict_dice_multiplied_by_std_labels["8"]["GreyMatter"]],'x',label="Grey Matter")
    byLabelNumber.plot(xaxis, [dict_dice_multiplied_by_std_labels["0"]["Hippocampus"],dict_dice_multiplied_by_std_labels["1"]["Hippocampus"],
                                        dict_dice_multiplied_by_std_labels["2"]["Hippocampus"],
                                        dict_dice_multiplied_by_std_labels["3"]["Hippocampus"],
                                        dict_dice_multiplied_by_std_labels["4"]["Hippocampus"],
                                        dict_dice_multiplied_by_std_labels["5"]["Hippocampus"],
                                        dict_dice_multiplied_by_std_labels["6"]["Hippocampus"],
                                        dict_dice_multiplied_by_std_labels["7"]["Hippocampus"],
                                        dict_dice_multiplied_by_std_labels["8"]["Hippocampus"]], '*',label="Hippocampus")
    byLabelNumber.plot(xaxis, [dict_dice_multiplied_by_std_labels["0"]["Thalamus"],
                               dict_dice_multiplied_by_std_labels["1"]["Thalamus"],
                                        dict_dice_multiplied_by_std_labels["2"]["Thalamus"],
                                        dict_dice_multiplied_by_std_labels["3"]["Thalamus"],
                                        dict_dice_multiplied_by_std_labels["4"]["Thalamus"],
                                        dict_dice_multiplied_by_std_labels["5"]["Thalamus"],
                                        dict_dice_multiplied_by_std_labels["6"]["Thalamus"],
                                        dict_dice_multiplied_by_std_labels["7"]["Thalamus"],
                                        dict_dice_multiplied_by_std_labels["8"]["Thalamus"]],'v', label="Thalamus")
    byLabelNumber.plot(xaxis, [dict_dice_multiplied_by_std_labels["0"]["WhiteMatter"],
                               dict_dice_multiplied_by_std_labels["1"]["WhiteMatter"],
                                        dict_dice_multiplied_by_std_labels["2"]["WhiteMatter"],
                                        dict_dice_multiplied_by_std_labels["3"]["WhiteMatter"],
                                        dict_dice_multiplied_by_std_labels["4"]["WhiteMatter"],
                                        dict_dice_multiplied_by_std_labels["5"]["WhiteMatter"],
                                        dict_dice_multiplied_by_std_labels["6"]["WhiteMatter"],
                                        dict_dice_multiplied_by_std_labels["7"]["WhiteMatter"],
                                        dict_dice_multiplied_by_std_labels["8"]["WhiteMatter"]],'s', label="White Matter")

    values_0 = np.array(list(dict_dice_multiplied_by_std_labels['0'].values()))
    values_1 = np.array(list(dict_dice_multiplied_by_std_labels['1'].values()))
    values_2 = np.array(list(dict_dice_multiplied_by_std_labels['2'].values()))
    values_3 = np.array(list(dict_dice_multiplied_by_std_labels['3'].values()))
    values_4 = np.array(list(dict_dice_multiplied_by_std_labels['4'].values()))
    values_5 = np.array(list(dict_dice_multiplied_by_std_labels['5'].values()))
    values_6 = np.array(list(dict_dice_multiplied_by_std_labels['6'].values()))
    values_7 = np.array(list(dict_dice_multiplied_by_std_labels['7'].values()))
    values_8 = np.array(list(dict_dice_multiplied_by_std_labels['8'].values()))

    values_0[np.isnan(values_0)] = 0.5
    values_1[np.isnan(values_1)] = 0.5
    values_2[np.isnan(values_2)] = 0.5
    values_3[np.isnan(values_3)] = 0.5
    values_4[np.isnan(values_4)] = 0.5
    values_5[np.isnan(values_5)] = 0.5
    values_6[np.isnan(values_6)] = 0.5
    values_7[np.isnan(values_7)] = 0.5
    values_8[np.isnan(values_8)] = 0.5




    if False:
        byLabelNumber.plot(xaxis, [np.nanmean(np.array(list(dict_dice_multiplied_by_std_labels['1'].values()))),
                               np.nanmean(np.array(list(dict_dice_multiplied_by_std_labels['2'].values()))),
                               np.nanmean(np.array(list(dict_dice_multiplied_by_std_labels['3'].values()))),
                               np.nanmean(np.array(list(dict_dice_multiplied_by_std_labels['4'].values()))),
                               np.nanmean(np.array(list(dict_dice_multiplied_by_std_labels['5'].values()))),
                               np.nanmean(np.array(list(dict_dice_multiplied_by_std_labels['6'].values()))),
                               np.nanmean(np.array(list(dict_dice_multiplied_by_std_labels['7'].values()))),
                               np.nanmean(np.array(list(dict_dice_multiplied_by_std_labels['8'].values())))], 'ro')

    byLabelNumber.plot(xaxis,[np.mean(values_0),np.mean(values_1), np.mean(values_2),
                              np.mean(values_3),np.mean(values_4),np.mean(values_5),np.mean(values_6)
                              ,np.mean(values_7),np.mean(values_8)], 'ro')
    byLabelNumber.legend()
    byLabelNumber.set_ylabel("(1- Mean Dice)* STD Dice")
    byLabelNumber.set_xlabel("Added feature")
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.50)
    plt.show()
    if False:
        t=time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        plt.savefig("sortedbylabel"+current_time+".png")



if __name__ == '__main__':
    main()
