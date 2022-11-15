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

    # Assign feature number to new data set
    data1 = pd.DataFrame(T1_results).assign(Features='T1')
    data2 = pd.DataFrame(T1_and_T2_results).assign(Features='T1 and T2')
    data3 = pd.DataFrame(standard_7_results).assign(Features='7 standard')

    # Concat data sets for boxplot
    cdf = pd.concat([data1, data2, data3])
    ax = sns.boxplot(x="LABEL", y="HDRFDST", hue="Features", data=cdf)
    plt.show()

if __name__ == '__main__':
    main()
