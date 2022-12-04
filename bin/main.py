"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import sys
import timeit
import warnings
import random

import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
import numpy as np
import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer
import pandas as pd
import matplotlib.pyplot as plt

try:
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load


def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Decision forest classifier model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_train_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())

    i_global = 10 # change this number
    robustness_best = float('inf')
    first_flag = 1

    for itr in range(2): # change this number
        # load atlas images
        putil.load_atlas_images(data_atlas_dir)

        print('-' * 5, 'Training...')

        # crawl the training image directories
        crawler = futil.FileSystemDataCrawler(data_train_dir,
                                              LOADING_KEYS,
                                              futil.BrainImageFilePathGenerator(),
                                              futil.DataDirectoryFilter())

        pre_process_params = {'skullstrip_pre': True,
                              'normalization_pre': True,
                              'registration_pre': True,
                              't1w_intensity_feature': True,
                              't2w_laplacian_feature': True,
                              't1w_laplacian_feature': True,
                              't2w_intensity_feature': True,
                              'coordinates_feature': True,
                              't1w_gradient_intensity_feature': True,
                              't2w_gradient_intensity_feature': True,
                              't1w_sobel_feature': False,
                              't2w_sobel_feature': False
                              }

        i_local = 0
        for key, state in pre_process_params.items():
            if i_local == i_global:
                pre_process_params[key] = True
            i_local += 1

        print(pre_process_params)

        # load images for training and pre-process
        images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

        # generate feature matrix and label vector
        data_train = np.concatenate([img.feature_matrix[0] for img in images])
        labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()

        forest = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1],
                                                    n_estimators=10,
                                                    max_depth=10)

        start_time = timeit.default_timer()
        forest.fit(data_train, labels_train)
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')
        # ---------------------------------------------------------------------------- #

        # create a result directory with timestamp
        if first_flag:
            t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            result_dir = os.path.join(result_dir,t)
            os.makedirs(result_dir, exist_ok=True)
            first_flag = 0

            # -------------------- Feature importance ------------------------ #
        # Gini Importance or Mean Decrease in Impurity (MDI) calculates each feature importance as the sum over the number
        # of splits (across all tress) that include the feature, proportionally to the number of samples it splits
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

        feature_names = ["Atlas coordinates 1", "Atlas coordinates 2", "Atlas coordinates 3", "T1 intensities",
                         "T2 intensities"]
        forest_importances = pd.Series(importances)  # add feature names here!

        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        figure_name="feature_importance_" + str(i_global)
        figure_path=os.path.join(result_dir, figure_name)
        plt.savefig(figure_path)

        print('-' * 5, 'Testing...')

        # initialize evaluator
        evaluator = putil.init_evaluator()

        # crawl the training image directories
        crawler = futil.FileSystemDataCrawler(data_test_dir,
                                              LOADING_KEYS,
                                              futil.BrainImageFilePathGenerator(),
                                              futil.DataDirectoryFilter())

        # load images for testing and pre-process
        pre_process_params['training'] = False
        images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

        images_prediction = []
        images_probabilities = []

        for img in images_test:
            print('-' * 10, 'Testing', img.id_)

            start_time = timeit.default_timer()
            predictions = forest.predict(img.feature_matrix[0])
            probabilities = forest.predict_proba(img.feature_matrix[0])
            print(' Time elapsed:', timeit.default_timer() - start_time, 's')

            # convert prediction and probabilities back to SimpleITK images
            image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8),
                                                                            img.image_properties)
            image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

            # evaluate segmentation without post-processing
            evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

            images_prediction.append(image_prediction)
            images_probabilities.append(image_probabilities)

        # post-process segmentation and evaluate with post-processing
        post_process_params = {'simple_post': True}
        images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities,
                                                         post_process_params, multi_process=True)

        for i, img in enumerate(images_test):
            evaluator.evaluate(images_post_processed[i], img.images[structure.BrainImageTypes.GroundTruth],
                               img.id_ + '-PP')

            # save results
            sitk.WriteImage(images_prediction[i], os.path.join(result_dir, images_test[i].id_ + '_SEG_'+ str(i_global) + '.mha'), True)
            sitk.WriteImage(images_post_processed[i], os.path.join(result_dir, images_test[i].id_ + '_SEG-PP_' + str(i_global) + '.mha'), True)

        # use two writers to report the results
        os.makedirs(result_dir, exist_ok=True)  # generate result directory, if it does not exists
        result_file = os.path.join(result_dir, 'results_' + str(i_global) + '.csv')
        writer.CSVWriter(result_file).write(evaluator.results)

        print('\nSubject-wise results...')
        writer.ConsoleWriter().write(evaluator.results)

        # report also mean and standard deviation among all subjects
        result_summary_file = os.path.join(result_dir, 'results_summary_' + str(i_global) + '.csv')
        functions = {'MEAN': np.mean, 'STD': np.std}
        writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
        print('\nAggregated statistic results...')
        writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

        # clear results such that the evaluator is ready for the next evaluation
        evaluator.clear()

        # Read in data and compare it to find best one
        data = pd.read_csv(result_summary_file, sep=';')
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
        if std == 0:
            std = 1
        amygdala_dice_std_times_mean = (1 - mean) * std

        greyMatter_idx = np.asarray(summary_label[:] == 'GreyMatter')
        mean = np.asarray(summary_value[np.logical_and(greyMatter_idx, np.logical_and(dice_idx, mean_idx))])
        std = np.asarray(summary_value[np.logical_and(greyMatter_idx, np.logical_and(dice_idx, std_idx))])
        if std == 0:
            std = 1
        greyMatter_dice_std_times_mean = (1 - mean) * std

        hippocampus_idx = np.asarray(summary_label[:] == 'Hippocampus')
        mean = np.asarray(summary_value[np.logical_and(hippocampus_idx, np.logical_and(dice_idx, mean_idx))])
        std = np.asarray(summary_value[np.logical_and(hippocampus_idx, np.logical_and(dice_idx, std_idx))])
        if std == 0:
            std = 1
        hippocampus_dice_std_times_mean = (1 - mean) * std

        thalamus_idx = np.asarray(summary_label[:] == 'Thalamus')
        mean = np.asarray(summary_value[np.logical_and(thalamus_idx, np.logical_and(dice_idx, mean_idx))])
        std = np.asarray(summary_value[np.logical_and(thalamus_idx, np.logical_and(dice_idx, std_idx))])
        if std == 0:
            std = 1
        thalamus_dice_std_times_mean = (1 - mean) * std

        whiteMatter_idx = np.asarray(summary_label[:] == 'WhiteMatter')
        mean = np.asarray(summary_value[np.logical_and(whiteMatter_idx, np.logical_and(dice_idx, mean_idx))])
        std = np.asarray(summary_value[np.logical_and(whiteMatter_idx, np.logical_and(dice_idx, std_idx))])
        if std == 0:
            std = 1
        whiteMatter_dice_std_times_mean = (1 - mean) * std

        new_robustness = np.mean([whiteMatter_dice_std_times_mean, thalamus_dice_std_times_mean, hippocampus_dice_std_times_mean, greyMatter_dice_std_times_mean, amygdala_dice_std_times_mean])
        if new_robustness == 0:
            new_robustness = float('inf')
        if robustness_best > new_robustness:
            i_global_best = i_global
            robustness_best = new_robustness
            print('Robustness best =' + str(i_global_best))
            print('Robustness best value =' + str(robustness_best))

        i_local = 0
        for key, state in pre_process_params.items():
            if i_local == i_global:
                result_file = os.path.join(result_dir, 'result_all_mean.txt')
                with open(result_file, 'a') as f:
                    f.write('\n' + str(key) + ': ' + str(new_robustness))
            i_local += 1

        # iterate features
        i_global += 1

    print(i_global_best)
    print(robustness_best)
    i_local = 0
    best_param = ''
    for key, state in pre_process_params.items():
        if i_local == i_global_best:
            best_param = key
            print(key)
        i_local += 1

    # write results into a file
    result_file = os.path.join(result_dir, 'result_best.txt')
    with open(result_file, 'w') as f:
        f.write('\nBest param: ' + best_param + '\n')
        f.write(str(pre_process_params))


if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/train/')),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/test/')),
        help='Directory with testing data.'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir)
