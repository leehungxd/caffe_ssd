#!/usr/bin/env python

"""
Parse training log

Evolved from parse_log.sh
"""

import os
import re
import extract_seconds
import argparse
import csv
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt

def parse_log(path_to_log):
    """Parse log file
    Returns (train_dict_list, test_dict_list)

    train_dict_list and test_dict_list are lists of dicts that define the table
    rows
    """

    regex_train_loss = re.compile('Iteration (\d+), loss = ([\.\deE+-]+)')
    # regex_train_output = re.compile('Train net output #(\d+): (\S+) = ([\.\deE+-]+)')
    regex_test_loss = re.compile('Test loss: ([\.\deE+-]+)')
    regex_test_output = re.compile('Test net output #(\d+): (\S+) = ([\.\deE+-]+)')
    regex_learning_rate = re.compile('Iteration (\d+), lr = ([-+]?[0-9]*\.?[0-9]+([eE]?[-+]?[0-9]+)?)')

    # Pick out lines of interest
    iteration = -1
    learning_rate = float('NaN')
    train_dict_list = []
    test_dict_list = []
    train_row = None
    test_row = None

    logfile_year = extract_seconds.get_log_created_year(path_to_log)
    with open(path_to_log) as f:
        start_time = extract_seconds.get_start_time(f, logfile_year)

        for line in f:
            iteration_match = regex_train_loss.search(line)
            if iteration_match:
                iteration = float(iteration_match.group(1))
                train_loss = float(iteration_match.group(2))
            if iteration == -1:
                # Only start parsing for other stuff if we've found the first
                # iteration
                continue

            try:
                time = extract_seconds.extract_datetime_from_line(line,
                                                                  logfile_year)
            except ValueError:
                # Skip lines with bad formatting, for example when resuming solver
                continue

            seconds = (time - start_time).total_seconds()

            learning_rate_match = regex_learning_rate.search(line)
            if learning_rate_match:
                learning_rate = float(learning_rate_match.group(1))

            train_dict_list, train_row = parse_line_for_net_output(
                regex_train_loss, train_row, train_dict_list,
                line, iteration, seconds, learning_rate
            )
            test_dict_list, test_row = parse_line_for_net_output(
                regex_test_output, test_row, test_dict_list,
                line, iteration, seconds, learning_rate
            )

    fix_initial_nan_learning_rate(train_dict_list)
    fix_initial_nan_learning_rate(test_dict_list)

    return train_dict_list, test_dict_list


def parse_line_for_net_output(regex_obj, row, row_dict_list,
                              line, iteration, seconds, learning_rate):
    """Parse a single line for training or test output

    Returns a a tuple with (row_dict_list, row)
    row: may be either a new row or an augmented version of the current row
    row_dict_list: may be either the current row_dict_list or an augmented
    version of the current row_dict_list
    """

    output_match = regex_obj.search(line)
    if output_match:
        if not row or row['NumIters'] != iteration:
            # Push the last row and start a new one
            if row:
                # If we're on a new iteration, push the last row
                # This will probably only happen for the first row; otherwise
                # the full row checking logic below will push and clear full
                # rows
                row_dict_list.append(row)

            row = OrderedDict([
                ('NumIters', iteration),
                ('Seconds', seconds),
                ('LearningRate', learning_rate)
            ])

        # output_num is not used; may be used in the future
        # output_num = output_match.group(1)
        output_name = output_match.group(2)
        output_val = output_match.group(3)
        row[output_name] = float(output_val)

    if row and len(row_dict_list) >= 1 and len(row) == len(row_dict_list[0]):
        # The row is full, based on the fact that it has the same number of
        # columns as the first row; append it to the list
        row_dict_list.append(row)
        row = None

    return row_dict_list, row


def fix_initial_nan_learning_rate(dict_list):
    """Correct initial value of learning rate

    Learning rate is normally not printed until after the initial test and
    training step, which means the initial testing and training rows have
    LearningRate = NaN. Fix this by copying over the LearningRate from the
    second row, if it exists.
    """

    if len(dict_list) > 1:
        dict_list[0]['LearningRate'] = dict_list[1]['LearningRate']


def save_csv_files(logfile_path, output_dir, train_dict_list, test_dict_list,
                   delimiter=',', verbose=False):
    """Save CSV files to output_dir

    If the input log file is, e.g., caffe.INFO, the names will be
    caffe.INFO.train and caffe.INFO.test
    """

    log_basename = os.path.basename(logfile_path)
    train_filename = os.path.join(output_dir, log_basename + '.train')
    write_csv(train_filename, train_dict_list, delimiter, verbose)

    test_filename = os.path.join(output_dir, log_basename + '.test')
    write_csv(test_filename, test_dict_list, delimiter, verbose)


def write_csv(output_filename, dict_list, delimiter, verbose=False):
    """Write a CSV file
    """

    if not dict_list:
        if verbose:
            print('Not writing %s; no lines to write' % output_filename)
        return

    dialect = csv.excel
    dialect.delimiter = delimiter

    with open(output_filename, 'w') as f:
        dict_writer = csv.DictWriter(f, fieldnames=dict_list[0].keys(),
                                     dialect=dialect)
        dict_writer.writeheader()
        dict_writer.writerows(dict_list)
    if verbose:
        print 'Wrote %s' % output_filename


def parse_args():
    description = ('Parse a Caffe training log into two CSV files '
                   'containing training and testing information')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('logfile_path',
                        help='Path to log file')

    parser.add_argument('output_dir',
                        help='Directory in which to place output CSV files')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print some extra info (e.g., output filenames)')

    parser.add_argument('--delimiter',
                        default=',',
                        help=('Column delimiter in output files '
                              '(default: \'%(default)s\')'))

    args = parser.parse_args()
    return args

def draw(train_log, test_log):
    # train_log = pd.read_csv("/home/lihang/projects/caffe/tools/extra/test/ResNet50_anngic_SSD_300x300.log.train")
    # test_log = pd.read_csv("/home/lihang/projects/caffe/tools/extra/test/ResNet50_anngic_SSD_300x300.log.test")

    _, ax1 = plt.subplots()
    ax1.set_title("train loss and test loss")
    numIters_train = [item['NumIters'] for item in train_log]
    loss_train = [item['mbox_loss'] for item in train_log]
    ax1.plot(numIters_train, loss_train, alpha=0.5)

    numIters_test = [item['NumIters'] for item in test_log]
    eval_test = [item['detection_eval'] for item in test_log]
    ax1.plot(numIters_test, eval_test, 'g')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    plt.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(test_log["NumIters"], test_log["acc/top-1"], 'r')
    ax2.plot(test_log["NumIters"], test_log["acc/top-5"], 'm')
    ax2.set_ylabel('test accuracy')
    plt.legend(loc='upper right')

    plt.show()

    print 'Done.'


# def main():
#     args = parse_args()
#     train_dict_list, test_dict_list = parse_log(args.logfile_path)
#     save_csv_files(args.logfile_path, args.output_dir, train_dict_list,
#                    test_dict_list, delimiter=args.delimiter)


if __name__ == '__main__':

    '''parse the test and train log into two files'''
    logfile_path = '/home/lihang/projects/caffe/models/shuffleNetV2_3h_1/anngic/SSD_300x300/shuffleNetV2_3h_1_anngic_SSD_300x300.log'
    train_dict_list, test_dict_list = parse_log(logfile_path)
    # output_dir = '/home/lihang/projects/caffe/tools/extra/test/'
    # save_csv_files(logfile_path, output_dir, train_dict_list, test_dict_list)

    draw(train_dict_list, test_dict_list)

