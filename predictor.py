#!/usr/bin/python

from sys import argv
from sklearn.ensemble import RandomForestRegressor
import os
import math
import numpy as np


def get_data(data_file):
    data_input = open(data_file, mode='r').readlines()
    label_data = []
    feature_data = []
    for each_line in data_input:
        split_line = each_line.rstrip('\n').split('\t')
        for each_column in range(1, len(split_line)):
            split_line[each_column] = split_line[each_column].split(':')[1]
        label_data.append(split_line[0])
        feature_data.append(split_line[1:])
    label_data = np.array(label_data)
    feature_data = np.array(feature_data)
    raw_data = np.array(data_input)
    return raw_data, label_data, feature_data


def train_test_file(train_data, test_data, fold=0):
    tmp_train_file = open('train_file-fold-{0}'.format(fold), mode='w')
    tmp_test_file = open('test_file-fold-{0}'.format(fold), mode='w')
    for line in train_data:
        tmp_train_file.write(line)
    for line in test_data:
        tmp_test_file.write(line)
    tmp_train_file.close()
    tmp_test_file.close()


def eval_mape(pair_data):
    time_window_count = len(pair_data)
    total_error_rate = 0
    for pair in pair_data:
        total_error_rate += abs((pair[0]-pair[1])/pair[0])
    return total_error_rate/time_window_count


class Predictor:

    def __init__(self, train_data, train_label, train_feature, test_data, test_label, test_feature, fold=0):
        self.train_data = train_data
        self.train_label = train_label
        self.train_feature = train_feature
        self.test_data = test_data
        self.test_label = test_label
        self.test_feature = test_feature
        self.fold = fold

    def select_predictor(self, pre):
        if pre == 'svr':
            return self.svr()
        elif pre == 'rf':
            return self.rf()

    def svr(self):
        train_test_file(self.train_data, self.test_data, self.fold)
        os.system('svm-scale -s scale-info train_file-fold-{0} > scale-train-fold-{0}'.format(self.fold))
        os.system('svm-scale -r scale-info test_file-fold-{0} > scale-test-fold-{0}'.format(self.fold))
        os.popen('svm-train -s 3 scale-train-fold-{0}'.format(self.fold))
        os.popen('svm-predict scale-test-fold-{0} scale-train-fold-{0}.model predict_svr-fold-{0}'.format(self.fold))
        pre_result = [float(n.rstrip('\n')) for n in open('predict_svr-fold-{0}'.format(self.fold), mode='r').readlines()]
        os.system('rm scale-info train_file-fold-{0} scale-train-fold-{0} test_file-fold-{0} scale-test-fold-{0} '
                  'scale-train-fold-{0}.model'.format(self.fold))
        pair_data = []
        for i in range(len(self.test_label)):
            pair_data.append([float(self.test_label[i]), float(pre_result[i])])
        return eval_mape(pair_data), pre_result

    def rf(self):
        rf_pre = RandomForestRegressor(n_estimators=40)
        rf_pre.fit(self.train_feature, self.train_label)
        pre_result = rf_pre.predict(self.test_feature)
        pair_data = []
        for i in range(len(self.test_label)):
            pair_data.append([float(self.test_label[i]), pre_result[i]])
        return eval_mape(pair_data), pre_result


if __name__ == '__main__':
    input_opts = argv[1:]
    i = 0
    cv_status = '0'
    while i < len(input_opts):
        if input_opts[i] == '-tr':
            train_file = input_opts[i+1]
        elif input_opts[i] == '-te':
            test_file = input_opts[i+1]
        elif input_opts[i] == '-cv':
            cv_test_file = input_opts[i+1]
        elif input_opts[i] == '-p':
            predictor = input_opts[i+1]
        i += 1
    result = []
    result_eval = []
    train_file_data, train_file_label, train_file_feature = get_data(train_file)
    cv_test_file_data, cv_test_file_label, cv_test_file_feature = get_data(cv_test_file)
    test_file_data, test_file_label, test_file_feature = get_data(test_file)
    split_line = 5228
    eval_pre = Predictor(train_file_data[:split_line], train_file_label[:split_line], train_file_feature[:split_line],
                         cv_test_file_data, cv_test_file_label, cv_test_file_feature)
    eval_result = []
    if predictor == 'rf':
        for i in range(10):
            tmp_eval, tmp_result = eval_pre.select_predictor(predictor)
            eval_result.append(tmp_eval)
    elif predictor == 'svr':
        tmp_eval, tmp_result = eval_pre.select_predictor(predictor)
        eval_result.append(tmp_eval)
    print 'Evaluation : {0}({1})'.format(np.average(eval_result), np.std(eval_result))

    single_pre = Predictor(train_file_data, train_file_label, train_file_feature, test_file_data, test_file_label
                           , test_file_feature)
    for i in xrange(10):
        tmp_eval, tmp_result = single_pre.select_predictor(predictor)
        result_eval.append(tmp_eval)
        result.append(tmp_result)
    output_file = open('predict_{0}'.format(predictor), mode='w')
    for line in xrange(len(result[0])):
        output_file.write('{0}\n'.format(np.average([n[line] for n in result])))
    output_file.close()
    print 'Test Evaluation : {0}'.format(np.average(result_eval))

