import datetime
from sys import argv
import pprint
import numpy as np
from collections import defaultdict

class NestedDict(dict):

    def __missing__(self, key):
        value = self[key] = type(self)()
        return value


def get_data(input_file):
    retrieved_data = NestedDict()
    input_data = open(input_file, mode='r').readlines()
    for line_count in xrange(1, len(input_data)):
        line_data = input_data[line_count].split('\t')
        time_stamp = line_data[0].split(',')[0]
        time_stamp_obj = datetime.datetime.strptime(time_stamp, '%Y-%m-%d %H:%M:%S')
        retrieved_data[time_stamp_obj]['1-0'] = int(line_data[1])
        retrieved_data[time_stamp_obj]['1-1'] = int(line_data[2])
        retrieved_data[time_stamp_obj]['2-0'] = int(line_data[3])
        retrieved_data[time_stamp_obj]['3-0'] = int(line_data[4])
        retrieved_data[time_stamp_obj]['3-1'] = int(line_data[5])
    return retrieved_data


def weekday_data(raw_data):
    retrieved_data = NestedDict()
    for gate_dir in ['1-0', '1-1', '2-0', '3-0', '3-1']:
        for w_day in xrange(7):
            for hour in xrange(24):
                    retrieved_data[gate_dir][w_day][hour] = defaultdict(list)
    for date in sorted(raw_data.keys()):
        for gate_dir in ['1-0', '1-1', '2-0', '3-0', '3-1']:
            retrieved_data[gate_dir][date.weekday()][date.hour][date.minute].append(raw_data[date][gate_dir])
    return retrieved_data


def week_model(week_data, gate_dir):
    model_data = NestedDict()
    for hour in xrange(24):
        for minute in xrange(0, 60, 20):
            tmp_workday_total = 0
            tmp_holiday_total = 0
            for w_day in xrange(5):
                tmp_workday_total += np.average(np.array(week_data[gate_dir][w_day][hour][minute]))
            for w_day in xrange(5, 7):
                tmp_holiday_total += np.average(np.array(week_data[gate_dir][w_day][hour][minute]))
            model_data['workday'][hour][minute] = tmp_workday_total/5
            model_data['holiday'][hour][minute] = tmp_holiday_total/2

    output_workday_file = open('{0}_workday'.format(gate_dir), mode='w')
    for hour in xrange(24):
        for minute in xrange(0, 60, 20):
            output_workday_file.write('{0}\n'.format(model_data['workday'][hour][minute]))
    output_workday_file.close()
    output_holiday_file = open('{0}_holiday'.format(gate_dir), mode='w')
    for hour in xrange(24):
        for minute in xrange(0, 60, 20):
            output_holiday_file.write('{0}\n'.format(model_data['holiday'][hour][minute]))
    output_holiday_file.close()


if __name__ == '__main__':
    raw_data = get_data(argv[1])
    week_data = weekday_data(raw_data)
    for gate_dir in ['1-0', '1-1', '2-0', '3-0', '3-1']:
        week_model(week_data, gate_dir)
