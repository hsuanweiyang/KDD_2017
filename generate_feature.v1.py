#!/usr/bin/python

import re
from sys import argv
import datetime
import pprint
import numpy as np


class NestedDict(dict):

    def __missing__(self, key):
        value = self[key] = type(self)()
        return value


def get_volume_data(data_file):
    volume_raw_data = open(data_file, mode='r').readlines()
    retrieved_data = NestedDict()
    for line_count in range(1, len(volume_raw_data)):
        each_data = re.findall('"(.*?)"', volume_raw_data[line_count])
        time_stamp = re.findall(r'\[(.*),(.*)\)', each_data[1])
        for index, time_window in enumerate(time_stamp):
            start_time = time_window[0]
            stop_time = time_window[1]
        retrieved_data[start_time][stop_time][each_data[0]][each_data[2]]['volume'] = int(each_data[3])
    return retrieved_data, volume_raw_data


def get_tollgate_lane_data(data_file):
    link_input = open(data_file, mode='r').readlines()
    link_data = NestedDict()
    for line_count in range(1, len(link_input)):
        each_data = re.findall('"(.*?)"', link_input[line_count])
        link_data[each_data[0]]['length'] = each_data[1]
        link_data[each_data[0]]['lanes'] = each_data[3]
        link_data[each_data[0]]['in_top'] = each_data[4]
    for link in link_data.keys():
        in_link = link_data[link]['in_top'].split(',')
        in_lanes = 0
        if in_link[0] != '':
            for in_id in in_link:
                in_lanes += int(link_data[in_id]['lanes'])
            link_data[link]['in_lanes'] = in_lanes
    return link_data


def generate_timebased_feature(raw_feature_data, start_time, stop_time, tollgate, direction):

    initial_start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    initial_stop_time = datetime.datetime.strptime(stop_time, '%Y-%m-%d %H:%M:%S')
    time_based_feature = {}

    def minute_feature():
        minutefeature_start = initial_start_time - datetime.timedelta(minutes=20)
        minutefeature_stop = initial_stop_time - datetime.timedelta(minutes=20)
        minute_total_volume = []
        minute_feature_output = {}
        while minutefeature_start.date() == initial_start_time.date():
            current_vol = raw_feature_data[minutefeature_start.strftime('%Y-%m-%d %H:%M:%S')] \
                [minutefeature_stop.strftime('%Y-%m-%d %H:%M:%S')][tollgate][direction]['volume']
            if current_vol:
                minute_total_volume.append(current_vol)
            minutefeature_start = minutefeature_start - datetime.timedelta(minutes=20)
            minutefeature_stop = minutefeature_stop - datetime.timedelta(minutes=20)
        if len(minute_total_volume) < 1:
            minute_total_volume.append(-1)
        minute_feature_output['feature_minute_avg'] = np.average(minute_total_volume)
        minute_feature_output['feature_minute_std'] = np.std(minute_total_volume)
        minute_feature_output['feature_minute_min'] = np.percentile(minute_total_volume, 0)
        minute_feature_output['feature_minute_q1'] = np.percentile(minute_total_volume, 25)
        minute_feature_output['feature_minute_q2'] = np.percentile(minute_total_volume, 50)
        minute_feature_output['feature_minute_q3'] = np.percentile(minute_total_volume, 75)
        minute_feature_output['feature_minute_max'] = np.percentile(minute_total_volume, 100)
        return minute_feature_output

    def day_feature():
        dayfeature_start = initial_start_time - datetime.timedelta(days=1)
        dayfeature_stop = initial_stop_time - datetime.timedelta(days=1)
        day_total_volume = []
        day_feature_output = {}
        while dayfeature_start.date() > datetime.date(2016, 9, 18):
            current_vol = raw_feature_data[dayfeature_start.strftime('%Y-%m-%d %H:%M:%S')] \
                [dayfeature_stop.strftime('%Y-%m-%d %H:%M:%S')][tollgate][direction]['volume']
            if current_vol:
                day_total_volume.append(current_vol)
            dayfeature_start = dayfeature_start - datetime.timedelta(days=1)
            dayfeature_stop = dayfeature_stop - datetime.timedelta(days=1)
        if len(day_total_volume) < 1:
            day_total_volume.append(-1)
        day_feature_output['feature_day_avg'] = np.average(day_total_volume)
        day_feature_output['feature_day_std'] = np.std(day_total_volume)
        day_feature_output['feature_day_min'] = np.percentile(day_total_volume, 0)
        day_feature_output['feature_day_q1'] = np.percentile(day_total_volume, 25)
        day_feature_output['feature_day_q2'] = np.percentile(day_total_volume, 50)
        day_feature_output['feature_day_q3'] = np.percentile(day_total_volume, 75)
        day_feature_output['feature_day_max'] = np.percentile(day_total_volume, 100)
        return day_feature_output

    def week_feature():
        weekfeature_start = initial_start_time - datetime.timedelta(weeks=1)
        weekfeature_stop = initial_stop_time - datetime.timedelta(weeks=1)
        week_total_volume = []
        week_feature_output = {}
        while weekfeature_start.date() > datetime.date(2016, 9, 18):
            current_vol = raw_feature_data[weekfeature_start.strftime('%Y-%m-%d %H:%M:%S')] \
                [weekfeature_stop.strftime('%Y-%m-%d %H:%M:%S')][tollgate][direction]['volume']
            if current_vol:
                week_total_volume.append(current_vol)
            weekfeature_start = weekfeature_start - datetime.timedelta(weeks=1)
            weekfeature_stop = weekfeature_stop - datetime.timedelta(weeks=1)
        if len(week_total_volume) < 1:
            week_total_volume.append(-1)
        week_feature_output['feature_week_avg'] = np.average(week_total_volume)
        '''
        week_feature_output['feature_week_std'] = np.std(week_total_volume)
        week_feature_output['feature_week_min'] = np.percentile(week_total_volume, 0)
        week_feature_output['feature_week_q1'] = np.percentile(week_total_volume, 25)
        week_feature_output['feature_week_q2'] = np.percentile(week_total_volume, 50)
        week_feature_output['feature_week_q3'] = np.percentile(week_total_volume, 75)
        week_feature_output['feature_week_max'] = np.percentile(week_total_volume, 100)
        '''
        return week_feature_output

    time_based_feature['minute'] = minute_feature()
    time_based_feature['day'] = day_feature()
    time_based_feature['week'] = week_feature()

    return time_based_feature


def generate_holiday_feature(start_time):
    holiday_feature_output = {}
    initial_start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    last_day = initial_start_time.date() - datetime.timedelta(days=1)
    today = initial_start_time.date()
    tomorrow = initial_start_time.date() + datetime.timedelta(days=1)
    special_holiday = [datetime.date(2016, 9, n) for n in range(15, 18)] + [datetime.date(2016, 10, n) for n in range(1, 8)]
    special_workday = [datetime.date(2016, 9, 18), datetime.date(2016, 10, 8), datetime.date(2016, 10, 9)]
    holiday_feature_output['weekday'] = initial_start_time.weekday()

    def day_state(day):
        """
            Simple:
                0: workday
                1: holiday
            Complex:
                0: 000, 100 => workday
                1: 111, 010 => normal holiday
                2: 101, 001 => day before holiday
                3: 011 => first day of holiday
                4: 110 =>

        """

        if day in special_holiday:
            day_status = 1
        elif day.weekday() >= 5 and day not in special_workday:
            day_status = 1
        else:
            day_status = 0
        return day_status

    last_day_state = day_state(last_day)
    today_simple_state = day_state(today)
    tomorrow_state = day_state(tomorrow)
    if today_simple_state == 0:
        if tomorrow_state == 0:
            today_complex_state = 0
        else:
            today_complex_state = 2
    else:
        if last_day_state + tomorrow_state == 2 or last_day_state + tomorrow_state == 0:
            today_complex_state = 1
        elif last_day_state == 0:
            today_complex_state = 3
        else:
            today_complex_state = 4

    holiday_feature_output['day_state_simple'] = today_simple_state
    holiday_feature_output['day_state_complex'] = today_complex_state
    return holiday_feature_output


def generate_train_feature_file(dict_input_data, raw_input_data, link_file, output_feature_file='train-feature_file'):
    output_file = open(output_feature_file, mode='w')
    tollgate_link = {'1': '113', '2': '117', '3': '122'}
    for line_count in range(1, len(raw_input_data)):
        each_data = re.findall('"(.*?)"', raw_input_data[line_count])
        time_stamp = re.findall(r'\[(.*),(.*)\)', each_data[1])
        for index, time_window in enumerate(time_stamp):
            start = time_window[0]
            stop = time_window[1]
        tollgate_id = each_data[0]
        direction_id = each_data[2]
        if direction_id == '1':
            op_direction_id = '0'
        else:
            op_direction_id = '1'
        tmp_out_str = '{0}\t1:{1}\t2:{2}'.format(dict_input_data[start][stop][tollgate_id][direction_id]['volume'],
                                                 tollgate_id, direction_id)
        tmp_index = 2

        # Feature : State of day(ie. workday or holiday)
        day_state_feature = generate_holiday_feature(start)
        for n in sorted(day_state_feature.keys()):
            tmp_index += 1

            tmp_out_str += '\t{0}:{1}'.format(tmp_index, day_state_feature[n])

        # Feature : Volume data of previous time window (min, day, week)
        history_feature = generate_timebased_feature(dict_input_data, start, stop, tollgate_id, direction_id)
        op_direction_history_feature = generate_timebased_feature(dict_input_data, start, stop, tollgate_id, op_direction_id)
        for n in sorted(history_feature.keys()):
            for i in sorted(history_feature[n].keys()):
                tmp_index += 1
                tmp_out_str += '\t{0}:{1}'.format(tmp_index, history_feature[n][i])
        for n in sorted(op_direction_history_feature.keys()):
            for i in sorted(op_direction_history_feature[n].keys()):
                tmp_index += 1
                tmp_out_str += '\t{0}:{1}'.format(tmp_index, op_direction_history_feature[n][i])

        # Feature : Lane data of nearest link of tollgate
        link_data = get_tollgate_lane_data(link_file)
        for link_feature in ['length', 'lanes', 'in_lanes']:
            tmp_index += 1
            tmp_out_str += '\t{0}:{1}'.format(tmp_index, link_data[tollgate_link[tollgate_id]][link_feature])

        output_file.write(tmp_out_str + '\n')


def generate_test_feature_file(dict_input_data, test_file, link_file, output_feature_file='test-feature_file'):
    output_file = open(output_feature_file, mode='w')
    test_input_data = open(test_file, mode='r').readlines()
    tollgate_link = {'1': '113', '2': '117', '3': '122'}
    for line_count in range(1, len(test_input_data)):

        each_data = test_input_data[line_count].split(',')
        time_stamp = re.findall(r'\[(.*),(.*)\)', test_input_data[line_count])
        for index, time_window in enumerate(time_stamp):
            start = time_window[0]
            stop = time_window[1]
        tollgate_id = each_data[0]
        direction_id = each_data[3]
        volume = each_data[4].split('\r')[0]
        if direction_id == '1':
            op_direction_id = '0'
        else:
            op_direction_id = '1'
        tmp_out_str = '{0}\t1:{1}\t2:{2}'.format(volume, tollgate_id, direction_id)
        tmp_index = 2

        # Feature : State of day(ie. workday or holiday)
        day_state_feature = generate_holiday_feature(start)
        for n in sorted(day_state_feature.keys()):
            tmp_index += 1
            tmp_out_str += '\t{0}:{1}'.format(tmp_index, day_state_feature[n])

        # Feature : Volume data of previous time window (min, day, week)
        history_feature = generate_timebased_feature(dict_input_data, start, stop, tollgate_id, direction_id)
        op_direction_history_feature = generate_timebased_feature(dict_input_data, start, stop, tollgate_id, op_direction_id)
        for n in sorted(history_feature.keys()):
            for i in sorted(history_feature[n].keys()):
                tmp_index += 1
                tmp_out_str += '\t{0}:{1}'.format(tmp_index, history_feature[n][i])
        for n in sorted(op_direction_history_feature.keys()):
            for i in sorted(op_direction_history_feature[n].keys()):
                tmp_index += 1
                tmp_out_str += '\t{0}:{1}'.format(tmp_index, op_direction_history_feature[n][i])

        # Feature : Lane data of nearest link of tollgate
        link_data = get_tollgate_lane_data(link_file)
        for link_feature in ['length', 'lanes', 'in_lanes']:
            tmp_index += 1
            tmp_out_str += '\t{0}:{1}'.format(tmp_index, link_data[tollgate_link[tollgate_id]][link_feature])

        output_file.write(tmp_out_str + '\n')


if __name__ == '__main__':
    input_opt = argv[1:]
    i = 0
    while i < len(input_opt):
        if input_opt[i] == '-tr':
            train_file = input_opt[i+1]
        elif input_opt[i] == '-te':
            test_file = input_opt[i+1]
        elif input_opt[i] == '-link':
            link_file = input_opt[i+1]
        elif input_opt[i] == '-h':
            print '-tr [train_file] -te [test_file(optional)]'
        i += 1
    data_dict, raw_data = get_volume_data(train_file)
    generate_train_feature_file(data_dict, raw_data, link_file)
    if test_file:
        generate_test_feature_file(data_dict, test_file, link_file)

