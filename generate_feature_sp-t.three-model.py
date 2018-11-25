#!/usr/bin/python
__author__ = 'hsuanwei'

import re
from sys import argv
import datetime
import pprint
import numpy as np
import math
import timeit


class NestedDict(dict):

    def __missing__(self, key):
        value = self[key] = type(self)()
        return value


def get_volume_data(data_file):
    volume_raw_data = open(data_file, mode='r').readlines()
    retrieved_data = NestedDict()
    for line_count in xrange(1, len(volume_raw_data)):
        each_data = re.findall('"(.*?)"', volume_raw_data[line_count])
        time_stamp = re.findall(r'\[(.*),(.*)\)', each_data[1])
        for index, time_window in enumerate(time_stamp):
            start_time = time_window[0]
        start_time_obj = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        if datetime.date(2016, 9, 30) < start_time_obj.date() < datetime.date(2016, 10, 10):
            continue
        retrieved_data[start_time][each_data[0]][each_data[2]]['volume'] = int(each_data[3])
    return retrieved_data, volume_raw_data


def get_tollgate_lane_data(data_file):
    global link_to_intersection
    link_to_intersection = {
        '1': {'B': ['113', '106', '121', '101', '116', '103', '111', '100', '105'],
              'C': ['113', '106', '121', '101', '116', '103', '111', '112', '104', '109', '102', '115']},
        '2': {'A': ['117', '120', '108', '107', '123', '110']},
        '3': {'A': ['122', '118', '114', '119', '108', '107', '123', '110'],
              'B': ['122', '103', '111', '100', '105'],
              'C': ['122', '103', '111', '112', '104', '109', '102', '115']}}
    link_input = open(data_file, mode='r').readlines()
    link_data = NestedDict()
    for line_count in xrange(1, len(link_input)):
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


def get_weather_data(data_file):
    weather_data = open(data_file, mode='r').readlines()
    retrieved_weather_data = NestedDict()
    for line_count in xrange(1, len(weather_data)):
        each_data = re.findall('"(.*?)"', weather_data[line_count])
        retrieved_weather_data[each_data[0]][each_data[1]]['pressure'] = each_data[2]
        retrieved_weather_data[each_data[0]][each_data[1]]['sea_pressure'] = each_data[3]
        retrieved_weather_data[each_data[0]][each_data[1]]['wind_direction'] = each_data[4]
        retrieved_weather_data[each_data[0]][each_data[1]]['wind_speed'] = each_data[5]
        retrieved_weather_data[each_data[0]][each_data[1]]['temperature'] = each_data[6]
        retrieved_weather_data[each_data[0]][each_data[1]]['rel_humidity'] = each_data[7]
        retrieved_weather_data[each_data[0]][each_data[1]]['precipitation'] = each_data[8]
    return retrieved_weather_data


def get_travel_time_data(trajectory_file):
    trajectory_raw_data = open(trajectory_file, mode='r').readlines()
    retrieved_data = NestedDict()
    for line_count in xrange(1, len(trajectory_raw_data)):
        line_data = re.findall('"(.*?)"', trajectory_raw_data[line_count])
        trace_start_time_obj = datetime.datetime.strptime(line_data[3], '%Y-%m-%d %H:%M:%S')
        travel_time = float(line_data[-1])
        trace_end_time_obj = trace_start_time_obj + datetime.timedelta(minutes=int(travel_time))
        time_window_minute = math.floor(trace_end_time_obj.minute/20)*20
        time_window = datetime.datetime(trace_end_time_obj.year, trace_end_time_obj.month, trace_end_time_obj.day,
                                        trace_end_time_obj.hour, int(time_window_minute))
        if trace_end_time_obj.date() < datetime.date(2016, 9, 18) or \
                                datetime.date(2016, 9, 30) < trace_end_time_obj.date() < datetime.date(2016, 10, 8):
            continue
        if not retrieved_data[line_data[1]][line_data[0]][time_window]:
            retrieved_data[line_data[1]][line_data[0]][time_window] = [travel_time]
        else:
            retrieved_data[line_data[1]][line_data[0]][time_window].append(travel_time)
    return retrieved_data


def generate_tollgate_velocity(travel_time_data, link_data, start_time, tollgate):
    tollgate_velocity = {}
    tmp_velocity = 0
    tmp_intersect_amount = 0
    for intersect in link_to_intersection[tollgate].keys():
        if len(travel_time_data[tollgate][intersect][start_time]) < 1:
            continue
        tmp_total_length = 0
        tmp_average_travel_time = np.average(np.array(travel_time_data[tollgate][intersect][start_time]))
        for link in link_to_intersection[tollgate][intersect]:
            tmp_total_length += float(link_data[link]['length'])
        tmp_velocity += tmp_total_length/tmp_average_travel_time
        tmp_intersect_amount += 1
    if tmp_velocity == 0:
        tollgate_velocity['average_velocity'] = -1
    else:
        tollgate_velocity['average_velocity'] = tmp_velocity/tmp_intersect_amount
    return tollgate_velocity


def generate_timebased_feature(raw_feature_data, start_time, tollgate, direction, mode='train'):

    initial_start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    time_based_feature = {}

    def minute_feature():
        minutefeature_start = initial_start_time - datetime.timedelta(minutes=20)
        minute_total_volume = []
        minute_feature_output = {}
        current_x = 1
        while minutefeature_start.date() > datetime.date(2016, 9, 18) and len(minute_total_volume) < 6:
            current_vol = raw_feature_data[minutefeature_start.strftime('%Y-%m-%d %H:%M:%S')][tollgate][direction]['volume']
            if mode == 'cv_test':
                if datetime.date(2016, 10, 17) < minutefeature_start.date() < datetime.date(2016, 10, 25) and \
                                minutefeature_start.hour not in [6, 7, 14, 15]:
                    current_x += 1
                    minutefeature_start = minutefeature_start - datetime.timedelta(minutes=20)
                    continue
            if current_vol:
                minute_total_volume.append(current_vol)
            minutefeature_start = minutefeature_start - datetime.timedelta(minutes=20)

        if len(minute_total_volume) < 1:
            minute_total_volume.append(-1)
        minute_coordinate = np.array(
            [[len(minute_total_volume) - index, n] for index, n in enumerate(minute_total_volume)])
        if len(minute_total_volume) < 1:
            minute_total_volume.append(-1)
        current_x += len(minute_coordinate)
        if len(minute_coordinate) == 1 or len(minute_coordinate) == 2:
            poly_degree = len(minute_coordinate) - 1
        elif len(minute_coordinate) < 5:
            poly_degree = 2
        else:
            poly_degree = 3
        poly_function = np.polyfit(minute_coordinate[:, 0], minute_coordinate[:, 1], poly_degree)
        p0_function = np.poly1d(poly_function)
        p1_function = np.polyder(p0_function)
        p2_function = np.polyder(p1_function)
        minute_feature_output['feature_minute_trend_predict'] = p0_function(current_x)
        minute_feature_output['feature_minute_trend_slope'] = p1_function(current_x - 1)
        minute_feature_output['feature_minute_trend_derivative'] = p2_function(current_x - 1)
        minute_feature_output['feature_minute_avg'] = np.average(minute_total_volume)
        minute_feature_output['feature_minute_std'] = np.std(minute_total_volume)
        minute_feature_output['feature_minute_min'] = np.percentile(minute_total_volume, 0)
        minute_feature_output['feature_minute_q1'] = np.percentile(minute_total_volume, 25)
        minute_feature_output['feature_minute_q2'] = np.percentile(minute_total_volume, 50)
        minute_feature_output['feature_minute_q3'] = np.percentile(minute_total_volume, 75)
        minute_feature_output['feature_minute_max'] = np.percentile(minute_total_volume, 100)
        return minute_feature_output

    def day_feature():
        day_step = 1
        #if initial_start_time.weekday() == 5 or 6:
        #    day_step = 7
        dayfeature_start = initial_start_time - datetime.timedelta(days=day_step)
        day_total_volume = []
        day_feature_output = {}
        current_x = 1
        while dayfeature_start.date() > datetime.date(2016, 9, 18):
            current_vol = raw_feature_data[dayfeature_start.strftime('%Y-%m-%d %H:%M:%S')][tollgate][direction]['volume']
            if mode == 'cv_test':
                if datetime.date(2016, 10, 17) < dayfeature_start.date() < datetime.date(2016, 10, 25):
                    current_x += 1
                    dayfeature_start = dayfeature_start - datetime.timedelta(days=day_step)
                    continue
            if current_vol:
                day_total_volume.append(current_vol)
            dayfeature_start = dayfeature_start - datetime.timedelta(days=day_step)
        if len(day_total_volume) < 1:
            day_total_volume.append(-1)
        day_coordinate = np.array(
            [[len(day_total_volume) - index, n] for index, n in enumerate(day_total_volume)])
        if len(day_total_volume) < 1:
            day_total_volume.append(-1)
        current_x += len(day_coordinate)
        if len(day_coordinate) == 1 or len(day_coordinate) == 2:
            poly_degree = len(day_coordinate) - 1
        elif len(day_coordinate) < 5:
            poly_degree = 2
        else:
            poly_degree = 3
        poly_function = np.polyfit(day_coordinate[:, 0], day_coordinate[:, 1], poly_degree)
        p0_function = np.poly1d(poly_function)
        p1_function = np.polyder(p0_function)
        p2_function = np.polyder(p1_function)
        day_feature_output['feature_day_trend_predict'] = p0_function(current_x)
        day_feature_output['feature_day_trend_slope'] = p1_function(current_x - 1)
        day_feature_output['feature_day_trend_derivative'] = p2_function(current_x - 1)
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
        week_total_volume = []
        week_feature_output = {}
        while weekfeature_start.date() > datetime.date(2016, 9, 18):
            current_vol = raw_feature_data[weekfeature_start.strftime('%Y-%m-%d %H:%M:%S')][tollgate][direction]['volume']
            if current_vol:
                week_total_volume.append(current_vol)
            weekfeature_start = weekfeature_start - datetime.timedelta(weeks=1)
        if len(week_total_volume) < 1:
            week_total_volume.append(-1)
        week_feature_output['feature_week_avg'] = np.average(week_total_volume)
        week_feature_output['feature_week_std'] = np.std(week_total_volume)
        week_feature_output['feature_week_min'] = np.percentile(week_total_volume, 0)
        week_feature_output['feature_week_max'] = np.percentile(week_total_volume, 100)
        '''
        week_feature_output['feature_week_q1'] = np.percentile(week_total_volume, 25)
        week_feature_output['feature_week_q2'] = np.percentile(week_total_volume, 50)
        week_feature_output['feature_week_q3'] = np.percentile(week_total_volume, 75)
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
    special_holiday = [datetime.date(2016, 9, n) for n in xrange(15, 18)] + [datetime.date(2016, 10, n) for n in xrange(1, 8)]
    special_workday = [datetime.date(2016, 9, 18), datetime.date(2016, 10, 8), datetime.date(2016, 10, 9)]

    for weekday in ['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6']:
        if initial_start_time.weekday() == int(weekday[-1]):
            holiday_feature_output[weekday] = 1
        else:
            holiday_feature_output[weekday] = 0

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
    '''
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
    '''
    holiday_feature_output['last'] = last_day_state
    holiday_feature_output['today'] = today_simple_state
    holiday_feature_output['tomorrow'] = tomorrow_state

    return holiday_feature_output


def generate_hour_feature(start_time):
    hour_feature_output = {}
    initial_start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    time_hour = initial_start_time.hour
    angle = 15*time_hour
    hour_feature_output['hour_x'] = round(math.cos(math.radians(angle)), 4)
    hour_feature_output['hour_y'] = round(math.sin(math.radians(angle)), 4)
    return hour_feature_output


def generate_weather_feature(weather_data, start_time):
    weather_feature_output = {}
    start_time_obj = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    time_flag = int(math.floor(start_time_obj.hour/3.0)*3)
    hour_obj = datetime.datetime(start_time_obj.year, start_time_obj.month, start_time_obj.day, time_flag)
    for feature in ['pressure', 'sea_pressure', 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation']:
        tmp_feature = weather_data[datetime.datetime.strftime(start_time_obj, '%Y-%m-%d')][str(hour_obj.hour)][feature]
        while len(tmp_feature) < 1:
            hour_obj = hour_obj - datetime.timedelta(hours=3)
            tmp_feature = weather_data[datetime.datetime.strftime(hour_obj, '%Y-%m-%d')][str(hour_obj.hour)][feature]
        weather_feature_output[feature] = tmp_feature
    return weather_feature_output


def generate_train_feature_file(dict_input_data, raw_input_data, link_data, weather_data, travel_time_data,
                                output_feature_file='train_feature_file'):
    output_file = open('{0}-{1}'.format(output_feature_file, datetime.datetime.today().strftime('%Y-%m-%d')), mode='w')
    cv_output_file = open('cv-test_{0}-{1}'.format(output_feature_file, datetime.datetime.today().strftime('%Y-%m-%d')), mode='w')
    feature_idx_file = open('index_{0}-{1}'.format(output_feature_file, datetime.datetime.today().strftime('%Y-%m-%d')), mode='w')
    for line_count in xrange(1, len(raw_input_data)):
        each_data = re.findall('"(.*?)"', raw_input_data[line_count])
        time_stamp = re.findall(r'\[(.*),(.*)\)', each_data[1])
        for index, time_window in enumerate(time_stamp):
            start = time_window[0]
        start_time_obj = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
        if datetime.date(2016, 9, 30) < start_time_obj.date() < datetime.date(2016, 10, 10) or \
                        start_time_obj.date() > datetime.date(2016, 10, 24):
            continue
        tollgate_id = each_data[0]
        direction_id = each_data[2]
        tmp_out_str = '{0}\t{1}\t1:{2}'.format(
            tollgate_id, dict_input_data[start][tollgate_id][direction_id]['volume'], direction_id)
        if line_count == 1:
            tmp_out_idx = '1:direction\n'
        tmp_index = 1

        # Feature : Hour in coordinate system
        hour_data = generate_hour_feature(start)
        for hour_feature in sorted(hour_data.keys()):
            tmp_index += 1
            tmp_out_str += '\t{0}:{1}'.format(tmp_index, hour_data[hour_feature])
            if line_count == 1:
                tmp_out_idx += '{0}:{1}\n'.format(tmp_index, hour_feature)

        # Feature : State of day(ie. workday or holiday)
        day_state_feature = generate_holiday_feature(start)
        for n in sorted(day_state_feature.keys()):
            tmp_index += 1
            tmp_out_str += '\t{0}:{1}'.format(tmp_index, day_state_feature[n])
            if line_count == 1:
                tmp_out_idx += '{0}:{1}\n'.format(tmp_index, n)

        # Feature : Weather
        weather_feature = generate_weather_feature(weather_data, start)
        for w_feature in sorted(weather_feature.keys()):
            tmp_index += 1
            tmp_out_str += '\t{0}:{1}'.format(tmp_index, weather_feature[w_feature])
            if line_count == 1:
                tmp_out_idx += '{0}:{1}\n'.format(tmp_index, w_feature)

        if datetime.date(2016, 10, 17) < start_time_obj.date() < datetime.date(2016, 10, 25) \
                and start_time_obj.hour in [8, 9, 17, 18]:
            tmp_cv_out_str = tmp_out_str
            tmp_cv_index = tmp_index

            # Feature(for cross validation test) : Volume data of previous time window (min, day, week)
            cv_history_feature = generate_timebased_feature(dict_input_data, start, tollgate_id, direction_id, 'cv_test')
            cv_near_gate_history_feature = generate_timebased_feature(dict_input_data, start, near_tollgate[tollgate_id],
                                                                      direction_id, 'cv_test')
            for n in sorted(cv_history_feature.keys()):
                for i in sorted(cv_history_feature[n].keys()):
                    tmp_cv_index += 1
                    tmp_cv_out_str += '\t{0}:{1}'.format(tmp_cv_index, cv_history_feature[n][i])
            cv_output_file.write(tmp_cv_out_str + '\n')

        # Feature : Volume data of previous time window (min, day, week)
        history_feature = generate_timebased_feature(dict_input_data, start, tollgate_id, direction_id)
        near_gate_history_feature = generate_timebased_feature(dict_input_data, start, near_tollgate[tollgate_id],
                                                               direction_id)
        for n in sorted(history_feature.keys()):
            for i in sorted(history_feature[n].keys()):
                tmp_index += 1
                tmp_out_str += '\t{0}:{1}'.format(tmp_index, history_feature[n][i])
                if line_count == 1:
                    tmp_out_idx += '{0}:{1}\n'.format(tmp_index, i)
        output_file.write(tmp_out_str + '\n')
    feature_idx_file.write(tmp_out_idx)
    feature_idx_file.close()
    output_file.close()
    cv_output_file.close()


def generate_test_feature_file(dict_input_data, test_file, link_data, weather_data, travel_time_data,
                               output_feature_file='test_feature_file'):
    output_file = open('{0}-{1}'.format(output_feature_file, datetime.datetime.today().strftime('%Y-%m-%d')), mode='w')
    test_input_data = open(test_file, mode='r').readlines()
    for line_count in xrange(0, len(test_input_data)):
        each_data = test_input_data[line_count].split(',')
        time_stamp = re.findall(r'\[(.*),(.*)\)', test_input_data[line_count])
        for index, time_window in enumerate(time_stamp):
            start = time_window[0]
        start_time_obj = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
        tollgate_id = each_data[0]
        direction_id = each_data[3]
        volume = each_data[4].split('\r')[0]
        tmp_out_str = '{0}\t{1}\t1:{2}'.format(tollgate_id, volume.rstrip(), direction_id)
        tmp_index = 1

        # Feature : Hour in coordinate system
        hour_data = generate_hour_feature(start)
        for hour_feature in sorted(hour_data.keys()):
            tmp_index += 1
            tmp_out_str += '\t{0}:{1}'.format(tmp_index, hour_data[hour_feature])

        # Feature : State of day(ie. workday or holiday)
        day_state_feature = generate_holiday_feature(start)
        for n in sorted(day_state_feature.keys()):
            tmp_index += 1
            tmp_out_str += '\t{0}:{1}'.format(tmp_index, day_state_feature[n])

        # Feature : Weather
        weather_feature = generate_weather_feature(weather_data, start)
        for w_feature in sorted(weather_feature.keys()):
            tmp_index += 1
            tmp_out_str += '\t{0}:{1}'.format(tmp_index, weather_feature[w_feature])

        # Feature : Volume data of previous time window (min, day, week)
        history_feature = generate_timebased_feature(dict_input_data, start, tollgate_id, direction_id)
        for n in sorted(history_feature.keys()):
            for i in sorted(history_feature[n].keys()):
                tmp_index += 1
                tmp_out_str += '\t{0}:{1}'.format(tmp_index, history_feature[n][i])
        output_file.write(tmp_out_str + '\n')
    output_file.close()

if __name__ == '__main__':
    input_opt = argv[1:]
    i = 0
    start_time = timeit.default_timer()
    while i < len(input_opt):
        if input_opt[i] == '-tr':
            train_file = input_opt[i+1]
        elif input_opt[i] == '-te':
            test_file = input_opt[i+1]
        elif input_opt[i] == '-link':
            link_file = input_opt[i+1]
        elif input_opt[i] == '-w':
            weather_file = input_opt[i+1]
        elif input_opt[i] == '-time':
            travel_file = input_opt[i+1]
        elif input_opt[i] == '-h':
            print '-tr [train_file] -te [test_file(optional)]'
        i += 1
    link_to_intersection = {}
    near_tollgate = {'1': '3', '2': '3', '3': '1'}
    data_dict, raw_data = get_volume_data(train_file)
    link_data = get_tollgate_lane_data(link_file)
    travel_data = get_travel_time_data(travel_file)
    weather_data = get_weather_data(weather_file)
    generate_train_feature_file(data_dict, raw_data, link_data, weather_data, travel_data)
    if test_file:
        generate_test_feature_file(data_dict, test_file, link_data, weather_data, travel_data)
    print '{0} sec'.format(timeit.default_timer() - start_time)
