import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed

import operator
import lmfit
from lmfit import Parameters

from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami

from scipy.stats import norm
from scipy.stats import multivariate_normal

from scipy.optimize import minimize



def select_data_points_radial_average_MA(x0, y0, Rmax, theta_range, file_name): # extract radial averaged temperature from one csv file
    # This method was originally developed by Mosfata, was adapted by HY to use for amplitude and phase estimation
    df_raw = pd.read_csv(file_name, sep=',', header=None, names=list(np.arange(0, 639)))
    raw_time_string = df_raw.iloc[2, 0][df_raw.iloc[2, 0].find('=') + 2:]  # '073:02:19:35.160000'
    raw_time_string = raw_time_string[raw_time_string.find(":") + 1:]  # '02:19:35.160000'
    strip_time = datetime.strptime(raw_time_string, '%H:%M:%S.%f')
    time_in_seconds = strip_time.hour * 3600 + strip_time.minute * 60 + strip_time.second + strip_time.microsecond / 10 ** 6

    theta_min = theta_range[0]*np.pi/180
    theta_max = theta_range[1]*np.pi/180

    theta_n = int(abs(theta_max-theta_min)/(2*np.pi)*180) # previous studies have shown that 2Pi requires above 100 theta points to yield accurate results

    theta = np.linspace(theta_min, theta_max, theta_n)  # The angles 1D array (rad)
    df_temp = df_raw.iloc[5:, :]

    #theta_n

    Tr = np.zeros((Rmax, theta_n))  # Initializing the radial temperature matrix (T)

    for i in range(Rmax):  # Looping through all radial points
        for j, theta_ in enumerate(theta):  # Looping through all angular points
            y = i * np.sin(theta_) + y0;
            x = i * np.cos(theta_) + x0  # Identifying the spatial 2D cartesian coordinates
            y1 = int(np.floor(y));
            y2 = y1 + 1;
            x1 = int(np.floor(x));
            x2 = x1 + 1  # Identifying the neighboring 4 points
            dy1 = (y2 - y) / (y2 - y1);
            dy2 = (y - y1) / (y2 - y1)  # Identifying the corresponding weights for the y-coordinates
            dx1 = (x2 - x) / (x2 - x1);
            dx2 = (x - x1) / (x2 - x1)  # Identifying the corresponding weights for the x-coordinates
            T11 = df_temp.iloc[y1, x1];
            T21 = df_temp.iloc[y2, x1]  # Identifying the corresponding temperatures for the y-coordinates
            T12 = df_temp.iloc[y1, x2];
            T22 = df_temp.iloc[y2, x2]  # Identifying the corresponding temperatures for the x-coordinates
            Tr[i, j] = dx1 * dy1 * T11 + dx1 * dy2 * T21 + dx2 * dy1 * T12 + dx2 * dy2 * T22 + 273.15  # Interpolated angular temperature matrix

    T_interpolate = np.mean(Tr, axis=1)

    return T_interpolate, time_in_seconds




def check_angular_uniformity(x0, y0, N_Rmax, pr, path, rec_name, output_name, method, num_cores, f_heating, R0, gap,
                             R_analysis,
                             exp_amp_phase_extraction_method):
    # we basically break entire disk into 6 regions, with interval of pi/3
    fig = plt.figure(figsize=(18.3, 12))
    colors = ['red', 'black', 'green', 'blue', 'orange', 'magenta']

    df_temperature_list = []
    df_amp_phase_list = []

    plt.subplot(231)

    for j in range(6):
        # note radial_temperature_average_disk_sample automatically checks if a dump file exist
        df_temperature = radial_temperature_average_disk_sample_several_ranges(x0, y0, N_Rmax,
                                                                               [[60*j, 60*(j+1)]],
                                                                               pr, path, rec_name, output_name, method,
                                                                               num_cores)
        df_amplitude_phase_measurement = batch_process_horizontal_lines(df_temperature, f_heating, R0, gap, R_analysis,
                                                                        exp_amp_phase_extraction_method)
        df_temperature_list.append(df_temperature)
        df_amp_phase_list.append(df_amplitude_phase_measurement)

        plt.scatter(df_amplitude_phase_measurement['r'],
                    df_amplitude_phase_measurement['amp_ratio'], facecolors='none',
                    s=60, linewidths=2, edgecolor=colors[j], label=str(60 * j) + ' to ' + str(60 * (j + 1)) + ' Degs')
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')

    plt.xlabel('R (m)', fontsize=12, fontweight='bold')
    plt.ylabel('Amplitude Ratio', fontsize=12, fontweight='bold')
    plt.title('f_heating = {} Hz, rec = {}'.format(f_heating, rec_name), fontsize=12, fontweight='bold')
    plt.legend()

    plt.subplot(232)

    for j in range(6):
        df_temperature = df_temperature_list[j]
        df_amplitude_phase_measurement = df_amp_phase_list[j]
        plt.scatter(df_amplitude_phase_measurement['r'],
                    df_amplitude_phase_measurement['phase_diff'], facecolors='none',
                    s=60, linewidths=2, edgecolor=colors[j], label=str(60 * j) + ' to ' + str(60 * (j + 1)) + ' Degs')
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')

    plt.xlabel('R (m)', fontsize=12, fontweight='bold')
    plt.ylabel('Phase difference (rad)', fontsize=12, fontweight='bold')
    plt.title('x0 = {}, y0 = {}, R0 = {}'.format(x0, y0, R0), fontsize=12, fontweight='bold')
    plt.legend()

    plt.subplot(233)

    for j in range(6):
        df_temperature = df_temperature_list[j]
        time_max = 10 * 1 / f_heating  # only show 10 cycles
        df_temperature = df_temperature.query('reltime<{:.2f}'.format(time_max))
        plt.plot(df_temperature['reltime'],
                 df_temperature.iloc[:, R0], linewidth=2, color=colors[j],
                 label=str(60 * j) + ' to ' + str(60 * (j + 1)) + ' Degs')
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')

    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Temperature (K)', fontsize=12, fontweight='bold')
    plt.title('R_ayalysis = {}, gap = {}'.format(R_analysis, gap), fontsize=12, fontweight='bold')
    plt.legend()

    plt.subplot(234)

    file_name_0 = [path + x for x in os.listdir(path)][0]
    n0 = file_name_0.rfind('//')
    n1 = file_name_0.rfind('.csv')
    frame_num = file_name_0[n0 + 2:n1]

    df_first_frame = pd.read_csv(file_name_0, sep=',', header=None, names=list(np.arange(0, 639)))
    df_first_frame_temperature = df_first_frame.iloc[5:, :]
    xmin = x0 - R0 - R_analysis
    xmax = x0 + R0 + R_analysis
    ymin = y0 - R0 - R_analysis
    ymax = y0 + R0 + R_analysis
    Z = np.array(df_first_frame_temperature.iloc[ymin:ymax, xmin:xmax])
    x = np.arange(xmin, xmax, 1)
    y = np.arange(ymin, ymax, 1)
    X, Y = np.meshgrid(x, y)

    # mid = int(np.shape(Z)[1] / 2)
    x3 = min(R0 + 10, R0 + R_analysis - 20)

    manual_locations = [(x0 - R0 + 15, y0 - R0 + 15), (x0 - R0, y0 - R0), (x0 - x3, y0 - x3)]
    # fig, ax = plt.subplots(figsize=(6, 6))
    ax = plt.gca()
    CS = ax.contour(X, Y, Z, 12)
    plt.plot([xmin, xmax], [y0, y0], ls='-.', color='k', lw=2)  # add a horizontal line cross x0,y0
    plt.plot([x0, x0], [ymin, ymax], ls='-.', color='k', lw=2)  # add a vertical line cross x0,y0

    circle1 = plt.Circle((x0, y0), R0, edgecolor='r', fill=False, linewidth=3, linestyle='-.')
    circle2 = plt.Circle((x0, y0), R0 + R_analysis, edgecolor='k', fill=False, linewidth=3, linestyle='-.')

    ax.invert_yaxis()
    ax.clabel(CS, inline=1, fontsize=12, manual=manual_locations)
    ax.add_artist(circle1)
    ax.add_artist(circle2)

    ax.set_xlabel('x (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('y (pixels)', fontsize=12, fontweight='bold')
    ax.set_title(frame_num, fontsize=12, fontweight='bold')

    plt.subplot(235)

    N_mid = int(len([path + x for x in os.listdir(path)])/3)
    file_name_0 = [path + x for x in os.listdir(path)][N_mid]
    n0 = file_name_0.rfind('//')
    n1 = file_name_0.rfind('.csv')
    frame_num = file_name_0[n0 + 2:n1]

    df_first_frame = pd.read_csv(file_name_0, sep=',', header=None, names=list(np.arange(0, 639)))
    df_first_frame_temperature = df_first_frame.iloc[5:, :]
    xmin = x0 - R0 - R_analysis
    xmax = x0 + R0 + R_analysis
    ymin = y0 - R0 - R_analysis
    ymax = y0 + R0 + R_analysis
    Z = np.array(df_first_frame_temperature.iloc[ymin:ymax, xmin:xmax])
    x = np.arange(xmin, xmax, 1)
    y = np.arange(ymin, ymax, 1)
    X, Y = np.meshgrid(x, y)

    # mid = int(np.shape(Z)[1] / 2)
    x3 = min(30, R_analysis - 10)

    manual_locations = [(x0 - 5, y0 - 5), (x0 - 15, y0 - 15), (x0 - x3, y0 - x3)]
    # fig, ax = plt.subplots(figsize=(6, 6))
    ax = plt.gca()
    CS = ax.contour(X, Y, Z, 12)
    plt.plot([xmin, xmax], [y0, y0], ls='-.', color='k', lw=2)  # add a horizontal line cross x0,y0
    plt.plot([x0, x0], [ymin, ymax], ls='-.', color='k', lw=2)  # add a vertical line cross x0,y0

    circle1 = plt.Circle((x0, y0), R0, edgecolor='r', fill=False, linewidth=3, linestyle='-.')
    circle2 = plt.Circle((x0, y0), R0 + R_analysis, edgecolor='k', fill=False, linewidth=3, linestyle='-.')

    ax.invert_yaxis()
    ax.clabel(CS, inline=1, fontsize=12, manual=manual_locations)
    ax.add_artist(circle1)
    ax.add_artist(circle2)

    ax.set_xlabel('x (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('y (pixels)', fontsize=12, fontweight='bold')
    ax.set_title(frame_num, fontsize=12, fontweight='bold')

    plt.subplot(236)

    file_name_0 = [path + x for x in os.listdir(path)][0]
    n0 = file_name_0.rfind('//')
    n1 = file_name_0.rfind('.csv')
    frame_num = file_name_0[n0 + 2:n1]

    df_first_frame = pd.read_csv(file_name_0, sep=',', header=None, names=list(np.arange(0, 639)))
    df_first_frame_temperature = df_first_frame.iloc[5:, :]
    Z = np.array(df_first_frame_temperature.iloc[y0 - R_analysis:y0 + R_analysis, x0 - R_analysis:x0 + R_analysis])
    x = np.arange(x0 - R_analysis, x0 + R_analysis, 1)
    y = np.arange(y0 - R_analysis, y0 + R_analysis, 1)
    X, Y = np.meshgrid(x, y)

    # mid = int(np.shape(Z)[1] / 2)
    x3 = min(30, R_analysis - 10)

    manual_locations = [(x0 - 5, y0 - 5), (x0 - 15, y0 - 15), (x0 - x3, y0 - x3)]
    # fig, ax = plt.subplots(figsize=(6, 6))
    ax = plt.gca()
    CS = ax.contour(X, Y, Z, 5)
    plt.plot([x0 - R_analysis, x0 + R_analysis], [y0, y0], ls='-.', color='k',
             lw=2)  # add a horizontal line cross x0,y0
    plt.plot([x0, x0], [y0 - R_analysis, y0 + R_analysis], ls='-.', color='k', lw=2)  # add a vertical line cross x0,y0
    ax.invert_yaxis()
    ax.clabel(CS, inline=1, fontsize=12, manual=manual_locations)
    ax.set_xlabel('x (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('y (pixels)', fontsize=12, fontweight='bold')
    ax.set_title(frame_num, fontsize=12, fontweight='bold')

    plt.show()

    amp_ratio_list = np.array([np.array(df_amp_phase_list[i]['amp_ratio']) for i in range(6)])
    phase_diff_list = np.array([np.array(df_amp_phase_list[i]['phase_diff']) for i in range(6)])
    weight = np.linspace(1, 0.6, len(np.std(amp_ratio_list, axis=0)))
    amp_std = np.std(amp_ratio_list, axis=0)
    phase_std = np.std(phase_diff_list, axis=0)
    weight_amp_phase_std = (amp_std + phase_std) * weight
    sum_std = np.sum(weight_amp_phase_std)

    return sum_std, fig



def select_data_points_radial_average_MA_match_model_grid(x0, y0, N_Rmax, pr, R_sample, theta_n,
                                                          file_name):  # N_Rmax: total number of computation nodes between center and edge
    # This method was originally developed by Mosfata, was adapted by HY to use for amplitude and phase estimation
    df_raw = pd.read_csv(file_name, sep=',', header=None, names=list(np.arange(0, 639)))
    raw_time_string = df_raw.iloc[2, 0][df_raw.iloc[2, 0].find('=') + 2:]  # '073:02:19:35.160000'
    raw_time_string = raw_time_string[raw_time_string.find(":") + 1:]  # '02:19:35.160000'
    strip_time = datetime.strptime(raw_time_string, '%H:%M:%S.%f')
    time_in_seconds = strip_time.hour * 3600 + strip_time.minute * 60 + strip_time.second + strip_time.microsecond / 10 ** 6
    theta = np.linspace(0, 2 * np.pi, theta_n)  # The angles 1D array (rad)
    df_temp = df_raw.iloc[5:, :]

    Tr = np.zeros((N_Rmax, theta_n))  # Initializing the radial temperature matrix (T)

    R = np.linspace(0, R_sample, N_Rmax)
    for i, R_ in enumerate(R):  # Looping through all radial points
        for j, theta_ in enumerate(theta):  # Looping through all angular points
            y = R_ / pr * np.sin(theta_) + y0;
            x = R_ / pr * np.cos(theta_) + x0  # Identifying the spatial 2D cartesian coordinates
            y1 = int(np.floor(y));
            y2 = y1 + 1;
            x1 = int(np.floor(x));
            x2 = x1 + 1  # Identifying the neighboring 4 points
            dy1 = (y2 - y) / (y2 - y1);
            dy2 = (y - y1) / (y2 - y1)  # Identifying the corresponding weights for the y-coordinates
            dx1 = (x2 - x) / (x2 - x1);
            dx2 = (x - x1) / (x2 - x1)  # Identifying the corresponding weights for the x-coordinates
            T11 = df_temp.iloc[y1, x1];
            T21 = df_temp.iloc[y2, x1]  # Identifying the corresponding temperatures for the y-coordinates
            T12 = df_temp.iloc[y1, x2];
            T22 = df_temp.iloc[y2, x2]  # Identifying the corresponding temperatures for the x-coordinates
            Tr[
                i, j] = dx1 * dy1 * T11 + dx1 * dy2 * T21 + dx2 * dy1 * T12 + dx2 * dy2 * T22 + 273.15  # Interpolated angular temperature matrix

    T_interpolate = np.mean(Tr, axis=1)

    return T_interpolate, time_in_seconds


def select_data_points_radial_average_HY(x0, y0, Rmax, file_name):
    df_raw = pd.read_csv(file_name, sep=',', header=None, names=list(np.arange(0, 639)))
    raw_time_string = df_raw.iloc[2, 0][df_raw.iloc[2, 0].find('=') + 2:]  # '073:02:19:35.160000'
    raw_time_string = raw_time_string[raw_time_string.find(":") + 1:]  # '02:19:35.160000'
    strip_time = datetime.strptime(raw_time_string, '%H:%M:%S.%f')
    time_in_seconds = strip_time.hour * 3600 + strip_time.minute * 60 + strip_time.second + strip_time.microsecond / 10 ** 6

    df_temp = df_raw.iloc[5:, :]
    x = []
    y = []
    angle = []
    r_pixel = []
    T_interpolate = np.zeros(Rmax)

    for r in range(Rmax):
        if r == 0:
            x.append(x0)
            y.append(y0)
            angle.append(0)
            r_pixel.append(r)
            T_interpolate[r] = df_temp.iloc[y0, x0]

        else:
            temp = []
            for i in np.arange(x0 - r - 2, x0 + r + 2, 1):
                for j in np.arange(y0 - r - 2, y0 + r + 2, 1):
                    d = ((i - x0) ** 2 + (j - y0) ** 2) ** (0.5)
                    if d >= r and d < r + 1:
                        x.append(i)
                        y.append(j)
                        r_pixel.append(r)
                        temp.append(
                            (df_temp.iloc[j, i] - T_interpolate[r - 1]) / (d - r + 1) + T_interpolate[r - 1] + 273.15)

            T_interpolate[r] = np.mean(temp)

    return T_interpolate, time_in_seconds


def radial_temperature_average_disk_sample_old(x0, y0, N_Rmax,theta_range, pr, path, rec_name, output_name, method,
                                           num_cores):  # unit in K
    # path= "C://Users//NTRG lab//Desktop//yuan//"
    # rec_name = "Rec-000011_e63", this is the folder which contains all csv data files

    dump_file_path = output_name + '_x0_{}_y0_{}_Rmax_{}_method_{}_theta_{}_{}'.format(x0, y0, N_Rmax, method,int(180/np.pi*theta_range[0]),int(180/np.pi*theta_range[1]))

    if (os.path.isfile(dump_file_path)):  # First check if a dump file exist:
        print('Found previous dump file :' + dump_file_path)
        temp_dump = pickle.load(open(dump_file_path, 'rb'))

    else:  # If not we obtain the dump file, note the dump file is averaged radial temperature

        file_names = [path + x for x in os.listdir(path)]
        s_time = time.time()

        if method == 'MA':  # default method, this one is much faster
            theta_n = 100  # default theta_n=100, however, if R increased significantly theta_n should also increase
            # joblib_output = Parallel(n_jobs=num_cores)(delayed(select_data_points_radial_average_MA)(x0,y0,Rmax,theta_n,file_name) for file_name in tqdm(file_names))
            joblib_output = Parallel(n_jobs=num_cores)(
                delayed(select_data_points_radial_average_MA)(x0, y0, N_Rmax, theta_range, file_name) for file_name in
                tqdm(file_names))

        else:
            joblib_output = Parallel(n_jobs=num_cores)(
                delayed(select_data_points_radial_average_HY)(x0, y0, N_Rmax, file_name) for file_name in
                tqdm(file_names))

        pickle.dump(joblib_output, open(dump_file_path, "wb"))  # create a dump file

        e_time = time.time()
        print('Time to process the entire dataset is {}'.format((e_time - s_time)))

        temp_dump = pickle.load(open(dump_file_path, 'rb'))

    temp_dump = np.array(temp_dump);
    temp_dump = temp_dump[temp_dump[:, 1].argsort()]  # sort based on the second column, which is relative time

    temp_profile = [];
    time_series = []
    for item in temp_dump:
        temp_profile.append(item[0])
        time_series.append(item[1])
    time_series = np.array(time_series)
    temp_data = np.array(temp_profile)
    time_data = time_series - min(time_series)  # such that time start from zero

    df_temperature = pd.DataFrame(
        data=temp_data)  # return a dataframe containing radial averaged temperature and relative time
    df_temperature['reltime'] = time_data

    return df_temperature  # note the column i of the df_temperature indicate the temperature in pixel i


def radial_temperature_average_disk_sample_several_ranges(x0, y0, N_Rmax, theta_range_list, pr, path, rec_name,
                                                          output_name, method,
                                                          num_cores):  # unit in K
    # path= "C://Users//NTRG lab//Desktop//yuan//"
    # rec_name = "Rec-000011_e63", this is the folder which contains all csv data files
    # note theta_range should be a 2D array [[0,pi/3],[pi/3*2,2pi]]
    df_temperature_list = []
    for theta_range in theta_range_list:
        dump_file_path = output_name + '_x0_{}_y0_{}_Rmax_{}_method_{}_theta_{}_{}'.format(x0, y0, N_Rmax, method, int(theta_range[0]), int(theta_range[1]))

        if (os.path.isfile(dump_file_path)):  # First check if a dump file exist:
            print('Found previous dump file :' + dump_file_path)
            temp_dump = pickle.load(open(dump_file_path, 'rb'))

        else:  # If not we obtain the dump file, note the dump file is averaged radial temperature

            file_names = [path + x for x in os.listdir(path)]
            s_time = time.time()

            if method == 'MA':  # default method, this one is much faster
                theta_n = 100  # default theta_n=100, however, if R increased significantly theta_n should also increase
                # joblib_output = Parallel(n_jobs=num_cores)(delayed(select_data_points_radial_average_MA)(x0,y0,Rmax,theta_n,file_name) for file_name in tqdm(file_names))
                joblib_output = Parallel(n_jobs=num_cores)(
                    delayed(select_data_points_radial_average_MA)(x0, y0, N_Rmax, theta_range, file_name) for file_name
                    in
                    tqdm(file_names))

            else:
                joblib_output = Parallel(n_jobs=num_cores)(
                    delayed(select_data_points_radial_average_HY)(x0, y0, N_Rmax, file_name) for file_name in
                    tqdm(file_names))

            pickle.dump(joblib_output, open(dump_file_path, "wb"))  # create a dump file

            e_time = time.time()
            print('Time to process the entire dataset is {}'.format((e_time - s_time)))

            temp_dump = pickle.load(open(dump_file_path, 'rb'))

        temp_dump = np.array(temp_dump);
        temp_dump = temp_dump[temp_dump[:, 1].argsort()]  # sort based on the second column, which is relative time

        temp_profile = [];
        time_series = []
        for item in temp_dump:
            temp_profile.append(item[0])
            time_series.append(item[1])
        time_series = np.array(time_series)
        temp_data = np.array(temp_profile)
        time_data = time_series - min(time_series)  # such that time start from zero

        df_temperature = pd.DataFrame(
            data=temp_data)  # return a dataframe containing radial averaged temperature and relative time
        df_temperature['reltime'] = time_data
        df_temperature_list.append(df_temperature)

    cols = df_temperature_list[0].columns
    data = np.array([np.array(df_temperature_list_.iloc[:, :]) for df_temperature_list_ in df_temperature_list])
    data_mean = np.mean(data, axis=0)
    df_averaged_temperature = pd.DataFrame(data=data_mean, columns=cols)

    return df_averaged_temperature  # note the column i of the df_temperature indicate the temperature in pixel i


def sin_func(t, amplitude, phase, bias, f_heating):
    return amplitude * np.sin(2 * np.pi * f_heating * t + phase) + bias

def sine_residual(params, x, data):
    amplitude = params['amplitude']
    phase = params['phase']
    bias = params['bias']
    freq = params['frequency']

    model = amplitude * np.sin(2 * np.pi * freq * x + phase) + bias

    return abs((data - model)/data) # before this is only data-model, this can't be right for certain situations


def amp_phase_one_pair(index, df_temperature, f_heating, gap, frequency_analysis_method):
    n_col = df_temperature.shape[1]
    tmin = min(df_temperature['reltime'])
    time = df_temperature['reltime'] - tmin
    dt = max(time)/len(time)

    if frequency_analysis_method == 'fft':

        A1 = df_temperature[index[0]]
        A2 = df_temperature[index[1]]

        fft_X1 = np.fft.fft(A1)
        fft_X2 = np.fft.fft(A2)

        T_total = np.max(time) - np.min(time)
        df = 1 / T_total

        N_0 = int(f_heating / df)

        magnitude_X1 = np.abs(fft_X1)
        magnitude_X2 = np.abs(fft_X2)

        phase_X1 = np.angle(fft_X1)
        phase_X2 = np.angle(fft_X2)

        N_f = 2
        N1, Amp1 = max(enumerate(magnitude_X1[N_0 - N_f:N_0 + N_f]), key=operator.itemgetter(1))
        N2, Amp2 = max(enumerate(magnitude_X2[N_0 - N_f:N_0 + N_f]), key=operator.itemgetter(1))

        Nf = N_0 + N1 - N_f
        amp_ratio = magnitude_X2[Nf] / magnitude_X1[Nf]
        phase_diff = abs(phase_X1[Nf] - phase_X2[Nf])

        phase_diff = phase_diff % np.pi
        if phase_diff > np.pi / 2:
            phase_diff = np.pi - phase_diff

        L = abs(index[0] - index[1]) * gap

    elif frequency_analysis_method == 'sine':

        fitting_params_initial = {'amplitude': 10, 'phase': 0.1, 'bias': 10}

        n_col = df_temperature.shape[1]
        tmin = min(df_temperature['reltime'])
        time = df_temperature['reltime'] - tmin

        A1 = df_temperature[index[0]]
        A2 = df_temperature[index[1]]

        x0 = np.array([10, 0.1, 10])  # amplitude,phase,bias

        params1 = Parameters()
        params1.add('amplitude', value=fitting_params_initial['amplitude'])
        params1.add('phase', value=fitting_params_initial['phase'])
        params1.add('bias', value=fitting_params_initial['bias'])
        params1.add('frequency', value=f_heating, vary=False)

        res1 = lmfit.minimize(sine_residual, params1, args=(time, A1))

        params2 = Parameters()
        params2.add('amplitude', value=fitting_params_initial['amplitude'])
        params2.add('phase', value=fitting_params_initial['phase'])
        params2.add('bias', value=fitting_params_initial['bias'])
        params2.add('frequency', value=f_heating, vary=False)
        res2 = lmfit.minimize(sine_residual, params2, args=(time, A2))

        amp1 = np.abs(res1.params['amplitude'].value)
        amp2 = np.abs(res2.params['amplitude'].value)

        p1 = res1.params['phase'].value
        p2 = res2.params['phase'].value

        amp_ratio = min(np.abs(amp1 / amp2), np.abs(amp2 / amp1))

        phase_diff = np.abs(p1 - p2)
        phase_diff = phase_diff % np.pi
        if phase_diff > np.pi / 2:
            phase_diff = np.pi - phase_diff

        T_total = np.max(time) - np.min(time)

        df = 1 / T_total

        L = abs(index[0] - index[1]) * gap  # here does not consider the offset between the heater and the region of analysis


    elif frequency_analysis_method == 'max_min':

        A1 = np.array(df_temperature[index[0]])
        A2 = np.array(df_temperature[index[1]])

        amp1,phase1 = find_amplitude_phase_min_max(A1,f_heating,dt)
        amp2, phase2 = find_amplitude_phase_min_max(A2, f_heating,dt)

        amp_ratio = amp2/amp1

        phase_diff = np.abs(phase1 - phase2)

        phase_diff = phase_diff % np.pi
        if phase_diff > np.pi / 2:
            phase_diff = np.pi - phase_diff

        L = abs(index[0] - index[1]) * gap

    return L, phase_diff, amp_ratio

def find_amplitude_phase_min_max(a,f_heating,dt):

    find_max = ((a[1:-1] - a[0:-2]) > 0) & ((a[1:-1] - a[2:]) > 0)
    find_min = ((a[1:-1] - a[0:-2]) < 0) & ((a[1:-1] - a[2:]) < 0)
    max_index = np.where(find_max == 1)
    min_index = np.where(find_min == 1)
    amp = a[max_index].mean() - a[min_index].mean()
    #phase_index = min_index[0][0]
    phase = min_index[0][0] * 2 * np.pi * f_heating*dt

    return amp, phase

def fit_amp_phase_one_batch(df_temperature, f_heating, R0, gap,frequency_analysis_method):
    N_lines = df_temperature.shape[1] - 1
    r_list = np.zeros(N_lines - 1)
    phase_diff_list = np.zeros(N_lines - 1)
    amp_ratio_list = np.zeros(N_lines - 1)
    r_ref_list = np.zeros(N_lines - 1)

    R_ring_inner_edge = R0  # R0 is the refernce radius, it should be minimumly equal to the radius of the light blocker

    for i in range(N_lines):

        if i > 0:
            index = [0, i]
            r_list[i - 1], phase_diff_list[i - 1], amp_ratio_list[i - 1] = amp_phase_one_pair(index, df_temperature,
                                                                                              f_heating, gap,frequency_analysis_method)
            r_list[i - 1] = r_list[i - 1] + R_ring_inner_edge
            r_ref_list[i - 1] = R_ring_inner_edge

    return r_list, r_ref_list, phase_diff_list, amp_ratio_list


def batch_process_horizontal_lines(df_temperature, f_heating, R0, gap, R_analysis,frequency_analysis_method):
    # N_line_groups = gap - 1
    N_line_groups = gap
    N_line_each_group = int(R_analysis / gap) - 1  # Don't have to analyze entire Rmax pixels
    r_list_all = []
    r_ref_list_all = []
    phase_diff_list_all = []
    amp_ratio_list_all = []
    N_frame_keep = df_temperature.shape[0]

    T_group = np.zeros((N_line_groups, N_line_each_group, N_frame_keep))

    for k in range(N_frame_keep):
        for j in range(N_line_groups):
            for i in range(N_line_each_group):
                T_group[j, i, k] = df_temperature.iloc[k, R0 + j + gap * i]

    for j in range(N_line_groups):
        df = pd.DataFrame(T_group[j, :, :].T)
        df['reltime'] = df_temperature['reltime']
        r_list, r_ref_list, phase_diff_list, amp_ratio_list = fit_amp_phase_one_batch(df, f_heating, R0 + j, gap,frequency_analysis_method)
        # print(df.shape[1]-1)

        r_list_all = r_list_all + list(r_list)
        r_ref_list_all = r_ref_list_all + list(r_ref_list)

        phase_diff_list_all = phase_diff_list_all + list(phase_diff_list)
        amp_ratio_list_all = amp_ratio_list_all + list(amp_ratio_list)

    df_result_IR = pd.DataFrame(data={'r': r_list_all, 'r_ref': r_ref_list_all, 'amp_ratio': amp_ratio_list_all,
                                      'phase_diff': phase_diff_list_all})

    return df_result_IR


def light_source_intensity(r, t, solar_simulator_settings, light_source_property):
    f_heating = solar_simulator_settings['f_heating']
    As = solar_simulator_settings['V_amplitude']
    V_DC = solar_simulator_settings['V_DC']

    ks = light_source_property['ks']  # solar simulator setting->current
    bs = light_source_property['bs']

    ka = light_source_property['ka']  # currect -> heat flux
    ba = light_source_property['ba']

    Amax = light_source_property['Amax']
    sigma_s = light_source_property['sigma_s']
    I = bs + ks * V_DC + ks * As * np.sin(2 * np.pi * f_heating * t)  # t: time, r: distance from center

    q = Amax * (ka * I + ba) / np.pi * (sigma_s / (r ** 2 + sigma_s ** 2))
    return q


def light_source_intensity_vecterize(r_array, t_array, N_Rs, solar_simulator_settings, light_source_property):
    f_heating = solar_simulator_settings['f_heating']
    As = solar_simulator_settings['V_amplitude']
    V_DC = solar_simulator_settings['V_DC']

    # N_Rs = 20
    ks = light_source_property['ks']  # solar simulator setting->current
    bs = light_source_property['bs']

    ka = light_source_property['ka']  # currect -> heat flux
    ba = light_source_property['ba']

    Amax = light_source_property['Amax']
    sigma_s = light_source_property['sigma_s']

    I = bs + ks * V_DC + ks * As * np.sin(
        2 * np.pi * f_heating * t_array[:, np.newaxis])  # t: time, r: distance from center

    q = Amax * (ka * I + ba) / np.pi * (sigma_s / (r_array[np.newaxis, :] ** 2 + sigma_s ** 2))
    q[:, N_Rs:] = 0.0

    return q


def radial_1D_explicit(sample_information, vacuum_chamber_setting, solar_simulator_settings,
                       light_source_property, numerical_simulation_setting):

    R = sample_information['R']
    Nr = numerical_simulation_setting['Nr']
    t_z = sample_information['t_z']

    Fo_criteria = numerical_simulation_setting['Fo_criteria']

    dr = R / Nr
    dz = t_z

    rho = sample_information['rho']
    cp_const = sample_information['cp_const']  # Cp = cp_const + cp_c1*T+ cp_c2*T**2+cp_c3*T**3, T unit in K
    cp_c1 = sample_information['cp_c1']
    cp_c2 = sample_information['cp_c2']
    cp_c3 = sample_information['cp_c3']
    alpha_r = sample_information['alpha_r']
    alpha_z = sample_information['alpha_z']

    T_initial = sample_information['T_initial']  # unit in K
    # k = alpha*rho*cp

    f_heating = solar_simulator_settings['f_heating']  # periodic heating frequency
    N_cycle = numerical_simulation_setting['N_cycle']

    T_sur1 = vacuum_chamber_setting['T_sur1']  # unit in K
    T_sur2 = vacuum_chamber_setting['T_sur2']  # unit in K

    frequency_analysis_method = numerical_simulation_setting['frequency_analysis_method']

    emissivity = sample_information['emissivity']  # assumed to be constant
    absorptivity = sample_information['absorptivity']  # assumed to be constant
    sigma_sb = 5.6703744 * 10 ** (-8)  # stefan-Boltzmann constant

    dt = min(Fo_criteria * (dr ** 2) / (alpha_r),
             1 / f_heating / 15)  # assume 15 samples per period, Fo_criteria default = 1/3
    t_total = 1 / f_heating * N_cycle  # total simulation time

    Nt = int(t_total / dt)  # total number of simulation time step
    time_simulation = dt * np.arange(Nt)
    r = np.arange(Nr)
    rm = r * dr

    Fo_r = alpha_r * dt / dr ** 2

    # Rs = vacuum_chamber_setting['Rs']  # the location where the solar light shines on the sample
    # N_Rs = int(Rs / R * Nr)

    N_Rs = int(vacuum_chamber_setting['N_Rs']) # max pixels that solar light shine upon

    T = T_initial * np.ones((Nt, Nr))

    vectorize = numerical_simulation_setting['vectorize']

    if vectorize:

        q_solar = light_source_intensity_vecterize(np.arange(Nr) * dr, np.arange(Nt) * dt, N_Rs,
                                                   solar_simulator_settings, light_source_property)
        T_temp = np.zeros(Nr)
        N_steady_count = 0
        time_index = Nt - 1
        N_one_cycle = int(Nt/N_cycle)
        #A_initial = 100

        for p in range(Nt - 1):  # p indicate time step

            cp = cp_const + cp_c1 * T[p, 0] + cp_c2 * T[p, 0] ** 2 + cp_c3 * T[p, 0] ** 3
            T[p + 1, 0] = T[p, 0] * (1 - 4 * Fo_r) + 4 * Fo_r * T[p, 1] - 2 * sigma_sb * emissivity * dt / (rho * cp * dz) * T[p, 0] ** 4 + \
                          dt / (rho * cp * dz) * (absorptivity * sigma_sb * T_sur1 ** 4 + absorptivity * sigma_sb * T_sur2 ** 4 + absorptivity *q_solar[p, 0])

            T[p + 1, 1:-1] = T[p, 1:-1] + Fo_r * (rm[1:-1] - dr / 2) / rm[1:-1] * (T[p, 0:-2] - T[p, 1:-1]) + Fo_r * (rm[1:-1] + dr / 2) / rm[1:-1] * (T[p, 2:] - T[p, 1:-1]) \
                             + dt / (rho * (cp_const + cp_c1 * T[p, 1:-1] + cp_c2 * T[p, 1:-1] ** 2 + cp_c3 * T[p, 1:-1] ** 3) * dz) * (absorptivity * sigma_sb * T_sur1 ** 4 \
                             + absorptivity * sigma_sb * T_sur2 ** 4 + absorptivity * q_solar[p,1:-1] - 2 * emissivity * sigma_sb * T[p,1:-1] ** 4)

            cp = cp_const + cp_c1 * T[p, -1] + cp_c2 * T[p, -1] ** 2 + cp_c3 * T[p, -1] ** 3
            T[p + 1, -1] = T[p, -1] * (1 - 2 * Fo_r) - 2 * dt * emissivity * sigma_sb / (rho * cp) * (1 / dz + 1 / dr) * T[p, -1] ** 4 + 2 * Fo_r * T[p, -2] + \
                           dt / (rho * cp * dz) * (absorptivity * sigma_sb * T_sur1 ** 4 + absorptivity * sigma_sb * T_sur2 ** 4 + absorptivity *q_solar[p, -1]) \
                           + 2 * dt / (rho * cp * dr) * absorptivity * sigma_sb * T_sur2 ** 4

            if (p>0) & (p % N_one_cycle ==0) & (frequency_analysis_method != 'fft'):
                A_max = np.max(T[p-N_one_cycle:p,:],axis = 0)
                A_min = np.min(T[p-N_one_cycle:p,:],axis = 0)
                if np.max(np.abs((T_temp[:] - T[p,:])/(A_max-A_min)))<2e-2:
                    N_steady_count += 1
                    if N_steady_count == 2: # only need 2 periods to calculate amplitude and phase
                        time_index = p
                        break
                T_temp = T[p, :]
    else:
        q_solar = np.zeros((Nt, Nr))  # heat flux of the solar simulator
        time_index = Nt - 1

        for i in range(N_Rs):  # truncated solar light
            for j in range(Nt):
                q_solar[j, i] = light_source_intensity(i * dr, j * dt, solar_simulator_settings, light_source_property)

        for p in range(Nt - 1):  # p indicate time step
            for m in range(Nr):  # m indicate node along radial direction
                rm = dr * m
                cp = cp_const + cp_c1 * T[p, m] + cp_c2 * T[p, m] ** 2 + cp_c3 * T[p, m] ** 3

                if m == 0:
                    # consider V1
                    T[p + 1, m] = T[p, m] + 4 * Fo_r * (T[p, m + 1] - T[p, m]) + \
                                  dt / (rho * cp * dz) * (
                                              absorptivity * sigma_sb * T_sur1 ** 4 + absorptivity * q_solar[
                                          p, m] - emissivity * sigma_sb * T[p, m] ** 4) + \
                                  dt / (rho * cp * dz) * (
                                              absorptivity * sigma_sb * T_sur2 ** 4 - emissivity * sigma_sb * T[
                                          p, m] ** 4)

                elif m == Nr - 1:
                    # consider V2
                    T[p + 1, m] = T[p, m] + 2 * (rm - dr / 2) / rm * Fo_r * (T[p, m - 1] - T[p, m]) + \
                                  dt / (rho * cp * dz) * (
                                              absorptivity * sigma_sb * T_sur1 ** 4 + absorptivity * q_solar[
                                          p, m] - emissivity * sigma_sb * T[p, m] ** 4) \
                                  + 2 * dt / (rho * cp * dr) * (
                                              absorptivity * sigma_sb * T_sur2 ** 4 - emissivity * sigma_sb * T[
                                          p, m] ** 4) \
                                  + dt / (rho * cp * dz) * (
                                              absorptivity * sigma_sb * T_sur2 ** 4 - emissivity * sigma_sb * T[
                                          p, m] ** 4)

                elif m >= 1 and m != Nr - 1:
                    # Consider E1
                    T[p + 1, m] = T[p, m,] + Fo_r * (rm - dr / 2) / rm * (T[p, m - 1] - T[p, m]) + Fo_r * (
                                rm + dr / 2) / rm * (T[p, m + 1] - T[p, m]) \
                                  + dt / (rho * cp * dz) * (
                                              absorptivity * sigma_sb * T_sur1 ** 4 + absorptivity * q_solar[
                                          p, m] - emissivity * sigma_sb * T[p, m] ** 4) \
                                  + dt / (rho * cp * dz) * (
                                              absorptivity * sigma_sb * T_sur2 ** 4 - emissivity * sigma_sb * T[
                                          p, m] ** 4)

    print('alpha_r = {:.2E}, sigma_s = {:.2E}, f_heating = {}, dt = {:.2E}, Nr = {}, Nt = {}, Fo_r = {:.2E}'.format(alpha_r,light_source_property['sigma_s'],f_heating, dt, Nr, Nt,Fo_r))

    return T[:time_index], time_simulation[:time_index], r,N_one_cycle


def simulation_result_amplitude_phase_extraction(df_temperature, df_amplitude_phase_measurement, sample_information,
                                                 vacuum_chamber_setting, solar_simulator_settings,
                                                 light_source_property, numerical_simulation_setting):
    #s_time = time.time()
    R = sample_information['R']
    Nr = numerical_simulation_setting['Nr']
    dr = R / Nr
    N_inner = int(df_amplitude_phase_measurement['r'].min())
    N_outer = int(df_amplitude_phase_measurement['r'].max())

    T_average = np.sum(
        [2 * np.pi * dr * m_ * dr * np.mean(df_temperature.iloc[:, m_]) for m_ in np.arange(N_inner, N_outer, 1)]) / (
                        ((dr * N_outer) ** 2 - (dr * N_inner) ** 2) * np.pi)
    sample_information['T_initial'] = T_average

    T_, time_T_, r_,N_one_cycle = radial_1D_explicit(sample_information, vacuum_chamber_setting, solar_simulator_settings,
                                         light_source_property, numerical_simulation_setting)
    #s1_time = time.time()

    f_heating = solar_simulator_settings['f_heating']
    gap = numerical_simulation_setting['gap']

    # I want max 400 samples per period
    N_skip_time = int(N_one_cycle / 400)

    df_temperature_simulation = pd.DataFrame(data=T_[-2*N_one_cycle::N_skip_time,:])  # return a dataframe containing radial averaged temperature and relative time
    df_temperature_simulation['reltime'] = time_T_[-2*N_one_cycle::N_skip_time]

    phase_diff_simulation = []
    amplitude_ratio_simulation = []

    for i in range(len(df_amplitude_phase_measurement)):
        L, phase_diff, amp_ratio = amp_phase_one_pair(
            [df_amplitude_phase_measurement.iloc[i, :]['r_ref'], df_amplitude_phase_measurement.iloc[i, :]['r']],
            df_temperature_simulation, f_heating, gap,numerical_simulation_setting['frequency_analysis_method'])
        amplitude_ratio_simulation.append(amp_ratio)
        phase_diff_simulation.append(phase_diff)

    #s2_time = time.time()
    df_amp_phase_simulated = pd.DataFrame(
        data={'amp_ratio': amplitude_ratio_simulation, 'phase_diff': phase_diff_simulation})

    df_amp_phase_simulated['r'] = df_amplitude_phase_measurement['r']
    df_amp_phase_simulated['r_ref'] = df_amplitude_phase_measurement['r_ref']

    #print("Model evaluation time {:.2f}s and amplitude and phase processing time {:.2f}s".format(s1_time-s_time,s2_time-s1_time))
    return df_amp_phase_simulated,df_temperature_simulation


def residual(params, df_temperature,df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
             solar_simulator_settings, light_source_property, numerical_simulation_setting):
    sample_information['alpha_r'] = params[0]

    df_amp_phase_simulated,df_temperature_simulation= simulation_result_amplitude_phase_extraction(df_temperature,df_amplitude_phase_measurement,
                                                                          sample_information, vacuum_chamber_setting,
                                                                          solar_simulator_settings,
                                                                          light_source_property,
                                                                          numerical_simulation_setting)

    phase_relative_error = [abs((measure - calculate) / measure) for measure, calculate in
                            zip(df_amplitude_phase_measurement['phase_diff'], df_amp_phase_simulated['phase_diff'])]
    amplitude_relative_error = [abs((measure - calculate) / measure) for measure, calculate in
                                zip(df_amplitude_phase_measurement['amp_ratio'], df_amp_phase_simulated['amp_ratio'])]

    return np.sum(amplitude_relative_error) + np.sum(phase_relative_error)

def residual_alpha_solar(params,df_temperature, df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
             solar_simulator_settings, light_source_property, numerical_simulation_setting):

    sample_information['alpha_r'] = params[0]
    light_source_property['sigma_s'] = params[1]

    df_amp_phase_simulated,df_temperature_simulation= simulation_result_amplitude_phase_extraction(df_temperature, df_amplitude_phase_measurement,
                                                                          sample_information, vacuum_chamber_setting,
                                                                          solar_simulator_settings,
                                                                          light_source_property,
                                                                          numerical_simulation_setting)

    phase_relative_error = [abs((measure - calculate) / measure) for measure, calculate in
                            zip(df_amplitude_phase_measurement['phase_diff'], df_amp_phase_simulated['phase_diff'])]
    amplitude_relative_error = [abs((measure - calculate) / measure) for measure, calculate in
                                zip(df_amplitude_phase_measurement['amp_ratio'], df_amp_phase_simulated['amp_ratio'])]

    return np.sum(amplitude_relative_error) + np.sum(phase_relative_error)


def residual_solar(params, df_temperature, df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
             solar_simulator_settings, light_source_property, numerical_simulation_setting):

    #sample_information['alpha_r'] = params[0]
    light_source_property['sigma_s'] = params[0]

    df_amp_phase_simulated,df_temperature_simulation= simulation_result_amplitude_phase_extraction(df_temperature, df_amplitude_phase_measurement,
                                                                          sample_information, vacuum_chamber_setting,
                                                                          solar_simulator_settings,
                                                                          light_source_property,
                                                                          numerical_simulation_setting)

    phase_relative_error = [abs((measure - calculate) / measure) for measure, calculate in
                            zip(df_amplitude_phase_measurement['phase_diff'], df_amp_phase_simulated['phase_diff'])]
    amplitude_relative_error = [abs((measure - calculate) / measure) for measure, calculate in
                                zip(df_amplitude_phase_measurement['amp_ratio'], df_amp_phase_simulated['amp_ratio'])]

    return np.sum(amplitude_relative_error) + np.sum(phase_relative_error)

def show_regression_results(param_name, regression_result, df_temperature,df_amplitude_phase_measurement, sample_information,
                            vacuum_chamber_setting, solar_simulator_settings, light_source_property,
                            numerical_simulation_setting):
    if param_name =='alpha':
        sample_information['alpha_r'] = regression_result
        title_text = 'alpha = {:.2E} m2/s, VDC = {} V, TS1 = {:.0f} K'.format(regression_result,solar_simulator_settings['V_DC'],vacuum_chamber_setting['T_sur1'])

    elif param_name == 'sigma_s':
        light_source_property['sigma_s'] = regression_result
        title_text = 'sigma_s = {:.2E}, VDC = {} V, TS1 = {:.0f} K'.format(regression_result, solar_simulator_settings['V_DC'],vacuum_chamber_setting['T_sur1'])

    df_amp_phase_simulated, df_temperature_simulation = simulation_result_amplitude_phase_extraction(df_temperature,
        df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting, solar_simulator_settings,
        light_source_property, numerical_simulation_setting)


    fig = plt.figure(figsize=(15, 5))
    # plt.scatter(df_result_IR_mosfata['r'],df_result_IR_mosfata['amp_ratio'],facecolors='none',edgecolors='k',label = 'Mostafa')
    plt.subplot(131)
    plt.scatter(df_amplitude_phase_measurement['r'], df_amplitude_phase_measurement['amp_ratio'], facecolors='none',
                edgecolors='r', label='measurement results')
    plt.scatter(df_amp_phase_simulated['r'], df_amp_phase_simulated['amp_ratio'], marker='+', label='regression results')

    plt.xlabel('R (m)', fontsize=14, fontweight='bold')
    plt.ylabel('Amplitude Ratio', fontsize=14, fontweight='bold')

    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=12)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=12)
        tick.label.set_fontweight('bold')
    plt.legend(prop={'weight': 'bold', 'size': 12})
    plt.title('{}, f = {} Hz'.format(solar_simulator_settings['rec_name'],solar_simulator_settings['f_heating']), fontsize=11, fontweight='bold')
    # plt.legend()

    plt.subplot(132)
    plt.scatter(df_amplitude_phase_measurement['r'], df_amplitude_phase_measurement['phase_diff'], facecolors='none',
                edgecolors='r', label='measurement results')
    plt.scatter(df_amp_phase_simulated['r'], df_amp_phase_simulated['phase_diff'], marker='+', label='regression results')

    plt.xlabel('R (m)', fontsize=14, fontweight='bold')
    plt.ylabel('Phase difference (rad)', fontsize=14, fontweight='bold')
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=12)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=12)
        tick.label.set_fontweight('bold')
    plt.legend(prop={'weight': 'bold', 'size': 12})

    plt.title(title_text, fontsize=11, fontweight='bold')



    plt.subplot(133)

    df_temperature_ = df_temperature.query('reltime>'+str(min(df_temperature_simulation['reltime']))+' and reltime< '+str(max(df_temperature_simulation['reltime'])))

    N_actual_data_per_period = int(len(df_temperature) / (max(df_temperature['reltime']) / (1 / solar_simulator_settings['f_heating'])))

    N_skip = int(N_actual_data_per_period/20) #only shows 20 data points per cycle maximum

    N_inner = int(df_amplitude_phase_measurement['r'].min())
    N_outer = int(df_amplitude_phase_measurement['r'].max())
    plt.plot(df_temperature_simulation['reltime'], df_temperature_simulation[N_inner], label='simulated R = '+str(N_inner)+' pixel')
    plt.scatter(df_temperature_['reltime'][::N_skip], df_temperature_[N_inner][::N_skip], label='measured R = '+str(N_inner)+' pixel, skip '+str(N_skip))

    plt.plot(df_temperature_simulation['reltime'], df_temperature_simulation[N_outer], label='simulated R = '+str(N_outer)+' pixel')
    plt.scatter(df_temperature_['reltime'][::N_skip], df_temperature_[N_outer][::N_skip], label='measured R = '+str(N_outer)+' pixel, skip '+str(N_skip))

    plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
    plt.ylabel('Temperature (K)', fontsize=14, fontweight='bold')


    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=12)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=12)
        tick.label.set_fontweight('bold')
    plt.legend(prop={'weight': 'bold', 'size': 12})
    R = sample_information['R']
    Nr = numerical_simulation_setting['Nr']
    dr = R/Nr

    T_average = np.sum([2 * np.pi * dr * m_ * dr * np.mean(df_temperature_.iloc[:, m_]) for m_ in np.arange(N_inner, N_outer, 1)]) / (
                            ((dr * N_outer) ** 2 - (dr * N_inner) ** 2) * np.pi)
    plt.title('Tmin:{:.0f}K, Tmax:{:.0f}K, Tmean:{:.0f}K'.format(np.mean(df_temperature_[N_outer]),np.mean(df_temperature_[N_inner]), T_average), fontsize=11, fontweight='bold')


    plt.tight_layout()


    plt.show()

    return fig,T_average


def high_T_Angstrom_execute_one_case(df_exp_condition, data_directory):

    rec_name = df_exp_condition['rec_name']
    path = data_directory + str(rec_name) + "//"

    output_name = rec_name
    num_cores = df_exp_condition['num_cores']

    method = df_exp_condition['average_method']  # indicate Mosfata's code
    # print(method)

    x0 = df_exp_condition['x0']  # in pixels
    y0 = df_exp_condition['y0']  # in pixels
    Rmax = df_exp_condition['Rmax']  # in pixels
    # x0,y0,N_Rmax,pr,path,rec_name,output_name,method,num_cores
    pr = df_exp_condition['pr']
    # df_temperature = radial_temperature_average_disk_sample_several_ranges(x0,y0,Rmax,[[0,np.pi/3],[2*np.pi/3,np.pi],[4*np.pi/3,5*np.pi/3],[5*np.pi/3,2*np.pi]],pr,path,rec_name,output_name,method,num_cores)

    # After obtaining temperature profile, next we obtain amplitude and phase
    f_heating = df_exp_condition['f_heating']
    # 1cm ->35
    R0 = df_exp_condition['R0']
    gap = df_exp_condition['gap']
    # Rmax = 125
    R_analysis = df_exp_condition['R_analysis']
    exp_amp_phase_extraction_method = df_exp_condition['exp_amp_phase_extraction_method']
    # df_amplitude_phase_measurement = batch_process_horizontal_lines(df_temperature,f_heating,R0,gap,R_analysis,'fft')

    sum_std, fig_symmetry = check_angular_uniformity(x0, y0, Rmax, pr, path, rec_name, output_name, method, num_cores,
                                                     f_heating, R0, gap, R_analysis,
                                                     exp_amp_phase_extraction_method)
    # print(sum_std)

    bb = df_exp_condition['anguler_range']
    bb = bb[1:-1]
    index = None
    angle_range = []
    while (index != -1):
        index = bb.find("],[")
        element = bb[:index]
        d = element.find(",")
        element_after_comma = element[d + 1:]
        element_before_comma = element[element.find("[") + 1:d]
        # print('Before = {} and after = {}'.format(element_before_comma,element_after_comma))
        bb = bb[index + 2:]
        angle_range.append([int(element_before_comma), int(element_after_comma)])

    df_temperature = radial_temperature_average_disk_sample_several_ranges(x0, y0, Rmax, angle_range, pr, path,
                                                                           rec_name, output_name, method, num_cores)

    df_amplitude_phase_measurement = batch_process_horizontal_lines(df_temperature, f_heating, R0, gap, R_analysis,
                                                                    exp_amp_phase_extraction_method)

    sample_information = {'R': df_exp_condition['R'], 't_z': df_exp_condition['t_z'], 'rho': df_exp_condition['rho'],
                          'cp_const': df_exp_condition['cp_const'], 'cp_c1':
                              df_exp_condition['cp_c1'], 'cp_c2': df_exp_condition['cp_c2'],
                          'cp_c3': df_exp_condition['cp_c3'], 'alpha_r': df_exp_condition['alpha_r'],
                          'alpha_z': df_exp_condition['alpha_z'], 'T_initial': df_exp_condition['T_initial'],
                          'emissivity': df_exp_condition['emissivity'],
                          'absorptivity': df_exp_condition['absorptivity']}
    # sample_information
    vacuum_chamber_setting = {'N_Rs': int(df_exp_condition['N_Rs']), 'R0': int(df_exp_condition['R0']),
                              'T_sur1': float(df_exp_condition['T_sur1']), 'T_sur2': float(df_exp_condition['T_sur2'])}
    # vacuum_chamber_setting

    numerical_simulation_setting = {'Nz': int(df_exp_condition['Nz']), 'Nr': int(df_exp_condition['Nr']),
                                    'equal_grid': df_exp_condition['equal_grid'],
                                    'N_cycle': int(df_exp_condition['N_cycle']),
                                    'vectorize': df_exp_condition['vectorize'],
                                    'Fo_criteria': float(df_exp_condition['Fo_criteria']),
                                    'frequency_analysis_method': df_exp_condition['frequency_analysis_method'],
                                    'gap': int(df_exp_condition['gap'])}
    # numerical_simulation_setting

    solar_simulator_settings = {'f_heating': float(df_exp_condition['f_heating']),
                                'V_amplitude': float(df_exp_condition['V_amplitude']),
                                'V_DC': float(df_exp_condition['V_DC']), 'rec_name': df_exp_condition['rec_name']}
    # solar_simulator_settings
    light_source_property = {'ks': float(df_exp_condition['ks']), 'bs': float(df_exp_condition['bs']),
                             'ka': float(df_exp_condition['ka']),
                             'ba': float(df_exp_condition['ba']), 'Amax': float(df_exp_condition['Amax']),
                             'sigma_s': float(df_exp_condition['sigma_s'])}
    # light_source_property

    regression_result = None

    if df_exp_condition['regression_result_type'] == 'alpha':
        res = minimize(residual, x0=float(df_exp_condition['p_initial']), args=(
        df_temperature, df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
        solar_simulator_settings, light_source_property, numerical_simulation_setting), method='nelder-mead', tol=2e-6)

        fig_regression,T_average = show_regression_results('alpha', res['final_simplex'][0][0][0], df_temperature,
                                                 df_amplitude_phase_measurement, sample_information,
                                                 vacuum_chamber_setting, solar_simulator_settings,
                                                 light_source_property,
                                                 numerical_simulation_setting)
        regression_result = res['final_simplex'][0][0][0]

    elif df_exp_condition['regression_result_type'] == 'sigma_s':
        res = minimize(residual_solar, x0=float(df_exp_condition['p_initial']), args=(
        df_temperature, df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting, solar_simulator_settings,
        light_source_property, numerical_simulation_setting), method='nelder-mead', tol=2e-6)
        fig_regression, T_average= show_regression_results('sigma_s', res['final_simplex'][0][0][0], df_temperature,
                                                 df_amplitude_phase_measurement, sample_information,
                                                 vacuum_chamber_setting, solar_simulator_settings,
                                                 light_source_property,
                                                 numerical_simulation_setting)

        regression_result = res['final_simplex'][0][0][0]

    return regression_result, fig_symmetry, fig_regression, sum_std,T_average


def sensitivity_model_output(f_heating, X_input_array,df_temperature, df_r_ref_locations, sample_information, vacuum_chamber_setting, numerical_simulation_setting,
                             solar_simulator_settings, light_source_property):
    # X_input array is 2D array produced by python salib
    # X_2eN5 =saltelli.sample(f_2eN5_problem,500,calc_second_order=False, seed=42)
    # df_r_ref_locations just indicate how to calculate amplitude and phase, where is reference line

    alpha_r = X_input_array[0]
    Amax = X_input_array[1]
    sigma_s = X_input_array[2]
    emissivity = X_input_array[3]
    T_sur1 = X_input_array[4]
    T_sur2 = X_input_array[5]
    rho = X_input_array[6]
    N_Rs = X_input_array[7]
    t_z = X_input_array[8]
    absorptivity = emissivity

    solar_simulator_settings['f_heating'] = f_heating
    sample_information['alpha_r'] = alpha_r
    sample_information['emissivity'] = emissivity
    sample_information['absorptivity'] = absorptivity
    sample_information['t_z'] = t_z
    sample_information['rho'] = rho

    vacuum_chamber_setting['N_Rs'] = N_Rs
    vacuum_chamber_setting['T_sur1'] = T_sur1
    vacuum_chamber_setting['T_sur2'] = T_sur2
    light_source_property['Amax'] = Amax
    light_source_property['sigma_s'] = sigma_s


    df_amp_phase_simulated, df_temperature_simulation = simulation_result_amplitude_phase_extraction(df_temperature,df_r_ref_locations,
                                                                                                     sample_information,
                                                                                                     vacuum_chamber_setting,
                                                                                                     solar_simulator_settings,
                                                                                                     light_source_property,
                                                                                                     numerical_simulation_setting)

    df_amp_phase_simulated['f_heating'] = np.array([f_heating for i in range(len(df_amp_phase_simulated))])

    df_amp_phase_simulated['alpha'] = np.array([alpha_r for i in range(len(df_amp_phase_simulated))])

    return df_amp_phase_simulated


def sensitivity_model_parallel(X_dump_file_name, f_heating_list, num_cores, df_temperature, df_r_ref_locations, sample_information, vacuum_chamber_setting, numerical_simulation_setting,
                             solar_simulator_settings, light_source_property):
    s_time = time.time()
    X_input_arrays = pickle.load(open(X_dump_file_name, 'rb')) # obtain pre-defined simulation conditions

    joblib_output = Parallel(n_jobs=num_cores, verbose=0)(
        delayed(sensitivity_model_output)(f_heating, X_input_array,df_temperature, df_r_ref_locations, sample_information, vacuum_chamber_setting, numerical_simulation_setting,
                             solar_simulator_settings, light_source_property) for X_input_array in tqdm(X_input_arrays) for f_heating
        in f_heating_list)

    pickle.dump(joblib_output, open("sensitivity_results_" + X_dump_file_name, "wb"))
    e_time = time.time()
    print(e_time - s_time)


def show_sensitivity_results_sobol(sobol_problem, parallel_results, f_heating):
    amp_ratio_results = np.array([np.array(parallel_result['amp_ratio']) for parallel_result in parallel_results])
    phase_diff_results = np.array([np.array(parallel_result['phase_diff']) for parallel_result in parallel_results])

    Si_amp_radius = np.array(
        [sobol.analyze(sobol_problem, amp_ratio_results[:, i], calc_second_order=False, print_to_console=False)['S1']
         for i in range(amp_ratio_results.shape[1])])
    # Just pay close attention that calc_second_order=False must be consistent with how X is defined!

    Si_phase_radius = np.array(
        [sobol.analyze(sobol_problem, phase_diff_results[:, i], calc_second_order=False, print_to_console=False)['S1']
         for i in range(phase_diff_results.shape[1])])

    plt.figure(figsize=(14, 6))

    plt.subplot(121)
    for i, name in enumerate(sobol_problem['names']):
        plt.plot(Si_amp_radius[:, i], label=name)

    plt.xlabel('R (pixel)', fontsize=14, fontweight='bold')
    plt.ylabel('Amp Ratio Sensitivity', fontsize=14, fontweight='bold')

    plt.suptitle('frequency = ' + str(f_heating) + ' Hz', fontsize=14, fontweight='bold')
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=12)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=12)
        tick.label.set_fontweight('bold')
    plt.legend(prop={'weight': 'bold', 'size': 12})

    plt.subplot(122)
    for i, name in enumerate(sobol_problem['names']):
        plt.plot(Si_phase_radius[:, i], label=name)

    plt.xlabel('R (pixel)', fontsize=14, fontweight='bold')
    plt.ylabel('Phase diff Sensitivity', fontsize=14, fontweight='bold')

    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=12)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=12)
        tick.label.set_fontweight('bold')
    plt.legend(prop={'weight': 'bold', 'size': 12})
    plt.show()




class MCMC_sampler:

    def __init__(self, df_temperature,df_amplitude_phase_measurement,
                 sample_information, vacuum_chamber_setting, numerical_simulation_setting,
                 solar_simulator_settings, light_source_property, prior_mu, prior_sigma,
                 transition_sigma, result_name, N_sample):

        self.sample_information = sample_information
        self.vacuum_chamber_setting = vacuum_chamber_setting
        self.numerical_simulation_setting = numerical_simulation_setting
        self.solar_simulator_settings = solar_simulator_settings
        self.light_source_property = light_source_property

        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.transition_sigma = transition_sigma
        self.df_temperature = df_temperature
        self.result_name = result_name

        self.df_amplitude_phase_measurement = df_amplitude_phase_measurement

        self.N_sample = N_sample

    def ln_prior(self, params):
        p_ln_alpha = norm.pdf(params['ln_alpha'], loc=self.prior_mu['ln_alpha'],
                              scale=self.prior_sigma['ln_alpha_sigma'])
        # p_ln_h = norm.pdf(params['ln_h'], loc=self.prior_mu['ln_h'], scale=self.prior_sigma['ln_h'])
        p_ln_sigma_dA = norm.pdf(params['ln_sigma_dA'], loc=self.prior_mu['ln_sigma_dA'],
                                 scale=self.prior_sigma['ln_sigma_dA_sigma'])
        p_ln_sigma_dP = norm.pdf(params['ln_sigma_dP'], loc=self.prior_mu['ln_sigma_dP'],
                                 scale=self.prior_sigma['ln_sigma_dP_sigma'])
        return np.log(p_ln_alpha) + np.log(p_ln_sigma_dA) + np.log(p_ln_sigma_dP)

    def ln_transformation_jacobian(self, params):
        jac_alpha = 1 / (np.exp(params['ln_alpha']))
        jac_sigma_dA = 1 / (np.exp(params['ln_sigma_dA']))
        jac_sigma_dP = 1 / (np.exp(params['ln_sigma_dP']))
        jac_rho = (1 + np.exp(2 * params['z'])) / (4 * np.exp(2 * params['z']))
        return np.log(jac_alpha) + np.log(jac_sigma_dA) + np.log(jac_sigma_dP) + np.log(jac_rho)

    def ln_likelihood(self, params):
        self.sample_information['alpha_r'] = np.exp(params['ln_alpha'])
        df_amp_phase_simulated, df_temperature_simulation = simulation_result_amplitude_phase_extraction(self.df_temperature,
            self.df_amplitude_phase_measurement, self.sample_information, self.vacuum_chamber_setting,
            self.solar_simulator_settings, self.light_source_property, self.numerical_simulation_setting)
        mean_measured = np.array(
            [self.df_amplitude_phase_measurement['amp_ratio'], self.df_amplitude_phase_measurement['phase_diff']])
        mean_theoretical = np.array([df_amp_phase_simulated['amp_ratio'], df_amp_phase_simulated['phase_diff']])

        sigma_dA = np.exp(params['ln_sigma_dA'])
        sigma_dP = np.exp(params['ln_sigma_dP'])
        rho_dA_dP = np.tanh(params['z'])

        cov_errs = [[sigma_dA ** 2, sigma_dA * sigma_dP * rho_dA_dP], [sigma_dA * sigma_dP * rho_dA_dP, sigma_dP ** 2]]

        return np.sum([np.log(multivariate_normal.pdf(mean_measured_, mean_theoretical_, cov_errs)) for
                       (mean_measured_, mean_theoretical_) in zip(mean_measured.T, mean_theoretical.T)])

    def rw_proposal(self, params):

        ln_sigma_dA = params['ln_sigma_dA']
        ln_sigma_dP = params['ln_sigma_dP']
        ln_alpha = params['ln_alpha']
        z = params['z']

        ln_alpha, ln_sigma_dA, ln_sigma_dP, z = np.random.normal(
            [ln_alpha, ln_sigma_dA, ln_sigma_dP, z], scale=self.transition_sigma)

        params_star = {'ln_alpha': ln_alpha, 'ln_sigma_dA': ln_sigma_dA, 'ln_sigma_dP': ln_sigma_dP, 'z': z}
        return params_star

    def rw_metropolis(self):

        n_accepted = 0
        n_rejected = 0

        params = self.prior_mu
        accepted = []
        posterior = np.exp(self.ln_likelihood(params) + self.ln_transformation_jacobian(params) + self.ln_prior(params))

        while (n_accepted < self.N_sample):
            params_star = self.rw_proposal(params)
            posterior_star = np.exp(
                self.ln_likelihood(params_star) + self.ln_transformation_jacobian(params_star) + self.ln_prior(
                    params_star))

            accept_ratio = min(1, posterior_star / posterior)
            u = np.random.rand()
            if u <= accept_ratio:  # accept the new state
                params = params_star
                posterior = posterior_star
                n_accepted += 1
                accepted.append(params)

                print('Accepted sample is {} and acceptance rate is {:.3f}.'.format(n_accepted, n_accepted / (
                            n_accepted + n_rejected)))

            else:  # reject the new state
                n_rejected += 1

        accepted = np.array(accepted)

        return accepted