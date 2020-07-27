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

#from scipy.optimize import minimize
from scipy import optimize
import matplotlib.ticker as mtick
from scipy.interpolate import interp1d


def plot_temperature_contour(x0, y0, path, file_name_0, file_name_1, R0, R_analysis):
    fig = plt.figure(figsize=(13, 6))

    df_first_frame = pd.read_csv(path + file_name_0, sep=',', header=None, names=list(np.arange(0, 639)))
    df_mid_frame = pd.read_csv(path + file_name_1, sep=',', header=None, names=list(np.arange(0, 639)))

    df_mid_frame_temperature = df_mid_frame.iloc[5:, :]
    df_first_frame_temperature = df_first_frame.iloc[5:, :]

    plt.subplot(121)

    xmin = x0 - R0 - R_analysis
    xmax = x0 + R0 + R_analysis
    ymin = y0 - R0 - R_analysis
    ymax = y0 + R0 + R_analysis
    Z = np.array(df_first_frame_temperature.iloc[ymin:ymax, xmin:xmax])
    x = np.arange(xmin, xmax, 1)
    y = np.arange(ymin, ymax, 1)
    X, Y = np.meshgrid(x, y)

    x3 = min(R0 + 10, R0 + R_analysis - 20)

    manual_locations = [(x0 - R0 + 15, y0 - R0 + 15), (x0 - R0, y0 - R0), (x0 - x3, y0 - x3)]

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
    ax.set_title("One Frame", fontsize=12, fontweight='bold')

    plt.subplot(122)

    xmin = x0 - R0 - R_analysis
    xmax = x0 + R0 + R_analysis
    ymin = y0 - R0 - R_analysis
    ymax = y0 + R0 + R_analysis
    Z = np.array(df_mid_frame_temperature.iloc[ymin:ymax, xmin:xmax])
    x = np.arange(xmin, xmax, 1)
    y = np.arange(ymin, ymax, 1)
    X, Y = np.meshgrid(x, y)

    x3 = min(R0 + 10, R0 + R_analysis - 20)

    manual_locations = [(x0 - R0 + 15, y0 - R0 + 15), (x0 - R0, y0 - R0), (x0 - x3, y0 - x3)]

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
    ax.set_title("Another Frame", fontsize=12, fontweight='bold')

    plt.show()


def select_data_points_radial_average_MA(x0, y0, Rmax, theta_range, file_name): # extract radial averaged temperature from one csv file
    # This method was originally developed by Mosfata, was adapted by HY to use for amplitude and phase estimation
    df_raw = pd.read_csv(file_name, sep=',', header=None, names=list(np.arange(0, 639)))
    raw_time_string = df_raw.iloc[2, 0][df_raw.iloc[2, 0].find('=') + 2:]  # '073:02:19:35.160000'

    # print(raw_time_string)
    day_info = int(raw_time_string[:raw_time_string.find(":")])
    raw_time_string = raw_time_string[raw_time_string.find(":") + 1:]  # '02:19:35.160000'
    # print(day_info)
    strip_time = datetime.strptime(raw_time_string, '%H:%M:%S.%f')
    # print(raw_time_string)
    time_in_seconds = day_info * 24 * 3600 + strip_time.hour * 3600 + strip_time.minute * 60 + strip_time.second + strip_time.microsecond / 10 ** 6
    # print(time_in_seconds)

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
                             exp_amp_phase_extraction_method,code_directory):
    # we basically break entire disk into 6 regions, with interval of pi/3
    fig = plt.figure(figsize=(18.3, 12))
    colors = ['red', 'black', 'green', 'blue', 'orange', 'magenta']

    df_temperature_list = []
    df_amp_phase_list = []

    plt.subplot(231)

    for j in range(6):
        # note radial_temperature_average_disk_sample automatically checks if a dump file exist
        df_temperature = radial_temperature_average_disk_sample_several_ranges(x0, y0, N_Rmax,
                                                                               [[60 * j, 60 * (j + 1)]],
                                                                               pr, path, rec_name, output_name, method,
                                                                               num_cores,code_directory)
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

    rep_csv_dump_path = code_directory+"temperature cache dump//" + rec_name + '_rep_dump'
    # rec_name
    if (os.path.isfile(rep_csv_dump_path)):  # First check if a dump file exist:
        print('Found previous dump file for representative temperature contour plots:' + rep_csv_dump_path)
        temp_dump = pickle.load(open(rep_csv_dump_path, 'rb'))
        df_first_frame = temp_dump[0]
        df_mid_frame = temp_dump[1]
        frame_num_first = temp_dump[2]
        frame_num_mid = temp_dump[3]

    else:  # If not we obtain the dump file, note the dump file is averaged radial temperature

        file_name_0 = [path + x for x in os.listdir(path)][0]
        n0 = file_name_0.rfind('//')
        n1 = file_name_0.rfind('.csv')
        frame_num_first = file_name_0[n0 + 2:n1]

        df_first_frame = pd.read_csv(file_name_0, sep=',', header=None, names=list(np.arange(0, 639)))

        N_mid = int(len([path + x for x in os.listdir(path)]) / 3)
        file_name_1 = [path + x for x in os.listdir(path)][N_mid]
        n2 = file_name_1.rfind('//')
        n3 = file_name_1.rfind('.csv')
        frame_num_mid = file_name_1[n2 + 2:n3]

        df_mid_frame = pd.read_csv(file_name_1, sep=',', header=None, names=list(np.arange(0, 639)))

        temp_dump = [df_first_frame, df_mid_frame, frame_num_first, frame_num_mid]

        pickle.dump(temp_dump, open(rep_csv_dump_path, "wb"))

    df_mid_frame_temperature = df_mid_frame.iloc[5:, :]
    df_first_frame_temperature = df_first_frame.iloc[5:, :]

    plt.subplot(234)

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
    ax.set_title(frame_num_first, fontsize=12, fontweight='bold')

    plt.subplot(235)

    xmin = x0 - R0 - R_analysis
    xmax = x0 + R0 + R_analysis
    ymin = y0 - R0 - R_analysis
    ymax = y0 + R0 + R_analysis
    Z = np.array(df_mid_frame_temperature.iloc[ymin:ymax, xmin:xmax])
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
    ax.set_title(frame_num_mid, fontsize=12, fontweight='bold')

    plt.subplot(236)
    T_mean_list = np.array(
        [np.array(df_temperature.iloc[:, R0:R0 + R_analysis].mean(axis=0)) for df_temperature in df_temperature_list])

    plt.plot(np.arange(R0, R0 + R_analysis), T_mean_list.mean(axis=0), linewidth=2)
    ax = plt.gca()
    ax.set_xlabel('R (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('T_mean (K)', fontsize=12, fontweight='bold')
    ax.set_title("Tmean vs R", fontsize=12, fontweight='bold')

    #plt.show()

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



def radial_temperature_average_disk_sample_several_ranges(x0, y0, N_Rmax, theta_range_list, pr, path, rec_name,
                                                          output_name, method,
                                                          num_cores,code_directory):  # unit in K
    # path= "C://Users//NTRG lab//Desktop//yuan//"
    # rec_name = "Rec-000011_e63", this is the folder which contains all csv data files
    # note theta_range should be a 2D array [[0,pi/3],[pi/3*2,2pi]]
    df_temperature_list = []
    for theta_range in theta_range_list:
        dump_file_path = code_directory+"temperature cache dump//"+output_name + '_x0_{}_y0_{}_Rmax_{}_method_{}_theta_{}_{}'.format(x0, y0, N_Rmax, method, int(theta_range[0]), int(theta_range[1]))

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
    R0 = vacuum_chamber_setting['R0']

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

    #print(N_cycle)
    #print(f_heating)
    dt = min(Fo_criteria * (dr ** 2) / (alpha_r), 1/f_heating/15)  # assume 15 samples per period, Fo_criteria default = 1/3


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
                if np.max(np.abs((T_temp[:] - T[p,:])/(A_max-A_min)))<1e-2:
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

    print('alpha_r = {:.2E}, sigma_s = {:.2E}, f_heating = {}, dt = {:.2E}, Nr = {}, Nt = {}, Fo_r = {:.2E}, R0 = {}'.format(alpha_r,light_source_property['sigma_s'],f_heating, dt, Nr, Nt,Fo_r,R0))

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



def residual_update(params, df_temperature,df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
             solar_simulator_settings, light_source_property, numerical_simulation_setting):


    error = None

    regression_module = numerical_simulation_setting['regression_module']
    regression_method = numerical_simulation_setting['regression_method']


    if numerical_simulation_setting['regression_result_type'] == 'sigma_s':
        if regression_module == 'lmfit':
            light_source_property['sigma_s'] = params['sigma_s'].value
        elif regression_module == 'scipy.optimize-NM':
            light_source_property['sigma_s'] = params[0]

    elif numerical_simulation_setting['regression_result_type'] == 'alpha_r':
        if regression_module == 'lmfit':
            sample_information['alpha_r'] = params['alpha_r'].value
            #print("params['alpha_r']: "+str(params['alpha_r']))
        elif regression_module == 'scipy.optimize-NM':
            sample_information['alpha_r'] = params[0]

    df_amp_phase_simulated,df_temperature_simulation= simulation_result_amplitude_phase_extraction(df_temperature,df_amplitude_phase_measurement,
                                                                          sample_information, vacuum_chamber_setting,
                                                                          solar_simulator_settings,
                                                                          light_source_property,
                                                                          numerical_simulation_setting)

    phase_relative_error = np.array([abs((measure - calculate) / measure) for measure, calculate in
                            zip(df_amplitude_phase_measurement['phase_diff'], df_amp_phase_simulated['phase_diff'])])
    amplitude_relative_error = np.array([abs((measure - calculate) / measure) for measure, calculate in
                                zip(df_amplitude_phase_measurement['amp_ratio'], df_amp_phase_simulated['amp_ratio'])])



    if regression_module == 'lmfit':

        if regression_method == 'amplitude':
            error = amplitude_relative_error
        elif regression_method == 'phase':
            error = phase_relative_error
        elif regression_method == 'amplitude-phase':
            error = amplitude_relative_error + amplitude_relative_error

        #return error

    elif regression_module == 'scipy.optimize-NM':
        if regression_method == 'amplitude':
            error = np.sum(amplitude_relative_error)
        elif regression_method == 'phase':
            error = np.sum(phase_relative_error)
        elif regression_method == 'amplitude-phase':
            error = np.sum(amplitude_relative_error)+ np.sum(amplitude_relative_error)

    return error



# def residual(params, df_temperature,df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
#              solar_simulator_settings, light_source_property, numerical_simulation_setting):
#     sample_information['alpha_r'] = params[0]
#
#     df_amp_phase_simulated,df_temperature_simulation= simulation_result_amplitude_phase_extraction(df_temperature,df_amplitude_phase_measurement,
#                                                                           sample_information, vacuum_chamber_setting,
#                                                                           solar_simulator_settings,
#                                                                           light_source_property,
#                                                                           numerical_simulation_setting)
#
#     phase_relative_error = [abs((measure - calculate) / measure) for measure, calculate in
#                             zip(df_amplitude_phase_measurement['phase_diff'], df_amp_phase_simulated['phase_diff'])]
#     amplitude_relative_error = [abs((measure - calculate) / measure) for measure, calculate in
#                                 zip(df_amplitude_phase_measurement['amp_ratio'], df_amp_phase_simulated['amp_ratio'])]
#
#     return np.sum(amplitude_relative_error) + np.sum(phase_relative_error)




#
# def residual_alpha_solar(params,df_temperature, df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
#              solar_simulator_settings, light_source_property, numerical_simulation_setting):
#
#     sample_information['alpha_r'] = params[0]
#     light_source_property['sigma_s'] = params[1]
#
#     df_amp_phase_simulated,df_temperature_simulation= simulation_result_amplitude_phase_extraction(df_temperature, df_amplitude_phase_measurement,
#                                                                           sample_information, vacuum_chamber_setting,
#                                                                           solar_simulator_settings,
#                                                                           light_source_property,
#                                                                           numerical_simulation_setting)
#
#     phase_relative_error = [abs((measure - calculate) / measure) for measure, calculate in
#                             zip(df_amplitude_phase_measurement['phase_diff'], df_amp_phase_simulated['phase_diff'])]
#     amplitude_relative_error = [abs((measure - calculate) / measure) for measure, calculate in
#                                 zip(df_amplitude_phase_measurement['amp_ratio'], df_amp_phase_simulated['amp_ratio'])]
#
#     return np.sum(amplitude_relative_error) + np.sum(phase_relative_error)



#
#
# def residual_solar(params, df_temperature, df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
#              solar_simulator_settings, light_source_property, numerical_simulation_setting):
#
#     #sample_information['alpha_r'] = params[0]
#     light_source_property['sigma_s'] = params[0]
#
#     df_amp_phase_simulated,df_temperature_simulation= simulation_result_amplitude_phase_extraction(df_temperature, df_amplitude_phase_measurement,
#                                                                           sample_information, vacuum_chamber_setting,
#                                                                           solar_simulator_settings,
#                                                                           light_source_property,
#                                                                           numerical_simulation_setting)
#
#     phase_relative_error = [abs((measure - calculate) / measure) for measure, calculate in
#                             zip(df_amplitude_phase_measurement['phase_diff'], df_amp_phase_simulated['phase_diff'])]
#     amplitude_relative_error = [abs((measure - calculate) / measure) for measure, calculate in
#                                 zip(df_amplitude_phase_measurement['amp_ratio'], df_amp_phase_simulated['amp_ratio'])]
#
#     return np.sum(amplitude_relative_error) + np.sum(phase_relative_error)


def show_regression_results(param_name, regression_result, df_temperature, df_amplitude_phase_measurement,
                            sample_information,vacuum_chamber_setting, solar_simulator_settings, light_source_property,numerical_simulation_setting):

    if param_name == 'alpha_r':
        sample_information['alpha_r'] = regression_result
        title_text = 'alpha = {:.2E} m2/s, VDC = {} V, TS1 = {:.0f} K'.format(regression_result,
                                                                              solar_simulator_settings['V_DC'],
                                                                              vacuum_chamber_setting['T_sur1'])

    elif param_name == 'sigma_s':
        light_source_property['sigma_s'] = regression_result
        title_text = 'sigma_s = {:.2E}, VDC = {} V, TS1 = {:.0f} K'.format(regression_result,
                                                                           solar_simulator_settings['V_DC'],
                                                                           vacuum_chamber_setting['T_sur1'])

    df_amp_phase_simulated, df_temperature_simulation = simulation_result_amplitude_phase_extraction(df_temperature,
                                                                                                     df_amplitude_phase_measurement,
                                                                                                     sample_information,
                                                                                                     vacuum_chamber_setting,
                                                                                                     solar_simulator_settings,
                                                                                                     light_source_property,
                                                                                                     numerical_simulation_setting)

    phase_relative_error = [abs((measure - calculate) / measure) for measure, calculate in
                            zip(df_amplitude_phase_measurement['phase_diff'], df_amp_phase_simulated['phase_diff'])]
    amplitude_relative_error = [abs((measure - calculate) / measure) for measure, calculate in
                                zip(df_amplitude_phase_measurement['amp_ratio'], df_amp_phase_simulated['amp_ratio'])]

    amp_residual_mean =  np.mean(amplitude_relative_error)
    phase_residual_mean = np.mean(phase_relative_error)
    total_residual_mean = amp_residual_mean + phase_residual_mean

    fig = plt.figure(figsize=(15, 5))
    # plt.scatter(df_result_IR_mosfata['r'],df_result_IR_mosfata['amp_ratio'],facecolors='none',edgecolors='k',label = 'Mostafa')
    plt.subplot(131)
    plt.scatter(df_amplitude_phase_measurement['r'], df_amplitude_phase_measurement['amp_ratio'], facecolors='none',
                edgecolors='r', label='measurement results')
    plt.scatter(df_amp_phase_simulated['r'], df_amp_phase_simulated['amp_ratio'], marker='+',
                label='regression results')

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
    plt.title('{}, f = {} Hz'.format(solar_simulator_settings['rec_name'], solar_simulator_settings['f_heating']),
              fontsize=11, fontweight='bold')
    # plt.legend()

    plt.subplot(132)
    plt.scatter(df_amplitude_phase_measurement['r'], df_amplitude_phase_measurement['phase_diff'], facecolors='none',
                edgecolors='r', label='measurement results')
    plt.scatter(df_amp_phase_simulated['r'], df_amp_phase_simulated['phase_diff'], marker='+',
                label='regression results')

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

    # df_temperature_ = df_temperature.query('reltime>'+str(min(df_temperature_simulation['reltime']))+' and reltime< '+str(max(df_temperature_simulation['reltime'])))
    df_temperature_simulation['reltime'] = df_temperature_simulation['reltime'] - min(
        df_temperature_simulation['reltime'])

    time_interval = max(df_temperature_simulation['reltime']) - min(df_temperature_simulation['reltime'])
    df_temperature_ = df_temperature.query('reltime<' + str(time_interval))
    # print(df_temperature_)
    N_actual_data_per_period = int(
        len(df_temperature_) / (max(df_temperature_['reltime']) / (1 / solar_simulator_settings['f_heating'])))

    # print("Number of data per period is {}".format(N_actual_data_per_period))
    N_skip = max(1, int(N_actual_data_per_period / 15))  # only shows 20 data points per cycle maximum
    # print(N_skip)
    N_inner = int(df_amplitude_phase_measurement['r'].min())
    N_outer = int(df_amplitude_phase_measurement['r'].max())
    plt.plot(df_temperature_simulation['reltime'], df_temperature_simulation[N_inner],
             label='simulated R = ' + str(N_inner) + ' pixel')
    plt.scatter(df_temperature_['reltime'][::N_skip], df_temperature_[N_inner][::N_skip],
                label='measured R = ' + str(N_inner) + ' pixel, skip ' + str(N_skip))

    plt.plot(df_temperature_simulation['reltime'], df_temperature_simulation[N_outer],
             label='simulated R = ' + str(N_outer) + ' pixel')
    plt.scatter(df_temperature_['reltime'][::N_skip], df_temperature_[N_outer][::N_skip],
                label='measured R = ' + str(N_outer) + ' pixel, skip ' + str(N_skip))

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
    dr = R / Nr

    T_average = np.sum(
        [2 * np.pi * dr * m_ * dr * np.mean(df_temperature_.iloc[:, m_]) for m_ in np.arange(N_inner, N_outer, 1)]) / (
                        ((dr * N_outer) ** 2 - (dr * N_inner) ** 2) * np.pi)
    plt.title('Tmin:{:.0f}K, Tmax:{:.0f}K, Tmean:{:.0f}K'.format(np.mean(df_temperature_[N_outer]),
                                                                 np.mean(df_temperature_[N_inner]), T_average),
              fontsize=11, fontweight='bold')

    plt.tight_layout()

   # plt.show()

    return fig, T_average, amp_residual_mean, phase_residual_mean,total_residual_mean


# def high_T_Angstrom_execute_one_case(df_exp_condition,df_sample_solar_simulator, data_directory,diagnostic_figure,df_temperature,df_amplitude_phase_measurement):
#     #this function read a row from an excel spread sheet and execute
#
#     rec_name = df_exp_condition['rec_name']
#
#     path = data_directory + str(rec_name) + "//"
#     output_name = rec_name
#     num_cores = df_exp_condition['num_cores']
#     method = df_exp_condition['average_method']  # indicate Mosfata's code
#     x0 = df_exp_condition['x0']  # in pixels
#     y0 = df_exp_condition['y0']  # in pixels
#     Rmax = df_exp_condition['Rmax']  # in pixels
#     # x0,y0,N_Rmax,pr,path,rec_name,output_name,method,num_cores
#     pr = df_exp_condition['pr']
#     # df_temperature = radial_temperature_average_disk_sample_several_ranges(x0,y0,Rmax,[[0,np.pi/3],[2*np.pi/3,np.pi],[4*np.pi/3,5*np.pi/3],[5*np.pi/3,2*np.pi]],pr,path,rec_name,output_name,method,num_cores)
#     # After obtaining temperature profile, next we obtain amplitude and phase
#     f_heating = df_exp_condition['f_heating']
#     # 1cm ->35
#     R0 = df_exp_condition['R0']
#     gap = df_exp_condition['gap']
#     # Rmax = 125
#     R_analysis = df_exp_condition['R_analysis']
#     exp_amp_phase_extraction_method = df_exp_condition['exp_amp_phase_extraction_method']
#     regression_method = df_exp_condition['regression_method']
#
#
#
#     regression_module = df_exp_condition['regression_module']
#
#     sample_information = {'R': df_exp_condition['R'], 't_z': df_exp_condition['t_z'], 'rho': df_exp_condition['rho'],
#                           'cp_const': df_exp_condition['cp_const'], 'cp_c1':
#                               df_exp_condition['cp_c1'], 'cp_c2': df_exp_condition['cp_c2'],
#                           'cp_c3': df_exp_condition['cp_c3'], 'alpha_r': df_exp_condition['alpha_r'],
#                           'alpha_z': df_exp_condition['alpha_z'], 'T_initial': df_exp_condition['T_initial'],
#                           'emissivity': df_exp_condition['emissivity'],
#                           'absorptivity': df_exp_condition['absorptivity']}
#     # sample_information
#     vacuum_chamber_setting = {'N_Rs': int(df_exp_condition['N_Rs']), 'R0': int(df_exp_condition['R0']),
#                               'T_sur1': float(df_exp_condition['T_sur1']), 'T_sur2': float(df_exp_condition['T_sur2'])}
#     # vacuum_chamber_setting
#
#     numerical_simulation_setting = {'Nz': int(df_exp_condition['Nz']), 'Nr': int(df_exp_condition['Nr']),
#                                     'equal_grid': df_exp_condition['equal_grid'],
#                                     'N_cycle': int(df_exp_condition['N_cycle']),
#                                     'vectorize': df_exp_condition['vectorize'],
#                                     'Fo_criteria': float(df_exp_condition['Fo_criteria']),
#                                     'frequency_analysis_method': df_exp_condition['frequency_analysis_method'],
#                                     'gap': int(df_exp_condition['gap']),
#                                     'regression_module': df_exp_condition['regression_module'],
#                                     'regression_method':df_exp_condition['regression_method'],
#                                     'regression_result_type': df_exp_condition['regression_result_type'],
#                                     'regression_residual_converging_criteria':df_exp_condition['regression_residual_converging_criteria']
#                                     }
#
#     # numerical_simulation_setting
#
#     solar_simulator_settings = {'f_heating': float(df_exp_condition['f_heating']),
#                                 'V_amplitude': float(df_exp_condition['V_amplitude']),
#                                 'V_DC': float(df_exp_condition['V_DC']), 'rec_name': df_exp_condition['rec_name']}
#     # solar_simulator_settings
#     light_source_property = {'ks': float(df_exp_condition['ks']), 'bs': float(df_exp_condition['bs']),
#                              'ka': float(df_exp_condition['ka']),
#                              'ba': float(df_exp_condition['ba']), 'Amax': float(df_exp_condition['Amax']),
#                              'sigma_s': float(df_exp_condition['sigma_s'])}
#     # light_source_property
#
#     regression_result = None
#
#     if regression_module == 'lmfit':
#
#         params = Parameters()
#
#         if df_exp_condition['regression_result_type'] == 'alpha_r':
#             params.add('alpha_r', value=float(df_exp_condition['p_initial']), min = 0.0)
#
#             out = lmfit.minimize(residual_update, params, args=(df_temperature, df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
#                                                                 solar_simulator_settings, light_source_property, numerical_simulation_setting),xtol = df_exp_condition['regression_residual_converging_criteria'])
#
#             regression_result = out.params['alpha_r'].value
#
#         elif df_exp_condition['regression_result_type'] == 'sigma_s':
#             params.add('sigma_s', value=float(df_exp_condition['p_initial']), min = 0.0)
#
#             out = lmfit.minimize(residual_update, params, args=(df_temperature, df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
#                                                                 solar_simulator_settings, light_source_property, numerical_simulation_setting),xtol = df_exp_condition['regression_residual_converging_criteria'])
#
#             regression_result = out.params['sigma_s'].value
#
#
#         #pass
#     elif regression_module =='scipy.optimize-NM':
#
#         res = optimize.minimize(residual_update, x0=float(df_exp_condition['p_initial']), args=(
#             df_temperature, df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
#             solar_simulator_settings, light_source_property, numerical_simulation_setting), method='nelder-mead',
#                        tol=df_exp_condition['regression_residual_converging_criteria'])
#         regression_result = res['final_simplex'][0][0][0]
#
#     fig_regression, T_average, amp_residual_mean, phase_residual_mean, total_residual_mean = show_regression_results(
#         df_exp_condition['regression_result_type'], regression_result, df_temperature,
#         df_amplitude_phase_measurement, sample_information,
#         vacuum_chamber_setting, solar_simulator_settings,
#         light_source_property,
#         numerical_simulation_setting)
#
#
#
#
#     # if df_exp_condition['regression_result_type'] == 'alpha_r':
#     #     res = minimize(residual, x0=float(df_exp_condition['p_initial']), args=(
#     #     df_temperature, df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
#     #     solar_simulator_settings, light_source_property, numerical_simulation_setting), method='nelder-mead', tol=2e-6)
#     #
#     #     fig_regression,T_average, amp_residual_mean, phase_residual_mean,total_residual_mean = show_regression_results('alpha_r', res['final_simplex'][0][0][0], df_temperature,
#     #                                              df_amplitude_phase_measurement, sample_information,
#     #                                              vacuum_chamber_setting, solar_simulator_settings,
#     #                                              light_source_property,
#     #                                              numerical_simulation_setting)
#     #     regression_result = res['final_simplex'][0][0][0]
#     #
#     # elif df_exp_condition['regression_result_type'] == 'sigma_s':
#     #     res = minimize(residual_solar, x0=float(df_exp_condition['p_initial']), args=(
#     #     df_temperature, df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting, solar_simulator_settings,
#     #     light_source_property, numerical_simulation_setting), method='nelder-mead', tol=2e-6)
#     #     fig_regression, T_average, amp_residual_mean, phase_residual_mean,total_residual_mean= show_regression_results('sigma_s', res['final_simplex'][0][0][0], df_temperature,
#     #                                              df_amplitude_phase_measurement, sample_information,
#     #                                              vacuum_chamber_setting, solar_simulator_settings,
#     #                                              light_source_property,
#     #                                              numerical_simulation_setting)
#     #
#     #     regression_result = res['final_simplex'][0][0][0]
#
#     print("recording {} completed.".format(rec_name))
#     return regression_result, diagnostic_figure, fig_regression,T_average,amp_residual_mean, phase_residual_mean,total_residual_mean


def high_T_Angstrom_execute_one_case(df_exp_condition, data_directory, code_directory, diagnostic_figure,
                                     df_temperature, df_amplitude_phase_measurement):
    # this is an excel file that contains sample's basic thermal properties and solar simulator characteristics

    df_sample_cp_rho_alpha_all = pd.read_excel(code_directory + "sample specifications//sample properties.xlsx",
                                               sheet_name="sample properties")
    df_solar_simulator_lorentzian = pd.read_excel(code_directory + "sample specifications//sample properties.xlsx",
                                                  sheet_name="solar simulator Lorentzian")

    df_solar_simulator_VI = pd.read_excel(code_directory + "sample specifications//sample properties.xlsx",
                                          sheet_name="solar simulator VI")


    sample_name = df_exp_condition['sample_name']
    df_sample_cp_rho_alpha = df_sample_cp_rho_alpha_all.query("sample_name=='{}'".format(sample_name))

    # this function read a row from an excel spread sheet and execute

    rec_name = df_exp_condition['rec_name']


    regression_module = df_exp_condition['regression_module']

    sample_information = {'R': df_exp_condition['sample_radius(m)'], 't_z': df_exp_condition['sample_thickness(m)'],
                          'rho': float(df_sample_cp_rho_alpha['rho']),
                          'cp_const': float(df_sample_cp_rho_alpha['cp_const']), 'cp_c1':
                              float(df_sample_cp_rho_alpha['cp_c1']), 'cp_c2': float(df_sample_cp_rho_alpha['cp_c2']),
                          'cp_c3': float(df_sample_cp_rho_alpha['cp_c3']), 'alpha_r': float(df_sample_cp_rho_alpha['alpha_r']),
                          'alpha_z': float(df_sample_cp_rho_alpha['alpha_z']), 'T_initial': None,
                          'emissivity': df_exp_condition['emissivity'],
                          'absorptivity': df_exp_condition['absorptivity']}
    # sample_information
    vacuum_chamber_setting = {'N_Rs': int(df_exp_condition['N_Rs']), 'R0': int(df_exp_condition['R0']),
                              'T_sur1': float(df_exp_condition['T_sur1']), 'T_sur2': float(df_exp_condition['T_sur2'])}
    # vacuum_chamber_setting

    numerical_simulation_setting = {'Nz': int(df_exp_condition['Nz']), 'Nr': int(df_exp_condition['Nr']),
                                    'equal_grid': df_exp_condition['equal_grid'],
                                    'N_cycle': int(df_exp_condition['N_cycle']),
                                    'vectorize': True,
                                    'Fo_criteria': float(df_exp_condition['Fo_criteria']),
                                    'frequency_analysis_method': df_exp_condition['frequency_analysis_method'],
                                    'gap': int(df_exp_condition['gap']),
                                    'regression_module': df_exp_condition['regression_module'],
                                    'regression_method': df_exp_condition['regression_method'],
                                    'regression_result_type': df_exp_condition['regression_result_type'],
                                    'regression_residual_converging_criteria': df_exp_condition[
                                        'regression_residual_converging_criteria']
                                    }
    # the code is executed using vectorized approach by default

    # numerical_simulation_setting
    focal_shift = df_exp_condition['focal_shift(cm)']

    locations_relative_focal_plane = df_solar_simulator_lorentzian['Distance from focal plane(cm)']
    Amax_relative_focal_plane = df_solar_simulator_lorentzian['Amax']
    sigma_relative_focal_plane = df_solar_simulator_lorentzian['sigma']

    f_Amax = interp1d(locations_relative_focal_plane, Amax_relative_focal_plane, kind='cubic')
    f_sigma = interp1d(locations_relative_focal_plane, sigma_relative_focal_plane, kind='cubic')

    solar_simulator_settings = {'f_heating': float(df_exp_condition['f_heating']),
                                'V_amplitude': float(df_exp_condition['V_amplitude']),
                                'V_DC': float(df_exp_condition['V_DC']), 'rec_name': rec_name}
    # solar_simulator_settings
    light_source_property = {'ks': float(df_solar_simulator_VI['ks']), 'bs': float(df_solar_simulator_VI['bs']),
                             'ka': float(df_solar_simulator_VI['ka']),
                             'ba': float(df_solar_simulator_VI['ba']), 'Amax': f_Amax(focal_shift),
                             'sigma_s': f_sigma(focal_shift)}
    # light_source_property

    regression_result = None

    if regression_module == 'lmfit':

        params = Parameters()

        if df_exp_condition['regression_result_type'] == 'alpha_r':
            params.add('alpha_r', value=float(df_exp_condition['p_initial']), min=0.0)

            out = lmfit.minimize(residual_update, params, args=(
            df_temperature, df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
            solar_simulator_settings, light_source_property, numerical_simulation_setting),
                                 xtol=df_exp_condition['regression_residual_converging_criteria'])

            regression_result = out.params['alpha_r'].value

        elif df_exp_condition['regression_result_type'] == 'sigma_s':
            params.add('sigma_s', value=float(df_exp_condition['p_initial']), min=0.0)

            out = lmfit.minimize(residual_update, params, args=(
            df_temperature, df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
            solar_simulator_settings, light_source_property, numerical_simulation_setting),
                                 xtol=df_exp_condition['regression_residual_converging_criteria'])

            regression_result = out.params['sigma_s'].value

    elif regression_module == 'scipy.optimize-NM':

        res = optimize.minimize(residual_update, x0=float(df_exp_condition['p_initial']), args=(
            df_temperature, df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
            solar_simulator_settings, light_source_property, numerical_simulation_setting), method='nelder-mead',
                                tol=df_exp_condition['regression_residual_converging_criteria'])
        regression_result = res['final_simplex'][0][0][0]

    fig_regression, T_average, amp_residual_mean, phase_residual_mean, total_residual_mean = show_regression_results(
        df_exp_condition['regression_result_type'], regression_result, df_temperature,
        df_amplitude_phase_measurement, sample_information,
        vacuum_chamber_setting, solar_simulator_settings,
        light_source_property,
        numerical_simulation_setting)

    print("recording {} completed.".format(rec_name))
    return regression_result, diagnostic_figure, fig_regression, T_average, amp_residual_mean, phase_residual_mean, total_residual_mean


def parallel_regression_batch_experimental_results(df_exp_condition_spreadsheet_filename, data_directory, num_cores,code_directory):
    df_exp_condition_spreadsheet = pd.read_excel(code_directory+"batch process information//" + df_exp_condition_spreadsheet_filename)
    # df_exp_condition_spreadsheet = df_exp_condition_spreadsheet[:5]
    parallel_result_summary
    diagnostic_figure_list = []
    df_temperature_list = []
    df_amplitude_phase_measurement_list = []

    for i in range(len(df_exp_condition_spreadsheet)):

        df_exp_condition = df_exp_condition_spreadsheet.iloc[i, :]

        rec_name = df_exp_condition['rec_name']
        path = data_directory + str(rec_name) + "//"

        output_name = rec_name
        # num_cores = df_exp_condition['num_cores']

        method = "MA"  # default uses Mosfata's code
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

        sum_std, diagnostic_figure = check_angular_uniformity(x0, y0, Rmax, pr, path, rec_name, output_name, method,
                                                              num_cores,
                                                              f_heating, R0, gap, R_analysis,
                                                              exp_amp_phase_extraction_method,code_directory)

        regression_method = df_exp_condition['regression_method']
        regression_module = df_exp_condition['regression_module'] # lmfit and scipy.optimize, these modules requires different return from residual function

        diagnostic_figure_list.append(diagnostic_figure)

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
                                                                               rec_name, output_name, method, num_cores,code_directory)

        df_amplitude_phase_measurement = batch_process_horizontal_lines(df_temperature, f_heating, R0, gap, R_analysis,
                                                                        exp_amp_phase_extraction_method)
        df_temperature_list.append(df_temperature)
        df_amplitude_phase_measurement_list.append(df_amplitude_phase_measurement)

    # regression_result, diagnostic_figure, regression_figure, sum_std,T_average = high_T_Angstrom_execute_one_case(df_exp_condition,data_directory)
    joblib_output = Parallel(n_jobs=num_cores, verbose=0)(
        delayed(high_T_Angstrom_execute_one_case)(df_exp_condition_spreadsheet.iloc[i, :], data_directory,code_directory,
                                                  diagnostic_figure_list[i], df_temperature_list[i],
                                                  df_amplitude_phase_measurement_list[i]) for i in
        tqdm(range(len(df_exp_condition_spreadsheet))))
    # joblib_output
    pickle.dump(joblib_output,open(code_directory+"result cache dump//regression_results_" + df_exp_condition_spreadsheet_filename, "wb"))



def parallel_result_summary(joblib_output,df_exp_condition_spreadsheet_filename,code_directory):
    regression_params = [joblib_output_[0] for joblib_output_ in joblib_output]
    T_average_list = [joblib_output_[3] for joblib_output_ in joblib_output]
    amp_ratio_residual_list = [joblib_output_[4] for joblib_output_ in joblib_output]
    phase_diff_residual_list = [joblib_output_[5] for joblib_output_ in joblib_output]
    amp_phase_residual_list = [joblib_output_[6] for joblib_output_ in joblib_output]

    df_exp_condition = pd.read_excel(code_directory+"batch process information//"+df_exp_condition_spreadsheet_filename)

    sigma_s_list = []
    alpha_list = []

    for i,regression_type in enumerate(df_exp_condition['regression_result_type']):
        if regression_type == 'sigma_s':
            sigma_s_list.append(joblib_output[i][0])
            alpha_list.append(df_exp_condition['alpha_r'][i])
        elif regression_type == 'alpha_r':
            sigma_s_list.append(df_exp_condition['sigma_s'][i])
            alpha_list.append(joblib_output[i][0])

    df_results_all = pd.DataFrame({'rec_name':df_exp_condition['rec_name'],'f_heating':df_exp_condition['f_heating'],'VDC':df_exp_condition['V_DC'],
                                   'sigma_s':sigma_s_list,'T_average':T_average_list,'R0':df_exp_condition['R0'],'alpha_r':alpha_list,
                                   'regression_parameter':df_exp_condition['regression_result_type'],'T_sur1':df_exp_condition['T_sur1'],'emissivity':df_exp_condition['emissivity'],
                                   'amp_res':amp_ratio_residual_list,'phase_res':phase_diff_residual_list,'amp_phase_res':amp_phase_residual_list})
    return df_results_all


def display_high_dimensional_regression_results(x_name, y_name, row_name, column_name, series_name, df_results_all, ylim):

    column_items = np.unique(df_results_all[column_name])
    row_items = np.unique(df_results_all[row_name])
    series_items= np.unique(df_results_all[series_name])
    #reg_parameter_name = np.unique(df_results_all['regression_parameter'])[0]
    f, axes = plt.subplots(len(row_items), len(column_items),
                           figsize=(int(len(column_items) * 4), int(len(row_items) * 3)))
    for i, row in enumerate(row_items):
        for j, column in enumerate(column_items):
            df_results_all_ = df_results_all.query("{}=={} and {} == {}".format(row_name, row, column_name, column))
            # VDC_list = np.unique(df_results_all_['VDC'])
            for series in series_items:
                df_ = df_results_all_.query("{}=={}".format(series_name, series))
                axes[i, j].scatter(df_[x_name], df_[y_name], label="{} = {:.1E}".format(series_name, series))
                axes[i, j].set_xlabel(x_name)
                axes[i, j].set_ylabel(y_name)
                axes[i, j].set_ylim(ylim)
                axes[i, j].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
                axes[i, j].set_title("{} = {:.1E},{} = {:.1E}".format(row_name, row, column_name, column))
    plt.tight_layout(h_pad=2)
    plt.legend()

    plt.show()


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
    N_Rs = int(X_input_array[7])
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

    df_amp_phase_simulated['alpha_r'] = np.array([alpha_r for i in range(len(df_amp_phase_simulated))])

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


def show_sensitivity_results_sobol(sobol_problem, parallel_results, f_heating, df_r_ref_locations,calc_second_order):
    amp_ratio_results = np.array([np.array(parallel_result['amp_ratio']) for parallel_result in parallel_results])
    phase_diff_results = np.array([np.array(parallel_result['phase_diff']) for parallel_result in parallel_results])

    Si_amp_radius = np.array(
        [sobol.analyze(sobol_problem, amp_ratio_results[:, i], calc_second_order=calc_second_order, print_to_console=False)['S1']
         for i in range(amp_ratio_results.shape[1])])
    # Just pay close attention that calc_second_order=False must be consistent with how X is defined!

    Si_phase_radius = np.array(
        [sobol.analyze(sobol_problem, phase_diff_results[:, i], calc_second_order=calc_second_order, print_to_console=False)['S1']
         for i in range(phase_diff_results.shape[1])])

    plt.figure(figsize=(14, 6))
    radius = df_r_ref_locations['r']
    plt.subplot(121)
    for i, name in enumerate(sobol_problem['names']):
        # plt.plot(Si_amp_radius[:, i], label=name)

        plt.scatter(radius, Si_amp_radius[:, i], label=name)

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
        # plt.plot(Si_phase_radius[:, i], label=name)
        plt.scatter(radius, Si_phase_radius[:, i], label=name)
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


def DOE_numerical_model_one_case(parameter_name_list, DOE_parameters, df_temperature, df_r_ref_locations,
                                 sample_information, vacuum_chamber_setting, solar_simulator_settings,
                                 light_source_property, numerical_simulation_setting):
    for i, (parameter_name, DOE_parameter_value) in enumerate(zip(parameter_name_list, DOE_parameters)):
        # print(parameter_name)
        if parameter_name in sample_information.keys():
            if parameter_name == 'emissivity':
                sample_information['absorptivity'] = DOE_parameter_value
            sample_information[parameter_name] = DOE_parameter_value
        elif parameter_name in vacuum_chamber_setting.keys():
            vacuum_chamber_setting[parameter_name] = DOE_parameter_value
        elif parameter_name in numerical_simulation_setting.keys():
            numerical_simulation_setting[parameter_name] = DOE_parameter_value
        elif parameter_name in solar_simulator_settings.keys():
            solar_simulator_settings[parameter_name] = DOE_parameter_value
        elif parameter_name in light_source_property.keys():
            light_source_property[parameter_name] = DOE_parameter_value

    df_amp_phase_simulated, df_temperature_simulation = simulation_result_amplitude_phase_extraction(df_temperature,
                                                                                                     df_r_ref_locations,
                                                                                                     sample_information,
                                                                                                     vacuum_chamber_setting,
                                                                                                     solar_simulator_settings,
                                                                                                     light_source_property,
                                                                                                     numerical_simulation_setting)

    return df_amp_phase_simulated



def parallel_2nd_level_DOE(parameter_name_list, full_factorial_combinations, df_r_ref_locations,num_cores, result_name,jupyter_directory, df_temperature,sample_information,vacuum_chamber_setting,solar_simulator_settings,light_source_property,numerical_simulation_setting):


    DOE_parameters_complete = [one_factorial_design for one_factorial_design in full_factorial_combinations]
    result_dump_path = jupyter_directory+"//sensitivity cache dump//" + result_name
    if (os.path.isfile(result_dump_path)):
        print("Previous dump file existed, check if duplicate! The previous results are loaded.")
        joblib_output = pickle.load(open(result_dump_path, 'rb'))
    else:
        joblib_output = Parallel(n_jobs=num_cores)(delayed(DOE_numerical_model_one_case)(parameter_name_list, DOE_parameters,df_temperature,df_r_ref_locations, sample_information,vacuum_chamber_setting,solar_simulator_settings,light_source_property,numerical_simulation_setting) for
                                                  DOE_parameters in tqdm(DOE_parameters_complete))

        pickle.dump(joblib_output, open(result_dump_path, "wb"))

    df_run_conditions = pd.DataFrame(columns=parameter_name_list,data =DOE_parameters_complete)
    amp_ratio_results = np.array([np.array(joblib_output_['amp_ratio']) for joblib_output_ in joblib_output])
    phase_diff_results = np.array([np.array(joblib_output_['phase_diff']) for joblib_output_ in joblib_output])

    df_results_amp_only = pd.DataFrame(columns=df_r_ref_locations['r'], data = amp_ratio_results)
    df_results_phase_only = pd.DataFrame(columns=df_r_ref_locations['r'], data = phase_diff_results)

    df_results_amp_ratio_complete = pd.concat([df_run_conditions,df_results_amp_only],axis = 1)
    df_results_phase_difference_complete = pd.concat([df_run_conditions,df_results_phase_only],axis = 1)

    return df_results_amp_ratio_complete, df_results_phase_difference_complete


def main_effects_2_level_DOE(df_original, f_heating, parameter_name_list, df_r_ref_locations):
    df_DOE = df_original.query('f_heating == {}'.format(f_heating))
    param_main_list = []
    N_index = len(parameter_name_list)

    for parameter_name in parameter_name_list:
        if parameter_name != 'f_heating':
            param_unique = np.unique(df_DOE[parameter_name])

            param_main = np.array(df_DOE.query('{}=={}'.format(parameter_name, param_unique[0])).iloc[:, N_index:].mean(
                axis=0)) - np.array(
                df_DOE.query('{}=={}'.format(parameter_name, param_unique[1])).iloc[:, N_index:].mean(axis=0))
            param_main_list.append(param_main)

    parameter_name_columns = parameter_name_list.copy()
    parameter_name_columns.remove('f_heating')  # remember python pass by reference, it modifies the original list!

    df_main_effect = pd.DataFrame(columns=parameter_name_columns, data=np.array(param_main_list).T)
    df_main_effect['r'] = df_r_ref_locations['r']
    return df_main_effect


def plot_main_effects_2nd_level_DOE(df_amp_ratio_DOE_origninal, df_phase_diff_DOE_orignal, f_heating,
                                    df_r_ref_locations, parameter_name_list):
    df_main_effect_amp_ratio = main_effects_2_level_DOE(df_amp_ratio_DOE_origninal, f_heating, parameter_name_list,
                                                        df_r_ref_locations)
    df_main_effect_phase_diff = main_effects_2_level_DOE(df_phase_diff_DOE_orignal, f_heating, parameter_name_list,
                                                         df_r_ref_locations)
    parameter_name_columns = parameter_name_list.copy()
    parameter_name_columns.remove('f_heating')

    plt.figure(figsize=(14, 6))

    plt.subplot(121)
    for parameter_name in parameter_name_columns:
        plt.scatter(df_main_effect_amp_ratio['r'], df_main_effect_amp_ratio[parameter_name], label=parameter_name)

    plt.xlabel('R(pixel)', fontsize=12, fontweight='bold')
    plt.ylabel('amplitude ratio main effect', fontsize=12, fontweight='bold')
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=12)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=12)
        tick.label.set_fontweight('bold')
    plt.legend(prop={'weight': 'bold', 'size': 12})

    plt.subplot(122)
    for parameter_name in parameter_name_columns:
        plt.scatter(df_main_effect_phase_diff['r'], df_main_effect_phase_diff[parameter_name], label=parameter_name)

    plt.xlabel('R(pixel)', fontsize=12, fontweight='bold')
    plt.ylabel('Phase difference main effect', fontsize=12, fontweight='bold')
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=12)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=12)
        tick.label.set_fontweight('bold')
    plt.legend(prop={'weight': 'bold', 'size': 12})

    plt.suptitle('DOE 2nd level factorial main effect, Frequency {} Hz'.format(f_heating), fontsize=14,
                 fontweight='bold')  # or plt.suptitle('Main title')

    plt.show()


def interaction_effects_2_level_DOE(df_original, f_heating, parameter_name_list, df_r_ref_locations, ylim,
                                    amp_or_phase):
    df_DOE = df_original.query('f_heating == {}'.format(f_heating))
    param_main_list = []
    N_index = len(parameter_name_list)

    dic_interaction = {}

    parameter_name_columns = parameter_name_list.copy()
    parameter_name_columns.remove('f_heating')

    N_cols = len(parameter_name_columns)
    N_rows = 1

    f, axes = plt.subplots(N_rows, N_cols, figsize=(N_cols * 5, N_rows * 5))
    j = 0

    while (parameter_name_columns != []):

        parameter_name_1 = parameter_name_columns.pop(0)
        parameter_1_unique = np.unique(
            df_DOE[parameter_name_1])  # calculated using Eq 4.4 on page 192, exp planning, analysis and opt

        for parameter_name_2 in parameter_name_columns:
            pair_name = parameter_name_1 + "_vs_" + parameter_name_2

            parameter_2_unique = np.unique(df_DOE[parameter_name_2])

            param_inter = 0.5 * (np.array(df_DOE.query(
                '{}=={} and {}=={}'.format(parameter_name_1, parameter_1_unique[1], parameter_name_2,
                                           parameter_2_unique[1])).iloc[:, N_index:].mean(
                axis=0)) - np.array(df_DOE.query(
                '{}=={} and {}=={}'.format(parameter_name_1, parameter_1_unique[0], parameter_name_2,
                                           parameter_2_unique[1])).iloc[:, N_index:].mean(axis=0))
                                 ) - 0.5 * (np.array(df_DOE.query(
                '{}=={} and {}=={}'.format(parameter_name_1, parameter_1_unique[1], parameter_name_2,
                                           parameter_2_unique[0])).iloc[:, N_index:].mean(
                axis=0)) - np.array(df_DOE.query(
                '{}=={} and {}=={}'.format(parameter_name_1, parameter_1_unique[0], parameter_name_2,
                                           parameter_2_unique[0])).iloc[:, N_index:].mean(axis=0)))

            dic_interaction[pair_name] = param_inter

            axes[j].scatter(df_r_ref_locations['r'], param_inter, label="{}".format(parameter_name_2))

        axes[j].set_xlabel("R (pixels)", fontweight='bold', fontsize=10)
        axes[j].set_ylim(ylim)
        axes[j].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
        axes[j].set_title(parameter_name_1)
        axes[j].legend()

        j += 1

    df_interaction = pd.DataFrame(dic_interaction)
    df_interaction['r'] = df_r_ref_locations['r']

    plt.suptitle("{} 2nd level DOE interaction effect, frequency = {} Hz".format(amp_or_phase, f_heating),
                 fontweight='bold', fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return df_interaction, f


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