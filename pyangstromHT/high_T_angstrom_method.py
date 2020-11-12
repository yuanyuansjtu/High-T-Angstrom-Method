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

#from SALib.sample import saltelli
#from SALib.analyze import sobol
#from SALib.test_functions import Ishigami

from scipy.stats import norm
from scipy.stats import multivariate_normal

#from scipy.optimize import minimize
from scipy import optimize
import matplotlib.ticker as mtick
from scipy.interpolate import interp1d,interp2d


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
#print("Yuan said hello")

def plot_temperature_contour(x0, y0, path, file_name_0, file_name_1, R0, R_analysis):
    fig = plt.figure(figsize=(13, 6))

    df_first_frame = pd.read_csv(path + file_name_0, skiprows=5,header = None)
    df_mid_frame = pd.read_csv(path + file_name_1, skiprows=5,header = None)

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
    CS = ax.contour(X, Y, Z, 18)
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
    CS = ax.contour(X, Y, Z, 18)
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


def show_one_contour(df_temperature_one_frame, x0, y0, R0, axes, x_adjust, y_adjust, R_zoom, n_contour):
    x0 = x0 + x_adjust
    y0 = y0 + y_adjust

    xmin = x0 - R0 - R_zoom
    xmax = x0 + R0 + R_zoom
    ymin = y0 - R0 - R_zoom
    ymax = y0 + R0 + R_zoom
    Z = np.array(df_temperature_one_frame.iloc[ymin:ymax, xmin:xmax])
    x = np.arange(xmin, xmax, 1)
    y = np.arange(ymin, ymax, 1)
    X, Y = np.meshgrid(x, y)

    manual_locations = [(x0 - R0 + 6, y0 - R0 + 6)]

    CS = axes.contour(X, Y, Z, n_contour)
    axes.plot([xmin, xmax], [y0, y0], ls='-.', color='k', lw=2)  # add a horizontal line cross x0,y0
    axes.plot([x0, x0], [ymin, ymax], ls='-.', color='k', lw=2)  # add a vertical line cross x0,y0

    circle1 = plt.Circle((x0, y0), R0, edgecolor='b', fill=False, linewidth=3, linestyle='-.')
    circle2 = plt.Circle((x0, y0), R0 + R_zoom, edgecolor='b', fill=False, linewidth=3, linestyle='-.')

    axes.invert_yaxis()
    axes.clabel(CS, inline=1, fontsize=12, manual=manual_locations)
    axes.add_artist(circle1)
    axes.add_artist(circle2)


def check_center_contour(code_directory, data_directory, rec_num, x0, y0, R0, R_analysis):
    path = data_directory + rec_num + "//"

    check_contour_csv_dump_path = code_directory + "temperature cache dump//" + rec_num + '_check_contour'

    if (os.path.isfile(check_contour_csv_dump_path)):  # First check if a dump file exist:
        print('Found previous dump file for contour_checking:' + check_contour_csv_dump_path)
        temp_dump = pickle.load(open(check_contour_csv_dump_path, 'rb'))
    else:

        N_files = len([path + x for x in os.listdir(path)])

        file_name_0 = [path + x for x in os.listdir(path)][0]
        n0 = file_name_0.rfind('//')
        n1 = file_name_0.rfind('.csv')
        frame_num_0 = file_name_0[n0 + 2:n1]
        df_frame_0 = pd.read_csv(file_name_0, skiprows=5, header=None)

        file_name_1 = [path + x for x in os.listdir(path)][int(N_files / 6)]
        n0 = file_name_1.rfind('//')
        n1 = file_name_1.rfind('.csv')
        frame_num_1 = file_name_1[n0 + 2:n1]
        df_frame_1 = pd.read_csv(file_name_1, skiprows=5, header=None)

        file_name_2 = [path + x for x in os.listdir(path)][int(N_files / 8)]
        n0 = file_name_2.rfind('//')
        n1 = file_name_2.rfind('.csv')
        frame_num_2 = file_name_2[n0 + 2:n1]
        df_frame_2 = pd.read_csv(file_name_2, skiprows=5, header=None)

        temp_dump = [df_frame_0, df_frame_1, df_frame_2]
        pickle.dump(temp_dump, open(check_contour_csv_dump_path, "wb"))

    n_x = 3
    n_y = 3

    # print("---------------------------------{}----------------------------------".format(rec_num))

    fig, axes = plt.subplots(n_y, n_x, sharex='all', sharey='all', figsize=(n_x * 6, n_y * 6 + 1))
    plt.subplots_adjust(wspace=0, hspace=0)

    for j in range(n_y):
        for i in range(n_x):
            show_one_contour(temp_dump[0], x0, y0, R0, axes[j, i], i - 1, j - 1, 15, 16)  # R_zoom = 15, n_contour = 16
    fig.suptitle("{}, x0 = {}, y0 = {}".format(rec_num, x0, y0), fontsize=12, fontweight='bold', y=0.90)

    fig1, axes1 = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(3 * 6, 1 * 6))
    plt.subplots_adjust(wspace=0, hspace=0)

    for j in range(3):
        show_one_contour(temp_dump[j], x0, y0, R0, axes1[j], 0, 0, R_analysis, 12)

    # print("---------------------------------NEW PLOT----------------------------------")


def batch_contour_plots(code_directory, data_directory, df_exp_condition_spreadsheet_filename):
    df_exp_condition_spreadsheet = pd.read_excel(
        code_directory + "batch process information//" + df_exp_condition_spreadsheet_filename)

    steady_state_figure_list = []
    df_temperature_list = []

    for i in range(len(df_exp_condition_spreadsheet)):
        df_exp_condition = df_exp_condition_spreadsheet.iloc[i, :]

        rec_num = df_exp_condition['rec_name']
        x0 = df_exp_condition['x0']  # in pixels
        y0 = df_exp_condition['y0']  # in pixels

        R0 = df_exp_condition['R0']  # in pixels
        R_analysis = df_exp_condition['R_analysis']

        check_center_contour(code_directory, data_directory, rec_num, x0, y0, R0,R_analysis)


# joblib_output
def regression_joblib_to_dataframe(joblib_output, code_directory, df_exp_condition_spreadsheet_filename,sigma_df):

    T_average_list = [joblib_output_[1] for joblib_output_ in joblib_output]
    T_min_list = [joblib_output_[2] for joblib_output_ in joblib_output]
    df_exp_condition = pd.read_excel(code_directory+"batch process information//" + df_exp_condition_spreadsheet_filename)

    # df_solar_simulator_lorentzian_sigma = pd.read_excel(code_directory + "sample specifications//sample properties.xlsx",
    #                                               sheet_name="Lorentzian sigma")

    locations_relative_focal_plane = sigma_df['focal_shift']

    sigma_relative_focal_plane = sigma_df['sigma_s']

    # f_Amax = interp1d(locations_relative_focal_plane, Amax_relative_focal_plane, kind='cubic')
    f_sigma = interp1d(locations_relative_focal_plane, sigma_relative_focal_plane, kind='cubic')

    sample_material = np.unique(df_exp_condition['sample_name'])[0]
    # Here we assume each spreadsheet only contains one material
    df_theoretical_thermal_diffusivity_all = pd.read_excel(
        code_directory + "sample specifications//sample properties.xlsx", sheet_name="thermal diffusivity")
    df_theoretical_thermal_diffusivity = df_theoretical_thermal_diffusivity_all.query(
        "Material=='{}'".format(sample_material))

    temperatures_theoretical = df_theoretical_thermal_diffusivity['Temperature C'] + 273.15  # converted to K
    theoretical_thermal_diffusivity = df_theoretical_thermal_diffusivity['Thermal diffsivity']

    f_alpha = interp1d(temperatures_theoretical, theoretical_thermal_diffusivity, kind='linear')

    sigma_s_list = []

    alpha_regression_list = []
    alpha_theoretical_list = []

    sigma_ray_tracing_list = []

    emissivity_front_list = []

    for i, regression_type in enumerate(df_exp_condition['regression_parameter']):

        if regression_type == 'sigma_s':
            sigma_s_list.append(joblib_output[i][0])
            focal_shift = df_exp_condition['focal_shift'][i]
            alpha_regression_list.append(f_alpha(T_average_list[i])) # this is wrong, need fixed!
            sigma_ray_tracing_list.append(f_sigma(focal_shift))
            alpha_theoretical_list.append(f_alpha(T_average_list[i]))
            emissivity_front_list.append(df_exp_condition['emissivity_front'][i])

        elif regression_type == 'alpha_r':
            focal_shift = df_exp_condition['focal_shift'][i]
            sigma_s_list.append(f_sigma(focal_shift))
            alpha_regression_list.append(joblib_output[i][0])
            sigma_ray_tracing_list.append(f_sigma(focal_shift))

            alpha_theoretical_list.append(f_alpha(T_average_list[i]))
            emissivity_front_list.append(df_exp_condition['emissivity_front'][i])

        elif regression_type == 'emissivity_front':
            focal_shift = df_exp_condition['focal_shift'][i]
            sigma_s_list.append(f_sigma(focal_shift))
            alpha_regression_list.append(f_alpha(T_average_list[i]))
            sigma_ray_tracing_list.append(f_sigma(focal_shift))

            alpha_theoretical_list.append(f_alpha(T_average_list[i]))
            emissivity_front_list.append(joblib_output[i][0])

    df_results_all = pd.DataFrame({'rec_name':df_exp_condition['rec_name'],'focal_distance(cm)':df_exp_condition['focal_shift'],'f_heating':df_exp_condition['f_heating'],'VDC':df_exp_condition['V_DC'],
                                   'sigma_s':sigma_s_list,'T_average(K)':T_average_list,'T_min(K)':T_min_list,'R0':df_exp_condition['R0'],'alpha_r':alpha_regression_list,'regression_parameter':df_exp_condition['regression_parameter']
                                   ,'alpha_theoretical':alpha_theoretical_list,'sigma_ray_tracing':sigma_ray_tracing_list,'regression_method':df_exp_condition['regression_method'],'emissivity_front':emissivity_front_list})


    return df_results_all


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
            x = i * np.cos(theta_) + x0;  # Identifying the spatial 2D cartesian coordinates
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


def select_data_points_radial_average_MA_vectorized(x0, y0, Rmax, theta_range, file_name): # extract radial averaged temperature from one csv file
    # This method was originally developed by Mosfata, was adapted by HY to use for amplitude and phase estimation
    df_raw = pd.read_csv(file_name, nrows=3, header = None)
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

    Tr = np.zeros((Rmax, theta_n))

    df_temp_one_frame = pd.read_csv(file_name, skiprows=5,header = None)
    arr_temp = df_temp_one_frame.to_numpy()
    R_pixel = np.array([j for j in range(Rmax)])

    for j, theta_ in enumerate(theta):
        X = R_pixel * np.cos(theta_) + x0
        Y = R_pixel * np.sin(theta_) + y0

        Y1 = np.array(np.floor(Y), dtype=int)
        Y2 = Y1 + 1
        X1 = np.array(np.floor(X), dtype=int)
        X2 = X1 + 1

        dy1 = (Y2 - Y) / (Y2 - Y1)
        dy2 = (Y - Y1) / (Y2 - Y1)  # Identifying the corresponding weights for the y-coordinates
        dx1 = (X2 - X) / (X2 - X1)
        dx2 = (X - X1) / (X2 - X1) # Identifying the corresponding weights for the x-coordinates

        T11 = arr_temp[(Y1, X1)]
        T21 = arr_temp[(Y2, X1)]
        T12 = arr_temp[(Y1, X2)]
        T22 = arr_temp[(Y2, X2)]

        Tr[:, j] = dx1 * dy1 * T11 + dx1 * dy2 * T21 + dx2 * dy1 * T12 + dx2 * dy2 * T22 + 273.15

    T_interpolate_vectorize = np.mean(Tr, axis=1)


    return T_interpolate_vectorize, time_in_seconds



def check_angular_uniformity(x0, y0, N_Rmax, pr, path, rec_name, output_name, method, num_cores, f_heating, R0, gap,
                             R_analysis, angle_range,focal_plane_location,VDC, exp_amp_phase_extraction_method,code_directory):
    # we basically break entire disk into 6 regions, with interval of pi/3
    fig = plt.figure(figsize=(18.3, 12))
    colors = ['red', 'black', 'green', 'blue', 'orange', 'magenta','brown','yellow']

    df_temperature_list = []
    df_amp_phase_list = []

    plt.subplot(231)



    df_temperature_list, df_averaged_temperature = radial_temperature_average_disk_sample_several_ranges(x0, y0, N_Rmax,angle_range,pr, path,
                                                                                                         rec_name, output_name, method, num_cores,code_directory)

    for j, angle in enumerate(angle_range):
    # note radial_temperature_average_disk_sample automatically checks if a dump file exist
        df_temperature = df_temperature_list[j]
        df_amplitude_phase_measurement = batch_process_horizontal_lines(df_temperature, f_heating, R0, gap, R_analysis,
                                                                        exp_amp_phase_extraction_method)
        df_temperature_list.append(df_temperature)
        df_amp_phase_list.append(df_amplitude_phase_measurement)

        plt.scatter(df_amplitude_phase_measurement['r'],
                    df_amplitude_phase_measurement['amp_ratio'], facecolors='none',
                    s=60, linewidths=2, edgecolor=colors[j], label=str(angle[0]) + ' to ' + str(angle[1]) + ' Degs')


    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')

    plt.xlabel('R (pixels)', fontsize=12, fontweight='bold')
    plt.ylabel('Amplitude Ratio', fontsize=12, fontweight='bold')
    plt.title('rec = {}'.format(rec_name), fontsize=12, fontweight='bold')
    plt.legend()

    plt.subplot(232)


    for j, angle in enumerate(angle_range):
        df_temperature = df_temperature_list[j]
        df_amplitude_phase_measurement = df_amp_phase_list[j]
        plt.scatter(df_amplitude_phase_measurement['r'],
                    df_amplitude_phase_measurement['phase_diff'], facecolors='none',
                    s=60, linewidths=2, edgecolor=colors[j], label=str(angle[0]) + ' to ' + str(angle[1]) + ' Degs')

    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')

    plt.xlabel('R (pixels)', fontsize=12, fontweight='bold')
    plt.ylabel('Phase difference (rad)', fontsize=12, fontweight='bold')
    #plt.title('x0 = {}, y0 = {}, R0 = {}'.format(x0, y0, R0), fontsize=12, fontweight='bold')
    plt.legend()

    plt.subplot(233)

    for j, angle in enumerate(angle_range):
        df_temperature = df_temperature_list[j]
        time_max = 10 * 1 / f_heating  # only show 10 cycles
        df_temperature = df_temperature.query('reltime<{:.2f}'.format(time_max))

        plt.plot(df_temperature['reltime'],
                 df_temperature.iloc[:, R0], linewidth=2, color=colors[j],label=str(angle[0]) + ' to ' + str(angle[1]) + ' Degs')



    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')

    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Temperature (K)', fontsize=12, fontweight='bold')

    plt.title('f_heating = {} Hz'.format(f_heating), fontsize=12, fontweight='bold')
    #plt.title('R_ayalysis = {}, gap = {}'.format(R_analysis, gap), fontsize=12, fontweight='bold')
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

        df_first_frame = pd.read_csv(file_name_0, skiprows=5,header = None)

        #N_mid = int(len([path + x for x in os.listdir(path)]) / 3)
        N_mid = 20
        file_name_1 = [path + x for x in os.listdir(path)][N_mid]
        n2 = file_name_1.rfind('//')
        n3 = file_name_1.rfind('.csv')
        frame_num_mid = file_name_1[n2 + 2:n3]

        df_mid_frame = pd.read_csv(file_name_1, skiprows=5,header = None)

        temp_dump = [df_first_frame, df_mid_frame, frame_num_first, frame_num_mid]

        pickle.dump(temp_dump, open(rep_csv_dump_path, "wb"))

    #df_mid_frame_temperature = df_mid_frame.iloc[5:, :]
    #df_first_frame_temperature = df_first_frame.iloc[5:, :]

    plt.subplot(234)

    xmin = x0 - R0 - R_analysis
    xmax = x0 + R0 + R_analysis
    ymin = y0 - R0 - R_analysis
    ymax = y0 + R0 + R_analysis
    Z = np.array(df_first_frame.iloc[ymin:ymax, xmin:xmax])
    x = np.arange(xmin, xmax, 1)
    y = np.arange(ymin, ymax, 1)
    X, Y = np.meshgrid(x, y)

    # mid = int(np.shape(Z)[1] / 2)
    x3 = min(R0 + 10, R0 + R_analysis - 20)

    manual_locations = [(x0 - R0 + 15, y0 - R0 + 15), (x0 - R0, y0 - R0), (x0 - x3, y0 - x3)]
    # fig, ax = plt.subplots(figsize=(6, 6))
    ax = plt.gca()
    CS = ax.contour(X, Y, Z, 18)
    plt.plot([xmin, xmax], [y0, y0], ls='-.', color='k', lw=2)  # add a horizontal line cross x0,y0
    plt.plot([x0, x0], [ymin, ymax], ls='-.', color='k', lw=2)  # add a vertical line cross x0,y0

    R_angle_show = R0 + R_analysis


    circle1 = plt.Circle((x0, y0), R0, edgecolor='r', fill=False, linewidth=3, linestyle='-.')
    circle2 = plt.Circle((x0, y0), R0 + R_analysis, edgecolor='k', fill=False, linewidth=3, linestyle='-.')

    circle3 = plt.Circle((x0, y0), int(0.01/pr), edgecolor='black', fill=False, linewidth=3, linestyle='dotted')

    ax.invert_yaxis()
    ax.clabel(CS, inline=1, fontsize=12, manual=manual_locations)
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)

    ax.set_xlabel('x (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('y (pixels)', fontsize=12, fontweight='bold')
    ax.set_title('x0 = {}, y0 = {}, R0 = {}'.format(x0, y0, R0), fontsize=12, fontweight='bold')

    for j, angle in enumerate(angle_range):
        plt.plot([x0, x0 + R_angle_show * np.cos(angle[0] * np.pi / 180)], [y0,
                            y0 + R_angle_show * np.sin(angle[0] * np.pi / 180)], ls='-.', color='blue', lw=2)
        # plt.plot([x0, x0 + R_angle_show * np.cos(angle[1] * np.pi / 180)], [y0,
        #                     y0 + R_angle_show * np.sin(angle[1] * np.pi / 180)], ls='dotted', color='blue', lw=2)

    plt.subplot(235)

    xmin = x0 - R0 - R_analysis
    xmax = x0 + R0 + R_analysis
    ymin = y0 - R0 - R_analysis
    ymax = y0 + R0 + R_analysis
    Z = np.array(df_mid_frame.iloc[ymin:ymax, xmin:xmax])
    x = np.arange(xmin, xmax, 1)
    y = np.arange(ymin, ymax, 1)
    X, Y = np.meshgrid(x, y)

    # mid = int(np.shape(Z)[1] / 2)
    x3 = min(R0 + 10, R0 + R_analysis - 20)

    manual_locations = [(x0 - R0 + 15, y0 - R0 + 15), (x0 - R0, y0 - R0), (x0 - x3, y0 - x3)]
    # fig, ax = plt.subplots(figsize=(6, 6))
    ax = plt.gca()
    CS = ax.contour(X, Y, Z, 18)
    plt.plot([xmin, xmax], [y0, y0], ls='-.', color='k', lw=2)  # add a horizontal line cross x0,y0
    plt.plot([x0, x0], [ymin, ymax], ls='-.', color='k', lw=2)  # add a vertical line cross x0,y0

    R_angle_show = R0 + R_analysis


    circle1 = plt.Circle((x0, y0), R0, edgecolor='r', fill=False, linewidth=3, linestyle='-.')
    circle2 = plt.Circle((x0, y0), R0 + R_analysis, edgecolor='k', fill=False, linewidth=3, linestyle='-.')

    circle3 = plt.Circle((x0, y0), int(0.01/pr), edgecolor='black', fill=False, linewidth=3, linestyle='dotted')



    ax.invert_yaxis()
    ax.clabel(CS, inline=1, fontsize=12, manual=manual_locations)
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)

    ax.set_xlabel('x (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('y (pixels)', fontsize=12, fontweight='bold')
    ax.set_title('R_ayalysis = {}, gap = {}'.format(R_analysis, gap), fontsize=12, fontweight='bold')
    #plt.title('R_ayalysis = {}, gap = {}'.format(R_analysis, gap), fontsize=12, fontweight='bold')

    for j, angle in enumerate(angle_range):
        plt.plot([x0, x0 + R_angle_show * np.cos(angle[0] * np.pi / 180)], [y0,
                            y0 + R_angle_show * np.sin(angle[0] * np.pi / 180)], ls='-.', color='blue', lw=2)
        # plt.plot([x0, x0 + R_angle_show * np.cos(angle[1] * np.pi / 180)], [y0,
        #                     y0 + R_angle_show * np.sin(angle[1] * np.pi / 180)], ls='dotted', color='blue', lw=2)


    plt.subplot(236)
    T_mean_list = np.array(
        [np.array(df_temperature.iloc[:, R0:R0 + R_analysis].mean(axis=0)) for df_temperature in df_temperature_list])
    T_max_list = np.array(
        [np.array(df_temperature.iloc[:, R0:R0 + R_analysis].max(axis=0)) for df_temperature in df_temperature_list])
    T_min_list = np.array(
        [np.array(df_temperature.iloc[:, R0:R0 + R_analysis].min(axis=0)) for df_temperature in df_temperature_list])

    plt.plot(np.arange(R0, R0 + R_analysis), T_mean_list.mean(axis=0), linewidth=2,label = "mean temperature")
    plt.plot(np.arange(R0, R0 + R_analysis), T_max_list.mean(axis=0), linewidth=2,label = "max temperature")
    plt.plot(np.arange(R0, R0 + R_analysis), T_min_list.mean(axis=0), linewidth=2,label = "min temperature")

    ax = plt.gca()
    ax.set_xlabel('R (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temperature (K)', fontsize=12, fontweight='bold')

    #plt.title('R_ayalysis = {}, gap = {}'.format(R_analysis, gap), fontsize=12, fontweight='bold')
    ax.set_title("focal = {} cm, VDC = {} V".format(focal_plane_location,VDC), fontsize=12, fontweight='bold')
    plt.legend()

    #ax.set_title("Temperature vs R", fontsize=12, fontweight='bold')

    #plt.show()

    amp_ratio_list = np.array([np.array(df_amp_phase_list[i]['amp_ratio']) for i in range(len(angle_range))])
    phase_diff_list = np.array([np.array(df_amp_phase_list[i]['phase_diff']) for i in range(len(angle_range))])
    weight = np.linspace(1, 0.6, len(np.std(amp_ratio_list, axis=0)))
    amp_std = np.std(amp_ratio_list, axis=0)
    phase_std = np.std(phase_diff_list, axis=0)
    weight_amp_phase_std = (amp_std + phase_std) * weight
    sum_std = np.sum(weight_amp_phase_std)

    return sum_std, fig


def coaxial_parallel_disk(r_i, r_j, L):
    # L: gal between 2 disks
    # i: sample
    # j: target
    # direction from i - > j
    R_i = r_i / L
    R_j = r_j / L
    S = 1 + (1 + R_j ** 2) / R_i ** 2

    F_ij = 0.5 * (S - np.sqrt(S ** 2 - 4 * (r_j / r_i) ** 2))

    return F_ij


def F_ring_LB_ring_sample(RI, RO, rm, dr, L):
    if rm == 0:
        F_RO_ring_rm = 2e-8
        F_RI_ring_rm = 1e-8

    else:
        F_RO_ring_rm = coaxial_parallel_disk(RO, rm + dr / 2, L) - coaxial_parallel_disk(RO, rm - dr / 2, L)
        F_RI_ring_rm = coaxial_parallel_disk(RI, rm + dr / 2, L) - coaxial_parallel_disk(RI, rm - dr / 2, L)

    return (RO ** 2 * F_RO_ring_rm - RI ** 2 * F_RI_ring_rm) / (RO ** 2 - RI ** 2)


def F_IR_shield_ring_sample(rm, dr, W_IR, R_IRS):
    # W distance between the IR shield to the sample
    # R_IRS Radius of the IR shield, typically similar to the sample, maybe 3.75"
    if rm == 0:
        F_IRS_ring_rm = 1e-8
    else:
        F_IRS_ring_rm = coaxial_parallel_disk(R_IRS, rm + dr / 2, W_IR) - coaxial_parallel_disk(R_IRS, rm - dr / 2, W_IR)

    return F_IRS_ring_rm


def F_back_chamber_wall_sample(rm, dr, W1, W2, R_chamber):
    #R_chamber = 48.641e-3  # diameter of vacuum chamber

    F_RO_W2 = coaxial_parallel_disk(rm + dr / 2, R_chamber, W1) - coaxial_parallel_disk(rm + dr / 2, R_chamber, W1 + W2)
    F_RI_W2 = coaxial_parallel_disk(rm - dr / 2, R_chamber, W1) - coaxial_parallel_disk(rm - dr / 2, R_chamber, W1 + W2)
    if rm == 0:
        F_ring_W2 = 1e-8
    else:
        F_ring_W2 = ((rm + dr / 2) ** 2 * F_RO_W2 - (rm - dr / 2) ** 2 * F_RI_W2) / (2 * rm * dr)

    F_W2_ring = rm * dr / (R_chamber * W2) * F_ring_W2

    return F_W2_ring


def calculator_coaxial_parallel_rings_view_factors(RI, RO, rm_array, dr, L):
    F_list = []
    for rm in rm_array:
        F_list.append(F_ring_LB_ring_sample(RI, RO, rm, dr, L))

    #     plt.plot(rm_array,F_list)
    #     plt.xlabel("Location on sample (m)")
    #     plt.ylabel("View factor")
    #     # This seems quite wrong, lol

    #     print("The total view factor between region on LB bounded by R = {:.2e} and R = {:.2e} is {}".format(RI, RO,np.sum(abs(np.array(F_list)))))

    return np.array(F_list)


def calculator_IR_shield_to_sample_view_factors(rm_array, dr, W_IR, R_IRS):
    F_list = []
    for rm in rm_array:
        F_list.append(F_IR_shield_ring_sample(rm, dr, W_IR, R_IRS))

    #     plt.plot(rm_array,F_list)
    #     plt.xlabel("Location on sample (m)")
    #     plt.ylabel("View factor")
    #     # This seems quite wrong, lol

    #     print("The total view factor of the IR shield is {}".format(np.sum(abs(np.array(F_list)))))

    return np.array(F_list)


def calculator_back_W2_ring_VF(rm_array, dr, W1, W2,R_chamber):
    F_list = []
    for rm in rm_array:
        F_list.append(F_back_chamber_wall_sample(rm, dr, W1, W2,R_chamber))

    #     plt.plot(rm_array,F_list)
    #     plt.xlabel("Location on sample (m)")
    #     plt.ylabel("View factor")
    #     # This seems quite wrong, lol

    #     print("The total view factor of the W2 is {}".format(np.sum(abs(np.array(F_list)))))

    return np.array(F_list)


def radiation_absorption_view_factor_calculations(code_directory,rm_array,dr,sample_information,solar_simulator_settings,vacuum_chamber_setting,numerical_simulation_setting,df_view_factor,df_LB_details_all):

    # The radioan absorption view factor contains the following cases:
    # (1) With light blocker: (a) Sensitivity analysis: Light blocker has the same temperature. (b) Regression analysis: Light blocker has radial varied temperature
    # (2) Without light blocker: (a) Sensitivity analysis: Surroundings are treated as the same temperature (not including glass and IR radiation shield). (b) Regression: Surrounding temperature are treated differently.
    # These cases are handled internally within the function


    sigma_sb = 5.67e-8 # stefan boltzmann constant

    VDC = solar_simulator_settings['V_DC']
    focal_shift = vacuum_chamber_setting['focal_shift']

    sample_name = sample_information['sample_name']
    absorptivity_front = sample_information['absorptivity_front']
    absorptivity_back = sample_information['absorptivity_back']



    if vacuum_chamber_setting['light_blocker'] == True:

        if (numerical_simulation_setting['analysis_mode'] != 'sensitivity') and (numerical_simulation_setting['analysis_mode'] != 'validation'):

            if sample_name == 'copper':
                df_LB_temp = df_LB_details_all.query("Material == '{}'".format('copper'))

            else:
                df_LB_temp = df_LB_details_all.query("Material == '{}'".format('graphite_poco'))

            T_LB1_C,T_LB2_C,T_LB3_C, T_LB_mean_C = interpolate_LB_temperatures(focal_shift, VDC, df_LB_temp)

            T_LB1 = T_LB1_C + 273.15
            T_LB2 = T_LB2_C + 273.15
            T_LB3 = T_LB3_C + 273.15

        else:
            # In sensitivity analysis light blocker temperature is a constant and can be specified
            T_LB1 = vacuum_chamber_setting['T_sur1']
            T_LB2 = vacuum_chamber_setting['T_sur1']
            T_LB3 = vacuum_chamber_setting['T_sur1']
            T_LB_mean_C = vacuum_chamber_setting['T_sur1']



        T_LBH = float(df_view_factor['T_LBH_C'])+273.15
        T_LBW = float(df_view_factor['T_LBW_C'])+273.15
        T_IRW = float(df_view_factor['T_IRW_C']) + 273.15
        T_W1 = float(df_view_factor['T_W1_C'])+273.15
        T_W2 = float(df_view_factor['T_W2_C'])+273.15

        e_LB = float(df_view_factor['e_LB'])
        e_LBW = float(df_view_factor['e_LBW'])
        e_LBH = float(df_view_factor['e_LBH'])
        e_IRW = float(df_view_factor['e_IRW'])
        e_W1 = float(df_view_factor['e_W1'])
        e_W2 = float(df_view_factor['e_W2'])

        L_LB_sample = float(df_view_factor['L_LB_sample'])
        R_LBH = float(df_view_factor['R_LBH'])
        R_LB1 = float(df_view_factor['R_LB1'])
        R_LB2 = float(df_view_factor['R_LB2'])
        R_LB3 = float(df_view_factor['R_LB3'])
        R_chamber = float(df_view_factor['R_chamber'])
        R_IRW = float(df_view_factor['R_IRW'])
        W1 = float(df_view_factor['W1'])
        W2 = float(df_view_factor['W2'])

        A_LBH_LB1 = np.pi * (R_LB1 ** 2 - R_LBH ** 2)
        A_LB1_LB2 = np.pi * (R_LB2 ** 2 - R_LB1 ** 2)
        A_LB2_LB3 = np.pi * (R_LB3 ** 2 - R_LB2 ** 2)
        A_LBH = np.pi * R_LBH ** 2
        A_LBW = 2 * np.pi * R_chamber * L_LB_sample

        A_W1 = 2 * np.pi * R_chamber * W1
        A_W2 = 2 * np.pi * R_chamber * W2
        A_IRW = np.pi * R_chamber ** 2


        Cm_front = calculator_coaxial_parallel_rings_view_factors(1e-4, R_LBH, rm_array, dr,L_LB_sample) * A_LBH * e_LBH * sigma_sb * T_LBH ** 4 * absorptivity_front \
        + calculator_coaxial_parallel_rings_view_factors(R_LBH, R_LB1, rm_array, dr,L_LB_sample) * A_LBH_LB1 * e_LB * sigma_sb * T_LB1 ** 4* absorptivity_front \
        + calculator_coaxial_parallel_rings_view_factors(R_LB1, R_LB2, rm_array, dr,L_LB_sample) * A_LB1_LB2 * e_LB * sigma_sb * T_LB2 ** 4* absorptivity_front \
        + calculator_coaxial_parallel_rings_view_factors(R_LB2, R_LB3, rm_array, dr,L_LB_sample) * A_LB2_LB3 * e_LB * sigma_sb * T_LB3 ** 4 * absorptivity_front\
        + calculator_back_W2_ring_VF(rm_array, dr, 1e-4, L_LB_sample, R_chamber) * A_LBW * e_LBW * sigma_sb * T_LBW ** 4 * absorptivity_front

        Cm_back = calculator_back_W2_ring_VF(rm_array, dr, 1e-4, W1, R_chamber) * A_W1 * e_W1 * sigma_sb * T_W1 ** 4 * absorptivity_back\
        + calculator_back_W2_ring_VF(rm_array, dr, W1, W2, R_chamber) * A_W2 * e_W2 * sigma_sb * T_W2 ** 4 * absorptivity_back \
        + calculator_IR_shield_to_sample_view_factors(rm_array, dr, W1 + W2, R_IRW) * A_IRW * e_IRW * sigma_sb * T_IRW ** 4 * absorptivity_back

        Cm_front[0] = 1e-8
        Cm_back[0] = 1e-8

    elif vacuum_chamber_setting['light_blocker'] == False:
        # No light blocker
        if numerical_simulation_setting['analysis_mode'] != 'sensitivity':
            T_W1 = float(df_view_factor['T_W1_C']) + 273.15
            T_W2 = float(df_view_factor['T_W2_C']) + 273.15
        elif numerical_simulation_setting['analysis_mode'] == 'sensitivity':
            # For sensitivity analysis these temperatures can be assigned manually
            T_W1 = vacuum_chamber_setting['T_sur1']
            T_W2 = vacuum_chamber_setting['T_sur1']

        T_W3 = float(df_view_factor['T_W3_C']) + 273.15
        T_glass = float(df_view_factor['T_glass_C']) + 273.15
        T_IRW = float(df_view_factor['T_IRW_C']) + 273.15

        W1 = float(df_view_factor['W1'])
        W2 = float(df_view_factor['W2'])
        W3 = float(df_view_factor['W3'])
        L_G = float(df_view_factor['L_G'])
        R_chamber = float(df_view_factor['R_chamber'])
        R_IR_I = float(df_view_factor['R_IR_I'])
        R_IR_O = R_chamber
        R_G_O = R_chamber
        L_IR = W2 + W3

        A_W2 = 2 * np.pi * R_chamber * W2
        A_W3 = 2 * np.pi * R_chamber * W3
        A_W1 = 2 * np.pi * R_chamber * W1
        A_glass = np.pi * R_G_O ** 2
        A_IR_shield = np.pi * (R_IR_O ** 2 - R_IR_I ** 2)

        VF_W1_front = calculator_back_W2_ring_VF(rm_array, dr, 1e-4, W1, R_chamber)
        VF_W2_back = calculator_back_W2_ring_VF(rm_array, dr, 1e-4, W2, R_chamber)
        VF_W3_back = calculator_back_W2_ring_VF(rm_array, dr, W2, W3, R_chamber)
        VF_glass = calculator_coaxial_parallel_rings_view_factors(1e-4, R_G_O, rm_array, dr, L_G)
        VF_IR_shield = calculator_coaxial_parallel_rings_view_factors(R_IR_I, R_IR_O, rm_array,dr, L_IR)


        Cm_front = 0.95*sigma_sb*VF_glass*A_glass*T_glass**4 + 0.95*sigma_sb*VF_W1_front*A_W1*T_W1**4
        Cm_back = sigma_sb*VF_W2_back*A_W2*T_W2**4 + sigma_sb*VF_W3_back*A_W3*T_W3**4 + 0.95*sigma_sb*VF_IR_shield*A_IR_shield*T_IRW**4

        Cm_front[0] = 1e-8
        Cm_back[0] = 1e-8

        T_LB_mean_C = T_W1

    return Cm_front,Cm_back, T_LB_mean_C


# def mean_LB_temperatures(df_LB_temperature_details, focal_shift, VDC, R_inner_pixel, R_outer_pixel):
#     T_LB1 = df_LB_temperature_details.query(
#         'VDC == {} and focal_shift == {} and R_pixels>{} and R_pixels<{}'.format(VDC, focal_shift, R_inner_pixel,
#                                                                                  R_outer_pixel))
#     T_LB1.sort_values(by=['R_pixels'])
#
#     R_array = np.array(T_LB1['R_pixels'])
#     R_array = np.insert(R_array, 0, R_inner_pixel)
#     Area_list = R_array[1:] ** 2 - R_array[:-1] * 2
#     T_mean_C = np.sum(Area_list * np.array(T_LB1['T_C'])) / np.sum(Area_list)
#
#     return T_mean_C


def mean_LB_temperatures(df_LB_temperature_details, focal_shift, VDC, R_inner_pixel, R_outer_pixel):
    R_available_array = df_LB_temperature_details.query(
        'VDC == {} and focal_shift == {}'.format(VDC, focal_shift))['R_pixels']
    R_pixel_min = min(R_available_array)

    if R_outer_pixel < R_pixel_min:
        T_mean_C = -1

    else:

        T_LB1 = df_LB_temperature_details.query(
            'VDC == {} and focal_shift == {} and R_pixels>{} and R_pixels<{}'.format(VDC, focal_shift, R_inner_pixel,
                                                                                     R_outer_pixel))

        T_LB1.sort_values(by=['R_pixels'])

        R_array = np.array(T_LB1['R_pixels'])
        R_array = np.insert(R_array, 0, R_inner_pixel)
        Area_list = R_array[1:] ** 2 - R_array[:-1] * 2
        T_mean_C = np.sum(Area_list * np.array(T_LB1['T_C'])) / np.sum(Area_list)

    return T_mean_C



def interpolate_LB_temperatures(actual_focal_shift,actual_VDC,df_LB_temperature_details_csv):

    VDC_array_unique = np.unique(np.array(df_LB_temperature_details_csv['VDC']))
    focal_shift_array_unique = np.unique(np.array(df_LB_temperature_details_csv['focal_shift']))

    R_LBH_pixel = 47
    R_LB1_pixel = 69 # 1/3 LB
    R_LB2_pixel = 138 # 2/3 LB
    R_LB3_pixel = 207 #3/3 LB

    T_RH_R1_list = []
    T_R1_R2_list = []
    T_R2_R3_list = []
    VDC_list = []
    focal_shift_list = []

    for VDC in VDC_array_unique:
        for focal_shift in focal_shift_array_unique:
            T_LBR1_LBR2 = mean_LB_temperatures(df_LB_temperature_details_csv,focal_shift,VDC,R_LB1_pixel,R_LB2_pixel)
            T_LBR2_LBR3 = mean_LB_temperatures(df_LB_temperature_details_csv,focal_shift,VDC,R_LB2_pixel,R_LB3_pixel)
            T_LBH_LBR1_temp = mean_LB_temperatures(df_LB_temperature_details_csv,focal_shift,VDC,R_LBH_pixel,R_LB1_pixel)

            if T_LBH_LBR1_temp>0: # just check if T_LBH_LBR1 is a number
                T_LBH_LBR1 = T_LBH_LBR1_temp
            else:
                T_LBH_LBR1 = 2*T_LBR1_LBR2 - T_LBR2_LBR3

            T_RH_R1_list.append(T_LBH_LBR1)
            T_R1_R2_list.append(T_LBR1_LBR2)
            T_R2_R3_list.append(T_LBR2_LBR3)

            VDC_list.append(VDC)
            focal_shift_list.append(focal_shift)

    df_mean_temp_C_LB_regions = pd.DataFrame({'VDC':VDC_list,'focal_shift':focal_shift_list,
                                             'T_RH_R1':T_RH_R1_list,'T_R1_R2':T_R1_R2_list,'T_R2_R3':T_R2_R3_list})

    LB_temp_focal_shift = df_mean_temp_C_LB_regions['focal_shift']
    LB_temp_VDC = df_mean_temp_C_LB_regions['VDC']
    T_RH_R1 = df_mean_temp_C_LB_regions['T_RH_R1']
    T_R1_R2 = df_mean_temp_C_LB_regions['T_R1_R2']
    T_R2_R3 = df_mean_temp_C_LB_regions['T_R2_R3']

    f_T_RH_R1 = interp2d(LB_temp_VDC, LB_temp_focal_shift, T_RH_R1, kind='linear')
    f_T_R1_R2 = interp2d(LB_temp_VDC, LB_temp_focal_shift, T_R1_R2, kind='linear')
    f_T_R2_R3 = interp2d(LB_temp_VDC, LB_temp_focal_shift, T_R2_R3, kind='linear')

    T_LB1_C = f_T_RH_R1(actual_VDC,actual_focal_shift)[0]
    T_LB2_C = f_T_R1_R2(actual_VDC,actual_focal_shift)[0]
    T_LB3_C = f_T_R2_R3(actual_VDC,actual_focal_shift)[0]

    T_LB_mean = (T_LB1_C*(R_LB1_pixel**2 - R_LBH_pixel**2) + T_LB2_C*(R_LB2_pixel**2 - R_LB1_pixel**2) + T_LB3_C*(R_LB3_pixel**2 - R_LB2_pixel**2))/(R_LB3_pixel**2 - R_LBH_pixel**2)
    return T_LB1_C, T_LB2_C, T_LB3_C, T_LB_mean


def select_data_points_radial_average_MA_match_model_grid(x0, y0, N_Rmax, pr, R_sample, theta_n,
                                                          file_name):  # N_Rmax: total number of computation nodes between center and edge
    # This method was originally developed by Mosfata, was adapted here for amplitude and phase estimation
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
    ##print(theta_range_list)
    df_temperature_list = []
    for theta_range in theta_range_list:
        dump_file_path = code_directory+"temperature cache dump//"+output_name + '_x0_{}_y0_{}_Rmax_{}_method_{}_theta_{}_{}'.format(int(x0), int(y0), N_Rmax, method, int(theta_range[0]), int(theta_range[1]))
        #print(dump_file_path)
        if (os.path.isfile(dump_file_path)):  # First check if a dump file exist:
            print('Found previous dump file :' + dump_file_path)
            temp_dump = pickle.load(open(dump_file_path, 'rb'))

        else:  # If not we obtain the dump file, note the dump file is averaged radial temperature

            file_names = [path + x for x in os.listdir(path)]
            s_time = time.time()

            if method == 'MA':  # default method, this one is much faster
                theta_n = 100  # default theta_n=100, however, if R increased significantly theta_n should also increase
                # joblib_output = Parallel(n_jobs=num_cores)(delayed(select_data_points_radial_average_MA)(x0,y0,Rmax,theta_n,file_name) for file_name in tqdm(file_names))
                #joblib_output = Parallel(n_jobs=num_cores)(delayed(select_data_points_radial_average_MA)(x0, y0, N_Rmax, theta_range, file_name) for file_name in tqdm(file_names))
                joblib_output = Parallel(n_jobs=num_cores)(delayed(select_data_points_radial_average_MA_vectorized)(x0, y0, N_Rmax, theta_range, file_name) for file_name in tqdm(file_names))

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

        df_temperature = pd.DataFrame(data=temp_data)  # return a dataframe containing radial averaged temperature and relative time
        df_temperature['reltime'] = time_data

        df_temperature_list.append(df_temperature)



    cols = df_temperature_list[0].columns
    data = np.array([np.array(df_temperature_list_.iloc[:, :]) for df_temperature_list_ in df_temperature_list])
    data_mean = np.mean(data, axis=0)
    df_averaged_temperature = pd.DataFrame(data=data_mean, columns=cols)

    return df_temperature_list, df_averaged_temperature  # note the column i of the df_temperature indicate the temperature in pixel i


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
    N_frame_keep = int(df_temperature.shape[0])

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


def interpolate_light_source_characteristic(sigma_df, df_solar_simulator_VQ, sigma, numerical_simulation_setting,vacuum_chamber_setting):

    focal_shift = vacuum_chamber_setting['focal_shift']

    if (numerical_simulation_setting['analysis_mode'] == 'regression') and (numerical_simulation_setting['regression_parameter'] == 'sigma_s'):
        sigma_s = sigma

    else:

        focal_shift_sigma_calib = sigma_df['focal_shift']
        sigma_s_calib = sigma_df['sigma_s']

        f_sigma_cubic = interp1d(focal_shift_sigma_calib, sigma_s_calib, kind='cubic')
        # focal_shift = vacuum_chamber_setting['focal_shift']
        sigma_s = float(f_sigma_cubic(focal_shift))

    V_calib = np.array(df_solar_simulator_VQ['V_DC'])
    focal_shift_calib = np.array(df_solar_simulator_VQ['focal_shift'])

    f_V_Q = interp2d(V_calib, focal_shift_calib, np.array(df_solar_simulator_VQ['E_total']), kind='cubic')

    absorptivity_front_calib = np.unique(df_solar_simulator_VQ['absorptivity_front'])[0]  # This is a very important parameter, it directly dictate the uncertainty of Amax


    R_sample = 0.0889 / 2

    VDC_array = np.unique(V_calib)
    Q_V_real = f_V_Q(VDC_array, focal_shift)

    Amax_fv = Q_V_real / (absorptivity_front_calib * sigma_s * np.log((sigma_s ** 2 + R_sample ** 2) / sigma_s ** 2))
    Amax = max(Amax_fv)

    regression_model = LinearRegression()
    # regression_model.fit(VDC_array, Q_V_real/Amax)

    regression_model.fit(np.reshape(VDC_array, (len(VDC_array), 1)),
                         np.reshape(Amax_fv / Amax, (len(Amax_fv / Amax), 1)))

    kvd = regression_model.coef_[0][0]
    bvd = regression_model.intercept_[0]

    return Amax, sigma_s, kvd, bvd



def light_source_intensity_Amax_fV_vecterize(r_array, t_array, solar_simulator_settings, vacuum_chamber_setting,light_source_property,
                                             df_solar_simulator_VQ, sigma_df):

    focal_shift = vacuum_chamber_setting['focal_shift']

    Amax, sigma_s, kvd, bvd = light_source_property['Amax'], light_source_property['sigma_s'], light_source_property[
        'kvd'], light_source_property['bvd']

    f_heating = solar_simulator_settings['f_heating']

    V_AC = solar_simulator_settings['V_amplitude']
    V_DC = solar_simulator_settings['V_DC']

    V_full_cycles = (V_DC + V_AC * np.sin(2 * np.pi * f_heating * t_array))

    q = (kvd * V_full_cycles[:, np.newaxis] + bvd) * Amax / np.pi * sigma_s / (
                sigma_s ** 2 + r_array[np.newaxis, :] ** 2)

    N_Rs = vacuum_chamber_setting['N_Rs']
    q[:, N_Rs:] = 0.0

    """
    V_calib = np.array(df_solar_simulator_VQ['V_DC'])
    focal_shift_calib = np.array(df_solar_simulator_VQ['focal_shift'])
    Amax_fv_calib = np.array(df_solar_simulator_VQ['Amax_fv'])
    f_calib = interp2d(V_calib, focal_shift_calib, Amax_fv_calib, kind='linear')

    focal_shift = vacuum_chamber_setting['focal_shift']

    f_heating = solar_simulator_settings['f_heating']
    N_Rs = vacuum_chamber_setting['N_Rs']

    V_AC = solar_simulator_settings['V_amplitude']
    V_DC = solar_simulator_settings['V_DC']

    V_full_cycles = V_DC + V_AC * np.sin(2 * np.pi * f_heating * t_array)

    idx = np.argsort(np.argsort(V_full_cycles))
    # warning! interp2d ony takes sorted input, does not work for sinosoidal function as it is! Must keep track of the index before sorting!
    Amax_fv_full_cycle = f_calib(V_full_cycles, focal_shift)[idx]

    q = Amax_fv_full_cycle[:, np.newaxis] / np.pi * (sigma_s / (sigma_s ** 2 + r_array[np.newaxis, :] ** 2))

    q[:, N_Rs:] = 0.0
    """

    """
    # alternative method, slightly faster, do not require intensive interpolation
    V_one_period = V_DC + V_AC * np.sin(2 * np.pi * f_heating * t_array_one_period)

    idx = np.argsort(np.argsort(V_one_period))
    # warning! interp2d ony takes sorted input, does not work for sinosoidal function as it is! Must keep track of the index before sorting!
    Amax_fv_one_cycle = f_calib(V_one_period,focal_shift)[idx] 

    Amax_fv_full_cycle = np.tile(Amax_fv_one_cycle,N_cycle)


    sigma_d = float(df_solar_simulator_sigma.query("focal_shift == {}".format(focal_shift))['sigma'])

    q = Amax_fv_full_cycle[:,np.newaxis]/np.pi*(sigma_d/(sigma_d**2 + r_array[np.newaxis, :] ** 2))

    q[:,N_Rs:] = 0.0
    """

    return q


def lorentzian_Amax_sigma_estimation(d,code_directory):

    df_solar_simulator_lorentzian = pd.read_excel(code_directory + "sample specifications//sample properties.xlsx",
                                                  sheet_name="solar simulator Lorentzian")

    locations_relative_focal_plane = df_solar_simulator_lorentzian['Distance from focal plane(cm)']
    Amax_relative_focal_plane = df_solar_simulator_lorentzian['Amax']
    sigma_relative_focal_plane = df_solar_simulator_lorentzian['sigma']

    f_Amax = interp1d(locations_relative_focal_plane, Amax_relative_focal_plane, kind='cubic')
    f_sigma = interp1d(locations_relative_focal_plane, sigma_relative_focal_plane, kind='cubic')

    return f_Amax(d), f_sigma(d)


def finite_difference_explicit_1D_const_alpha(sample_information, vacuum_chamber_setting, solar_simulator_settings,
                       light_source_property, numerical_simulation_setting, df_solar_simulator_VQ,
                       sigma_df,code_directory,df_view_factor,df_LB_details_all):

    T_LBW = float(df_view_factor['T_LBW_C'][0]) + 273.15

    R = sample_information['R'] # sample radius
    Nr = numerical_simulation_setting['Nr'] # number of discretization along radial direction
    t_z = sample_information['t_z'] # sample thickness

    dr = R / Nr
    r = np.arange(Nr)
    rm = r * dr

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
    cp_test = cp_const + cp_c1 * T_initial + cp_c2 * T_initial ** 2 + cp_c3 * T_initial ** 3

    f_heating = solar_simulator_settings['f_heating']  # periodic heating frequency
    N_cycle = numerical_simulation_setting['N_cycle']
    N_stable_cycle_output = numerical_simulation_setting['N_stable_cycle_output']

    simulated_amp_phase_extraction_method = numerical_simulation_setting['simulated_amp_phase_extraction_method']

    emissivity_front = sample_information['emissivity_front']  # assumed to be constant
    emissivity_back = sample_information['emissivity_back']  # assumed to be constant

    absorptivity_front = sample_information['absorptivity_front']  # assumed to be constant

    sigma_sb = 5.6703744 * 10 ** (-8)  # stefan-Boltzmann constant


    # The radioan absorption view factor contains the following cases:
    # (1) With light blocker: (a) Sensitivity analysis: Light blocker has the same temperature. (b) Regression analysis: Light blocker has radial varied temperature
    # (2) Without light blocker: (a) Sensitivity analysis: Surroundings are treated as the same temperature (not including glass and IR radiation shield). (b) Regression: Surrounding temperature are treated differently.
    # These cases are handled internally within the function
    # Needs to distinguish front and back
    Cm_front,Cm_back, T_LB_mean_C = radiation_absorption_view_factor_calculations(code_directory, rm, dr, sample_information,
                                                                    solar_simulator_settings,
                                                                    vacuum_chamber_setting,
                                                                    numerical_simulation_setting,
                                                                    df_view_factor, df_LB_details_all)


    Fo_criteria = numerical_simulation_setting['Fo_criteria']

    dt = min(Fo_criteria * (dr ** 2) / (alpha_r),
             1 / f_heating / 15)  # assume 15 samples per period, Fo_criteria default = 1/3

    t_total = 1 / f_heating * N_cycle  # total simulation time

    Nt = int(t_total / dt)  # total number of simulation time step
    time_simulation = dt * np.arange(Nt)

    Fo_r = alpha_r * dt / dr ** 2

    T = T_initial * np.ones((Nt, Nr))
    dz = t_z
    q_solar = light_source_intensity_Amax_fV_vecterize(np.arange(Nr) * dr, np.arange(Nt) * dt,
                                                       solar_simulator_settings, vacuum_chamber_setting,
                                                       light_source_property,
                                                       df_solar_simulator_VQ, sigma_df)
    T_temp = np.zeros(Nr)
    N_steady_count = 0
    time_index = Nt - 1
    N_one_cycle = int(Nt / N_cycle)

    Cm = Cm_front + Cm_back

    for p in range(Nt - 1):  # p indicate time step

        # The node at center of the disk
        cp_center = cp_const + cp_c1 * T[p, 0] + cp_c2 * T[p, 0] ** 2 + cp_c3 * T[p, 0] ** 3
        T[p + 1, 0] = T[p, 0] * (1 - 4 * Fo_r) + 4 * Fo_r * T[p, 1] - sigma_sb * (
                    emissivity_back + emissivity_front) * dt / (rho * cp_center * dz) * T[p, 0] ** 4 + \
                      dt / (rho * cp_center * dz) * (absorptivity_front * q_solar[p, 0] + 4 * Cm[0] / (np.pi * dr ** 2))

        # The node in the middle of disk
        cp_mid = cp_const + cp_c1 * T[p, 1:-1] + cp_c2 * T[p, 1:-1] ** 2 + cp_c3 * T[p, 1:-1] ** 3
        T[p + 1, 1:-1] = T[p, 1:-1] + Fo_r * (rm[1:-1] - dr / 2) / rm[1:-1] * (T[p, 0:-2] - T[p, 1:-1]) + Fo_r * (
                    rm[1:-1] + dr / 2) / rm[1:-1] * (T[p, 2:] - T[p, 1:-1]) \
                         + dt / (rho * cp_mid * dz) * (
                                     absorptivity_front * q_solar[p, 1:-1] + Cm[1:-1] / (2 * np.pi * rm[1:-1] * dr) - (
                                         emissivity_front + emissivity_back) * sigma_sb * T[p, 1:-1] ** 4)

        # The node at the edge of the disk
        cp_edge = cp_const + cp_c1 * T[p, -1] + cp_c2 * T[p, -1] ** 2 + cp_c3 * T[p, -1] ** 3
        T[p + 1, -1] = T[p, -1] * (1 - 2 * Fo_r) + 2 * Fo_r * T[p, -2] + absorptivity_front * dt * q_solar[p, -1] / (
                    dz * rho * cp_edge) + 2 * rm[-1] * absorptivity_front * sigma_sb * dt * T_LBW ** 4 / (
                                   (rm[-1] - dr / 2) * dr * rho * cp_edge) + Cm[-1] * dt / (
                                   np.pi * (rm[-1] - dr / 2) * dr * dz * rho * cp_edge) - \
                       (emissivity_front + emissivity_back) * sigma_sb * dt / (dz * rho * cp_edge) * T[p, -1] ** 4 - 2 * rm[
                           -1] * emissivity_back * sigma_sb * dt * T[p, -1] ** 4 / ((rm[-1] - dr / 2) * dr * rho * cp_edge)


        if (p > 0) & (p % N_one_cycle == 0) & (simulated_amp_phase_extraction_method != 'fft'):
            A_max = np.max(T[p - N_one_cycle:p, :], axis=0)
            A_min = np.min(T[p - N_one_cycle:p, :], axis=0)
            if np.max(np.abs((T_temp[:] - T[p, :]) / (A_max - A_min))) < 2e-3:
                N_steady_count += 1
                if N_steady_count == N_stable_cycle_output:  # only need 2 periods to calculate amplitude and phase
                    time_index = p
                    print("stable temperature profile has been obtained @ iteration N= {}!".format(
                        int(p / N_one_cycle)))
            T_temp = T[p, :]

        if (p == Nt - 2) and (N_steady_count<N_stable_cycle_output):
            time_index = p
            print("Error! No stable temperature profile was obtained!")

    print(
        'alpha_r = {:.2E}, sigma_s = {:.2E}, Amax = {}, f_heating = {}, focal plane = {}, Rec = {}, T_sur1 = {:.1f}, e_front = {:.2f}.'.format(
            alpha_r, light_source_property['sigma_s'], light_source_property['Amax'], f_heating,
            vacuum_chamber_setting['focal_shift'],
            sample_information['rec_name'], vacuum_chamber_setting['T_sur1'],
            sample_information['emissivity_front']))

    return T, time_simulation, r, N_one_cycle, q_solar


def finite_difference_explicit_1D_const_cp_alpha(sample_information, vacuum_chamber_setting, solar_simulator_settings,
                       light_source_property, numerical_simulation_setting, df_solar_simulator_VQ,
                       sigma_df,code_directory,df_view_factor,df_LB_details_all):

    T_LBW = float(df_view_factor['T_LBW_C'][0]) + 273.15

    R = sample_information['R']  # sample radius
    Nr = numerical_simulation_setting['Nr']  # number of discretization along radial direction
    t_z = sample_information['t_z']  # sample thickness

    dr = R / Nr
    r = np.arange(Nr)
    rm = r * dr

    rho = sample_information['rho']
    cp_const = sample_information['cp_const']  # Cp = cp_const + cp_c1*T+ cp_c2*T**2+cp_c3*T**3, T unit in K
    cp_c1 = sample_information['cp_c1']
    cp_c2 = sample_information['cp_c2']
    cp_c3 = sample_information['cp_c3']


    T_initial = sample_information['T_initial']  # unit in K
    T_min = sample_information['T_min']

    cp = cp_const + cp_c1 * T_min + cp_c2 * T_min ** 2 + cp_c3 * T_min ** 3 # We just assume the specific heat on the sample is a constant, using properties estimated at T_min


    alpha_r = sample_information['alpha_r']

    f_heating = solar_simulator_settings['f_heating']  # periodic heating frequency
    N_cycle = numerical_simulation_setting['N_cycle']
    N_stable_cycle_output = numerical_simulation_setting['N_stable_cycle_output']

    simulated_amp_phase_extraction_method = numerical_simulation_setting['simulated_amp_phase_extraction_method']

    emissivity_front = sample_information['emissivity_front']  # assumed to be constant
    emissivity_back = sample_information['emissivity_back']  # assumed to be constant

    absorptivity_front = sample_information['absorptivity_front']  # assumed to be constant

    sigma_sb = 5.6703744 * 10 ** (-8)  # stefan-Boltzmann constant

    # The radioan absorption view factor contains the following cases:
    # (1) With light blocker: (a) Sensitivity analysis: Light blocker has the same temperature. (b) Regression analysis: Light blocker has radial varied temperature
    # (2) Without light blocker: (a) Sensitivity analysis: Surroundings are treated as the same temperature (not including glass and IR radiation shield). (b) Regression: Surrounding temperature are treated differently.
    # These cases are handled internally within the function
    # Needs to distinguish front and back
    Cm_front, Cm_back, T_LB_mean_C = radiation_absorption_view_factor_calculations(code_directory, rm, dr,
                                                                                   sample_information,
                                                                                   solar_simulator_settings,
                                                                                   vacuum_chamber_setting,
                                                                                   numerical_simulation_setting,
                                                                                   df_view_factor, df_LB_details_all)

    Fo_criteria = numerical_simulation_setting['Fo_criteria']

    dt = Fo_criteria * (dr ** 2) / (alpha_r)  # assume 15 samples per period, Fo_criteria default = 1/3

    if f_heating >0:
        t_total = 1 / f_heating * N_cycle  # total simulation time
    elif f_heating ==0:
        t_total = 100 # just simulate 100s

    Nt = int(t_total / dt)  # total number of simulation time step
    time_simulation = dt * np.arange(Nt)

    Fo_r = alpha_r * dt / dr ** 2

    T = T_initial * np.ones((Nt, Nr))
    dz = t_z
    q_solar = light_source_intensity_Amax_fV_vecterize(np.arange(Nr) * dr, np.arange(Nt) * dt,
                                                       solar_simulator_settings, vacuum_chamber_setting,
                                                       light_source_property,
                                                       df_solar_simulator_VQ, sigma_df)
    T_temp = np.zeros(Nr)
    N_steady_count = 0
    time_index = Nt - 1
    N_one_cycle = int(Nt / N_cycle)

    Cm = Cm_front + Cm_back

    for p in range(Nt - 1):  # p indicate time step

        # The node at center of the disk
        #cp_center = cp_const + cp_c1 * T[p, 0] + cp_c2 * T[p, 0] ** 2 + cp_c3 * T[p, 0] ** 3
        cp_center = cp
        T[p + 1, 0] = T[p, 0] * (1 - 4 * Fo_r) + 4 * Fo_r * T[p, 1] - sigma_sb * (
                emissivity_back + emissivity_front) * dt / (rho * cp_center * dz) * T[p, 0] ** 4 + \
                      dt / (rho * cp_center * dz) * (absorptivity_front * q_solar[p, 0] + 4 * Cm[0] / (np.pi * dr ** 2))

        # The node in the middle of disk
        #cp_mid = cp_const + cp_c1 * T[p, 1:-1] + cp_c2 * T[p, 1:-1] ** 2 + cp_c3 * T[p, 1:-1] ** 3
        cp_mid = cp
        T[p + 1, 1:-1] = T[p, 1:-1] + Fo_r * (rm[1:-1] - dr / 2) / rm[1:-1] * (T[p, 0:-2] - T[p, 1:-1]) + Fo_r * (
                rm[1:-1] + dr / 2) / rm[1:-1] * (T[p, 2:] - T[p, 1:-1]) \
                         + dt / (rho * cp_mid * dz) * (
                                 absorptivity_front * q_solar[p, 1:-1] + Cm[1:-1] / (2 * np.pi * rm[1:-1] * dr) - (
                                 emissivity_front + emissivity_back) * sigma_sb * T[p, 1:-1] ** 4)

        # The node at the edge of the disk
        #cp_edge = cp_const + cp_c1 * T[p, -1] + cp_c2 * T[p, -1] ** 2 + cp_c3 * T[p, -1] ** 3
        cp_edge = cp
        T[p + 1, -1] = T[p, -1] * (1 - 2 * Fo_r) + 2 * Fo_r * T[p, -2] + absorptivity_front * dt * q_solar[p, -1] / (
                dz * rho * cp_edge) + 2 * rm[-1] * absorptivity_front * sigma_sb * dt * T_LBW ** 4 / (
                               (rm[-1] - dr / 2) * dr * rho * cp_edge) + Cm[-1] * dt / (
                               np.pi * (rm[-1] - dr / 2) * dr * dz * rho * cp_edge) - \
                       (emissivity_front + emissivity_back) * sigma_sb * dt / (dz * rho * cp_edge) * T[p, -1] ** 4 - 2 * \
                       rm[
                           -1] * emissivity_back * sigma_sb * dt * T[p, -1] ** 4 / (
                                   (rm[-1] - dr / 2) * dr * rho * cp_edge)

        if (p > 0) & (p % N_one_cycle == 0) & (simulated_amp_phase_extraction_method != 'fft'):
            A_max = np.max(T[p - N_one_cycle:p, :], axis=0)
            A_min = np.min(T[p - N_one_cycle:p, :], axis=0)
            if np.max(np.abs((T_temp[:] - T[p, :]) / (A_max - A_min))) < 2e-3:
                N_steady_count += 1
                if N_steady_count == N_stable_cycle_output:  # only need 2 periods to calculate amplitude and phase
                    time_index = p
                    print("stable temperature profile has been obtained @ iteration N= {}!".format(
                        int(p / N_one_cycle)))
            T_temp = T[p, :]

        if (p == Nt - 2) and (N_steady_count<N_stable_cycle_output):
            time_index = p
            print("Error! No stable temperature profile was obtained!")

    print(
        'alpha_r = {:.2E}, sigma_s = {:.2E}, Amax = {}, f_heating = {}, focal plane = {}, Rec = {}, T_sur1 = {:.1f}, e_front = {:.2f}.'.format(
            alpha_r, light_source_property['sigma_s'], light_source_property['Amax'], f_heating,
            vacuum_chamber_setting['focal_shift'],
            sample_information['rec_name'], vacuum_chamber_setting['T_sur1'],
            sample_information['emissivity_front']))

    return T, time_simulation, r, N_one_cycle, q_solar

def finite_difference_explicit_1D_variable_properties(sample_information, vacuum_chamber_setting, solar_simulator_settings,
                       light_source_property, numerical_simulation_setting, df_solar_simulator_VQ,
                       sigma_df,code_directory,df_view_factor,df_LB_details_all):

    T_LBW = float(df_view_factor['T_LBW_C'][0]) + 273.15

    R = sample_information['R']  # sample radius
    Nr = numerical_simulation_setting['Nr']  # number of discretization along radial direction
    t_z = sample_information['t_z']  # sample thickness

    dr = R / Nr
    r = np.arange(Nr)
    rm = r * dr

    rho = sample_information['rho']
    cp_const = sample_information['cp_const']  # Cp = cp_const + cp_c1*T+ cp_c2*T**2+cp_c3*T**3, T unit in K
    cp_c1 = sample_information['cp_c1']
    cp_c2 = sample_information['cp_c2']
    cp_c3 = sample_information['cp_c3']

    T_initial = sample_information['T_initial']  # unit in K


    alpha_r = sample_information['alpha_r']

    f_heating = solar_simulator_settings['f_heating']  # periodic heating frequency
    N_cycle = numerical_simulation_setting['N_cycle']
    N_stable_cycle_output = numerical_simulation_setting['N_stable_cycle_output']

    simulated_amp_phase_extraction_method = numerical_simulation_setting['simulated_amp_phase_extraction_method']

    emissivity_front = sample_information['emissivity_front']  # assumed to be constant
    emissivity_back = sample_information['emissivity_back']  # assumed to be constant

    absorptivity_front = sample_information['absorptivity_front']  # assumed to be constant

    sigma_sb = 5.6703744 * 10 ** (-8)  # stefan-Boltzmann constant

    # The radioan absorption view factor contains the following cases:
    # (1) With light blocker: (a) Sensitivity analysis: Light blocker has the same temperature. (b) Regression analysis: Light blocker has radial varied temperature
    # (2) Without light blocker: (a) Sensitivity analysis: Surroundings are treated as the same temperature (not including glass and IR radiation shield). (b) Regression: Surrounding temperature are treated differently.
    # These cases are handled internally within the function
    # Needs to distinguish front and back
    Cm_front, Cm_back, T_LB_mean_C = radiation_absorption_view_factor_calculations(code_directory, rm, dr,
                                                                                   sample_information,
                                                                                   solar_simulator_settings,
                                                                                   vacuum_chamber_setting,
                                                                                   numerical_simulation_setting,
                                                                                   df_view_factor, df_LB_details_all)


    Fo_criteria = numerical_simulation_setting['Fo_criteria']
    alpha_r_A = float(sample_information['alpha_r_A'])
    alpha_r_B = float(sample_information['alpha_r_B'])

    T_min = sample_information['T_min']
    dz = t_z

    alpha_max = 1 / (alpha_r_A * T_min + alpha_r_B)

    dr = R / Nr
    dt = min(Fo_criteria * (dr ** 2) / (alpha_max),
             1 / f_heating / 15)  # assume 15 samples per period, Fo_criteria default = 1/3

    t_total = 1 / f_heating * N_cycle  # total simulation time

    Nt = int(t_total / dt)  # total number of simulation time step
    time_simulation = dt * np.arange(Nt)
    r = np.arange(Nr)
    rm = r * dr

    T = T_initial * np.ones((Nt, Nr))


    # Why is this done here???
    Amax, sigma_s, kvd, bvd = interpolate_light_source_characteristic(sigma_df, df_solar_simulator_VQ,
                                                                      light_source_property['sigma_s'],
                                                                      numerical_simulation_setting,
                                                                      vacuum_chamber_setting)
    light_source_property['Amax'] = Amax

    q_solar = light_source_intensity_Amax_fV_vecterize(np.arange(Nr) * dr, np.arange(Nt) * dt,
                                                       solar_simulator_settings, vacuum_chamber_setting,
                                                       light_source_property,
                                                       df_solar_simulator_VQ, sigma_df)
    T_temp = np.zeros(Nr)
    N_steady_count = 0
    time_index = Nt - 1
    N_one_cycle = int(Nt / N_cycle)

    Cm = Cm_front + Cm_back

    for p in range(Nt - 1):  # p indicate time step

        # The node at center of the disk
        cp_c = cp_const + cp_c1 * T[p, 0] + cp_c2 * T[p, 0] ** 2 + cp_c3 * T[p, 0] ** 3
        alpha_r_c = 1 / (alpha_r_A * T[p, 0] + alpha_r_B)

        T[p + 1, 0] = T[p, 0] * (1 - 4 * (alpha_r_c * dt / dr ** 2)) + 4 * (alpha_r_c * dt / dr ** 2) * T[
            p, 1] - sigma_sb * (emissivity_back + emissivity_front) * dt / (rho * cp_c * dz) * T[p, 0] ** 4 + \
                      dt / (rho * cp_c * dz) * (absorptivity_front * q_solar[p, 0] + 4 * Cm[0] / (np.pi * dr ** 2))

        # The node in the middle of disk
        cp_mid = cp_const + cp_c1 * T[p, 1:-1] + cp_c2 * T[p, 1:-1] ** 2 + cp_c3 * T[p, 1:-1] ** 3
        alpha_mid = 1 / (alpha_r_A * T[p, 1:-1] + alpha_r_B)
        T[p + 1, 1:-1] = T[p, 1:-1] + (alpha_mid * dt / dr ** 2) * (rm[1:-1] - dr / 2) / rm[1:-1] * (
                    T[p, 0:-2] - T[p, 1:-1]) + (alpha_mid * dt / dr ** 2) * (rm[1:-1] + dr / 2) / rm[1:-1] * (
                                     T[p, 2:] - T[p, 1:-1]) \
                         + dt / (rho * cp_mid * dz) * (
                                     absorptivity_front * q_solar[p, 1:-1] + Cm[1:-1] / (2 * np.pi * rm[1:-1] * dr) - (
                                         emissivity_front + emissivity_back) * sigma_sb * T[p, 1:-1] ** 4)

        # The node at the edge of the disk
        cp_edge = cp_const + cp_c1 * T[p, -1] + cp_c2 * T[p, -1] ** 2 + cp_c3 * T[p, -1] ** 3
        alpha_r_e = 1 / (alpha_r_A * T[p, -1] + alpha_r_B)
        T[p + 1, -1] = T[p, -1] * (1 - 2 * (alpha_r_e * dt / dr ** 2)) + 2 * (alpha_r_e * dt / dr ** 2) * T[
            p, -2] + absorptivity_front * dt * q_solar[p, -1] / (dz * rho * cp_edge) + 2 * rm[
                           -1] * absorptivity_front * sigma_sb * dt * T_LBW ** 4 / (
                                   (rm[-1] - dr / 2) * dr * rho * cp_edge) + Cm[-1] * dt / (
                                   np.pi * (rm[-1] - dr / 2) * dr * dz * rho * cp_edge) - \
                       (emissivity_front + emissivity_back) * sigma_sb * dt / (dz * rho * cp_edge) * T[p, -1] ** 4 - 2 * \
                       rm[-1] * emissivity_back * sigma_sb * dt * T[p, -1] ** 4 / (
                                   (rm[-1] - dr / 2) * dr * rho * cp_edge)

        if (p > 0) & (p % N_one_cycle == 0) & (simulated_amp_phase_extraction_method != 'fft'):
            A_max = np.max(T[p - N_one_cycle:p, :], axis=0)
            A_min = np.min(T[p - N_one_cycle:p, :], axis=0)
            if np.max(np.abs((T_temp[:] - T[p, :]) / (A_max - A_min))) < 2e-3:
                N_steady_count += 1
                if N_steady_count == N_stable_cycle_output:  # only need 2 periods to calculate amplitude and phase
                    time_index = p
                    print("stable temperature profile has been obtained @ iteration N= {}!".format(
                        int(p / N_one_cycle)))
            T_temp = T[p, :]

        if (p == Nt - 2) and (N_steady_count<N_stable_cycle_output):
            time_index = p
            print("Error! No stable temperature profile was obtained!")

    print(
        'alpha_r = {:.2E}, sigma_s = {:.2E}, Amax = {}, f_heating = {}, focal plane = {}, Rec = {}, T_sur1 = {:.1f}, e_front = {:.2f}.'.format(
            alpha_r, light_source_property['sigma_s'], light_source_property['Amax'], f_heating,
            vacuum_chamber_setting['focal_shift'],
            sample_information['rec_name'], T_LB_mean_C,
            sample_information['emissivity_front']))

    return T, time_simulation, r, N_one_cycle, q_solar


def finite_difference_explicit_2D_const_alpha(sample_information, vacuum_chamber_setting, solar_simulator_settings,
                       light_source_property, numerical_simulation_setting, df_solar_simulator_VQ,
                       sigma_df,code_directory,df_view_factor,df_LB_details_all):

    T_LBW = float(df_view_factor['T_LBW_C'][0]) + 273.15

    R = sample_information['R'] # sample radius
    Nr = numerical_simulation_setting['Nr'] # number of discretization along radial direction
    t_z = sample_information['t_z'] # sample thickness

    dr = R / Nr
    r = np.arange(Nr)
    rm = r * dr

    rho = sample_information['rho']
    cp_const = sample_information['cp_const']  # Cp = cp_const + cp_c1*T+ cp_c2*T**2+cp_c3*T**3, T unit in K
    cp_c1 = sample_information['cp_c1']
    cp_c2 = sample_information['cp_c2']
    cp_c3 = sample_information['cp_c3']
    alpha_r = sample_information['alpha_r']
    alpha_z = sample_information['alpha_z']


    T_initial = sample_information['T_initial']  # unit in K


    f_heating = solar_simulator_settings['f_heating']  # periodic heating frequency
    N_cycle = numerical_simulation_setting['N_cycle']
    N_stable_cycle_output = numerical_simulation_setting['N_stable_cycle_output']

    simulated_amp_phase_extraction_method = numerical_simulation_setting['simulated_amp_phase_extraction_method']

    emissivity_front = sample_information['emissivity_front']  # assumed to be constant
    emissivity_back = sample_information['emissivity_back']  # assumed to be constant

    absorptivity_front = sample_information['absorptivity_front']  # assumed to be constant

    sigma_sb = 5.6703744 * 10 ** (-8)  # stefan-Boltzmann constant


    # The radioan absorption view factor contains the following cases:
    # (1) With light blocker: (a) Sensitivity analysis: Light blocker has the same temperature. (b) Regression analysis: Light blocker has radial varied temperature
    # (2) Without light blocker: (a) Sensitivity analysis: Surroundings are treated as the same temperature (not including glass and IR radiation shield). (b) Regression: Surrounding temperature are treated differently.
    # These cases are handled internally within the function
    # Needs to distinguish front and back
    Cm_front,Cm_back, T_LB_mean_C = radiation_absorption_view_factor_calculations(code_directory, rm, dr, sample_information,
                                                                    solar_simulator_settings,
                                                                    vacuum_chamber_setting,
                                                                    numerical_simulation_setting,
                                                                    df_view_factor, df_LB_details_all)


    Fo_criteria = numerical_simulation_setting['Fo_criteria']


    Nz = numerical_simulation_setting['Nz']
    dz = t_z / Nz



    dr = R / Nr
    dt = min(Fo_criteria * (dr ** 2) / (alpha_r), Fo_criteria * (dz ** 2) / (alpha_z))

    t_total = 1 / f_heating * N_cycle  # total simulation time

    Nt = int(t_total / dt)  # total number of simulation time step
    time_simulation = dt * np.arange(Nt)
    r = np.arange(Nr)
    rm = r * dr

    T = T_initial * np.ones((Nt, Nr, Nz))

    q_solar = light_source_intensity_Amax_fV_vecterize(np.arange(Nr) * dr, np.arange(Nt) * dt,
                                                       solar_simulator_settings, vacuum_chamber_setting,
                                                       light_source_property,
                                                       df_solar_simulator_VQ, sigma_df)

    T_temp = np.zeros(Nr)
    N_steady_count = 0
    N_one_cycle = int(Nt / N_cycle)

    for p in range(Nt - 1):  # p indicate time step

        # This scheme update the entire nodes

        # node V1
        cp_V1 = cp_const + cp_c1 * T[p, 0, 0] + cp_c2 * T[p, 0, 0] ** 2 + cp_c3 * T[p, 0, 0] ** 3
        alpha_r_V1 = alpha_r  # 1 / (alpha_r_A * T[p, 0, 0] + alpha_r_B)
        alpha_z_V1 = alpha_z  # 1 / (alpha_z_A * T[p, 0, 0] + alpha_z_B)
        T[p + 1, 0, 0] = T[p, 0, 0] + 4 * alpha_r_V1 * dt / dr ** 2 * (
                T[p, 1, 0] - T[p, 0, 0]) + 2 * alpha_z_V1 * dt / dz ** 2 * (T[p, 0, 1] - T[p, 0, 0]) \
                         + dt / (dz * rho * cp_V1) * (
                                 2 * absorptivity_front * q_solar[p, 0] + 4 * Cm_front[0] / (
                                 np.pi * dr ** 2) - 2 * emissivity_front * sigma_sb * T[p, 0, 0] ** 4)

        # node E4
        cp_E4 = cp_const + cp_c1 * T[p, 0, 1:-1] + cp_c2 * T[p, 0, 1:-1] ** 2 + cp_c3 * T[p, 0, 1:-1] ** 3
        alpha_r_E4 = alpha_r  # 1 / (alpha_r_A * T[p, 0, 1:-1] + alpha_r_B)
        alpha_z_E4 = alpha_z  # 1 / (alpha_z_A * T[p, 0, 1:-1] + alpha_z_B)
        T[p + 1, 0, 1:-1] = T[p, 0, 1:-1] + alpha_z_E4 * dt / dz ** 2 * (
                T[p, 0, 0:-2] + T[p, 0, 2:] - 2 * T[p, 0, 1:-1]) + 4 * alpha_r_E4 * dt / dr ** 2 * (
                                    T[p, 1, 1:-1] - T[p, 0, 1:-1])

        # node V4
        cp_V4 = cp_const + cp_c1 * T[p, 0, -1] + cp_c2 * T[p, 0, -1] ** 2 + cp_c3 * T[p, 0, -1] ** 3
        alpha_r_V4 = alpha_r  # 1 / (alpha_r_A * T[p, 0, -1] + alpha_r_B)
        alpha_z_V4 = alpha_z  # 1 / (alpha_z_A * T[p, 0, -1] + alpha_z_B)
        T[p + 1, 0, -1] = T[p, 0, -1] + 4 * alpha_r_V4 * dt / dr ** 2 * (
                T[p, 1, -1] - T[p, 0, -1]) + 2 * alpha_z_V4 * dt / dz ** 2 * (T[p, 0, -2] - T[p, 0, -1]) \
                          + dt / (dz * rho * cp_V4) * (
                                  4 * Cm_back[0] / (np.pi * dr ** 2) - 2 * emissivity_back * sigma_sb * T[
                              p, 0, -1] ** 4)

        # node E1
        cp_E1 = cp_const + cp_c1 * T[p, 1:-1, 0] + cp_c2 * T[p, 1:-1, 0] ** 2 + cp_c3 * T[p, 1:-1, 0] ** 3
        # print(cp_E1)
        alpha_r_E1 = alpha_r  # 1 / (alpha_r_A * T[p, 1:-1, 0] + alpha_r_B)
        alpha_z_E1 = alpha_z  # 1 / (alpha_z_A * T[p, 1:-1, 0] + alpha_z_B)

        T[p + 1, 1:-1, 0] = T[p, 1:-1, 0] + dt / (dz * rho * cp_E1) * (
                2 * absorptivity_front * q_solar[p, 1:-1] + Cm_front[1:-1] / (
                np.pi * rm[1:-1] * dr) - 2 * emissivity_front * sigma_sb * T[p, 1:-1, 0] ** 4) \
                            + alpha_r_E1 * dt / (rm[1:-1] * dr ** 2) * (
                                    (rm[1:-1] - dr / 2) * (T[p, 0:-2, 0] - T[p, 1:-1, 0]) + (
                                    rm[1:-1] + dr / 2) * (T[p, 2:, 0] - T[p, 1:-1, 0])) \
                            + alpha_z_E1 * dt / (dz ** 2) * (T[p, 1:-1, 1] - T[p, 1:-1, 0])

        # node B, this need to be tested very carefully
        cp_B = cp_const + cp_c1 * T[p, 1:-1, 1:-1] + cp_c2 * T[p, 1:-1, 1:-1] ** 2 + cp_c3 * T[p, 1:-1, 1:-1] ** 3
        alpha_r_Body = alpha_r  # 1 / (alpha_r_A * T[p, 1:-1, 1:-1] + alpha_r_B)
        alpha_z_Body = alpha_z  # 1 / (alpha_z_A * T[p, 1:-1, 1:-1] + alpha_z_B)
        T[p + 1, 1:-1, 1:-1] = T[p, 1:-1, 1:-1] + (alpha_r_Body * dt / (rm[1:-1] * dr ** 2) * (
                    (rm[1:-1] - dr / 2) * (T[p, 0:-2, 1:-1] - T[p, 1:-1, 1:-1]).T + (rm[1:-1] + dr / 2) * (
                        T[p, 2:, 1:-1] - T[p, 1:-1, 1:-1]).T)).T \
                               + alpha_z_Body * dt / dz ** 2 * (
                                       T[p, 1:-1, 0:-2] + T[p, 1:-1, 2:] - 2 * T[p, 1:-1, 1:-1])

        # node E3
        cp_E3 = cp_const + cp_c1 * T[p, 1:-1, -1] + cp_c2 * T[p, 1:-1, -1] ** 2 + cp_c3 * T[p, 1:-1, -1] ** 3
        alpha_r_E3 = alpha_r  # 1 / (alpha_r_A * T[p, 1:-1, -1] + alpha_r_B)
        alpha_z_E3 = alpha_z  # 1 / (alpha_z_A * T[p, 1:-1, -1] + alpha_z_B)
        T[p + 1, 1:-1, -1] = T[p, 1:-1, -1] + dt / (dz * rho * cp_E3) * (
                Cm_back[1:-1] / (np.pi * rm[1:-1] * dr) - 2 * emissivity_back * sigma_sb * T[p, 1:-1,
                                                                                           -1] ** 4) \
                             + alpha_r_E3 * dt / (rm[1:-1] * dr ** 2) * (
                                     (rm[1:-1] - dr / 2) * (T[p, 0:-2, -1] - T[p, 1:-1, -1]) + (
                                     rm[1:-1] + dr / 2) * (T[p, 2:, -1] - T[p, 1:-1, -1])) \
                             + alpha_z_E3 * dt / dz ** 2 * (T[p, 1:-1, -2] - T[p, 1:-1, -1])

        # node V2
        cp_V2 = cp_const + cp_c1 * T[p, -1, 0] + cp_c2 * T[p, -1, 0] ** 2 + cp_c3 * T[p, -1, 0] ** 3
        alpha_r_V2 = alpha_r  # 1 / (alpha_r_A * T[p, -1, 0] + alpha_r_B)
        alpha_z_V2 = alpha_z  # 1 / (alpha_z_A * T[p, -1, 0] + alpha_z_B)
        T[p + 1, -1, 0] = T[p, -1, 0] + dt / (dz * rho * cp_V2) * (
                2 * absorptivity_front * q_solar[p, -1] + 2 * Cm_front[-1] / (
                np.pi * rm[-1] * dr) - 2 * sigma_sb * emissivity_front * T[p, -1, 0] ** 4) \
                          + dt / (dr * rho * cp_V2) * (
                                  2 * sigma_sb * T_LBW ** 4 - 2 * sigma_sb * emissivity_front * T[
                              p, -1, 0] ** 4) + 2 * alpha_r_V2 * dt / dr ** 2 * (T[p, -2, 0] - T[p, -1, 0]) \
                          + 2 * alpha_z_V2 * dt / dz ** 2 * (T[p, -1, 1] - T[p, -1, 0])

        # node E2
        cp_E2 = cp_const + cp_c1 * T[p, -1, 1:-1] + cp_c2 * T[p, -1, 1:-1] ** 2 + cp_c3 * T[p, -1, 1:-1] ** 3
        alpha_r_E2 = alpha_r  # 1 / (alpha_r_A * T[p, -1, 1:-1] + alpha_r_B)
        alpha_z_E2 = alpha_z  # 1 / (alpha_z_A * T[p, -1, 1:-1] + alpha_z_B)
        T[p + 1, -1, 1:-1] = T[p, -1, 1:-1] + 2 * alpha_r_E2 * dt / dr ** 2 * (
                T[p, -2, 1:-1] - T[p, -1, 1:-1]) + 2 * alpha_z_E2 * dt / dz ** 2 * (
                                     T[p, -1, 0:-2] + T[p, -1, 2:] - 2 * T[p, -1, 1:-1]) \
                             + 2 * dt / (dr * rho * cp_E2) * (-emissivity_front * sigma_sb * T[
                                                                                             p, -1,
                                                                                             1:-1] ** 4 + absorptivity_front * sigma_sb * T_LBW ** 4)

        # node V3
        cp_V3 = cp_const + cp_c1 * T[p, -1, -1] + cp_c2 * T[p, -1, -1] ** 2 + cp_c3 * T[p, -1, -1] ** 3
        alpha_r_V3 = alpha_r  # 1 / (alpha_r_A * T[p, -1, -1] + alpha_r_B)
        alpha_z_V3 = alpha_z  # 1 / (alpha_z_A * T[p, -1, -1] + alpha_z_B)
        T[p + 1, -1, -1] = T[p, -1, -1] + dt / (dz * rho * cp_V3) * (
                2 * Cm_back[-1] / (np.pi * rm[-1] * dr) - 2 * sigma_sb * emissivity_back * T[
            p, -1, -1] ** 4) \
                           + dt / (dr * rho * cp_V3) * (
                                   2 * sigma_sb * T_LBW ** 4 - 2 * sigma_sb * emissivity_back * T[
                               p, -1, -1] ** 4) + 2 * alpha_r_V3 * dt / dr ** 2 * (
                                   T[p, -2, -1] - T[p, -1, -1]) \
                           + 2 * alpha_z_V3 * dt / dz ** 2 * (T[p, -1, -2] - T[p, -1, -1])

        if (p > 0) & (p % N_one_cycle == 0) & (simulated_amp_phase_extraction_method != 'fft'):
            A_max = np.max(T[p - N_one_cycle:p, :, -1], axis=0)
            A_min = np.min(T[p - N_one_cycle:p, :, -1], axis=0)
            if np.max(np.abs((T_temp[:] - T[p, :, -1]) / (A_max - A_min))) < 2e-3:
                N_steady_count += 1
                if N_steady_count == N_stable_cycle_output:  # only need 2 periods to calculate amplitude and phase
                    time_index = p
                    print("stable temperature profile has been obtained @ iteration N= {}!".format(
                        int(p / N_one_cycle)))
            T_temp = T[p, :, -1]

        if (p == Nt - 2) and (N_steady_count < N_stable_cycle_output):
            time_index = p
            # T_temp = T[p, :]
            print("Error! No stable temperature profile was obtained!")

    print(
        'alpha_r = {:.2E}, focal plane = {}, sigma_s = {:.2E}, f_heating = {}, Rec = {}, T_sur1 = {:.1f}, e_front = {:.2f}.'.format(
            alpha_r, vacuum_chamber_setting['focal_shift'], light_source_property['sigma_s'], f_heating,
            sample_information['rec_name'], vacuum_chamber_setting['T_sur1'],
            sample_information['emissivity_front']))

    return T, time_simulation, r, N_one_cycle, q_solar




def finite_difference_explicit_2D_variable_properties(sample_information, vacuum_chamber_setting, solar_simulator_settings,
                       light_source_property, numerical_simulation_setting, df_solar_simulator_VQ,
                       sigma_df,code_directory,df_view_factor,df_LB_details_all):

    T_LBW = float(df_view_factor['T_LBW_C'][0]) + 273.15

    R = sample_information['R'] # sample radius
    Nr = numerical_simulation_setting['Nr'] # number of discretization along radial direction
    t_z = sample_information['t_z'] # sample thickness

    dr = R / Nr
    r = np.arange(Nr)
    rm = r * dr

    rho = sample_information['rho']
    cp_const = sample_information['cp_const']  # Cp = cp_const + cp_c1*T+ cp_c2*T**2+cp_c3*T**3, T unit in K
    cp_c1 = sample_information['cp_c1']
    cp_c2 = sample_information['cp_c2']
    cp_c3 = sample_information['cp_c3']
    alpha_r = sample_information['alpha_r']
    alpha_z = sample_information['alpha_z']


    T_initial = sample_information['T_initial']  # unit in K


    f_heating = solar_simulator_settings['f_heating']  # periodic heating frequency
    N_cycle = numerical_simulation_setting['N_cycle']
    N_stable_cycle_output = numerical_simulation_setting['N_stable_cycle_output']

    simulated_amp_phase_extraction_method = numerical_simulation_setting['simulated_amp_phase_extraction_method']

    emissivity_front = sample_information['emissivity_front']  # assumed to be constant
    emissivity_back = sample_information['emissivity_back']  # assumed to be constant

    absorptivity_front = sample_information['absorptivity_front']  # assumed to be constant

    sigma_sb = 5.6703744 * 10 ** (-8)  # stefan-Boltzmann constant


    # The radioan absorption view factor contains the following cases:
    # (1) With light blocker: (a) Sensitivity analysis: Light blocker has the same temperature. (b) Regression analysis: Light blocker has radial varied temperature
    # (2) Without light blocker: (a) Sensitivity analysis: Surroundings are treated as the same temperature (not including glass and IR radiation shield). (b) Regression: Surrounding temperature are treated differently.
    # These cases are handled internally within the function
    # Needs to distinguish front and back
    Cm_front,Cm_back, T_LB_mean_C = radiation_absorption_view_factor_calculations(code_directory, rm, dr, sample_information,
                                                                    solar_simulator_settings,
                                                                    vacuum_chamber_setting,
                                                                    numerical_simulation_setting,
                                                                    df_view_factor, df_LB_details_all)

    Fo_criteria = numerical_simulation_setting['Fo_criteria']
    alpha_r_A = float(sample_information['alpha_r_A'])
    alpha_r_B = float(sample_information['alpha_r_B'])

    alpha_z_A = float(sample_information['alpha_z_A'])
    alpha_z_B = float(sample_information['alpha_z_B'])
    Nz = numerical_simulation_setting['Nz']
    dz = t_z / Nz

    T_min = sample_information['T_min']
    alpha_r_max = 1 / (alpha_r_A * T_min + alpha_r_B)
    alpha_z_max = 1 / (alpha_z_A * T_min + alpha_z_B)

    dr = R / Nr
    dt = min(Fo_criteria * (dr ** 2) / (alpha_r_max), Fo_criteria * (dz ** 2) / (alpha_z_max))

    t_total = 1 / f_heating * N_cycle  # total simulation time

    Nt = int(t_total / dt)  # total number of simulation time step
    time_simulation = dt * np.arange(Nt)
    r = np.arange(Nr)
    rm = r * dr

    T = T_initial * np.ones((Nt, Nr, Nz))

    q_solar = light_source_intensity_Amax_fV_vecterize(np.arange(Nr) * dr, np.arange(Nt) * dt,
                                                       solar_simulator_settings, vacuum_chamber_setting,
                                                       light_source_property,
                                                       df_solar_simulator_VQ, sigma_df)

    T_temp = np.zeros(Nr)
    N_steady_count = 0
    time_index = Nt - 1
    N_one_cycle = int(Nt / N_cycle)

    for p in range(Nt - 1):  # p indicate time step

        # This scheme update the entire nodes

        # node V1
        cp_V1 = cp_const + cp_c1 * T[p, 0, 0] + cp_c2 * T[p, 0, 0] ** 2 + cp_c3 * T[p, 0, 0] ** 3
        alpha_r_V1 = 1 / (alpha_r_A * T[p, 0, 0] + alpha_r_B)
        alpha_z_V1 = 1 / (alpha_z_A * T[p, 0, 0] + alpha_z_B)
        T[p + 1, 0, 0] = T[p, 0, 0] + 4 * alpha_r_V1 * dt / dr ** 2 * (
                T[p, 1, 0] - T[p, 0, 0]) + 2 * alpha_z_V1 * dt / dz ** 2 * (T[p, 0, 1] - T[p, 0, 0]) \
                         + dt / (dz * rho * cp_V1) * (
                                 2 * absorptivity_front * q_solar[p, 0] + 4 * Cm_front[0] / (
                                 np.pi * dr ** 2) - 2 * emissivity_front * sigma_sb * T[p, 0, 0] ** 4)

        # node E4
        cp_E4 = cp_const + cp_c1 * T[p, 0, 1:-1] + cp_c2 * T[p, 0, 1:-1] ** 2 + cp_c3 * T[p, 0, 1:-1] ** 3
        alpha_r_E4 = 1 / (alpha_r_A * T[p, 0, 1:-1] + alpha_r_B)
        alpha_z_E4 = 1 / (alpha_z_A * T[p, 0, 1:-1] + alpha_z_B)
        T[p + 1, 0, 1:-1] = T[p, 0, 1:-1] + alpha_z_E4 * dt / dz ** 2 * (
                T[p, 0, 0:-2] + T[p, 0, 2:] - 2 * T[p, 0, 1:-1]) + 4 * alpha_r_E4 * dt / dr ** 2 * (
                                    T[p, 1, 1:-1] - T[p, 0, 1:-1])

        # node V4
        cp_V4 = cp_const + cp_c1 * T[p, 0, -1] + cp_c2 * T[p, 0, -1] ** 2 + cp_c3 * T[p, 0, -1] ** 3
        alpha_r_V4 = 1 / (alpha_r_A * T[p, 0, -1] + alpha_r_B)
        alpha_z_V4 = 1 / (alpha_z_A * T[p, 0, -1] + alpha_z_B)
        T[p + 1, 0, -1] = T[p, 0, -1] + 4 * alpha_r_V4 * dt / dr ** 2 * (
                T[p, 1, -1] - T[p, 0, -1]) + 2 * alpha_z_V4 * dt / dz ** 2 * (T[p, 0, -2] - T[p, 0, -1]) \
                          + dt / (dz * rho * cp_V4) * (
                                  4 * Cm_back[0] / (np.pi * dr ** 2) - 2 * emissivity_back * sigma_sb * T[
                              p, 0, -1] ** 4)

        # node E1
        cp_E1 = cp_const + cp_c1 * T[p, 1:-1, 0] + cp_c2 * T[p, 1:-1, 0] ** 2 + cp_c3 * T[p, 1:-1, 0] ** 3
        # print(cp_E1)
        alpha_r_E1 = 1 / (alpha_r_A * T[p, 1:-1, 0] + alpha_r_B)
        alpha_z_E1 = 1 / (alpha_z_A * T[p, 1:-1, 0] + alpha_z_B)

        T[p + 1, 1:-1, 0] = T[p, 1:-1, 0] + dt / (dz * rho * cp_E1) * (
                2 * absorptivity_front * q_solar[p, 1:-1] + Cm_front[1:-1] / (
                np.pi * rm[1:-1] * dr) - 2 * emissivity_front * sigma_sb * T[p, 1:-1, 0] ** 4) \
                            + alpha_r_E1 * dt / (rm[1:-1] * dr ** 2) * (
                                    (rm[1:-1] - dr / 2) * (T[p, 0:-2, 0] - T[p, 1:-1, 0]) + (
                                    rm[1:-1] + dr / 2) * (T[p, 2:, 0] - T[p, 1:-1, 0])) \
                            + alpha_z_E1 * dt / (dz ** 2) * (T[p, 1:-1, 1] - T[p, 1:-1, 0])

        # node B, this need to be tested very carefully
        cp_B = cp_const + cp_c1 * T[p, 1:-1, 1:-1] + cp_c2 * T[p, 1:-1, 1:-1] ** 2 + cp_c3 * T[p, 1:-1, 1:-1] ** 3
        alpha_r_Body = 1 / (alpha_r_A * T[p, 1:-1, 1:-1] + alpha_r_B)
        alpha_z_Body = 1 / (alpha_z_A * T[p, 1:-1, 1:-1] + alpha_z_B)
        T[p + 1, 1:-1, 1:-1] = T[p, 1:-1, 1:-1] + (alpha_r_Body.T * dt / (rm[1:-1] * dr ** 2) * (
                (rm[1:-1] - dr / 2) * (T[p, 0:-2, 1:-1] - T[p, 1:-1, 1:-1]).T + (rm[1:-1] + dr / 2) * (
                    T[p, 2:, 1:-1] - T[p, 1:-1, 1:-1]).T)).T \
                               + alpha_z_Body * dt / dz ** 2 * (
                                           T[p, 1:-1, 0:-2] + T[p, 1:-1, 2:] - 2 * T[p, 1:-1, 1:-1])

        # node E3
        cp_E3 = cp_const + cp_c1 * T[p, 1:-1, -1] + cp_c2 * T[p, 1:-1, -1] ** 2 + cp_c3 * T[p, 1:-1, -1] ** 3
        alpha_r_E3 = 1 / (alpha_r_A * T[p, 1:-1, -1] + alpha_r_B)
        alpha_z_E3 = 1 / (alpha_z_A * T[p, 1:-1, -1] + alpha_z_B)
        T[p + 1, 1:-1, -1] = T[p, 1:-1, -1] + dt / (dz * rho * cp_E3) * (
                Cm_back[1:-1] / (np.pi * rm[1:-1] * dr) - 2 * emissivity_back * sigma_sb * T[p, 1:-1,
                                                                                           -1] ** 4) \
                             + alpha_r_E3 * dt / (rm[1:-1] * dr ** 2) * (
                                     (rm[1:-1] - dr / 2) * (T[p, 0:-2, -1] - T[p, 1:-1, -1]) + (
                                     rm[1:-1] + dr / 2) * (T[p, 2:, -1] - T[p, 1:-1, -1])) \
                             + alpha_z_E3 * dt / dz ** 2 * (T[p, 1:-1, -2] - T[p, 1:-1, -1])

        # node V2
        cp_V2 = cp_const + cp_c1 * T[p, -1, 0] + cp_c2 * T[p, -1, 0] ** 2 + cp_c3 * T[p, -1, 0] ** 3
        alpha_r_V2 = 1 / (alpha_r_A * T[p, -1, 0] + alpha_r_B)
        alpha_z_V2 = 1 / (alpha_z_A * T[p, -1, 0] + alpha_z_B)
        T[p + 1, -1, 0] = T[p, -1, 0] + dt / (dz * rho * cp_V2) * (
                2 * absorptivity_front * q_solar[p, -1] + 2 * Cm_front[-1] / (
                np.pi * rm[-1] * dr) - 2 * sigma_sb * emissivity_front * T[p, -1, 0] ** 4) \
                          + dt / (dr * rho * cp_V2) * (
                                  2 * sigma_sb * T_LBW ** 4 - 2 * sigma_sb * emissivity_front * T[
                              p, -1, 0] ** 4) + 2 * alpha_r_V2 * dt / dr ** 2 * (T[p, -2, 0] - T[p, -1, 0]) \
                          + 2 * alpha_z_V2 * dt / dz ** 2 * (T[p, -1, 1] - T[p, -1, 0])

        # node E2
        cp_E2 = cp_const + cp_c1 * T[p, -1, 1:-1] + cp_c2 * T[p, -1, 1:-1] ** 2 + cp_c3 * T[p, -1, 1:-1] ** 3
        alpha_r_E2 = 1 / (alpha_r_A * T[p, -1, 1:-1] + alpha_r_B)
        alpha_z_E2 = 1 / (alpha_z_A * T[p, -1, 1:-1] + alpha_z_B)
        T[p + 1, -1, 1:-1] = T[p, -1, 1:-1] + 2 * alpha_r_E2 * dt / dr ** 2 * (
                T[p, -2, 1:-1] - T[p, -1, 1:-1]) + 2 * alpha_z_E2 * dt / dz ** 2 * (
                                     T[p, -1, 0:-2] + T[p, -1, 2:] - 2 * T[p, -1, 1:-1]) \
                             + 2 * dt / (dr * rho * cp_E2) * (-emissivity_front * sigma_sb * T[
                                                                                             p, -1,
                                                                                             1:-1] ** 4 + absorptivity_front * sigma_sb * T_LBW ** 4)

        # node V3
        cp_V3 = cp_const + cp_c1 * T[p, -1, -1] + cp_c2 * T[p, -1, -1] ** 2 + cp_c3 * T[p, -1, -1] ** 3
        alpha_r_V3 = 1 / (alpha_r_A * T[p, -1, -1] + alpha_r_B)
        alpha_z_V3 = 1 / (alpha_z_A * T[p, -1, -1] + alpha_z_B)
        T[p + 1, -1, -1] = T[p, -1, -1] + dt / (dz * rho * cp_V3) * (
                2 * Cm_back[-1] / (np.pi * rm[-1] * dr) - 2 * sigma_sb * emissivity_back * T[
            p, -1, -1] ** 4) \
                           + dt / (dr * rho * cp_V3) * (
                                   2 * sigma_sb * T_LBW ** 4 - 2 * sigma_sb * emissivity_back * T[
                               p, -1, -1] ** 4) + 2 * alpha_r_V3 * dt / dr ** 2 * (
                                   T[p, -2, -1] - T[p, -1, -1]) \
                           + 2 * alpha_z_V3 * dt / dz ** 2 * (T[p, -1, -2] - T[p, -1, -1])


        if (p > 0) & (p % N_one_cycle == 0) & (simulated_amp_phase_extraction_method != 'fft'):
            A_max = np.max(T[p - N_one_cycle:p, :, -1], axis=0)
            A_min = np.min(T[p - N_one_cycle:p, :, -1], axis=0)
            if np.max(np.abs((T_temp[:] - T[p, :, -1]) / (A_max - A_min))) < 2e-3:
                N_steady_count += 1
                if N_steady_count == N_stable_cycle_output:  # only need 2 periods to calculate amplitude and phase
                    time_index = p
                    print("stable temperature profile has been obtained @ iteration N= {}!".format(
                        int(p / N_one_cycle)))
            T_temp = T[p, :, -1]

        if (p == Nt - 2) and (N_steady_count < N_stable_cycle_output):
            time_index = p
            # T_temp = T[p, :]
            print("Error! No stable temperature profile was obtained!")

    print(
        'alpha_r = {:.2E}, focal plane = {}, sigma_s = {:.2E}, f_heating = {}, Rec = {}, T_sur1 = {:.1f}, e_front = {:.2f}.'.format(
            alpha_r, vacuum_chamber_setting['focal_shift'], light_source_property['sigma_s'], f_heating,
            sample_information['rec_name'], vacuum_chamber_setting['T_sur1'],
            sample_information['emissivity_front']))

    return T, time_simulation, r, N_one_cycle, q_solar


def radial_finite_difference_explicit(sample_information, vacuum_chamber_setting, solar_simulator_settings,
                       light_source_property, numerical_simulation_setting, df_solar_simulator_VQ,
                       sigma_df,code_directory,df_view_factor,df_LB_details_all):

    if (numerical_simulation_setting['analysis_mode'] == 'sensitivity') or (
                (numerical_simulation_setting['analysis_mode'] == 'regression') and (numerical_simulation_setting['regression_parameter'] == 'alpha_r')):
            # For sensitivity analysis and regression over thermal diffusivity, thermal diffusivity is treated as a constant throughout the sample
        if numerical_simulation_setting['axial_conduction'] == False:
            T, time_simulation, r, N_one_cycle, q_solar = finite_difference_explicit_1D_const_alpha(sample_information, vacuum_chamber_setting, solar_simulator_settings,
                       light_source_property, numerical_simulation_setting, df_solar_simulator_VQ,
                       sigma_df,code_directory,df_view_factor,df_LB_details_all)
        else:
            T, time_simulation, r, N_one_cycle, q_solar = finite_difference_explicit_2D_const_alpha(
                    sample_information, vacuum_chamber_setting, solar_simulator_settings,
                    light_source_property, numerical_simulation_setting, df_solar_simulator_VQ,
                    sigma_df, code_directory, df_view_factor, df_LB_details_all)

    elif (numerical_simulation_setting['analysis_mode'] != 'sensitivity') and (numerical_simulation_setting['analysis_mode'] != 'validation') and (numerical_simulation_setting['regression_parameter'] != 'alpha_r'):
        if numerical_simulation_setting['axial_conduction'] == False:
            T, time_simulation, r, N_one_cycle, q_solar = finite_difference_explicit_1D_variable_properties(sample_information, vacuum_chamber_setting, solar_simulator_settings,
                       light_source_property, numerical_simulation_setting, df_solar_simulator_VQ,
                       sigma_df,code_directory,df_view_factor,df_LB_details_all)
        else:
            T, time_simulation, r, N_one_cycle, q_solar = finite_difference_explicit_2D_variable_properties(
                    sample_information, vacuum_chamber_setting, solar_simulator_settings,
                    light_source_property, numerical_simulation_setting, df_solar_simulator_VQ,
                    sigma_df, code_directory, df_view_factor, df_LB_details_all)

    elif (numerical_simulation_setting['analysis_mode'] == 'validation'):
        if numerical_simulation_setting['axial_conduction'] == False:
            T, time_simulation, r, N_one_cycle, q_solar = finite_difference_explicit_1D_const_cp_alpha(sample_information, vacuum_chamber_setting, solar_simulator_settings,
                       light_source_property, numerical_simulation_setting, df_solar_simulator_VQ,
                       sigma_df,code_directory,df_view_factor,df_LB_details_all)

    return T, time_simulation, r, N_one_cycle, q_solar




#
# #old function in Nov-8-2020
# def radial_finite_difference_explicit(sample_information, vacuum_chamber_setting, solar_simulator_settings,
#                        light_source_property, numerical_simulation_setting, df_solar_simulator_VQ,
#                        sigma_df,code_directory,df_view_factor,df_LB_details_all):
#
#     #df_view_factor = pd.read_csv(code_directory + "sample specifications//view factors and T.csv")
#
#     T_LBW = float(df_view_factor['T_LBW_C'][0]) + 273.15
#
#     R = sample_information['R'] # sample radius
#     Nr = numerical_simulation_setting['Nr'] # number of discretization along radial direction
#     t_z = sample_information['t_z'] # sample thickness
#
#     dr = R / Nr
#     r = np.arange(Nr)
#     rm = r * dr
#
#     rho = sample_information['rho']
#     cp_const = sample_information['cp_const']  # Cp = cp_const + cp_c1*T+ cp_c2*T**2+cp_c3*T**3, T unit in K
#     cp_c1 = sample_information['cp_c1']
#     cp_c2 = sample_information['cp_c2']
#     cp_c3 = sample_information['cp_c3']
#     alpha_r = sample_information['alpha_r']
#     alpha_z = sample_information['alpha_z']
#     R0 = vacuum_chamber_setting['R0']
#
#     T_initial = sample_information['T_initial']  # unit in K
#     # k = alpha*rho*cp
#     cp_test = cp_const + cp_c1 * T_initial + cp_c2 * T_initial ** 2 + cp_c3 * T_initial ** 3
#
#     f_heating = solar_simulator_settings['f_heating']  # periodic heating frequency
#     N_cycle = numerical_simulation_setting['N_cycle']
#     N_stable_cycle_output = numerical_simulation_setting['N_stable_cycle_output']
#
#     simulated_amp_phase_extraction_method = numerical_simulation_setting['simulated_amp_phase_extraction_method']
#
#     emissivity_front = sample_information['emissivity_front']  # assumed to be constant
#     emissivity_back = sample_information['emissivity_back']  # assumed to be constant
#
#     absorptivity_front = sample_information['absorptivity_front']  # assumed to be constant
#
#     sigma_sb = 5.6703744 * 10 ** (-8)  # stefan-Boltzmann constant
#
#
#     # The radioan absorption view factor contains the following cases:
#     # (1) With light blocker: (a) Sensitivity analysis: Light blocker has the same temperature. (b) Regression analysis: Light blocker has radial varied temperature
#     # (2) Without light blocker: (a) Sensitivity analysis: Surroundings are treated as the same temperature (not including glass and IR radiation shield). (b) Regression: Surrounding temperature are treated differently.
#     # These cases are handled internally within the function
#     # Needs to distinguish front and back
#     Cm_front,Cm_back, T_LB_mean_C = radiation_absorption_view_factor_calculations(code_directory, rm, dr, sample_information,
#                                                                     solar_simulator_settings,
#                                                                     vacuum_chamber_setting,
#                                                                     numerical_simulation_setting,
#                                                                     df_view_factor, df_LB_details_all)
#
#     if (numerical_simulation_setting['analysis_mode'] == 'sensitivity') or (
#             (numerical_simulation_setting['analysis_mode'] == 'regression') and (numerical_simulation_setting['regression_parameter'] == 'alpha_r')):
#         # For sensitivity analysis and regression over thermal diffusivity, thermal diffusivity is treated as a constant throughout the sample
#
#         if numerical_simulation_setting['axial_conduction'] == False:
#         # Is the problem 1D?
#             Fo_criteria = numerical_simulation_setting['Fo_criteria']
#
#             dt = min(Fo_criteria * (dr ** 2) / (alpha_r),
#                      1 / f_heating / 15)  # assume 15 samples per period, Fo_criteria default = 1/3
#
#             t_total = 1 / f_heating * N_cycle  # total simulation time
#
#             Nt = int(t_total / dt)  # total number of simulation time step
#             time_simulation = dt * np.arange(Nt)
#
#             Fo_r = alpha_r * dt / dr ** 2
#
#             T = T_initial * np.ones((Nt, Nr))
#             dz = t_z
#             q_solar = light_source_intensity_Amax_fV_vecterize(np.arange(Nr) * dr, np.arange(Nt) * dt,
#                                                                    solar_simulator_settings, vacuum_chamber_setting,
#                                                                    light_source_property,
#                                                                    df_solar_simulator_VQ, sigma_df)
#             T_temp = np.zeros(Nr)
#             N_steady_count = 0
#             time_index = Nt - 1
#             N_one_cycle = int(Nt / N_cycle)
#
#             Cm = Cm_front+Cm_back
#
#             for p in range(Nt - 1):  # p indicate time step
#
#                 # The node at center of the disk
#                 cp_center = cp_const + cp_c1 * T[p, 0] + cp_c2 * T[p, 0] ** 2 + cp_c3 * T[p, 0] ** 3
#                 T[p + 1, 0] = T[p, 0] * (1 - 4 * Fo_r) + 4 * Fo_r * T[p, 1] - sigma_sb * (emissivity_back + emissivity_front) * dt / (rho * cp_center * dz) * T[p, 0] ** 4 + \
#                               dt / (rho * cp_center * dz) * (absorptivity_front * q_solar[p, 0] + 4*Cm[0]/(np.pi*dr**2))
#
#
#                 # The node in the middle of disk
#                 cp_mid = cp_const + cp_c1 * T[p, 1:-1] + cp_c2 * T[p, 1:-1] ** 2 + cp_c3 * T[p, 1:-1] ** 3
#                 T[p + 1, 1:-1] = T[p, 1:-1] + Fo_r * (rm[1:-1] - dr / 2) / rm[1:-1] * (T[p, 0:-2] - T[p, 1:-1]) + Fo_r * (rm[1:-1] + dr / 2) / rm[1:-1] * (T[p, 2:] - T[p, 1:-1]) \
#                                  + dt / (rho * cp_mid * dz) * (absorptivity_front * q_solar[p,1:-1] + Cm[1:-1]/(2*np.pi*rm[1:-1]*dr) - (emissivity_front + emissivity_back) * sigma_sb * T[p,1:-1] ** 4)
#
#
#                 # The node at the edge of the disk
#                 cp_edge = cp_const + cp_c1 * T[p, -1] + cp_c2 * T[p, -1] ** 2 + cp_c3 * T[p, -1] ** 3
#                 T[p + 1, -1] = T[p, -1] * (1 - 2 * Fo_r) + 2 * Fo_r *T [p, -2] + absorptivity_front*dt*q_solar[p, -1]/(dz*rho*cp_edge) + 2*rm[-1]*absorptivity_front*sigma_sb*dt*T_LBW**4/((rm[-1]-dr/2)*dr*rho*cp_edge) + Cm[-1]*dt/(np.pi*(rm[-1]-dr/2)*dr*dz*rho*cp_edge) - \
#                                (emissivity_front+emissivity_back)*sigma_sb*dt/(dz*rho*cp_edge)*T[p, -1]**4 - 2*rm[-1]*emissivity_back*sigma_sb*dt*T[p, -1]**4/((rm[-1]-dr/2)*dr*rho*cp_edge)
#
#
#                 if (p > 0) & (p % N_one_cycle == 0) & (simulated_amp_phase_extraction_method != 'fft'):
#                     A_max = np.max(T[p - N_one_cycle:p, :], axis=0)
#                     A_min = np.min(T[p - N_one_cycle:p, :], axis=0)
#                     if np.max(np.abs((T_temp[:] - T[p, :]) / (A_max - A_min))) < 2e-3:
#                         N_steady_count += 1
#                         if N_steady_count == N_stable_cycle_output:  # only need 2 periods to calculate amplitude and phase
#                             time_index = p
#                             print("stable temperature profile has been obtained @ iteration N= {}!".format(
#                                 int(p / N_one_cycle)))
#                             break
#                     T_temp = T[p, :]
#
#                 if p == Nt - 2:
#                     time_index = p
#                     # T_temp = T[p, :]
#                     print("Error! No stable temperature profile was obtained!")
#
#             # print('alpha_r = {:.2E}, sigma_s = {:.2E}, Amax = {}, f_heating = {}, focal plane = {}, Rec = {}, T_sur1 = {:.1f}, e_front = {:.2f}.'.format(
#             #         alpha_r,  light_source_property['sigma_s'],light_source_property['Amax'], f_heating,vacuum_chamber_setting['focal_shift'],
#             #         sample_information['rec_name'], vacuum_chamber_setting['T_sur1'],
#             #         sample_information['emissivity_front']))
#
#             return T[:time_index, :], time_simulation[:time_index], r, N_one_cycle, q_solar[:time_index,:]
#
#
#         elif numerical_simulation_setting['axial_conduction'] == True:
#             # 2D conduction
#             Fo_criteria = numerical_simulation_setting['Fo_criteria']
#
#             # alpha_r_A = float(sample_information['alpha_r_A'])
#             # alpha_r_B = float(sample_information['alpha_r_B'])
#             #
#             # alpha_z_A = float(sample_information['alpha_z_A'])
#             # alpha_z_B = float(sample_information['alpha_z_B'])
#             alpha_r = float(sample_information['alpha_r'])
#             alpha_z = float(sample_information['alpha_z'])
#             Nz = numerical_simulation_setting['Nz']
#             dz = t_z / Nz
#
#             T_min = sample_information['T_min']
#             # alpha_r_max = 1 / (alpha_r_A * T_min + alpha_r_B)
#             # alpha_z_max = 1 / (alpha_z_A * T_min + alpha_z_B)
#
#             dr = R / Nr
#             dt = min(Fo_criteria * (dr ** 2) / (alpha_r), Fo_criteria * (dz ** 2) / (alpha_z))
#
#             t_total = 1 / f_heating * N_cycle  # total simulation time
#
#             Nt = int(t_total / dt)  # total number of simulation time step
#             time_simulation = dt * np.arange(Nt)
#             r = np.arange(Nr)
#             rm = r * dr
#
#             T = T_initial * np.ones((Nt, Nr, Nz))
#
#             q_solar = light_source_intensity_Amax_fV_vecterize(np.arange(Nr) * dr, np.arange(Nt) * dt,
#                                                                solar_simulator_settings, vacuum_chamber_setting,
#                                                                light_source_property,
#                                                                df_solar_simulator_VQ, sigma_df)
#
#             T_temp = np.zeros(Nr)
#             N_steady_count = 0
#             time_index = Nt - 1
#             N_one_cycle = int(Nt / N_cycle)
#
#             for p in range(Nt - 1):  # p indicate time step
#
#                 # This scheme update the entire nodes
#
#                 # node V1
#                 cp_V1 = cp_const + cp_c1 * T[p, 0, 0] + cp_c2 * T[p, 0, 0] ** 2 + cp_c3 * T[p, 0, 0] ** 3
#                 alpha_r_V1 = alpha_r #1 / (alpha_r_A * T[p, 0, 0] + alpha_r_B)
#                 alpha_z_V1 = alpha_z #1 / (alpha_z_A * T[p, 0, 0] + alpha_z_B)
#                 T[p + 1, 0, 0] = T[p, 0, 0] + 4 * alpha_r_V1 * dt / dr ** 2 * (
#                         T[p, 1, 0] - T[p, 0, 0]) + 2 * alpha_z_V1 * dt / dz ** 2 * (T[p, 0, 1] - T[p, 0, 0]) \
#                                  + dt / (dz * rho * cp_V1) * (
#                                          2 * absorptivity_front * q_solar[p, 0] + 4 * Cm_front[0] / (
#                                          np.pi * dr ** 2) - 2 * emissivity_front * sigma_sb * T[p, 0, 0] ** 4)
#
#                 # node E4
#                 cp_E4 = cp_const + cp_c1 * T[p, 0, 1:-1] + cp_c2 * T[p, 0, 1:-1] ** 2 + cp_c3 * T[p, 0, 1:-1] ** 3
#                 alpha_r_E4 = alpha_r # 1 / (alpha_r_A * T[p, 0, 1:-1] + alpha_r_B)
#                 alpha_z_E4 = alpha_z #1 / (alpha_z_A * T[p, 0, 1:-1] + alpha_z_B)
#                 T[p + 1, 0, 1:-1] = T[p, 0, 1:-1] + alpha_z_E4 * dt / dz ** 2 * (
#                         T[p, 0, 0:-2] + T[p, 0, 2:] - 2 * T[p, 0, 1:-1]) + 4 * alpha_r_E4 * dt / dr ** 2 * (
#                                             T[p, 1, 1:-1] - T[p, 0, 1:-1])
#
#                 # node V4
#                 cp_V4 = cp_const + cp_c1 * T[p, 0, -1] + cp_c2 * T[p, 0, -1] ** 2 + cp_c3 * T[p, 0, -1] ** 3
#                 alpha_r_V4 = alpha_r #1 / (alpha_r_A * T[p, 0, -1] + alpha_r_B)
#                 alpha_z_V4 = alpha_z #1 / (alpha_z_A * T[p, 0, -1] + alpha_z_B)
#                 T[p + 1, 0, -1] = T[p, 0, -1] + 4 * alpha_r_V4 * dt / dr ** 2 * (
#                         T[p, 1, -1] - T[p, 0, -1]) + 2 * alpha_z_V4 * dt / dz ** 2 * (T[p, 0, -2] - T[p, 0, -1]) \
#                                   + dt / (dz * rho * cp_V4) * (
#                                           4 * Cm_back[0] / (np.pi * dr ** 2) - 2 * emissivity_back * sigma_sb * T[
#                                       p, 0, -1] ** 4)
#
#                 # node E1
#                 cp_E1 = cp_const + cp_c1 * T[p, 1:-1, 0] + cp_c2 * T[p, 1:-1, 0] ** 2 + cp_c3 * T[p, 1:-1, 0] ** 3
#                 # print(cp_E1)
#                 alpha_r_E1 = alpha_r #1 / (alpha_r_A * T[p, 1:-1, 0] + alpha_r_B)
#                 alpha_z_E1 = alpha_z # 1 / (alpha_z_A * T[p, 1:-1, 0] + alpha_z_B)
#
#                 T[p + 1, 1:-1, 0] = T[p, 1:-1, 0] + dt / (dz * rho * cp_E1) * (
#                         2 * absorptivity_front * q_solar[p, 1:-1] + Cm_front[1:-1] / (
#                         np.pi * rm[1:-1] * dr) - 2 * emissivity_front * sigma_sb * T[p, 1:-1, 0] ** 4) \
#                                     + alpha_r_E1 * dt / (rm[1:-1] * dr ** 2) * (
#                                             (rm[1:-1] - dr / 2) * (T[p, 0:-2, 0] - T[p, 1:-1, 0]) + (
#                                             rm[1:-1] + dr / 2) * (T[p, 2:, 0] - T[p, 1:-1, 0])) \
#                                     + alpha_z_E1 * dt / (dz ** 2) * (T[p, 1:-1, 1] - T[p, 1:-1, 0])
#
#                 # node B, this need to be tested very carefully
#                 cp_B = cp_const + cp_c1 * T[p, 1:-1, 1:-1] + cp_c2 * T[p, 1:-1, 1:-1] ** 2 + cp_c3 * T[p, 1:-1,1:-1] ** 3
#                 alpha_r_Body = alpha_r #1 / (alpha_r_A * T[p, 1:-1, 1:-1] + alpha_r_B)
#                 alpha_z_Body = alpha_z # 1 / (alpha_z_A * T[p, 1:-1, 1:-1] + alpha_z_B)
#                 T[p + 1, 1:-1, 1:-1] = T[p, 1:-1, 1:-1] + (alpha_r_Body* dt / (rm[1:-1] * dr ** 2) * ((rm[1:-1] - dr / 2) * (T[p, 0:-2, 1:-1] - T[p, 1:-1, 1:-1]).T + (rm[1:-1] + dr / 2) * (T[p, 2:, 1:-1] - T[p, 1:-1, 1:-1]).T)).T \
#                                        + alpha_z_Body * dt / dz ** 2 * (
#                                                    T[p, 1:-1, 0:-2] + T[p, 1:-1, 2:] - 2 * T[p, 1:-1, 1:-1])
#
#                 # node E3
#                 cp_E3 = cp_const + cp_c1 * T[p, 1:-1, -1] + cp_c2 * T[p, 1:-1, -1] ** 2 + cp_c3 * T[p, 1:-1, -1] ** 3
#                 alpha_r_E3 =  alpha_r # 1 / (alpha_r_A * T[p, 1:-1, -1] + alpha_r_B)
#                 alpha_z_E3 =  alpha_z #1 / (alpha_z_A * T[p, 1:-1, -1] + alpha_z_B)
#                 T[p + 1, 1:-1, -1] = T[p, 1:-1, -1] + dt / (dz * rho * cp_E3) * (
#                         Cm_back[1:-1] / (np.pi * rm[1:-1] * dr) - 2 * emissivity_back * sigma_sb * T[p, 1:-1,
#                                                                                                    -1] ** 4) \
#                                      + alpha_r_E3 * dt / (rm[1:-1] * dr ** 2) * (
#                                              (rm[1:-1] - dr / 2) * (T[p, 0:-2, -1] - T[p, 1:-1, -1]) + (
#                                              rm[1:-1] + dr / 2) * (T[p, 2:, -1] - T[p, 1:-1, -1])) \
#                                      + alpha_z_E3 * dt / dz ** 2 * (T[p, 1:-1, -2] - T[p, 1:-1, -1])
#
#                 # node V2
#                 cp_V2 = cp_const + cp_c1 * T[p, -1, 0] + cp_c2 * T[p, -1, 0] ** 2 + cp_c3 * T[p, -1, 0] ** 3
#                 alpha_r_V2 = alpha_r # 1 / (alpha_r_A * T[p, -1, 0] + alpha_r_B)
#                 alpha_z_V2 = alpha_z # 1 / (alpha_z_A * T[p, -1, 0] + alpha_z_B)
#                 T[p + 1, -1, 0] = T[p, -1, 0] + dt / (dz * rho * cp_V2) * (
#                         2 * absorptivity_front * q_solar[p, -1] + 2 * Cm_front[-1] / (
#                         np.pi * rm[-1] * dr) - 2 * sigma_sb * emissivity_front * T[p, -1, 0] ** 4) \
#                                   + dt / (dr * rho * cp_V2) * (
#                                           2 * sigma_sb * T_LBW ** 4 - 2 * sigma_sb * emissivity_front * T[
#                                       p, -1, 0] ** 4) + 2 * alpha_r_V2 * dt / dr ** 2 * (T[p, -2, 0] - T[p, -1, 0]) \
#                                   + 2 * alpha_z_V2 * dt / dz ** 2 * (T[p, -1, 1] - T[p, -1, 0])
#
#                 # node E2
#                 cp_E2 = cp_const + cp_c1 * T[p, -1, 1:-1] + cp_c2 * T[p, -1, 1:-1] ** 2 + cp_c3 * T[p, -1, 1:-1] ** 3
#                 alpha_r_E2 = alpha_r # 1 / (alpha_r_A * T[p, -1, 1:-1] + alpha_r_B)
#                 alpha_z_E2 = alpha_z # 1 / (alpha_z_A * T[p, -1, 1:-1] + alpha_z_B)
#                 T[p + 1, -1, 1:-1] = T[p, -1, 1:-1] + 2 * alpha_r_E2 * dt / dr ** 2 * (
#                         T[p, -2, 1:-1] - T[p, -1, 1:-1]) + 2 * alpha_z_E2 * dt / dz ** 2 * (
#                                              T[p, -1, 0:-2] + T[p, -1, 2:] - 2 * T[p, -1, 1:-1]) \
#                                      + 2 * dt / (dr * rho * cp_E2) * (-emissivity_front * sigma_sb * T[
#                                                                                                      p, -1,
#                                                                                                      1:-1] ** 4 + absorptivity_front * sigma_sb * T_LBW ** 4)
#
#                 # node V3
#                 cp_V3 = cp_const + cp_c1 * T[p, -1, -1] + cp_c2 * T[p, -1, -1] ** 2 + cp_c3 * T[p, -1, -1] ** 3
#                 alpha_r_V3 = alpha_r #1 / (alpha_r_A * T[p, -1, -1] + alpha_r_B)
#                 alpha_z_V3 = alpha_z # 1 / (alpha_z_A * T[p, -1, -1] + alpha_z_B)
#                 T[p + 1, -1, -1] = T[p, -1, -1] + dt / (dz * rho * cp_V3) * (
#                         2 * Cm_back[-1] / (np.pi * rm[-1] * dr) - 2 * sigma_sb * emissivity_back * T[
#                     p, -1, -1] ** 4) \
#                                    + dt / (dr * rho * cp_V3) * (
#                                            2 * sigma_sb * T_LBW ** 4 - 2 * sigma_sb * emissivity_back * T[
#                                        p, -1, -1] ** 4) + 2 * alpha_r_V3 * dt / dr ** 2 * (
#                                            T[p, -2, -1] - T[p, -1, -1]) \
#                                    + 2 * alpha_z_V3 * dt / dz ** 2 * (T[p, -1, -2] - T[p, -1, -1])
#
#                 if (p > 0) & (p % N_one_cycle == 0) & (simulated_amp_phase_extraction_method != 'fft'):
#                     A_max = np.max(T[p - N_one_cycle:p, :, -1], axis=0)
#                     A_min = np.min(T[p - N_one_cycle:p, :, -1], axis=0)
#                     if np.max(np.abs((T_temp[:] - T[p, :, -1]) / (A_max - A_min))) < 2e-3:
#                         N_steady_count += 1
#                         if N_steady_count == N_stable_cycle_output:  # only need 2 periods to calculate amplitude and phase
#                             time_index = p
#                             print("stable temperature profile has been obtained @ iteration N= {}!".format(
#                                 int(p / N_one_cycle)))
#                             break
#                     T_temp = T[p, :, -1]
#
#                 if p == Nt - 2:
#                     time_index = p
#                     # T_temp = T[p, :]
#                     print("Error! No stable temperature profile was obtained!")
#
#             print('alpha_r = {:.2E}, focal plane = {}, sigma_s = {:.2E}, f_heating = {}, Rec = {}, T_sur1 = {:.1f}, e_front = {:.2f}.'.format(
#                     alpha_r, vacuum_chamber_setting['focal_shift'], light_source_property['sigma_s'], f_heating,
#                     sample_information['rec_name'], vacuum_chamber_setting['T_sur1'],
#                     sample_information['emissivity_front']))
#
#         return T[:time_index, :, :], time_simulation[:time_index], r, N_one_cycle, q_solar[:time_index, :]
#
#
#
#
#
#     elif (numerical_simulation_setting['analysis_mode'] != 'sensitivity') and (numerical_simulation_setting['regression_parameter'] != 'alpha_r'):
#
#         if numerical_simulation_setting['axial_conduction'] == False:
#             #1D problem here
#             Fo_criteria = numerical_simulation_setting['Fo_criteria']
#             alpha_r_A = float(sample_information['alpha_r_A'])
#             alpha_r_B = float(sample_information['alpha_r_B'])
#
#             T_min = sample_information['T_min']
#             dz = t_z
#
#             alpha_max = 1 / (alpha_r_A * T_min + alpha_r_B)
#
#             dr = R / Nr
#             dt = min(Fo_criteria * (dr ** 2) / (alpha_max),
#                      1 / f_heating / 15)  # assume 15 samples per period, Fo_criteria default = 1/3
#
#             t_total = 1 / f_heating * N_cycle  # total simulation time
#
#             Nt = int(t_total / dt)  # total number of simulation time step
#             time_simulation = dt * np.arange(Nt)
#             r = np.arange(Nr)
#             rm = r * dr
#
#             T = T_initial * np.ones((Nt, Nr))
#
#             # 1D conduction here
#
#
#
#             # Note the initial sigma_s value was set as float(df_exp_condition['p_initial']). If regression parameter was not sigma this does not matter, because it will take sigma from a different place anyway
#
#
#             #   Let's bypass this for now, the entire code is quite messy
#             Amax, sigma_s, kvd, bvd = interpolate_light_source_characteristic(sigma_df, df_solar_simulator_VQ,light_source_property['sigma_s'],
#                                                                               numerical_simulation_setting,
#                                                                               vacuum_chamber_setting)
#             light_source_property['Amax'] = Amax
#
#
#             q_solar = light_source_intensity_Amax_fV_vecterize(np.arange(Nr) * dr, np.arange(Nt) * dt,
#                                                                    solar_simulator_settings, vacuum_chamber_setting,
#                                                                    light_source_property,
#                                                                    df_solar_simulator_VQ, sigma_df)
#             T_temp = np.zeros(Nr)
#             N_steady_count = 0
#             time_index = Nt - 1
#             N_one_cycle = int(Nt / N_cycle)
#
#             Cm = Cm_front + Cm_back
#
#             for p in range(Nt - 1):  # p indicate time step
#
#                 # The node at center of the disk
#                 cp_c = cp_const + cp_c1 * T[p, 0] + cp_c2 * T[p, 0] ** 2 + cp_c3 * T[p, 0] ** 3
#                 alpha_r_c = 1/(alpha_r_A*T[p, 0]+alpha_r_B)
#
#                 T[p + 1, 0] = T[p, 0] * (1 - 4 * (alpha_r_c*dt/dr**2)) + 4 * (alpha_r_c*dt/dr**2) * T[p, 1] - sigma_sb * (emissivity_back + emissivity_front) * dt / (rho * cp_c * dz) * T[p, 0] ** 4 + \
#                               dt / (rho * cp_c * dz) * (absorptivity_front * q_solar[p, 0] + 4*Cm[0]/(np.pi*dr**2))
#
#
#                 # The node in the middle of disk
#                 cp_mid = cp_const + cp_c1 * T[p, 1:-1] + cp_c2 * T[p, 1:-1] ** 2 + cp_c3 * T[p, 1:-1] ** 3
#                 alpha_mid = 1 / (alpha_r_A * T[p, 1:-1] + alpha_r_B)
#                 T[p + 1, 1:-1] = T[p, 1:-1] + (alpha_mid*dt/dr**2) * (rm[1:-1] - dr / 2) / rm[1:-1] * (T[p, 0:-2] - T[p, 1:-1]) + (alpha_mid*dt/dr**2) * (rm[1:-1] + dr / 2) / rm[1:-1] * (T[p, 2:] - T[p, 1:-1]) \
#                                  + dt / (rho * cp_mid * dz) * (absorptivity_front * q_solar[p,1:-1] + Cm[1:-1]/(2*np.pi*rm[1:-1]*dr) - (emissivity_front + emissivity_back) * sigma_sb * T[p,1:-1] ** 4)
#
#
#                 # The node at the edge of the disk
#                 cp_edge = cp_const + cp_c1 * T[p, -1] + cp_c2 * T[p, -1] ** 2 + cp_c3 * T[p, -1] ** 3
#                 alpha_r_e = 1/(alpha_r_A*T[p, -1]+alpha_r_B)
#                 T[p + 1, -1] = T[p, -1] * (1 - 2 * (alpha_r_e*dt/dr**2)) + 2 * (alpha_r_e*dt/dr**2) *T[p, -2] + absorptivity_front*dt*q_solar[p, -1]/(dz*rho*cp_edge) + 2*rm[-1]*absorptivity_front*sigma_sb*dt*T_LBW**4/((rm[-1]-dr/2)*dr*rho*cp_edge) + Cm[-1]*dt/(np.pi*(rm[-1]-dr/2)*dr*dz*rho*cp_edge) - \
#                                (emissivity_front+emissivity_back)*sigma_sb*dt/(dz*rho*cp_edge)*T[p, -1]**4 - 2*rm[-1]*emissivity_back*sigma_sb*dt*T[p, -1]**4/((rm[-1]-dr/2)*dr*rho*cp_edge)
#
#
#                 if (p > 0) & (p % N_one_cycle == 0) & (simulated_amp_phase_extraction_method != 'fft'):
#                     A_max = np.max(T[p - N_one_cycle:p, :], axis=0)
#                     A_min = np.min(T[p - N_one_cycle:p, :], axis=0)
#                     if np.max(np.abs((T_temp[:] - T[p, :]) / (A_max - A_min))) < 2e-3:
#                         N_steady_count += 1
#                         if N_steady_count == N_stable_cycle_output:  # only need 2 periods to calculate amplitude and phase
#                             time_index = p
#                             print("stable temperature profile has been obtained @ iteration N= {}!".format(
#                                 int(p / N_one_cycle)))
#                             break
#                     T_temp = T[p, :]
#
#                 if p == Nt - 2:
#                     time_index = p
#                     # T_temp = T[p, :]
#                     print("Error! No stable temperature profile was obtained!")
#
#             print('alpha_r = {:.2E}, sigma_s = {:.2E}, Amax = {}, f_heating = {}, focal plane = {}, Rec = {}, T_sur1 = {:.1f}, e_front = {:.2f}.'.format(
#                     alpha_r,  light_source_property['sigma_s'],light_source_property['Amax'], f_heating,vacuum_chamber_setting['focal_shift'],
#                     sample_information['rec_name'], vacuum_chamber_setting['T_sur1'],
#                     sample_information['emissivity_front']))
#
#             return T[:time_index, :], time_simulation[:time_index], r, N_one_cycle,q_solar[:time_index,:]
#
#
#
#         elif numerical_simulation_setting['axial_conduction'] == True:
#             # 2D conduction
#             Fo_criteria = numerical_simulation_setting['Fo_criteria']
#             alpha_r_A = float(sample_information['alpha_r_A'])
#             alpha_r_B = float(sample_information['alpha_r_B'])
#
#             alpha_z_A = float(sample_information['alpha_z_A'])
#             alpha_z_B = float(sample_information['alpha_z_B'])
#             Nz = numerical_simulation_setting['Nz']
#             dz = t_z/Nz
#
#             T_min = sample_information['T_min']
#             alpha_r_max = 1 / (alpha_r_A * T_min + alpha_r_B)
#             alpha_z_max = 1 / (alpha_z_A * T_min + alpha_z_B)
#
#             dr = R / Nr
#             dt = min(Fo_criteria * (dr ** 2) / (alpha_r_max),Fo_criteria * (dz ** 2) / (alpha_z_max))
#
#             t_total = 1 / f_heating * N_cycle  # total simulation time
#
#             Nt = int(t_total / dt)  # total number of simulation time step
#             time_simulation = dt * np.arange(Nt)
#             r = np.arange(Nr)
#             rm = r * dr
#
#             T = T_initial * np.ones((Nt, Nr, Nz))
#
#             q_solar = light_source_intensity_Amax_fV_vecterize(np.arange(Nr) * dr, np.arange(Nt) * dt,
#                                                                    solar_simulator_settings, vacuum_chamber_setting,
#                                                                    light_source_property,
#                                                                    df_solar_simulator_VQ, sigma_df)
#
#             T_temp = np.zeros(Nr)
#             N_steady_count = 0
#             time_index = Nt - 1
#             N_one_cycle = int(Nt / N_cycle)
#
#             for p in range(Nt - 1):  # p indicate time step
#
#
#                 # This scheme update the entire nodes
#
#                 # node V1
#                 cp_V1 = cp_const + cp_c1 * T[p, 0, 0] + cp_c2 * T[p, 0, 0] ** 2 + cp_c3 * T[p, 0, 0] ** 3
#                 alpha_r_V1 = 1 / (alpha_r_A * T[p, 0, 0] + alpha_r_B)
#                 alpha_z_V1 = 1 / (alpha_z_A * T[p, 0, 0] + alpha_z_B)
#                 T[p + 1, 0, 0] = T[p, 0, 0] + 4 * alpha_r_V1 * dt / dr ** 2 * (
#                             T[p, 1, 0] - T[p, 0, 0]) + 2 * alpha_z_V1 * dt / dz ** 2 * (T[p, 0, 1] - T[p, 0, 0]) \
#                                  + dt / (dz * rho * cp_V1) * (
#                                              2 * absorptivity_front * q_solar[p, 0] + 4 * Cm_front[0] / (
#                                                  np.pi * dr ** 2) - 2 * emissivity_front * sigma_sb * T[p, 0, 0] ** 4)
#
#                 # node E4
#                 cp_E4 = cp_const + cp_c1 * T[p, 0, 1:-1] + cp_c2 * T[p, 0, 1:-1] ** 2 + cp_c3 * T[p, 0, 1:-1] ** 3
#                 alpha_r_E4 = 1 / (alpha_r_A * T[p, 0, 1:-1] + alpha_r_B)
#                 alpha_z_E4 = 1 / (alpha_z_A * T[p, 0, 1:-1] + alpha_z_B)
#                 T[p + 1, 0, 1:-1] = T[p, 0, 1:-1] + alpha_z_E4 * dt / dz ** 2 * (
#                             T[p, 0, 0:-2] + T[p, 0, 2:] - 2 * T[p, 0, 1:-1]) + 4 * alpha_r_E4 * dt / dr ** 2 * (
#                                              T[p, 1, 1:-1] - T[p, 0, 1:-1])
#
#                 # node V4
#                 cp_V4 = cp_const + cp_c1 * T[p, 0, -1] + cp_c2 * T[p, 0, -1] ** 2 + cp_c3 * T[p, 0, -1] ** 3
#                 alpha_r_V4 = 1 / (alpha_r_A * T[p, 0, -1] + alpha_r_B)
#                 alpha_z_V4 = 1 / (alpha_z_A * T[p, 0, -1] + alpha_z_B)
#                 T[p + 1, 0, -1] = T[p, 0, -1] + 4 * alpha_r_V4 * dt / dr ** 2 * (
#                             T[p, 1, -1] - T[p, 0, -1]) + 2 * alpha_z_V4 * dt / dz ** 2 * (T[p, 0, -2] - T[p, 0, -1]) \
#                                   + dt / (dz * rho * cp_V4) * (
#                                               4 * Cm_back[0] / (np.pi * dr ** 2) - 2 * emissivity_back * sigma_sb * T[
#                                           p, 0, -1] ** 4)
#
#                 # node E1
#                 cp_E1 = cp_const + cp_c1 * T[p, 1:-1, 0] + cp_c2 * T[p, 1:-1, 0] ** 2 + cp_c3 * T[p, 1:-1, 0] ** 3
#                 # print(cp_E1)
#                 alpha_r_E1 = 1 / (alpha_r_A * T[p, 1:-1, 0] + alpha_r_B)
#                 alpha_z_E1 = 1 / (alpha_z_A * T[p, 1:-1, 0] + alpha_z_B)
#
#                 T[p + 1, 1:-1, 0] = T[p, 1:-1, 0] + dt / (dz * rho * cp_E1) * (
#                             2 * absorptivity_front * q_solar[p, 1:-1] + Cm_front[1:-1] / (
#                                 np.pi * rm[1:-1] * dr) - 2 * emissivity_front * sigma_sb * T[p, 1:-1, 0] ** 4) \
#                                     + alpha_r_E1 * dt / (rm[1:-1] * dr ** 2) * (
#                                                 (rm[1:-1] - dr / 2) * (T[p, 0:-2, 0] - T[p, 1:-1, 0]) + (
#                                                     rm[1:-1] + dr / 2) * (T[p, 2:, 0] - T[p, 1:-1, 0])) \
#                                     + alpha_z_E1 * dt / (dz ** 2) * (T[p, 1:-1, 1] - T[p, 1:-1, 0])
#
#                 # node B, this need to be tested very carefully
#                 cp_B = cp_const + cp_c1 * T[p, 1:-1, 1:-1] + cp_c2 * T[p, 1:-1, 1:-1] ** 2 + cp_c3 * T[p, 1:-1, 1:-1] ** 3
#                 alpha_r_Body = 1 / (alpha_r_A * T[p, 1:-1, 1:-1] + alpha_r_B)
#                 alpha_z_Body = 1 / (alpha_z_A * T[p, 1:-1, 1:-1] + alpha_z_B)
#                 T[p + 1, 1:-1, 1:-1] = T[p, 1:-1, 1:-1] + (alpha_r_Body.T * dt / (rm[1:-1] * dr ** 2) * (
#                             (rm[1:-1] - dr / 2) * (T[p, 0:-2, 1:-1] - T[p, 1:-1, 1:-1]).T + (rm[1:-1] + dr / 2) * (T[p, 2:, 1:-1] - T[p, 1:-1, 1:-1]).T)).T \
#                                     + alpha_z_Body * dt / dz ** 2 * (T[p, 1:-1, 0:-2] + T[p, 1:-1, 2:] - 2 * T[p, 1:-1, 1:-1])
#
#                 # node E3
#                 cp_E3 = cp_const + cp_c1 * T[p, 1:-1, -1] + cp_c2 * T[p, 1:-1, -1] ** 2 + cp_c3 * T[p, 1:-1, -1] ** 3
#                 alpha_r_E3 = 1 / (alpha_r_A * T[p, 1:-1, -1] + alpha_r_B)
#                 alpha_z_E3 = 1 / (alpha_z_A * T[p, 1:-1, -1] + alpha_z_B)
#                 T[p + 1, 1:-1, -1] = T[p, 1:-1, -1] + dt / (dz * rho * cp_E3) * (
#                             Cm_back[1:-1] / (np.pi * rm[1:-1] * dr) - 2 * emissivity_back * sigma_sb * T[p, 1:-1,
#                                                                                                        -1] ** 4) \
#                                      + alpha_r_E3 * dt / (rm[1:-1] * dr ** 2) * (
#                                                  (rm[1:-1] - dr / 2) * (T[p, 0:-2, -1] - T[p, 1:-1, -1]) + (
#                                                      rm[1:-1] + dr / 2) * (T[p, 2:, -1] - T[p, 1:-1, -1])) \
#                                      + alpha_z_E3 * dt / dz ** 2 * (T[p, 1:-1, -2] - T[p, 1:-1, -1])
#
#                 # node V2
#                 cp_V2 = cp_const + cp_c1 * T[p, -1, 0] + cp_c2 * T[p, -1, 0] ** 2 + cp_c3 * T[p, -1, 0] ** 3
#                 alpha_r_V2 = 1 / (alpha_r_A * T[p, -1, 0] + alpha_r_B)
#                 alpha_z_V2 = 1 / (alpha_z_A * T[p, -1, 0] + alpha_z_B)
#                 T[p + 1, -1, 0] = T[p, -1, 0] + dt / (dz * rho * cp_V2) * (
#                             2 * absorptivity_front * q_solar[p, -1] + 2 * Cm_front[-1] / (
#                                 np.pi * rm[-1] * dr) - 2 * sigma_sb * emissivity_front * T[p, -1, 0] ** 4) \
#                                   + dt / (dr * rho * cp_V2) * (
#                                               2 * sigma_sb * T_LBW ** 4 - 2 * sigma_sb * emissivity_front * T[
#                                           p, -1, 0] ** 4) + 2 * alpha_r_V2 * dt / dr ** 2 * (T[p, -2, 0] - T[p, -1, 0]) \
#                                   + 2 * alpha_z_V2 * dt / dz ** 2 * (T[p, -1, 1] - T[p, -1, 0])
#
#                 # node E2
#                 cp_E2 = cp_const + cp_c1 * T[p, -1, 1:-1] + cp_c2 * T[p, -1, 1:-1] ** 2 + cp_c3 * T[p, -1, 1:-1] ** 3
#                 alpha_r_E2 = 1 / (alpha_r_A * T[p, -1, 1:-1] + alpha_r_B)
#                 alpha_z_E2 = 1 / (alpha_z_A * T[p, -1, 1:-1] + alpha_z_B)
#                 T[p + 1, -1, 1:-1] = T[p, -1, 1:-1] + 2 * alpha_r_E2 * dt / dr ** 2 * (
#                             T[p, -2, 1:-1] - T[p, -1, 1:-1]) + 2 * alpha_z_E2 * dt / dz ** 2 * (
#                                               T[p, -1, 0:-2] + T[p, -1, 2:] - 2 * T[p, -1, 1:-1]) \
#                                   + 2 * dt / (dr * rho * cp_E2) * (-emissivity_front * sigma_sb * T[
#                     p, -1, 1:-1] ** 4 + absorptivity_front * sigma_sb * T_LBW ** 4)
#
#                 # node V3
#                 cp_V3 = cp_const + cp_c1 * T[p, -1, -1] + cp_c2 * T[p, -1, -1] ** 2 + cp_c3 * T[p, -1, -1] ** 3
#                 alpha_r_V3 = 1 / (alpha_r_A * T[p, -1, -1] + alpha_r_B)
#                 alpha_z_V3 = 1 / (alpha_z_A * T[p, -1, -1] + alpha_z_B)
#                 T[p + 1, -1, -1] = T[p, -1, -1] + dt / (dz * rho * cp_V3) * (
#                             2 * Cm_back[-1] / (np.pi * rm[-1] * dr) - 2 * sigma_sb * emissivity_back * T[
#                         p, -1, -1] ** 4) \
#                                    + dt / (dr * rho * cp_V3) * (
#                                                2 * sigma_sb * T_LBW ** 4 - 2 * sigma_sb * emissivity_back * T[
#                                            p, -1, -1] ** 4) + 2 * alpha_r_V3 * dt / dr ** 2 * (
#                                                T[p, -2, -1] - T[p, -1, -1]) \
#                                    + 2 * alpha_z_V3 * dt / dz ** 2 * (T[p, -1, -2] - T[p, -1, -1])
#
#
#
#                 """
#
#                 # This scheme first update radial direction, then update z direction
#                 # The result of this scheme somehow deviate from 1D solution
#                 for j in range(Nz):
#
#                     if j==0:
#                         # node E1
#                         cp_E1 = cp_const + cp_c1 * T[p,1:-1, 0] + cp_c2 *  T[p,1:-1, 0] ** 2 + cp_c3 *  T[p,1:-1, 0] ** 3
#                         #print(cp_E1)
#                         alpha_r_E1 = 1 / (alpha_r_A * T[p,1:-1, 0] + alpha_r_B)
#                         alpha_z_E1 = 1 / (alpha_z_A * T[p,1:-1, 0] + alpha_z_B)
#
#                         T[p+1,1:-1,0] = T[p,1:-1,0] + dt/(dz*rho*cp_E1)*(2*absorptivity_front*q_solar[p,1:-1]+Cm_front[1:-1]/(np.pi*rm[1:-1]*dr) -2*emissivity_front*sigma_sb*T[p,1:-1,0]**4) \
#                             + alpha_r_E1*dt/(rm[1:-1]*dr**2)*((rm[1:-1]-dr/2)*(T[p,0:-2,0] - T[p,1:-1,0]) + (rm[1:-1]+dr/2)*(T[p,2:,0] - T[p,1:-1,0])) \
#                             + alpha_z_E1*dt/(dz**2)*(T[p,1:-1,1] - T[p,1:-1,0])
#
#
#                         # node V1
#                         cp_V1 = cp_const + cp_c1 * T[p,0, 0] + cp_c2 *  T[p,0, 0] ** 2 + cp_c3 *  T[p,0, 0] ** 3
#                         alpha_r_V1 = 1 / (alpha_r_A * T[p,0, 0] + alpha_r_B)
#                         alpha_z_V1 = 1 / (alpha_z_A * T[p,0, 0] + alpha_z_B)
#                         T[p+1,0,0] = T[p,0,0] + 4*alpha_r_V1*dt/dr**2*(T[p,1,0] - T[p,0,0]) + 2*alpha_z_V1*dt/dz**2*(T[p,0,1]-T[p,0,0]) \
#                             + dt/(dz*rho*cp_V1)*(2*absorptivity_front*q_solar[p,0]+4*Cm_front[0]/(np.pi*dr**2)-2*emissivity_front*sigma_sb*T[p,0,0]**4)
#
#                         # node V2
#                         cp_V2 = cp_const + cp_c1 * T[p,-1, 0] + cp_c2 *  T[p,-1, 0] ** 2 + cp_c3 *  T[p,-1, 0] ** 3
#                         alpha_r_V2 = 1 / (alpha_r_A * T[p,-1, 0] + alpha_r_B)
#                         alpha_z_V2 = 1 / (alpha_z_A * T[p,-1, 0] + alpha_z_B)
#                         T[p+1,-1,0] = T[p,-1,0] + dt/(dz*rho*cp_V2)*(2*absorptivity_front*q_solar[p,-1] + 2*Cm_front[-1]/(np.pi*rm[-1]*dr)-2*sigma_sb*emissivity_front*T[p,-1,0]**4) \
#                             + dt/(dr*rho*cp_V2)*(2*sigma_sb*T_LBW**4 - 2*sigma_sb*emissivity_front*T[p,-1,0]**4) + 2*alpha_r_V2*dt/dr**2*(T[p,-2,0]-T[p,-1,0]) \
#                             + 2*alpha_z_V2*dt/dz**2*(T[p,-1,1]-T[p,-1,0])
#
#
#                     elif (j>0) and (j<Nz-1):
#                         # node E4
#                         cp_E4 = cp_const + cp_c1 * T[p,0, j] + cp_c2 * T[p,0, j] ** 2 + cp_c3 * T[p,0, j] ** 3
#                         alpha_r_E4 = 1 / (alpha_r_A * T[p,0, j] + alpha_r_B)
#                         alpha_z_E4 = 1 / (alpha_z_A * T[p,0, j] + alpha_z_B)
#                         T[p+1,0,j] = T[p,0,j] + alpha_z_E4*dt/dz**2*(T[p,0,j-1] + T[p,0,j+1] - 2*T[p,0,j]) + 4*alpha_r_E4*dt/dr**2*(T[p,1,j]-T[p,0,j])
#
#                         # node B, this need to be tested very carefully
#                         cp_B = cp_const + cp_c1 * T[p, 1:-1, j] + cp_c2 * T[p, 1:-1, j] ** 2 + cp_c3 * T[p, 1:-1, j] ** 3
#                         alpha_r_Body = 1 / (alpha_r_A * T[p, 1:-1, j] + alpha_r_B)
#                         alpha_z_Body = 1 / (alpha_z_A * T[p, 1:-1, j] + alpha_z_B)
#                         T[p+1,1:-1,j] = T[p,1:-1,j] + alpha_r_Body*dt/(rm[1:-1]*dr**2)*((rm[1:-1]-dr/2)*(T[p,0:-2,j] - T[p,1:-1,j])+ (rm[1:-1]+dr/2)*(T[p,2:,j] - T[p,1:-1,j]))\
#                         + alpha_z_Body*dt/dz**2*(T[p,1:-1,j-1] + T[p,1:-1,j+1] - 2*T[p,1:-1,j])
#
#
#                         # node E2
#                         cp_E2 = cp_const + cp_c1 * T[p, -1, j] + cp_c2 * T[p, -1,j] ** 2 + cp_c3 * T[p, -1, j] ** 3
#                         alpha_r_E2 = 1 / (alpha_r_A * T[p, -1, j] + alpha_r_B)
#                         alpha_z_E2 = 1 / (alpha_z_A * T[p, -1, j] + alpha_z_B)
#                         T[p+1,-1,j] = T[p,-1,j] + 2*alpha_r_E2*dt/dr**2*(T[p,-2,j]-T[p,-1,j]) + 2*alpha_z_E2*dt/dz**2*(T[p,-1,j-1]+T[p,-1,j+1]-2*T[p,-1,j]) \
#                             + 2*dt/(dr*rho*cp_E2)*(-emissivity_front*sigma_sb*T[p,-1,j]**4 + absorptivity_front*sigma_sb*T_LBW**4)
#
#
#                     elif j == Nz-1:
#                         #node V4
#                         cp_V4 = cp_const + cp_c1 * T[p,0, -1] + cp_c2 *  T[p,0, -1] ** 2 + cp_c3 *  T[p,0, -1] ** 3
#                         alpha_r_V4 = 1 / (alpha_r_A * T[p,0, -1] + alpha_r_B)
#                         alpha_z_V4 = 1 / (alpha_z_A * T[p,0, -1] + alpha_z_B)
#                         T[p+1,0,-1] = T[p,0,-1] + 4*alpha_r_V4*dt/dr**2*(T[p,1,-1] - T[p,0,-1]) + 2*alpha_z_V4*dt/dz**2*(T[p,0,-2]-T[p,0,-1]) \
#                             + dt/(dz*rho*cp_V4)*(4*Cm_back[0]/(np.pi*dr**2)-2*emissivity_back*sigma_sb*T[p,0,-1]**4)
#
#                         #node E3
#                         cp_E3 = cp_const + cp_c1 * T[p,1:-1, -1] + cp_c2 *  T[p,1:-1, -1] ** 2 + cp_c3 *  T[p,1:-1, -1] ** 3
#                         alpha_r_E3 = 1 / (alpha_r_A * T[p,1:-1, -1] + alpha_r_B)
#                         alpha_z_E3 = 1 / (alpha_z_A * T[p,1:-1, -1] + alpha_z_B)
#                         T[p+1,1:-1,-1] = T[p,1:-1,-1] + dt/(dz*rho*cp_E3)*(Cm_back[1:-1]/(np.pi*rm[1:-1]*dr)-2*emissivity_back*sigma_sb*T[p,1:-1,-1]**4) \
#                                              + alpha_r_E3*dt/(rm[1:-1]*dr**2)*((rm[1:-1]-dr/2)*(T[p,0:-2,-1] - T[p,1:-1,-1])+(rm[1:-1]+dr/2)*(T[p,2:,-1] - T[p,1:-1,-1])) \
#                                              + alpha_z_E3*dt/dz**2*(T[p,1:-1,-2]-T[p,1:-1,-1])
#
#                         # node V3
#                         cp_V3 = cp_const + cp_c1 * T[p,-1, -1] + cp_c2 *  T[p,-1, -1] ** 2 + cp_c3 *  T[p,-1, -1] ** 3
#                         alpha_r_V3 = 1 / (alpha_r_A *  T[p,-1, -1] + alpha_r_B)
#                         alpha_z_V3 = 1 / (alpha_z_A *  T[p,-1, -1] + alpha_z_B)
#                         T[p+1,-1,-1] = T[p,-1,-1] + dt/(dz*rho*cp_V3)*(2*Cm_back[-1]/(np.pi*rm[-1]*dr) - 2*sigma_sb*emissivity_back*T[p,-1,-1]**4) \
#                                            + dt/(dr*rho*cp_V3)*(2*sigma_sb*T_LBW**4 - 2*sigma_sb*emissivity_back*T[p,-1,-1]**4) + 2*alpha_r_V3*dt/dr**2*(T[p,-2,-1]-T[p,-1,-1]) \
#                                            + 2*alpha_z_V3*dt/dz**2*(T[p,-1,-2]-T[p,-1,-1])
#
#                 """
#
#
#                 """
#
#                 for i in range(Nr):
#                 # This scheme first update Z direction, then radial direction.
#                 # This scheme gives the same result as 1D explicit code, but takes much longer to run
#
#                     if i==0:
#                         # node V1
#                         cp_V1 = cp_const + cp_c1 * T[p, 0, 0] + cp_c2 * T[p, 0, 0] ** 2 + cp_c3 * T[p, 0, 0] ** 3
#                         alpha_r_V1 = 1 / (alpha_r_A * T[p, 0, 0] + alpha_r_B)
#                         alpha_z_V1 = 1 / (alpha_z_A * T[p, 0, 0] + alpha_z_B)
#                         T[p + 1, 0, 0] = T[p, 0, 0] + 4 * alpha_r_V1 * dt / dr ** 2 * (
#                                     T[p, 1, 0] - T[p, 0, 0]) + 2 * alpha_z_V1 * dt / dz ** 2 * (T[p, 0, 1] - T[p, 0, 0]) \
#                                          + dt / (dz * rho * cp_V1) * (
#                                                      2 * absorptivity_front * q_solar[p, 0] + 4 * Cm_front[0] / (
#                                                          np.pi * dr ** 2) - 2 * emissivity_front * sigma_sb * T[p, 0, 0] ** 4)
#
#                         #Node E4
#                         cp_E4 = cp_const + cp_c1 * T[p,i, 1:-1] + cp_c2 * T[p,i, 1:-1] ** 2 + cp_c3 * T[p,i, 1:-1] ** 3
#                         alpha_r_E4 = 1 / (alpha_r_A * T[p,i, 1:-1] + alpha_r_B)
#                         alpha_z_E4 = 1 / (alpha_z_A * T[p,i, 1:-1] + alpha_z_B)
#                         T[p+1,i,1:-1] = T[p,i,1:-1] + alpha_z_E4*dt/dz**2*(T[p,i,0:-2] + T[p,i,2:] - 2*T[p,i,1:-1]) + 4*alpha_r_E4*dt/dr**2*(T[p,i+1,1:-1]-T[p,i,1:-1])
#
#
#                         #node V4
#                         cp_V4 = cp_const + cp_c1 * T[p,0, -1] + cp_c2 *  T[p,0, -1] ** 2 + cp_c3 *  T[p,0, -1] ** 3
#                         alpha_r_V4 = 1 / (alpha_r_A * T[p,0, -1] + alpha_r_B)
#                         alpha_z_V4 = 1 / (alpha_z_A * T[p,0, -1] + alpha_z_B)
#                         T[p+1,0,-1] = T[p,0,-1] + 4*alpha_r_V4*dt/dr**2*(T[p,1,-1] - T[p,0,-1]) + 2*alpha_z_V4*dt/dz**2*(T[p,0,-2]-T[p,0,-1]) \
#                             + dt/(dz*rho*cp_V4)*(4*Cm_back[0]/(np.pi*dr**2)-2*emissivity_back*sigma_sb*T[p,0,-1]**4)
#
#                     elif (i>0) and (i<Nr-1):
#
#                         # E1
#                         cp_E1 = cp_const + cp_c1 * T[p,i, 0] + cp_c2 *  T[p,i, 0] ** 2 + cp_c3 *  T[p,i, 0] ** 3
#                         #print(cp_E1)
#                         alpha_r_E1 = 1 / (alpha_r_A * T[p,i, 0] + alpha_r_B)
#                         alpha_z_E1 = 1 / (alpha_z_A * T[p,i, 0] + alpha_z_B)
#
#                         T[p+1,i,0] = T[p,i,0] + dt/(dz*rho*cp_E1)*(2*absorptivity_front*q_solar[p,i]+Cm_front[i]/(np.pi*rm[i]*dr) -2*emissivity_front*sigma_sb*T[p,i,0]**4) \
#                             + alpha_r_E1*dt/(rm[i]*dr**2)*((rm[i]-dr/2)*(T[p,i-1,0] - T[p,i,0]) + (rm[i]+dr/2)*(T[p,i+1,0] - T[p,i,0])) \
#                             + alpha_z_E1*dt/(dz**2)*(T[p,i,1] - T[p,i,0])
#
#
#                         # node B, this need to be tested very carefully
#                         cp_B = cp_const + cp_c1 * T[p, i, 1:-1] + cp_c2 * T[p, i, 1:-1] ** 2 + cp_c3 * T[p, i, 1:-1] ** 3
#                         alpha_r_Body = 1 / (alpha_r_A * T[p, i, 1:-1] + alpha_r_B)
#                         alpha_z_Body = 1 / (alpha_z_A * T[p, i, 1:-1] + alpha_z_B)
#                         T[p+1, i, 1:-1] = T[p, i, 1:-1] + alpha_r_Body*dt/(rm[i]*dr**2)*((rm[i]-dr/2)*(T[p,i-1,1:-1] - T[p,i,1:-1])+ (rm[i]+dr/2)*(T[p,i+1,1:-1] - T[p,i,1:-1]))\
#                         + alpha_z_Body*dt/dz**2*(T[p,i,0:-2] + T[p,i,2:] - 2*T[p,i,1:-1])
#
#
#                         #node E3
#                         cp_E3 = cp_const + cp_c1 * T[p,i, -1] + cp_c2 *  T[p,i, -1] ** 2 + cp_c3 *   T[p,i, -1] ** 3
#                         alpha_r_E3 = 1 / (alpha_r_A *  T[p,i, -1] + alpha_r_B)
#                         alpha_z_E3 = 1 / (alpha_z_A *  T[p,i, -1] + alpha_z_B)
#                         T[p+1, i, -1] =  T[p,i, -1] + dt/(dz*rho*cp_E3)*(Cm_back[i]/(np.pi*rm[i]*dr)-2*emissivity_back*sigma_sb*T[p,i,-1]**4) \
#                                              + alpha_r_E3*dt/(rm[i]*dr**2)*((rm[i]-dr/2)*(T[p,i-1,-1] - T[p,i,-1])+(rm[i]+dr/2)*(T[p,i+1,-1] - T[p,i,-1])) \
#                                              + alpha_z_E3*dt/dz**2*(T[p,i,-2]-T[p,i,-1])
#
#                     elif i == Nr-1:
#
#                         # node V2
#                         cp_V2 = cp_const + cp_c1 * T[p, -1, 0] + cp_c2 * T[p, -1, 0] ** 2 + cp_c3 * T[p, -1, 0] ** 3
#                         alpha_r_V2 = 1 / (alpha_r_A * T[p, -1, 0] + alpha_r_B)
#                         alpha_z_V2 = 1 / (alpha_z_A * T[p, -1, 0] + alpha_z_B)
#                         T[p + 1, -1, 0] = T[p, -1, 0] + dt / (dz * rho * cp_V2) * (
#                                     2 * absorptivity_front * q_solar[p, -1] + 2 * Cm_front[-1] / (
#                                         np.pi * rm[-1] * dr) - 2 * sigma_sb * emissivity_front * T[p, -1, 0] ** 4) \
#                                           + dt / (dr * rho * cp_V2) * (
#                                                       2 * sigma_sb * T_LBW ** 4 - 2 * sigma_sb * emissivity_front * T[
#                                                   p, -1, 0] ** 4) + 2 * alpha_r_V2 * dt / dr ** 2 * (T[p, -2, 0] - T[p, -1, 0]) \
#                                           + 2 * alpha_z_V2 * dt / dz ** 2 * (T[p, -1, 1] - T[p, -1, 0])
#
#
#                         # node E2
#                         cp_E2 = cp_const + cp_c1 * T[p, -1, 1:-1] + cp_c2 * T[p, -1, 1:-1] ** 2 + cp_c3 * T[p, -1, 1:-1] ** 3
#                         alpha_r_E2 = 1 / (alpha_r_A * T[p, -1, 1:-1] + alpha_r_B)
#                         alpha_z_E2 = 1 / (alpha_z_A * T[p, -1, 1:-1] + alpha_z_B)
#                         T[p+1, -1, 1:-1] = T[p, -1, 1:-1] + 2*alpha_r_E2*dt/dr**2*(T[p,-2,1:-1]-T[p,-1,1:-1]) + 2*alpha_z_E2*dt/dz**2*(T[p,-1,0:-2]+T[p,-1,2:]-2*T[p,-1,1:-1]) \
#                             + 2*dt/(dr*rho*cp_E2)*(-emissivity_front*sigma_sb*T[p,-1,1:-1]**4 + absorptivity_front*sigma_sb*T_LBW**4)
#
#
#
#                         # node V3
#                         cp_V3 = cp_const + cp_c1 * T[p,-1, -1] + cp_c2 *  T[p,-1, -1] ** 2 + cp_c3 *  T[p,-1, -1] ** 3
#                         alpha_r_V3 = 1 / (alpha_r_A *  T[p,-1, -1] + alpha_r_B)
#                         alpha_z_V3 = 1 / (alpha_z_A *  T[p,-1, -1] + alpha_z_B)
#                         T[p+1,-1,-1] = T[p,-1,-1] + dt/(dz*rho*cp_V3)*(2*Cm_back[-1]/(np.pi*rm[-1]*dr) - 2*sigma_sb*emissivity_back*T[p,-1,-1]**4) \
#                                            + dt/(dr*rho*cp_V3)*(2*sigma_sb*T_LBW**4 - 2*sigma_sb*emissivity_back*T[p,-1,-1]**4) + 2*alpha_r_V3*dt/dr**2*(T[p,-2,-1]-T[p,-1,-1]) \
#                                            + 2*alpha_z_V3*dt/dz**2*(T[p,-1,-2]-T[p,-1,-1])
#
#                         """
#
#
#
#                 if (p > 0) & (p % N_one_cycle == 0) & (simulated_amp_phase_extraction_method != 'fft'):
#                     A_max = np.max(T[p - N_one_cycle:p, :,-1], axis=0)
#                     A_min = np.min(T[p - N_one_cycle:p, :,-1], axis=0)
#                     if np.max(np.abs((T_temp[:] - T[p, :,-1]) / (A_max - A_min))) < 2e-3:
#                         N_steady_count += 1
#                         if N_steady_count == N_stable_cycle_output:  # only need 2 periods to calculate amplitude and phase
#                             time_index = p
#                             print("stable temperature profile has been obtained @ iteration N= {}!".format(
#                                 int(p / N_one_cycle)))
#                             break
#                     T_temp = T[p, :,-1]
#
#                 if p == Nt - 2:
#                     time_index = p
#                             # T_temp = T[p, :]
#                     print("Error! No stable temperature profile was obtained!")
#
#         return T[:time_index, :,:], time_simulation[:time_index], r, N_one_cycle, q_solar[:time_index,:]








def simulation_result_amplitude_phase_extraction(sample_information,
                                                 vacuum_chamber_setting, solar_simulator_settings,
                                                 light_source_property, numerical_simulation_setting,code_directory,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all):
    R0 = vacuum_chamber_setting['R0']
    R_analysis = vacuum_chamber_setting['R_analysis']
    simulated_amp_phase_extraction_method = numerical_simulation_setting['simulated_amp_phase_extraction_method']
    N_stable_cycle_output = numerical_simulation_setting['N_stable_cycle_output']


    T_temp, time_T_, r_,N_one_cycle, q_solar = radial_finite_difference_explicit(sample_information, vacuum_chamber_setting, solar_simulator_settings,
                       light_source_property, numerical_simulation_setting,df_solar_simulator_VQ, sigma_df,code_directory,df_view_factor,df_LB_details_all)

    f_heating = solar_simulator_settings['f_heating']
    gap = numerical_simulation_setting['gap']

    # I want max 50 samples per period
    N_skip_time = max(int(N_one_cycle /numerical_simulation_setting['N_simulated_sample_each_cycle']),1) # avoid N_skip to be zero

    if numerical_simulation_setting['axial_conduction'] == True:
        T_ = T_temp[:,:,-1] # only take temperature profiles at the last node facing IR
    else:
        T_ = T_temp # 1D heat conduction keep the temperature profile as is

    df_temperature_simulation_steady_oscillation = pd.DataFrame(data=T_[-N_stable_cycle_output*N_one_cycle::N_skip_time,:])  # return a dataframe containing radial averaged temperature and relative time
    df_temperature_simulation_steady_oscillation['reltime'] = time_T_[-N_stable_cycle_output*N_one_cycle::N_skip_time]

    df_temperature_transient = pd.DataFrame(data = T_[::N_skip_time,:])
    df_temperature_transient['reltime'] = time_T_[::N_skip_time]

    df_solar_simulator_heat_flux = pd.DataFrame(data = q_solar[-N_stable_cycle_output*N_one_cycle::N_skip_time,:])
    df_solar_simulator_heat_flux['reltime'] = time_T_[-N_stable_cycle_output*N_one_cycle::N_skip_time]
    df_amp_phase_simulated = batch_process_horizontal_lines(df_temperature_simulation_steady_oscillation, f_heating, R0, gap, R_analysis, simulated_amp_phase_extraction_method) # The default frequency analysis for simulated temperature profile is sine

    return df_amp_phase_simulated,df_temperature_simulation_steady_oscillation,df_solar_simulator_heat_flux,df_temperature_transient



def rw_mcmc_likelihood(params, df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
            solar_simulator_settings, light_source_property, numerical_simulation_setting,code_directory,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all,sigma_dA,sigma_dP):
    sample_information['alpha_r'] = params[0]

    df_amp_phase_simulated,df_temperature_simulation,df_light_source,df_temperature_transient = simulation_result_amplitude_phase_extraction(sample_information, vacuum_chamber_setting,
                                                                          solar_simulator_settings,
                                                                          light_source_property,
                                                                          numerical_simulation_setting,code_directory, df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all)


    P_dA_log = np.log(norm.pdf(np.array(df_amplitude_phase_measurement['amp_ratio']),np.array(df_amp_phase_simulated['amp_ratio']), sigma_dA))
    P_dP_log = np.log(norm.pdf(np.array(df_amplitude_phase_measurement['phase_diff']),np.array(df_amp_phase_simulated['phase_diff']), sigma_dP))


    return np.sum(P_dA_log)+np.sum(P_dP_log)


def MCMC_acceptance(x,x_new):
    if x_new > x:
        return True
    else:
        accept = np.random.uniform(0, 1)
        return (accept < (np.exp(x_new - x)))

def rw_proposal(params,step_alpha_r):
    alpha_r_new = np.random.normal(params[0], step_alpha_r)
    return [alpha_r_new]


def rw_metropolis_hastings(params_int,N_mcmc_sample_max, step_alpha_r,sigma_dA,sigma_dP, df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
            solar_simulator_settings, light_source_property, numerical_simulation_setting,code_directory,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all):

    params = params_int
    accepted = []
    n_sample = 0
    n_reject = 1
    temp_data = []
    params_likelihood = rw_mcmc_likelihood(params, df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
            solar_simulator_settings, light_source_property, numerical_simulation_setting,code_directory,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all,sigma_dA,sigma_dP)


    while(n_sample<N_mcmc_sample_max):

        params_new = rw_proposal(params,step_alpha_r)
        params_new_likelihood = rw_mcmc_likelihood(params, df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
            solar_simulator_settings, light_source_property, numerical_simulation_setting,code_directory,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all,sigma_dA,sigma_dP)

        if (MCMC_acceptance(params_likelihood,params_new_likelihood)):
            n_sample += 1
            accept_rate = n_sample / (n_reject + n_sample)
            params = params_new

            temp_data = params_new + [time.time(), accept_rate]
            accepted.append(temp_data)
            print('iter ' + str(n_sample) + ', accepted: ' + str(
                params_new) + ', acceptance rate: ' + "{0:.4g}".format(accept_rate))
        else:
            n_reject += 1

        params_likelihood = params_new_likelihood

    return accepted





def residual_update(params, df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
             solar_simulator_settings, light_source_property, numerical_simulation_setting,code_directory,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all):


    error = None

    regression_module = numerical_simulation_setting['regression_module']
    regression_method = numerical_simulation_setting['regression_method']


    if numerical_simulation_setting['regression_parameter'] == 'sigma_s':
        if regression_module == 'lmfit':
            light_source_property['sigma_s'] = params['sigma_s'].value
        elif regression_module == 'scipy.optimize-NM':
            light_source_property['sigma_s'] = params[0]

    elif numerical_simulation_setting['regression_parameter'] == 'alpha_r':
        if regression_module == 'lmfit':
            sample_information['alpha_r'] = params['alpha_r'].value
            #print("params['alpha_r']: "+str(params['alpha_r']))
        elif regression_module == 'scipy.optimize-NM':
            sample_information['alpha_r'] = params[0]

    elif numerical_simulation_setting['regression_parameter'] == 'emissivity_front':
        if regression_module == 'lmfit':
            sample_information['emissivity_front'] = params['emissivity_front'].value
            #print("params['alpha_r']: "+str(params['alpha_r']))
        elif regression_module == 'scipy.optimize-NM':
            sample_information['emissivity_front'] = params[0]

    df_amp_phase_simulated,df_temperature_simulation,df_light_source,df_temperature_transient = simulation_result_amplitude_phase_extraction(sample_information, vacuum_chamber_setting,
                                                                          solar_simulator_settings,
                                                                          light_source_property,
                                                                          numerical_simulation_setting,code_directory, df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all)

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

    elif regression_module == 'scipy.optimize-NM':
        if regression_method == 'amplitude':
            error = np.sum(amplitude_relative_error)
        elif regression_method == 'phase':
            error = np.sum(phase_relative_error)
        elif regression_method == 'amplitude-phase':
            error = np.sum(amplitude_relative_error)+ np.sum(amplitude_relative_error)

    return error




def show_regression_results(param_name, regression_result, df_temperature, df_amplitude_phase_measurement,
                            sample_information,vacuum_chamber_setting, solar_simulator_settings, light_source_property,numerical_simulation_setting,code_directory,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all):

    if param_name == 'alpha_r':
        sample_information['alpha_r'] = regression_result
        title_text = 'alpha = {:.2E} m2/s, DC = {} V, d = {} cm'.format(regression_result,
                                                                              solar_simulator_settings['V_DC'],
                                                                              vacuum_chamber_setting['focal_shift'])

    elif param_name == 'sigma_s':
        light_source_property['sigma_s'] = regression_result
        title_text = 'sigma_s = {:.2E}, DC = {} V, d = {} cm'.format(regression_result,
                                                                           solar_simulator_settings['V_DC'],
                                                                           vacuum_chamber_setting['focal_shift'])

    df_amp_phase_simulated, df_temperature_simulation,df_light_source,df_temperature_transient = simulation_result_amplitude_phase_extraction(sample_information,
                                                                                                     vacuum_chamber_setting,
                                                                                                     solar_simulator_settings,
                                                                                                     light_source_property,
                                                                                                     numerical_simulation_setting,code_directory,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all)

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

    plt.xlabel('R (pixels)', fontsize=14, fontweight='bold')
    plt.ylabel('Amplitude Ratio', fontsize=14, fontweight='bold')

    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=12)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=12)
        tick.label.set_fontweight('bold')
    plt.legend(prop={'weight': 'bold', 'size': 12})
    plt.title('{}, f = {} Hz'.format(sample_information['rec_name'], solar_simulator_settings['f_heating']),
              fontsize=11, fontweight='bold')
    # plt.legend()

    plt.subplot(132)
    plt.scatter(df_amplitude_phase_measurement['r'], df_amplitude_phase_measurement['phase_diff'], facecolors='none',
                edgecolors='r', label='measurement results')
    plt.scatter(df_amp_phase_simulated['r'], df_amp_phase_simulated['phase_diff'], marker='+',
                label='regression results')

    plt.xlabel('R (pixels)', fontsize=14, fontweight='bold')
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


def result_visulization_one_case(df_exp_condition_i, code_directory, data_directory, df_result_i,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all,df_sample_cp_rho_alpha_all):

    fig = plt.figure(figsize=(17, 16))
    colors = ['red', 'black', 'green', 'blue', 'orange', 'magenta', 'brown', 'yellow', 'purple', 'cornflowerblue']

    x0 = int(df_exp_condition_i['x0'])
    y0 = int(df_exp_condition_i['y0'])
    pr = float(df_exp_condition_i['pr'])
    Rmax = int(df_exp_condition_i['Rmax'])

    rec_name = df_exp_condition_i['rec_name']
    simulated_amp_phase_extraction_method = df_exp_condition_i['simulated_amp_phase_extraction_method']
    f_heating = float(df_exp_condition_i['f_heating'])

    R0 = int(df_exp_condition_i['R0'])
    gap = int(df_exp_condition_i['gap'])
    R_analysis = int(df_exp_condition_i['R_analysis'])

    bb = df_exp_condition_i['anguler_range']
    bb = bb[1:-1]  # reac_excel read an element as an array
    index = None
    anguler_range = []
    while (index != -1):
        index = bb.find("],[")
        element = bb[:index]
        d = element.find(",")
        element_after_comma = element[d + 1:]
        element_before_comma = element[element.find("[") + 1:d]
        # print('Before = {} and after = {}'.format(element_before_comma,element_after_comma))
        bb = bb[index + 2:]
        anguler_range.append([int(element_before_comma), int(element_after_comma)])


    focal_shift = float(df_exp_condition_i['focal_shift'])
    V_DC = float(df_exp_condition_i['V_DC'])
    exp_amp_phase_extraction_method = df_exp_condition_i['exp_amp_phase_extraction_method']

    output_name = rec_name
    path = data_directory + rec_name + "//"

    df_temperature_list = []
    df_amp_phase_list = []

    plt.subplot(331)
    df_temperature_list, df_averaged_temperature = radial_temperature_average_disk_sample_several_ranges(x0, y0, Rmax,
                                                                                                         anguler_range,
                                                                                                         pr, path,
                                                                                                         rec_name,
                                                                                                         output_name,
                                                                                                         'MA', 4,
                                                                                                         code_directory)
    for j, angle in enumerate(anguler_range):
        # note radial_temperature_average_disk_sample automatically checks if a dump file exist
        df_temperature = df_temperature_list[j]
        df_amplitude_phase_measurement = batch_process_horizontal_lines(df_temperature, f_heating, R0, gap, R_analysis,
                                                                        exp_amp_phase_extraction_method)
        df_temperature_list.append(df_temperature)
        df_amp_phase_list.append(df_amplitude_phase_measurement)

        plt.scatter(df_amplitude_phase_measurement['r'],
                    df_amplitude_phase_measurement['amp_ratio'], facecolors='none',
                    s=60, linewidths=2, edgecolor=colors[j], label=str(angle[0]) + ' to ' + str(angle[1]) + ' Degs')

    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')

    plt.xlabel('R (pixels)', fontsize=12, fontweight='bold')
    plt.ylabel('Amplitude Ratio', fontsize=12, fontweight='bold')
    plt.title('rec = {}'.format(rec_name), fontsize=12, fontweight='bold')
    plt.legend()

    plt.subplot(332)

    for j, angle in enumerate(anguler_range):
        df_temperature = df_temperature_list[j]
        df_amplitude_phase_measurement = df_amp_phase_list[j]
        plt.scatter(df_amplitude_phase_measurement['r'],
                    df_amplitude_phase_measurement['phase_diff'], facecolors='none',
                    s=60, linewidths=2, edgecolor=colors[j], label=str(angle[0]) + ' to ' + str(angle[1]) + ' Degs')

    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')

    plt.xlabel('R (pixels)', fontsize=12, fontweight='bold')
    plt.ylabel('Phase difference (rad)', fontsize=12, fontweight='bold')
    # plt.title('x0 = {}, y0 = {}, R0 = {}'.format(x0, y0, R0), fontsize=12, fontweight='bold')
    plt.legend()


    plt.subplot(333)

    for j, angle in enumerate(anguler_range):
        df_temperature = df_temperature_list[j]
        time_max = 6 * 1 / f_heating  # only show 10 cycles
        df_temperature = df_temperature.query('reltime<{:.2f}'.format(time_max))

        plt.plot(df_temperature['reltime'],
                 df_temperature.iloc[:, R0], linewidth=2, color=colors[j],
                 label=str(angle[0]) + ' to ' + str(angle[1]) + ' Degs')

    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')

    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Temperature (K)', fontsize=12, fontweight='bold')

    plt.title('f_heating = {} Hz'.format(f_heating), fontsize=12, fontweight='bold')
    # plt.title('R_ayalysis = {}, gap = {}'.format(R_analysis, gap), fontsize=12, fontweight='bold')
    plt.legend()

    rep_csv_dump_path = code_directory + "temperature cache dump//" + rec_name + '_rep_dump'
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

        df_first_frame = pd.read_csv(file_name_0, skiprows=5, header=None)

        N_mid = int(len([path + x for x in os.listdir(path)]) / 3)
        #N_mid = 20
        file_name_1 = [path + x for x in os.listdir(path)][N_mid]
        n2 = file_name_1.rfind('//')
        n3 = file_name_1.rfind('.csv')
        frame_num_mid = file_name_1[n2 + 2:n3]

        df_mid_frame = pd.read_csv(file_name_1, skiprows=5, header=None)

        temp_dump = [df_first_frame, df_mid_frame, frame_num_first, frame_num_mid]

        pickle.dump(temp_dump, open(rep_csv_dump_path, "wb"))

    plt.subplot(334)

    xmin = x0 - R0 - R_analysis
    xmax = x0 + R0 + R_analysis
    ymin = y0 - R0 - R_analysis
    ymax = y0 + R0 + R_analysis
    Z = np.array(df_first_frame.iloc[ymin:ymax, xmin:xmax])
    x = np.arange(xmin, xmax, 1)
    y = np.arange(ymin, ymax, 1)
    X, Y = np.meshgrid(x, y)

    # mid = int(np.shape(Z)[1] / 2)
    x3 = min(R0 + 10, R0 + R_analysis - 20)

    manual_locations = [(x0 - R0 + 15, y0 - R0 + 15), (x0 - R0, y0 - R0), (x0 - x3, y0 - x3)]
    # fig, ax = plt.subplots(figsize=(6, 6))
    ax = plt.gca()
    CS = ax.contour(X, Y, Z, 18)
    plt.plot([xmin, xmax], [y0, y0], ls='-.', color='k', lw=2)  # add a horizontal line cross x0,y0
    plt.plot([x0, x0], [ymin, ymax], ls='-.', color='k', lw=2)  # add a vertical line cross x0,y0

    R_angle_show = R0 + R_analysis

    circle1 = plt.Circle((x0, y0), R0, edgecolor='r', fill=False, linewidth=3, linestyle='-.')
    circle2 = plt.Circle((x0, y0), R0 + R_analysis, edgecolor='k', fill=False, linewidth=3, linestyle='-.')

    circle3 = plt.Circle((x0, y0), int(0.01 / pr), edgecolor='black', fill=False, linewidth=3, linestyle='dotted')

    ax.invert_yaxis()
    ax.clabel(CS, inline=1, fontsize=12, manual=manual_locations)
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)

    ax.set_xlabel('x (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('y (pixels)', fontsize=12, fontweight='bold')
    ax.set_title('x0 = {}, y0 = {}, R0 = {}'.format(x0, y0, R0), fontsize=12, fontweight='bold')

    for j, angle in enumerate(anguler_range):
        plt.plot([x0, x0 + R_angle_show * np.cos(angle[0] * np.pi / 180)], [y0,
                                                                            y0 + R_angle_show * np.sin(
                                                                                angle[0] * np.pi / 180)], ls='-.',
                 color='blue', lw=2)
        # plt.plot([x0, x0 + R_angle_show * np.cos(angle[1] * np.pi / 180)], [y0,
        #                     y0 + R_angle_show * np.sin(angle[1] * np.pi / 180)], ls='dotted', color='blue', lw=2)

    plt.subplot(335)

    xmin = x0 - R0 - R_analysis
    xmax = x0 + R0 + R_analysis
    ymin = y0 - R0 - R_analysis
    ymax = y0 + R0 + R_analysis
    Z = np.array(df_mid_frame.iloc[ymin:ymax, xmin:xmax])
    x = np.arange(xmin, xmax, 1)
    y = np.arange(ymin, ymax, 1)
    X, Y = np.meshgrid(x, y)

    # mid = int(np.shape(Z)[1] / 2)
    x3 = min(R0 + 10, R0 + R_analysis - 20)

    manual_locations = [(x0 - R0 + 15, y0 - R0 + 15), (x0 - R0, y0 - R0), (x0 - x3, y0 - x3)]
    # fig, ax = plt.subplots(figsize=(6, 6))
    ax = plt.gca()
    CS = ax.contour(X, Y, Z, 18)
    plt.plot([xmin, xmax], [y0, y0], ls='-.', color='k', lw=2)  # add a horizontal line cross x0,y0
    plt.plot([x0, x0], [ymin, ymax], ls='-.', color='k', lw=2)  # add a vertical line cross x0,y0

    R_angle_show = R0 + R_analysis

    circle1 = plt.Circle((x0, y0), R0, edgecolor='r', fill=False, linewidth=3, linestyle='-.')
    circle2 = plt.Circle((x0, y0), R0 + R_analysis, edgecolor='k', fill=False, linewidth=3, linestyle='-.')

    circle3 = plt.Circle((x0, y0), int(0.01 / pr), edgecolor='black', fill=False, linewidth=3, linestyle='dotted')

    ax.invert_yaxis()
    ax.clabel(CS, inline=1, fontsize=12, manual=manual_locations)
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)

    ax.set_xlabel('x (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('y (pixels)', fontsize=12, fontweight='bold')
    ax.set_title('R_ayalysis = {}, gap = {}'.format(R_analysis, gap), fontsize=12, fontweight='bold')
    # plt.title('R_ayalysis = {}, gap = {}'.format(R_analysis, gap), fontsize=12, fontweight='bold')

    for j, angle in enumerate(anguler_range):
        plt.plot([x0, x0 + R_angle_show * np.cos(angle[0] * np.pi / 180)], [y0,
                                                                            y0 + R_angle_show * np.sin(
                                                                                angle[0] * np.pi / 180)], ls='-.',
                 color='blue', lw=2)
        # plt.plot([x0, x0 + R_angle_show * np.cos(angle[1] * np.pi / 180)], [y0,
        #                     y0 + R_angle_show * np.sin(angle[1] * np.pi / 180)], ls='dotted', color='blue', lw=2)

    plt.subplot(336)
    T_mean_list = np.array(
        [np.array(df_temperature.iloc[:, R0:R0 + R_analysis].mean(axis=0)) for df_temperature in df_temperature_list])
    T_max_list = np.array(
        [np.array(df_temperature.iloc[:, R0:R0 + R_analysis].max(axis=0)) for df_temperature in df_temperature_list])
    T_min_list = np.array(
        [np.array(df_temperature.iloc[:, R0:R0 + R_analysis].min(axis=0)) for df_temperature in df_temperature_list])

    plt.plot(np.arange(R0, R0 + R_analysis), T_mean_list.mean(axis=0), linewidth=2, label="mean temperature")
    plt.plot(np.arange(R0, R0 + R_analysis), T_max_list.mean(axis=0), linewidth=2, label="max temperature")
    plt.plot(np.arange(R0, R0 + R_analysis), T_min_list.mean(axis=0), linewidth=2, label="min temperature")

    ax = plt.gca()
    ax.set_xlabel('R (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temperature (K)', fontsize=12, fontweight='bold')

    # plt.title('R_ayalysis = {}, gap = {}'.format(R_analysis, gap), fontsize=12, fontweight='bold')
    ax.set_title("focal = {} cm, VDC = {} V".format(focal_shift, V_DC), fontsize=12, fontweight='bold')
    plt.legend()



    # df_sample_cp_rho_alpha_all = pd.read_excel(code_directory + "sample specifications//sample properties.xlsx",
    #                                            sheet_name="sample properties")


    #df_solar_simulator_VQ = pd.read_csv(code_directory + "sample specifications//9_14_Amax_Fv_d_correlations.csv")

    #sigma_df = pd.read_csv(code_directory + "sample specifications//Lorentzian sigma.csv")

    sample_name = df_exp_condition_i['sample_name']

    df_sample_cp_rho_alpha = df_sample_cp_rho_alpha_all.query("sample_name=='{}'".format(sample_name))

    T_average = df_result_i['T_average(K)']
    T_min = df_result_i['T_min(K)']



    sample_information = {'R': df_exp_condition_i['sample_radius(m)'], 't_z': df_exp_condition_i['sample_thickness(m)'],
                          'rho': float(df_sample_cp_rho_alpha['rho']),
                          'cp_const': float(df_sample_cp_rho_alpha['cp_const']), 'cp_c1':
                              float(df_sample_cp_rho_alpha['cp_c1']), 'cp_c2': float(df_sample_cp_rho_alpha['cp_c2']),
                          'cp_c3': float(df_sample_cp_rho_alpha['cp_c3']), 'alpha_r': df_result_i['alpha_r'],
                          'alpha_z': float(df_sample_cp_rho_alpha['alpha_z']), 'T_initial': T_average,
                          'emissivity_front': df_exp_condition_i['emissivity_front'],
                          'absorptivity_front': df_exp_condition_i['absorptivity_front'],
                          'emissivity_back': df_exp_condition_i['emissivity_back'],
                          'absorptivity_back': df_exp_condition_i['absorptivity_back'],
                          'sample_name': sample_name,'T_min':T_min,'alpha_r_A':float(df_sample_cp_rho_alpha['alpha_r_A']),'alpha_r_B':float(df_sample_cp_rho_alpha['alpha_r_B']),'alpha_z_A':float(df_sample_cp_rho_alpha['alpha_z_A']),'alpha_z_B':float(df_sample_cp_rho_alpha['alpha_z_B'])}

    # sample_information
    # Note that T_sur1 is read in degree C, must be converted to K.

    if df_exp_condition_i['regression_parameter'] == 'sigma_s':
        T_sur1 = float(df_exp_condition_i['T_sur1'])
        T_sur2 = float(df_exp_condition_i['T_sur2'])
    else:
        T_sur1 = None
        T_sur2 = None


    vacuum_chamber_setting = {'N_Rs': int(df_exp_condition_i['N_Rs']), 'R0': R0,
                              'T_sur1': T_sur1, 'T_sur2': T_sur2,
                              'focal_shift': focal_shift, 'R_analysis': R_analysis,'light_blocker':df_exp_condition_i['light_blocker']}
    # vacuum_chamber_setting

    #regression_method: either using amplitude and phase, or one of these two

    # numerical_simulation_setting = {'Nz': int(df_exp_condition_i['Nz']), 'Nr': int(df_exp_condition_i['Nr']),
    #                                 'equal_grid': df_exp_condition_i['equal_grid'],
    #                                 'N_cycle': int(df_exp_condition_i['N_cycle']),
    #                                 'vectorize': True,
    #                                 'Fo_criteria': float(df_exp_condition_i['Fo_criteria']),
    #                                 'simulated_amp_phase_extraction_method': df_exp_condition_i[
    #                                     'simulated_amp_phase_extraction_method'],
    #                                 'gap': int(df_exp_condition_i['gap']),
    #                                 'regression_module': df_exp_condition_i['regression_module'],
    #                                 'regression_method': df_exp_condition_i['regression_method'],
    #                                 'regression_parameter': df_exp_condition_i['regression_parameter'],
    #                                 'regression_residual_converging_criteria': df_exp_condition_i[
    #                                     'regression_residual_converging_criteria'],
    #                                 'view_factor_setting': df_exp_condition_i['view_factor_setting']
    #                                 }

    numerical_simulation_setting = {'Nz': int(df_exp_condition_i['Nz']), 'Nr': int(df_exp_condition_i['Nr']),
                                    'equal_grid': df_exp_condition_i['equal_grid'],
                                    'N_cycle': int(df_exp_condition_i['N_cycle']),
                                    'Fo_criteria': float(df_exp_condition_i['Fo_criteria']),
                                    'simulated_amp_phase_extraction_method': df_exp_condition_i['simulated_amp_phase_extraction_method'],
                                    'gap': int(df_exp_condition_i['gap']),
                                    'regression_module': df_exp_condition_i['regression_module'],
                                    'regression_method': df_exp_condition_i['regression_method'],
                                    'regression_parameter': df_exp_condition_i['regression_parameter'],
                                    'regression_residual_converging_criteria': df_exp_condition_i[
                                        'regression_residual_converging_criteria'],'view_factor_setting':df_exp_condition_i['view_factor_setting'],
                                    'axial_conduction':df_exp_condition_i['axial_conduction'],
                                    'analysis_mode':df_exp_condition_i['analysis_mode'],'N_stable_cycle_output':int(2)}



    # the code is executed using vectorized approach by default

    # numerical_simulation_setting
    solar_simulator_settings = {'f_heating': float(df_exp_condition_i['f_heating']),
                                'V_amplitude': float(df_exp_condition_i['V_amplitude']),
                                'V_DC': V_DC, 'rec_name': rec_name}

    param_name = df_exp_condition_i['regression_parameter']

    if param_name == 'alpha_r':
        sample_information['alpha_r'] = df_result_i['alpha_r']
        sigma_temp = None
        title_text = 'alpha = {:.2E} m2/s, DC = {} V, d = {} cm'.format(df_result_i['alpha_r'],
                                                                        solar_simulator_settings['V_DC'],
                                                                        vacuum_chamber_setting['focal_shift'])

    elif param_name == 'sigma_s':
        #light_source_property['sigma_s'] = df_result_i['sigma_s']
        sigma_temp = df_result_i['sigma_s']
        title_text = 'sigma_s = {:.2E}, DC = {} V, d = {} cm'.format(df_result_i['sigma_s'],
                                                                     solar_simulator_settings['V_DC'],
                                                                     vacuum_chamber_setting['focal_shift'])


    elif param_name == 'emissivity_front':
        sigma_temp = None
        sample_information['emissivity_front'] = df_result_i['emissivity_front']
        sample_information['absorptivity_front'] = df_result_i['emissivity_front']
        title_text = 'emissivity_front = {:.2E}, DC = {} V, d = {} cm'.format(df_result_i['emissivity_front'],
                                                                     solar_simulator_settings['V_DC'],
                                                                     vacuum_chamber_setting['focal_shift'])


    # solar_simulator_settings


    Amax, sigma_s, kvd, bvd = interpolate_light_source_characteristic(sigma_df, df_solar_simulator_VQ, sigma_temp, numerical_simulation_setting,vacuum_chamber_setting)

    light_source_property = {'Amax': Amax, 'sigma_s': sigma_s, 'kvd': kvd, 'bvd': bvd}
    # light_source_property




    df_amp_phase_simulated, df_temperature_simulation,df_light_source,df_temperature_transient = simulation_result_amplitude_phase_extraction(sample_information,
                                                                                                     vacuum_chamber_setting,
                                                                                                     solar_simulator_settings,
                                                                                                     light_source_property,
                                                                                                     numerical_simulation_setting,
                                                                                                     code_directory,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all)

    phase_relative_error = [abs((measure - calculate) / measure) for measure, calculate in
                            zip(df_amplitude_phase_measurement['phase_diff'], df_amp_phase_simulated['phase_diff'])]
    amplitude_relative_error = [abs((measure - calculate) / measure) for measure, calculate in
                                zip(df_amplitude_phase_measurement['amp_ratio'], df_amp_phase_simulated['amp_ratio'])]

    amp_residual_mean = np.mean(amplitude_relative_error)
    phase_residual_mean = np.mean(phase_relative_error)
    total_residual_mean = amp_residual_mean + phase_residual_mean

    # fig = plt.figure(figsize=(15, 5))
    # plt.scatter(df_result_IR_mosfata['r'],df_result_IR_mosfata['amp_ratio'],facecolors='none',edgecolors='k',label = 'Mostafa')
    plt.subplot(337)
    plt.scatter(df_amplitude_phase_measurement['r'], df_amplitude_phase_measurement['amp_ratio'], facecolors='none',
                edgecolors='r', label='measurement results')
    plt.scatter(df_amp_phase_simulated['r'], df_amp_phase_simulated['amp_ratio'], marker='+',
                label='regression results')

    plt.xlabel('R (pixels)', fontsize=12, fontweight='bold')
    plt.ylabel('Amplitude Ratio', fontsize=12, fontweight='bold')

    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=12)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=12)
        tick.label.set_fontweight('bold')
    plt.legend(prop={'weight': 'bold', 'size': 12})
    plt.title('{}, f = {} Hz'.format(sample_information['rec_name'], solar_simulator_settings['f_heating']),
              fontsize=11, fontweight='bold')
    # plt.legend()

    plt.subplot(338)
    plt.scatter(df_amplitude_phase_measurement['r'], df_amplitude_phase_measurement['phase_diff'], facecolors='none',
                edgecolors='r', label='measurement results')
    plt.scatter(df_amp_phase_simulated['r'], df_amp_phase_simulated['phase_diff'], marker='+',
                label='regression results')

    plt.xlabel('R (pixels)', fontsize=14, fontweight='bold')
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

    plt.subplot(339)

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

    plt.show()

    return fig


def parallel_result_visualization(df_exp_condition_spreadsheet_filename, df_results_all, num_cores, data_directory,
                                  code_directory,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all,df_sample_cp_rho_alpha_all):
    df_exp_conditions = pd.read_excel(
        code_directory + "batch process information//" + df_exp_condition_spreadsheet_filename)

    regression_fig_joblib_output = Parallel(n_jobs=num_cores, verbose=0)(
        delayed(result_visulization_one_case)(df_exp_conditions.iloc[i, :], code_directory, data_directory,
                                              df_results_all.iloc[i, :], df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all,df_sample_cp_rho_alpha_all) for i in
        tqdm(range(len(df_exp_conditions))))

    pickle.dump(regression_fig_joblib_output,
                open(code_directory + "regression figure dump//regression_figs_" + df_exp_condition_spreadsheet_filename,
                     "wb"))


def high_T_Angstrom_execute_one_case(df_exp_condition, data_directory, code_directory, df_amplitude_phase_measurement,df_temperature,df_sample_cp_rho_alpha_all,df_thermal_diffusivity_temperature_all, df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all):


    sample_name = df_exp_condition['sample_name']
    df_sample_cp_rho_alpha = df_sample_cp_rho_alpha_all.query("sample_name=='{}'".format(sample_name))

    # this function read a row from an excel spread sheet and execute

    rec_name = df_exp_condition['rec_name']
    view_factor_setting = df_exp_condition['view_factor_setting']

    regression_module = df_exp_condition['regression_module']
    N_simulated_sample_each_cycle = 50

    focal_shift = float(df_exp_condition['focal_shift'])
    VDC = float(df_exp_condition['V_DC'])

    T_sur1 = float(df_exp_condition['T_sur1'])
    T_sur2 = float(df_exp_condition['T_sur2'])

    # We need to evaluate the sample's average temperature in the region of analysis, and feed in a thermal diffusivity value from a reference source, this will be important for sigma measurement

    R0 = int(df_exp_condition['R0'])
    R_analysis  = int(df_exp_condition['R_analysis'])

    Nr = int(df_exp_condition['Nr'])

    alpha_r_A = float(df_sample_cp_rho_alpha['alpha_r_A'])
    alpha_r_B = float(df_sample_cp_rho_alpha['alpha_r_B'])
    alpha_z_A = float(df_sample_cp_rho_alpha['alpha_z_A'])
    alpha_z_B = float(df_sample_cp_rho_alpha['alpha_z_B'])

    #df_thermal_diffusivity_temperature = df_thermal_diffusivity_temperature_all.query("Material=='{}'".format(sample_name))
    #f_thermal_duffisivity_T = interp1d(df_thermal_diffusivity_temperature['Temperature C'], df_thermal_diffusivity_temperature['Thermal diffsivity'])
    #alpha_r = f_thermal_duffisivity_T(T_average - 273.15) #convert back to C

    N_stable_cycle_output = 2 # by default, we only analyze 2 cycle using sine fitting method

    T_average = np.sum(
        [2 * np.pi *  m_ *  np.mean(df_temperature.iloc[:, m_]) for m_ in np.arange(R0, R_analysis+R0, 1)]) / (
                        ((R_analysis+R0) ** 2 - (R0) ** 2) * np.pi) # unit in K

    #np.min(df_temperature_list[0].loc[:, df_temperature_list[0].columns != 'reltime'].min())

    T_min = np.min(df_temperature.iloc[:, Nr-15])

    alpha_r = 1/(alpha_r_A*T_average+alpha_r_B)

    sample_information = {'R': df_exp_condition['sample_radius(m)'], 't_z': df_exp_condition['sample_thickness(m)'],
                          'rho': float(df_sample_cp_rho_alpha['rho']),
                          'cp_const': float(df_sample_cp_rho_alpha['cp_const']), 'cp_c1':
                              float(df_sample_cp_rho_alpha['cp_c1']), 'cp_c2': float(df_sample_cp_rho_alpha['cp_c2']),
                          'cp_c3': float(df_sample_cp_rho_alpha['cp_c3']), 'alpha_r': alpha_r,
                          'alpha_z': float(df_sample_cp_rho_alpha['alpha_z']), 'T_initial': T_average,
                          'emissivity_front': df_exp_condition['emissivity_front'],
                          'absorptivity_front': df_exp_condition['absorptivity_front'],
                          'emissivity_back': df_exp_condition['emissivity_back'],
                          'absorptivity_back': df_exp_condition['absorptivity_back'],
                          'sample_name':sample_name,'T_min':T_min,'alpha_r_A':alpha_r_A,'alpha_r_B':alpha_r_B,'alpha_z_A':alpha_z_A,'alpha_z_B':alpha_z_B,'rec_name': rec_name}

    # sample_information
    # Note that T_sur1 is read in degree C, must be converted to K.
    # Indicate where light_blocker is used or not, option here: True, False

    vacuum_chamber_setting = {'N_Rs': int(df_exp_condition['N_Rs']), 'R0': R0,
                              'T_sur1': T_sur1, 'T_sur2': T_sur2,
                              'focal_shift':focal_shift,'R_analysis':R_analysis,'light_blocker':df_exp_condition['light_blocker']}
    # vacuum_chamber_setting

    numerical_simulation_setting = {'Nz': int(df_exp_condition['Nz']), 'Nr': Nr,
                                    'equal_grid': df_exp_condition['equal_grid'],
                                    'N_cycle': int(df_exp_condition['N_cycle']),
                                    'Fo_criteria': float(df_exp_condition['Fo_criteria']),
                                    'simulated_amp_phase_extraction_method': df_exp_condition['simulated_amp_phase_extraction_method'],
                                    'gap': int(df_exp_condition['gap']),
                                    'regression_module': df_exp_condition['regression_module'],
                                    'regression_method': df_exp_condition['regression_method'],
                                    'regression_parameter': df_exp_condition['regression_parameter'],
                                    'regression_residual_converging_criteria': df_exp_condition[
                                        'regression_residual_converging_criteria'],'view_factor_setting':df_exp_condition['view_factor_setting'],
                                    'axial_conduction':df_exp_condition['axial_conduction'],
                                    'analysis_mode':df_exp_condition['analysis_mode'],'N_stable_cycle_output':N_stable_cycle_output,'N_simulated_sample_each_cycle':N_simulated_sample_each_cycle}

    # numerical_simulation_setting
    solar_simulator_settings = {'f_heating': float(df_exp_condition['f_heating']),
                                'V_amplitude': float(df_exp_condition['V_amplitude']),
                                'V_DC': VDC}

    # solar_simulator_settings

    Amax, sigma_s, kvd, bvd = interpolate_light_source_characteristic(sigma_df, df_solar_simulator_VQ, float(df_exp_condition['p_initial']), numerical_simulation_setting,vacuum_chamber_setting)
    # Note the initial sigma_s value was set as float(df_exp_condition['p_initial']). If regression parameter was not sigma this does not matter, because it will take sigma from a different place anyway


    light_source_property = {'Amax': Amax, 'sigma_s': sigma_s, 'kvd': kvd, 'bvd': bvd}
    # light_source_property

    regression_result = None

    if regression_module == 'lmfit':

        params = Parameters()

        if df_exp_condition['regression_parameter'] == 'alpha_r':
            params.add('alpha_r', value=float(df_exp_condition['p_initial']), min=1e-8)

            out = lmfit.minimize(residual_update, params, args=(df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
            solar_simulator_settings, light_source_property, numerical_simulation_setting,code_directory,df_solar_simulator_VQ,sigma_df, df_view_factor,df_LB_details_all),
                                 xtol=df_exp_condition['regression_residual_converging_criteria'])

            regression_result = out.params['alpha_r'].value

        elif df_exp_condition['regression_parameter'] == 'sigma_s':
            params.add('sigma_s', value=float(df_exp_condition['p_initial']), min=1e-4, max= 5e-1)

            out = lmfit.minimize(residual_update, params, args=(df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
            solar_simulator_settings, light_source_property, numerical_simulation_setting,code_directory,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all),
                                 xtol=df_exp_condition['regression_residual_converging_criteria'])

            regression_result = out.params['sigma_s'].value

        elif df_exp_condition['regression_parameter'] == 'emissivity_front':
            params.add('emissivity_front', value=float(df_exp_condition['p_initial']), min=0.05, max = 1)

            out = lmfit.minimize(residual_update, params, args=(df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
            solar_simulator_settings, light_source_property, numerical_simulation_setting,code_directory,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all),
                                 xtol=df_exp_condition['regression_residual_converging_criteria'])

            regression_result = out.params['emissivity_front'].value


    elif regression_module == 'scipy.optimize-NM':

        res = optimize.minimize(residual_update, x0=float(df_exp_condition['p_initial']), args=(df_amplitude_phase_measurement, sample_information, vacuum_chamber_setting,
            solar_simulator_settings, light_source_property, numerical_simulation_setting,code_directory,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all), method='nelder-mead',
                                tol=df_exp_condition['regression_residual_converging_criteria'])
        regression_result = res['final_simplex'][0][0][0]


    print("recording {} completed.".format(rec_name))
    return regression_result, T_average, T_min


def steady_temperature_profile_check(x0, y0, N_Rmax, theta_range_list, pr, path, rec_name, output_name,
                                     method, num_cores, code_directory, R0, R_analysis, focal_plane_relative_location):
    df_temperature_list, df_averaged_temperature = radial_temperature_average_disk_sample_several_ranges(x0, y0, N_Rmax,
                                                                                                         theta_range_list,
                                                                                                         pr, path,
                                                                                                         rec_name,
                                                                                                         output_name,
                                                                                                         method,
                                                                                                         num_cores,
                                                                                                         code_directory)

    df_averaged_temperature = df_averaged_temperature.loc[:, df_averaged_temperature.columns != 'reltime']
    df_averaged_temperature = df_averaged_temperature.mean(axis=0)

    rep_csv_dump_path = code_directory + "temperature cache dump//" + rec_name + '_rep_dump'
    # rec_name
    if (os.path.isfile(rep_csv_dump_path)):  # First check if a dump file exist:
        print('Found previous dump file for representative temperature contour plots:' + rep_csv_dump_path)
        temp_dump = pickle.load(open(rep_csv_dump_path, 'rb'))
        df_first_frame = temp_dump[0]
        frame_num_first = temp_dump[1]

    else:  # If not we obtain the dump file, note the dump file is averaged radial temperature

        file_name_0 = [path + x for x in os.listdir(path)][0]
        n0 = file_name_0.rfind('//')
        n1 = file_name_0.rfind('.csv')
        frame_num_first = file_name_0[n0 + 2:n1]

        df_first_frame = pd.read_csv(file_name_0, skiprows=5, header=None)

        temp_dump = [df_first_frame, frame_num_first]

        pickle.dump(temp_dump, open(rep_csv_dump_path, "wb"))

    fig = plt.figure(figsize=(13, 6))

    plt.subplot(121)
    plt.plot(df_averaged_temperature)

    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')

    plt.xlabel('R (pixels)', fontsize=12, fontweight='bold')
    plt.ylabel('Temperature (K)', fontsize=12, fontweight='bold')

    plt.title("focal rel = {:.1f} cm".format(focal_plane_relative_location), fontsize=12, fontweight='bold')

    # plt.title('f_heating = {} Hz, rec = {}'.format(f_heating, rec_name), fontsize=12, fontweight='bold')

    plt.subplot(122)

    # R_analysis = 45
    angle_range = theta_range_list

    xmin = x0 - R0 - R_analysis
    xmax = x0 + R0 + R_analysis
    ymin = y0 - R0 - R_analysis
    ymax = y0 + R0 + R_analysis
    Z = np.array(df_first_frame.iloc[ymin:ymax, xmin:xmax])
    x = np.arange(xmin, xmax, 1)
    y = np.arange(ymin, ymax, 1)
    X, Y = np.meshgrid(x, y)

    x3 = min(R0 + 10, R0 + R_analysis - 20)

    manual_locations = [(x0 - R0 + 15, y0 - R0 + 15), (x0 - R0, y0 - R0), (x0 - x3, y0 - x3)]

    ax = plt.gca()
    CS = ax.contour(X, Y, Z, 12)
    plt.plot([xmin, xmax], [y0, y0], ls='-.', color='k', lw=2)  # add a horizontal line cross x0,y0
    plt.plot([x0, x0], [ymin, ymax], ls='-.', color='k', lw=2)  # add a vertical line cross x0,y0

    R_angle_show = R0 + R_analysis

    circle1 = plt.Circle((x0, y0), R0, edgecolor='r', fill=False, linewidth=3, linestyle='-.')
    circle2 = plt.Circle((x0, y0), R0 + R_analysis, edgecolor='k', fill=False, linewidth=3, linestyle='-.')

    circle3 = plt.Circle((x0, y0), int(0.01 / pr), edgecolor='black', fill=False, linewidth=3, linestyle='dotted')

    ax.invert_yaxis()
    ax.clabel(CS, inline=1, fontsize=12, manual=manual_locations)
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)

    ax.set_xlabel('x (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('y (pixels)', fontsize=12, fontweight='bold')
    print(frame_num_first)
    ax.set_title(frame_num_first, fontsize=12, fontweight='bold')

    for j, angle in enumerate(angle_range):
        plt.plot([x0, x0 + R_angle_show * np.cos(angle[0] * np.pi / 180)], [y0,
                                                                            y0 + R_angle_show * np.sin(
                                                                                angle[0] * np.pi / 180)], ls='-.',
                 color='blue', lw=2)

    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')

    plt.xlabel('R (pixels)', fontsize=12, fontweight='bold')
    plt.ylabel('Temperature (K)', fontsize=12, fontweight='bold')

    return fig, df_averaged_temperature


def parallel_temperature_average_batch_experimental_results_steady_state(df_exp_condition_spreadsheet_filename,
                                                                         data_directory, num_cores, code_directory):
    df_exp_condition_spreadsheet = pd.read_excel(
        code_directory + "batch process information//" + df_exp_condition_spreadsheet_filename)

    steady_state_figure_list = []
    df_temperature_list = []

    for i in range(len(df_exp_condition_spreadsheet)):

        df_exp_condition = df_exp_condition_spreadsheet.iloc[i, :]

        rec_name = df_exp_condition['rec_name']
        path = data_directory + str(rec_name) + "//"

        output_name = rec_name

        method = "MA"  # default uses Mosfata's code

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
        focal_plane_relative_location = df_exp_condition['focal_shift']

        bb = df_exp_condition['anguler_range']
        bb = bb[1:-1]  # reac_excel read an element as an array
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

        steady_state_figs, T_average_time_and_radius = steady_temperature_profile_check(x0, y0, Rmax, angle_range, pr,
                                                                                        path, rec_name, output_name,
                                                                                        method, num_cores,
                                                                                        code_directory, R0, R_analysis,
                                                                                        focal_plane_relative_location)

        df_temperature_list.append(T_average_time_and_radius)
        steady_state_figure_list.append(steady_state_figs)

    return steady_state_figure_list, df_temperature_list



def parallel_temperature_average_batch_experimental_results(df_exp_condition_spreadsheet_filename, data_directory, num_cores,code_directory):
    df_exp_condition_spreadsheet = pd.read_excel(code_directory+"batch process information//" + df_exp_condition_spreadsheet_filename)

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

        x0 = int(df_exp_condition['x0'])  # in pixels
        y0 = int(df_exp_condition['y0'])  # in pixels
        Rmax = int(df_exp_condition['Rmax'])  # in pixels
        # x0,y0,N_Rmax,pr,path,rec_name,output_name,method,num_cores
        pr = df_exp_condition['pr']
        # df_temperature = radial_temperature_average_disk_sample_several_ranges(x0,y0,Rmax,[[0,np.pi/3],[2*np.pi/3,np.pi],[4*np.pi/3,5*np.pi/3],[5*np.pi/3,2*np.pi]],pr,path,rec_name,output_name,method,num_cores)

        # After obtaining temperature profile, next we obtain amplitude and phase
        f_heating = df_exp_condition['f_heating']
        # 1cm ->35
        R0 = int(df_exp_condition['R0'])
        gap = int(df_exp_condition['gap'])
        # Rmax = 125
        R_analysis = int(df_exp_condition['R_analysis'])
        exp_amp_phase_extraction_method = df_exp_condition['exp_amp_phase_extraction_method']

        focal_plane_location = df_exp_condition['focal_shift']
        VDC = df_exp_condition['V_DC']

        bb = df_exp_condition['anguler_range']
        bb = bb[1:-1] # reac_excel read an element as an array
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
            angle_range.append([int(element_before_comma),int(element_after_comma)])

        # sum_std, diagnostic_figure = check_angular_uniformity(x0, y0, Rmax, pr, path, rec_name, output_name, method,
        #                                                       num_cores, f_heating, R0, gap, R_analysis,angle_range,focal_plane_location, VDC, exp_amp_phase_extraction_method,code_directory)
        #
        # diagnostic_figure_list.append(diagnostic_figure)

        df_temperature_list_all_ranges, df_temperature = radial_temperature_average_disk_sample_several_ranges(x0, y0, Rmax, angle_range, pr, path,
                                                                               rec_name, output_name, method, num_cores,code_directory)

        df_amplitude_phase_measurement = batch_process_horizontal_lines(df_temperature, f_heating, R0, gap, R_analysis,
                                                                        exp_amp_phase_extraction_method)
        df_temperature_list.append(df_temperature)


        df_amplitude_phase_measurement_list.append(df_amplitude_phase_measurement)


    return df_temperature_list, df_amplitude_phase_measurement_list


def parallel_regression_batch_experimental_results(df_exp_condition_spreadsheet_filename, data_directory, num_cores,code_directory,df_sample_cp_rho_alpha_all,df_thermal_diffusivity_temperature_all, df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all):
    df_exp_condition_spreadsheet = pd.read_excel(code_directory+"batch process information//" + df_exp_condition_spreadsheet_filename)

    df_temperature_list, df_amplitude_phase_measurement_list = parallel_temperature_average_batch_experimental_results(df_exp_condition_spreadsheet_filename, data_directory, num_cores,code_directory)

    joblib_output = Parallel(n_jobs=num_cores, verbose=0)(
        delayed(high_T_Angstrom_execute_one_case)(df_exp_condition_spreadsheet.iloc[i, :], data_directory,code_directory, df_amplitude_phase_measurement_list[i], df_temperature_list[i],df_sample_cp_rho_alpha_all,df_thermal_diffusivity_temperature_all, df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all) for i in
        tqdm(range(len(df_exp_condition_spreadsheet))))

    pickle.dump(joblib_output,open(code_directory+"result cache dump//regression_results_" + df_exp_condition_spreadsheet_filename, "wb"))




def parallel_result_summary(joblib_output,df_exp_condition_spreadsheet_filename,code_directory):
    regression_params = [joblib_output_[0] for joblib_output_ in joblib_output]
    T_average_list = [joblib_output_[3] for joblib_output_ in joblib_output]
    amp_ratio_residual_list = [joblib_output_[4] for joblib_output_ in joblib_output]
    phase_diff_residual_list = [joblib_output_[5] for joblib_output_ in joblib_output]
    amp_phase_residual_list = [joblib_output_[6] for joblib_output_ in joblib_output]

    df_exp_condition = pd.read_excel(df_exp_condition_spreadsheet_filename)

    sigma_s_list = []
    alpha_list = []

    for i,regression_type in enumerate(df_exp_condition['regression_parameter']):
        if regression_type == 'sigma_s':
            sigma_s_list.append(joblib_output[i][0])
            alpha_list.append(df_exp_condition['alpha_r'][i])
        elif regression_type == 'alpha_r':
            sigma_s_list.append(df_exp_condition['sigma_s'][i])
            alpha_list.append(joblib_output[i][0])

    df_results_all = pd.DataFrame({'rec_name':df_exp_condition['rec_name'],'f_heating':df_exp_condition['f_heating'],'VDC':df_exp_condition['V_DC'],
                                   'sigma_s':sigma_s_list,'T_average':T_average_list,'R0':df_exp_condition['R0'],'alpha_r':alpha_list,
                                   'regression_parameter':df_exp_condition['regression_parameter'],'T_sur1':df_exp_condition['T_sur1'],'emissivity':df_exp_condition['emissivity'],
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
                if type(series) == str:
                    df_ = df_results_all_.query("{}=='{}'".format(series_name, series))
                    axes[i, j].scatter(df_[x_name], df_[y_name], label="{} = '{}'".format(series_name, series))
                else:
                    df_ = df_results_all_.query("{}=={}".format(series_name, series))
                    axes[i, j].scatter(df_[x_name], df_[y_name], label="{} = {:.1E}".format(series_name, series))

                axes[i, j].set_xlabel(x_name)
                axes[i, j].set_ylabel(y_name)
                axes[i, j].set_ylim(ylim)
                axes[i, j].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
                axes[i, j].set_title("{} = {:.1E},{} = {:.1E}".format(row_name, row, column_name, column))

            if y_name == 'alpha_r':
                axes[i, j].scatter(df_results_all_[x_name], df_results_all_['alpha_theoretical'], label='reference')
    plt.tight_layout(h_pad=2)
    plt.legend()

    plt.show()


def display_high_dimensional_regression_results_one_row(x_name, y_name, column_name, series_name, df_results_all, ylim):
    column_items = np.unique(df_results_all[column_name])
    series_items = np.unique(df_results_all[series_name])
    f, axes = plt.subplots(1, len(column_items),
                           figsize=(int(len(column_items) * 5), 5),sharex=True, sharey=True)
    for j, column in enumerate(column_items):
        df_results_all_ = df_results_all.query("{} == {}".format(column_name, column))

        for series in series_items:
            if type(series) == str:
                df_ = df_results_all_.query("{}=='{}'".format(series_name, series))
                axes[j].scatter(df_[x_name], df_[y_name], label="{} = '{}'".format(series_name, series))
            else:
                df_ = df_results_all_.query("{}=={}".format(series_name, series))
                axes[j].scatter(df_[x_name], df_[y_name], label="{} = {:.1E}".format(series_name, series))

            axes[j].set_xlabel(x_name)
            axes[j].set_ylabel(y_name)
            axes[j].set_ylim(ylim)
            axes[j].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            axes[j].set_title("{} = {:.1E}".format(column_name, column))

        if y_name == 'alpha_r':
            axes[j].scatter(df_results_all_[x_name], df_results_all_['alpha_theoretical'], label='reference')

    plt.tight_layout(h_pad=2)
    plt.legend()

    plt.show()


def display_high_dimensional_regression_results_one_row_one_column(x_name, y_name, series_name, df_results_all, ylim):
    # column_items = np.unique(df_results_all[column_name])
    series_items = np.unique(df_results_all[series_name])
    plt.figure(figsize=(8, 6))

    for series in series_items:

        if type(series) == str:
            df_ = df_results_all.query("{}=='{}'".format(series_name, series))
            plt.scatter(df_[x_name], df_[y_name], label="'{}' = '{}'".format(series_name, series))
        else:
            df_ = df_results_all.query("{}=={}".format(series_name, series))
            plt.scatter(df_[x_name], df_[y_name], label="{} = {:.1E}".format(series_name, series))
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.ylim(ylim)
        axes = plt.gca()
        axes.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

    if y_name == 'alpha_r':
        plt.scatter(df_results_all[x_name], df_results_all['alpha_theoretical'], label='reference')

    plt.tight_layout(h_pad=2)
    plt.legend()

    plt.show()


def create_circular_mask(h, w, center, radius):

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def calculate_total_emission(file_name, x0, y0, R_sample, pr, emissivity_front, emissivity_back):
    df_temp_ = pd.read_csv(file_name, skiprows=5, header=None)
    df_temp = df_temp_ + 273.15  # convert C to K
    h, w = df_temp.shape
    mask = create_circular_mask(h, w, center=[x0, y0], radius=R_sample)
    IR_rec_array = np.array(df_temp)
    IR_rec_array[~mask] = 0

    sigma_sb = 5.67e-8  # Stefan-Boltzmann constant

    E_front = np.sum(emissivity_front * sigma_sb * IR_rec_array ** 4 * pr ** 2)
    E_back = np.sum(emissivity_back * sigma_sb * IR_rec_array ** 4 * pr ** 2)

    E_total = E_front + E_back

    return E_total


def batch_process_steady_state_emission_spread_sheet(data_directory, code_directory,
                                                     df_exp_condition_spreadsheet_filename, num_cores):
    df_exp_condition_spreadsheet = pd.read_excel(
        code_directory + "batch process information//" + df_exp_condition_spreadsheet_filename)
    emission_list = []

    for i in range(len(df_exp_condition_spreadsheet)):
        df_exp_condition = df_exp_condition_spreadsheet.iloc[i, :]

        rec_name = df_exp_condition['rec_name']
        path = data_directory + str(rec_name) + "//"

        output_name = rec_name
        x0 = df_exp_condition['x0']  # in pixels
        y0 = df_exp_condition['y0']  # in pixels
        Rmax = df_exp_condition['Rmax']  # in pixels
        # x0,y0,N_Rmax,pr,path,rec_name,output_name,method,num_cores
        pr = df_exp_condition['pr']
        # df_temperature = radial_temperature_average_disk_sample_several_ranges(x0,y0,Rmax,[[0,np.pi/3],[2*np.pi/3,np.pi],[4*np.pi/3,5*np.pi/3],[5*np.pi/3,2*np.pi]],pr,path,rec_name,output_name,method,num_cores)

        emissivity_front = df_exp_condition['emissivity_front']
        emissivity_back = df_exp_condition['emissivity_back']

        # focal_plane_location = df_exp_condition['focal_shift']
        # VDC = df_exp_condition['V_DC']

        file_names = [path + x for x in os.listdir(path)]

        joblib_output = Parallel(n_jobs=num_cores
                                 )(
            delayed(calculate_total_emission)(file_name, x0, y0, Rmax, pr, emissivity_front, emissivity_back) for
            file_name in tqdm(file_names))

        E_mean = np.mean(joblib_output)
        emission_list.append(E_mean)

    df_DC_results = pd.DataFrame({'rec_name': df_exp_condition_spreadsheet['rec_name'],
                                  'focal_shift': df_exp_condition_spreadsheet['focal_shift_cm'],
                                  'V_DC': df_exp_condition_spreadsheet['V_DC'], 'E_total': emission_list,
                                  'emissivity_front': df_exp_condition_spreadsheet['emissivity_front'],
                                  'emissivity_back': df_exp_condition_spreadsheet['emissivity_back'],
                                  'absorptivity_front': df_exp_condition_spreadsheet['absorptivity_front']})

    return df_DC_results


def compute_Amax_fv(df_DC_results, df_sigma_results):
    estimated_sigma_SC = []
    Amax_fv_list = []

    for i in range(len(df_DC_results)):
        df_exp_condition_ = df_DC_results.iloc[i, :]
        focal_shift = df_exp_condition_['focal_shift']
        sigma_SC = float(df_sigma_results.query("focal_shift =={}".format(focal_shift))['sigma_s'])
        # print(sigma_SC)
        estimated_sigma_SC.append(sigma_SC)

        absorptivity_front = df_exp_condition_['absorptivity_front']

        E_total = df_exp_condition_['E_total']

        R_sample = 0.0889 / 2

        Amax_fv = E_total / (absorptivity_front * sigma_SC * np.log((sigma_SC ** 2 + R_sample ** 2) / sigma_SC ** 2))

        Amax_fv_list.append(Amax_fv)

    df_updated_DC_results = df_DC_results.copy()
    df_updated_DC_results['sigma_SC'] = estimated_sigma_SC
    df_updated_DC_results['Amax_fv'] = Amax_fv_list

    return df_updated_DC_results











def sensitivity_model_output(f_heating, X_input_array,df_temperature, df_r_ref_locations, sample_information, vacuum_chamber_setting, numerical_simulation_setting,
                             solar_simulator_settings, light_source_property,code_directory,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all):
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


    df_amp_phase_simulated, df_temperature_simulation, df_light_source, df_temperature_transient= simulation_result_amplitude_phase_extraction(sample_information,
                                                                                                     vacuum_chamber_setting,
                                                                                                     solar_simulator_settings,
                                                                                                     light_source_property,
                                                                                                     numerical_simulation_setting,code_directory,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all)

    df_amp_phase_simulated['f_heating'] = np.array([f_heating for i in range(len(df_amp_phase_simulated))])

    df_amp_phase_simulated['alpha_r'] = np.array([alpha_r for i in range(len(df_amp_phase_simulated))])

    return df_amp_phase_simulated


def sensitivity_model_parallel(X_dump_file_name, f_heating_list, num_cores, df_temperature, df_r_ref_locations, sample_information, vacuum_chamber_setting, numerical_simulation_setting,
                             solar_simulator_settings, light_source_property,code_directory,df_solar_simulator_VQ,sigma_df):
    s_time = time.time()
    X_input_arrays = pickle.load(open(X_dump_file_name, 'rb')) # obtain pre-defined simulation conditions

    joblib_output = Parallel(n_jobs=num_cores, verbose=0)(
        delayed(sensitivity_model_output)(f_heating, X_input_array,df_temperature, df_r_ref_locations, sample_information, vacuum_chamber_setting, numerical_simulation_setting,
                             solar_simulator_settings, light_source_property,code_directory,df_solar_simulator_VQ,sigma_df) for X_input_array in tqdm(X_input_arrays) for f_heating
        in f_heating_list)

    pickle.dump(joblib_output, open("sensitivity_results_" + X_dump_file_name, "wb"))
    e_time = time.time()
    print(e_time - s_time)


# def show_sensitivity_results_sobol(sobol_problem, parallel_results, f_heating, df_r_ref_locations,calc_second_order):
#     amp_ratio_results = np.array([np.array(parallel_result['amp_ratio']) for parallel_result in parallel_results])
#     phase_diff_results = np.array([np.array(parallel_result['phase_diff']) for parallel_result in parallel_results])
#
#     Si_amp_radius = np.array(
#         [sobol.analyze(sobol_problem, amp_ratio_results[:, i], calc_second_order=calc_second_order, print_to_console=False)['S1']
#          for i in range(amp_ratio_results.shape[1])])
#     # Just pay close attention that calc_second_order=False must be consistent with how X is defined!
#
#     Si_phase_radius = np.array(
#         [sobol.analyze(sobol_problem, phase_diff_results[:, i], calc_second_order=calc_second_order, print_to_console=False)['S1']
#          for i in range(phase_diff_results.shape[1])])
#
#     plt.figure(figsize=(14, 6))
#     radius = df_r_ref_locations['r']
#     plt.subplot(121)
#     for i, name in enumerate(sobol_problem['names']):
#         # plt.plot(Si_amp_radius[:, i], label=name)
#
#         plt.scatter(radius, Si_amp_radius[:, i], label=name)
#
#     plt.xlabel('R (pixel)', fontsize=14, fontweight='bold')
#     plt.ylabel('Amp Ratio Sensitivity', fontsize=14, fontweight='bold')
#
#     plt.suptitle('frequency = ' + str(f_heating) + ' Hz', fontsize=14, fontweight='bold')
#     ax = plt.gca()
#     for tick in ax.xaxis.get_major_ticks():
#         tick.label.set_fontsize(fontsize=12)
#         tick.label.set_fontweight('bold')
#     for tick in ax.yaxis.get_major_ticks():
#         tick.label.set_fontsize(fontsize=12)
#         tick.label.set_fontweight('bold')
#     plt.legend(prop={'weight': 'bold', 'size': 12})
#
#     plt.subplot(122)
#     for i, name in enumerate(sobol_problem['names']):
#         # plt.plot(Si_phase_radius[:, i], label=name)
#         plt.scatter(radius, Si_phase_radius[:, i], label=name)
#     plt.xlabel('R (pixel)', fontsize=14, fontweight='bold')
#     plt.ylabel('Phase diff Sensitivity', fontsize=14, fontweight='bold')
#
#     ax = plt.gca()
#     for tick in ax.xaxis.get_major_ticks():
#         tick.label.set_fontsize(fontsize=12)
#         tick.label.set_fontweight('bold')
#     for tick in ax.yaxis.get_major_ticks():
#         tick.label.set_fontsize(fontsize=12)
#         tick.label.set_fontweight('bold')
#     plt.legend(prop={'weight': 'bold', 'size': 12})
#     plt.show()
#



def DOE_numerical_model_one_case(parameter_name_list, DOE_parameters,sample_information, vacuum_chamber_setting, solar_simulator_settings,
                                 light_source_property, numerical_simulation_setting,code_directory,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all):
    for i, (parameter_name, DOE_parameter_value) in enumerate(zip(parameter_name_list, DOE_parameters)):
        # print(parameter_name)
        if parameter_name in sample_information.keys():
            sample_information[parameter_name] = DOE_parameter_value
        elif parameter_name in vacuum_chamber_setting.keys():
            vacuum_chamber_setting[parameter_name] = DOE_parameter_value
        elif parameter_name in numerical_simulation_setting.keys():
            numerical_simulation_setting[parameter_name] = DOE_parameter_value
        elif parameter_name in solar_simulator_settings.keys():
            solar_simulator_settings[parameter_name] = DOE_parameter_value
            if parameter_name == 'f_heating':
                if DOE_parameter_value >=0.08:
                    numerical_simulation_setting['N_cycle'] = 25
                elif DOE_parameter_value <0.08 and DOE_parameter_value>=0.04:
                    numerical_simulation_setting['N_cycle'] = 15
                elif DOE_parameter_value <0.04 and DOE_parameter_value>0.005:
                    numerical_simulation_setting['N_cycle'] = 6
                elif DOE_parameter_value < 0.005:
                    numerical_simulation_setting['N_cycle'] = 4
        elif parameter_name in light_source_property.keys():
            light_source_property[parameter_name] = DOE_parameter_value

    df_amp_phase_simulated, df_temperature_simulation,df_light_source, df_temperature_transient = simulation_result_amplitude_phase_extraction(sample_information,
                                                                                                     vacuum_chamber_setting,
                                                                                                     solar_simulator_settings,
                                                                                                     light_source_property,
                                                                                                     numerical_simulation_setting,code_directory,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all)

    return df_amp_phase_simulated

def parallel_2nd_level_DOE(parameter_name_list, full_factorial_combinations,num_cores, result_name,code_directory,sample_information,
                           vacuum_chamber_setting,solar_simulator_settings,light_source_property,numerical_simulation_setting,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all):


    DOE_parameters_complete = [one_factorial_design for one_factorial_design in full_factorial_combinations]

    #jupyter_directory = os.getcwd()
    result_dump_path = code_directory+"Sensitivity result jupyter notebook//sensitivity cache dump//" + result_name
    if (os.path.isfile(result_dump_path)):
        print("Previous dump file existed, check if duplicate! The previous results are loaded.")
        joblib_output = pickle.load(open(result_dump_path, 'rb'))
    else:
        print("No dump file found")
        joblib_output = Parallel(n_jobs=num_cores)(delayed(DOE_numerical_model_one_case)(parameter_name_list, DOE_parameters,sample_information,vacuum_chamber_setting,
                                                                                         solar_simulator_settings,light_source_property,numerical_simulation_setting,code_directory,
                                                                                         df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all) for
                                                  DOE_parameters in tqdm(DOE_parameters_complete))
        print("run complete, now dump files")
        pickle.dump(joblib_output, open(result_dump_path, "wb"))

    df_run_conditions = pd.DataFrame(columns=parameter_name_list,data =DOE_parameters_complete)
    amp_ratio_results = np.array([np.array(joblib_output_['amp_ratio']) for joblib_output_ in joblib_output])
    phase_diff_results = np.array([np.array(joblib_output_['phase_diff']) for joblib_output_ in joblib_output])

    df_results_amp_only = pd.DataFrame(columns=joblib_output[0]['r'], data = amp_ratio_results)
    df_results_phase_only = pd.DataFrame(columns=joblib_output[0]['r'], data = phase_diff_results)

    df_results_amp_ratio_complete = pd.concat([df_run_conditions,df_results_amp_only],axis = 1)
    df_results_phase_difference_complete = pd.concat([df_run_conditions,df_results_phase_only],axis = 1)

    return df_results_amp_ratio_complete, df_results_phase_difference_complete



def main_effects_2_level_DOE(df_original, f_heating, parameter_name_list):
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
    df_main_effect['r'] = df_DOE.columns[len(parameter_name_list):]

    df_main_effect_sum = df_main_effect.copy()
    df_main_effect_percentage = df_main_effect.copy()
    df_main_effect_sum = df_main_effect_sum.drop('r', 1)
    normalizing_factor = abs(df_main_effect_sum).sum(axis=1)

    for col in parameter_name_columns:
        if col != 'r':
            df_main_effect_percentage[col] = abs(df_main_effect_percentage[col]) / normalizing_factor

    # result_summation_abs = abs(df_main_effect).sum(axis=0)
    # min_sensitivity_column = result_summation_abs.idxmin()
    # df_relative_sensitivity = df_main_effect.copy()
    # df_original_copy_min_column = df_relative_sensitivity[min_sensitivity_column].copy()
    # parameter_name_list_copy = parameter_name_list.copy()
    # parameter_name_list_copy.remove('f_heating')
    # for param in parameter_name_columns:
    #     df_relative_sensitivity[param] = df_relative_sensitivity[param] / df_original_copy_min_column

    return df_main_effect, df_main_effect_percentage


def plot_main_effects_2nd_level_DOE(df_amp_ratio_DOE_origninal, df_phase_diff_DOE_orignal, f_heating, parameter_name_list):
    df_main_effect_amp_ratio, df_relative_sensitivity_amp_ratio = main_effects_2_level_DOE(df_amp_ratio_DOE_origninal, f_heating, parameter_name_list)
    df_main_effect_phase_diff, df_relative_sensitivity_phase_diff = main_effects_2_level_DOE(df_phase_diff_DOE_orignal, f_heating, parameter_name_list)
    parameter_name_columns = parameter_name_list.copy()
    parameter_name_columns.remove('f_heating')

    plt.figure(figsize=(12, 12))

    plt.subplot(221)
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

    plt.subplot(222)
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


    plt.subplot(223)
    for parameter_name in parameter_name_columns:
        plt.scatter(df_relative_sensitivity_amp_ratio['r'], df_relative_sensitivity_amp_ratio[parameter_name], label=parameter_name)

    plt.xlabel('R(pixel)', fontsize=12, fontweight='bold')
    plt.ylabel('Amplitude ratio main effect relative', fontsize=12, fontweight='bold')
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=12)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=12)
        tick.label.set_fontweight('bold')
    plt.legend(prop={'weight': 'bold', 'size': 12})



    plt.subplot(224)
    for parameter_name in parameter_name_columns:
        plt.scatter(df_relative_sensitivity_phase_diff['r'], df_relative_sensitivity_phase_diff[parameter_name], label=parameter_name)

    plt.xlabel('R(pixel)', fontsize=12, fontweight='bold')
    plt.ylabel('Phase difference main effect relative', fontsize=12, fontweight='bold')
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


def main_effect_2nd_level_DOE_one_row(df_results_main_effect, y_axis_label, f_heating_list, parameter_name_list):
    f, axes = plt.subplots(2, len(f_heating_list),
                           figsize=(int(len(f_heating_list) * 5), 10), sharex=True, sharey='row')

    df_main_effect_list = []
    df_main_effect_relative_list = []

    parameter_name_columns = parameter_name_list.copy()
    parameter_name_columns.remove('f_heating')

    for i, f_heating in enumerate(f_heating_list):
        df_main_effect, df_main_effect_relative = main_effects_2_level_DOE(df_results_main_effect, f_heating,
                                                                           parameter_name_list)
        df_main_effect_list.append(df_main_effect)
        df_main_effect_relative_list.append(df_main_effect_relative)


        for parameter_name in parameter_name_columns:
            if parameter_name == 'alpha_r':
                legend_temp = 'Thermal diffusivity: '+ r'$\alpha$'
            elif parameter_name == 'Amax':
                legend_temp = 'Peak nominal intensity: '+ r'$A_{max}$'
            elif parameter_name == 'sigma_s':
                legend_temp = 'Intensity distribution: '+r'$\sigma$'
            elif parameter_name == 'emissivity_front':
                legend_temp = 'Front emissivity: '+ r'$\epsilon$'
            elif parameter_name == 'absorptivity_front':
                legend_temp = 'Front absorptivity: '+r'$\eta$'
            elif parameter_name == 'T_sur1':
                legend_temp = 'Front temperature'

            axes[0, i].scatter(df_main_effect['r'], df_main_effect[parameter_name], label=legend_temp)
            axes[1, i].scatter(df_main_effect_relative['r'], df_main_effect_relative[parameter_name],
                               label=legend_temp)

        axes[0, i].set_xlabel('R(pixel)', fontsize=12, fontweight='bold')
        axes[0, i].set_title("f_heating = {} Hz".format(f_heating), fontsize=12, fontweight='bold')

        axes[1, i].set_xlabel('R(pixel)', fontsize=12, fontweight='bold')
        axes[1, i].set_title("f_heating = {} Hz".format(f_heating), fontsize=12, fontweight='bold')

        if i == 0:
            axes[0, i].set_ylabel('{}'.format(y_axis_label), fontsize=12, fontweight='bold')
            axes[1, i].set_ylabel('{}'.format(y_axis_label), fontsize=12, fontweight='bold')

        axes[0, i].legend(prop={'weight': 'bold', 'size': 12})
        axes[1, i].legend(prop={'weight': 'bold', 'size': 12})

        for tick in axes[0, i].xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize=12)
            tick.label.set_fontweight('bold')
        for tick in axes[0, i].yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize=12)
            tick.label.set_fontweight('bold')
        for tick in axes[1, i].xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize=12)
            tick.label.set_fontweight('bold')
        for tick in axes[1, i].yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize=12)
            tick.label.set_fontweight('bold')

    plt.tight_layout(h_pad=1)

    return df_main_effect_list, df_main_effect_relative_list



# def amp_phase_DOE_sensitivity(parameter_name_list, full_factorial_combinations, df_r_ref_locations,num_cores, result_name,df_temperature,sample_information,vacuum_chamber_setting,solar_simulator_settings,light_source_property,numerical_simulation_setting,df_solar_simulator_VQ,sigma_df):
#
#     jupyter_directory = os.getcwd() # note if you run this in Pycharm then manually change, sorry
#     df_results_amp_ratio_complete, df_results_phase_difference_complete = parallel_2nd_level_DOE(parameter_name_list, full_factorial_combinations, df_r_ref_locations,num_cores,
#                                                                                                  result_name,jupyter_directory,df_temperature,sample_information,vacuum_chamber_setting,
#                                                                                                  solar_simulator_settings,light_source_property,numerical_simulation_setting,df_solar_simulator_VQ,sigma_df)
#
#
#     y_axis_label = 'Amplitude ratio ME'
#     main_effect_2nd_level_DOE_one_row(df_results_amp_ratio_complete,y_axis_label,f_heating_list,parameter_name_list,df_r_ref_locations)
#
#     y_axis_label = 'Phase diff ME'
#     main_effect_2nd_level_DOE_one_row(df_results_phase_difference_complete,y_axis_label,f_heating_list,parameter_name_list,df_r_ref_locations)






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
        df_amp_phase_simulated, df_temperature_simulation,df_light_source, df_temperature_transient = simulation_result_amplitude_phase_extraction(self.df_temperature,
            self.df_amplitude_phase_measurement, self.sample_information, self.vacuum_chamber_setting,
            self.solar_simulator_settings, self.light_source_property, self.numerical_simulation_setting,self.code_directory)
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