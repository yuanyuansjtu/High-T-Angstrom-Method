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
#from lmfit import Parameters
import chaospy as cp
import random
from lmfit import minimize, Parameters

from scipy.stats import uniform
import matplotlib.ticker as mticker
import matplotlib as mpl
from scipy.optimize import minimize as minimize2

import pymc3 as pm
import theano
import theano.tensor as tt

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

    circle1 = plt.Circle((x0, y0), R0, edgecolor='r', fill=False, linewidth=3, linestyle='-.')
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
    df_exp_condition_spreadsheet = pd.read_csv(
        code_directory + "batch process information//" + df_exp_condition_spreadsheet_filename)

    steady_state_figure_list = []
    df_temperature_list = []

    for i in range(len(df_exp_condition_spreadsheet)):
        df_exp_condition = df_exp_condition_spreadsheet.iloc[i, :]

        rec_num = df_exp_condition['rec_name']
        x0 = df_exp_condition['x0_pixels']  # in pixels
        y0 = df_exp_condition['y0_pixels']  # in pixels

        R0 = df_exp_condition['R0_pixels']  # in pixels
        R_analysis = df_exp_condition['R_analysis_pixels']

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
        code_directory + "sample specifications//Sample and light source properties.xlsx", sheet_name="thermal diffusivity")
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
    A_max_list = []

    for i, regression_type in enumerate(df_exp_condition['regression_parameter']):

        if regression_type == 'sigma_s':
            sigma_s_list.append(joblib_output[i][0])
            focal_shift = df_exp_condition['focal_shift'][i]
            alpha_regression_list.append(f_alpha(T_average_list[i])) # this is wrong, need fixed!
            sigma_ray_tracing_list.append(f_sigma(focal_shift))
            alpha_theoretical_list.append(f_alpha(T_average_list[i]))
            emissivity_front_list.append(df_exp_condition['emissivity_front'][i])
            A_max_list.append(joblib_output[i][5])

        elif regression_type == 'alpha_r':
            focal_shift = df_exp_condition['focal_shift'][i]
            sigma_s_list.append(f_sigma(focal_shift))
            alpha_regression_list.append(joblib_output[i][0])
            sigma_ray_tracing_list.append(f_sigma(focal_shift))

            alpha_theoretical_list.append(f_alpha(T_average_list[i]))
            emissivity_front_list.append(df_exp_condition['emissivity_front'][i])
            A_max_list.append(joblib_output[i][5])

        elif regression_type == 'emissivity_front':
            focal_shift = df_exp_condition['focal_shift'][i]
            sigma_s_list.append(f_sigma(focal_shift))
            alpha_regression_list.append(f_alpha(T_average_list[i]))
            sigma_ray_tracing_list.append(f_sigma(focal_shift))

            alpha_theoretical_list.append(f_alpha(T_average_list[i]))
            emissivity_front_list.append(joblib_output[i][0])
            A_max_list.append(joblib_output[i][5])

    df_results_all = pd.DataFrame({'rec_name':df_exp_condition['rec_name'],'focal_distance(cm)':df_exp_condition['focal_shift'],'f_heating':df_exp_condition['f_heating'],'VDC':df_exp_condition['V_DC'],
                                   'sigma_s':sigma_s_list,'T_average(K)':T_average_list,'T_min(K)':T_min_list,'R0':df_exp_condition['R0'],'alpha_r':alpha_regression_list,'regression_parameter':df_exp_condition['regression_parameter']
                                   ,'alpha_theoretical':alpha_theoretical_list,'sigma_ray_tracing':sigma_ray_tracing_list,'regression_method':df_exp_condition['regression_method'],'emissivity_front':emissivity_front_list,'gap':df_exp_condition['gap'],'R_analysis':df_exp_condition['R_analysis'],'Amax':A_max_list})


    return df_results_all


# joblib_output
def mcmc_joblib_to_dataframe(joblib_output, code_directory, df_exp_condition_spreadsheet_filename,sigma_df):

    T_average_list = [joblib_output_[1] for joblib_output_ in joblib_output]
    T_min_list = [joblib_output_[2] for joblib_output_ in joblib_output]
    df_exp_condition = pd.read_excel(code_directory+"batch process information//" + df_exp_condition_spreadsheet_filename)


    locations_relative_focal_plane = sigma_df['focal_shift']

    sigma_relative_focal_plane = sigma_df['sigma_s']

    # f_Amax = interp1d(locations_relative_focal_plane, Amax_relative_focal_plane, kind='cubic')
    f_sigma = interp1d(locations_relative_focal_plane, sigma_relative_focal_plane, kind='linear')

    sample_material = np.unique(df_exp_condition['sample_name'])[0]
    # Here we assume each spreadsheet only contains one material
    df_theoretical_thermal_diffusivity_all = pd.read_excel(
        code_directory + "sample specifications//Sample and light source properties.xlsx", sheet_name="thermal diffusivity")
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
    A_max_list = []
    parameter_std_list = []

    for i, regression_type in enumerate(df_exp_condition['regression_parameter']):

        if regression_type == 'sigma_s':
            sigma_s_list.append(joblib_output[i][0][200:].mean())
            focal_shift = df_exp_condition['focal_shift'][i]
            alpha_regression_list.append(f_alpha(T_average_list[i])) # this is wrong, need fixed!
            sigma_ray_tracing_list.append(f_sigma(focal_shift))
            alpha_theoretical_list.append(f_alpha(T_average_list[i]))
            emissivity_front_list.append(df_exp_condition['emissivity_front'][i])
            A_max_list.append(joblib_output[i][5])
            parameter_std_list.append(joblib_output[i][0][200:].std())
            #print(parameter_std_list)


        elif regression_type == 'alpha_r':
            focal_shift = df_exp_condition['focal_shift'][i]
            sigma_s_list.append(f_sigma(focal_shift))
            alpha_regression_list.append(joblib_output[i][0][200:].mean())
            sigma_ray_tracing_list.append(f_sigma(focal_shift))

            alpha_theoretical_list.append(f_alpha(T_average_list[i]))
            emissivity_front_list.append(df_exp_condition['emissivity_front'][i])
            A_max_list.append(joblib_output[i][5])
            parameter_std_list.append(joblib_output[i][0].std())


        elif regression_type == 'emissivity_front':
            focal_shift = df_exp_condition['focal_shift'][i]
            sigma_s_list.append(f_sigma(focal_shift))
            alpha_regression_list.append(f_alpha(T_average_list[i]))
            sigma_ray_tracing_list.append(f_sigma(focal_shift))

            alpha_theoretical_list.append(f_alpha(T_average_list[i]))
            emissivity_front_list.append(joblib_output[i][0])
            A_max_list.append(joblib_output[i][5])


    df_results_all = pd.DataFrame({'rec_name': df_exp_condition['rec_name'], 'focal_distance(cm)': df_exp_condition['focal_shift'],
                 'f_heating': df_exp_condition['f_heating'], 'VDC': df_exp_condition['V_DC'],
                 'sigma_s': sigma_s_list,'parameter_std':parameter_std_list, 'T_average(K)': T_average_list, 'T_min(K)': T_min_list,
                 'R0': df_exp_condition['R0'], 'alpha_r': alpha_regression_list,
                 'regression_parameter': df_exp_condition['regression_parameter']
                    , 'alpha_theoretical': alpha_theoretical_list,
                 'regression_method': df_exp_condition['regression_method'], 'emissivity_front': emissivity_front_list,
                 'gap': df_exp_condition['gap'], 'R_analysis': df_exp_condition['R_analysis'], 'Amax': A_max_list})

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

    theta_n = int(abs(theta_max-theta_min)/(2*np.pi)*180) # previous studies have shown that 2Pi requires above 100 theta points to yield accurate results, here we uses 180 points

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

def radiation_absorption_view_factor_calculations_network(code_directory,rm_array,dr,sample_information,solar_simulator_settings,vacuum_chamber_setting,numerical_simulation_setting,df_view_factor,df_LB_details_all):

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
    emissivity_front = sample_information['emissivity_front']

    T_bias = sample_information['T_bias']

    if vacuum_chamber_setting['light_blocker'] == True:

        if (numerical_simulation_setting['analysis_mode'] != 'sensitivity') and ((numerical_simulation_setting['analysis_mode'] != 'validation_implicit_variable_properties') and (numerical_simulation_setting['analysis_mode'] != 'validation_implicit_const_alpha')):

            if sample_name == 'copper':
                df_LB_temp = df_LB_details_all.query("Material == '{}'".format('copper'))

            else:
                df_LB_temp = df_LB_details_all.query("Material == '{}'".format('graphite_poco'))

            T_LB1_C,T_LB2_C,T_LB3_C, T_LB_mean_C = interpolate_LB_temperatures(focal_shift, VDC, df_LB_temp)
            print("Case variable light blocker temperature!")
            T_LB1 = T_LB1_C + 273.15+T_bias
            T_LB2 = T_LB2_C + 273.15+T_bias
            T_LB3 = T_LB3_C + 273.15+T_bias

        else:
            # In sensitivity analysis light blocker temperature is a constant and can be specified
            print("Light blocker temperature is set as constant, the same as Tsur1.")
            T_LB1 = vacuum_chamber_setting['T_sur1']+T_bias
            T_LB2 = vacuum_chamber_setting['T_sur1']+T_bias
            T_LB3 = vacuum_chamber_setting['T_sur1']+T_bias
            T_LB_mean_C = vacuum_chamber_setting['T_sur1']+T_bias


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


        F_list_LBH_R1 = calculator_coaxial_parallel_rings_view_factors(R_LBH, R_LB1, rm_array, dr, L_LB_sample)
        F_list_R1_R2 = calculator_coaxial_parallel_rings_view_factors(R_LB1, R_LB2, rm_array, dr, L_LB_sample)
        F_list_R2_R3 = calculator_coaxial_parallel_rings_view_factors(R_LB2, R_LB3, rm_array, dr, L_LB_sample)
        F_list_LBH = calculator_coaxial_parallel_rings_view_factors(1e-4, R_LBH, rm_array, dr, L_LB_sample)

        Am = 2 * np.pi * rm_array * dr
        Am[0] = np.pi * (dr / 2) ** 2

        c_LB1 = 1 / ((1 - emissivity_front) / (emissivity_front * Am) + 1 / (A_LBH_LB1 * F_list_LBH_R1) + (1 - e_LB) / (e_LB * A_LBH_LB1))

        c_LB2 = 1 / ((1 - emissivity_front) / (emissivity_front * Am) + 1 / (A_LB1_LB2 * F_list_R1_R2) + (1 - e_LB) / (e_LB * A_LB1_LB2))

        c_LB3 = 1 / ((1 - emissivity_front) / (emissivity_front * Am) + 1 / (A_LB2_LB3 * F_list_R2_R3) + (1 - e_LB) / (e_LB * A_LB2_LB3))

        c_LBH = 1 / ((1 - emissivity_front) / (emissivity_front * Am) + 1 / (A_LBH * F_list_LBH) + (1 - e_LBH) / (e_LBH * A_LBH))

        Cm_LB = c_LB1+c_LB2+c_LB3+c_LBH
        qm_LB = (c_LB1*T_LB1**4 + c_LB2*T_LB2**4 + c_LB3*T_LB3**4 + c_LBH*T_LBH**4)*sigma_sb

        # Cm_front = calculator_coaxial_parallel_rings_view_factors(1e-4, R_LBH, rm_array, dr,L_LB_sample) * A_LBH * e_LBH * sigma_sb * T_LBH ** 4 * absorptivity_front \
        # + calculator_coaxial_parallel_rings_view_factors(R_LBH, R_LB1, rm_array, dr,L_LB_sample) * A_LBH_LB1 * e_LB * sigma_sb * T_LB1 ** 4* absorptivity_front \
        # + calculator_coaxial_parallel_rings_view_factors(R_LB1, R_LB2, rm_array, dr,L_LB_sample) * A_LB1_LB2 * e_LB * sigma_sb * T_LB2 ** 4* absorptivity_front \
        # + calculator_coaxial_parallel_rings_view_factors(R_LB2, R_LB3, rm_array, dr,L_LB_sample) * A_LB2_LB3 * e_LB * sigma_sb * T_LB3 ** 4 * absorptivity_front\
        # + calculator_back_W2_ring_VF(rm_array, dr, 1e-4, L_LB_sample, R_chamber) * A_LBW * e_LBW * sigma_sb * T_LBW ** 4 * absorptivity_front

        Cm_back = calculator_back_W2_ring_VF(rm_array, dr, 1e-4, W1, R_chamber) * A_W1 * e_W1 * sigma_sb * T_W1 ** 4 * absorptivity_back\
        + calculator_back_W2_ring_VF(rm_array, dr, W1, W2, R_chamber) * A_W2 * e_W2 * sigma_sb * T_W2 ** 4 * absorptivity_back \
        + calculator_IR_shield_to_sample_view_factors(rm_array, dr, W1 + W2, R_IRW) * A_IRW * e_IRW * sigma_sb * T_IRW ** 4 * absorptivity_back

        #Cm_front[0] = 1e-8
        Cm_back[0] = 1e-8

    # elif vacuum_chamber_setting['light_blocker'] == False:
    #     # No light blocker
    #     if numerical_simulation_setting['analysis_mode'] != 'sensitivity':
    #         T_W1 = float(df_view_factor['T_W1_C']) + 273.15
    #         T_W2 = float(df_view_factor['T_W2_C']) + 273.15
    #     elif numerical_simulation_setting['analysis_mode'] == 'sensitivity':
    #         # For sensitivity analysis these temperatures can be assigned manually
    #         T_W1 = vacuum_chamber_setting['T_sur1']
    #         T_W2 = vacuum_chamber_setting['T_sur1']
    #
    #     T_W3 = float(df_view_factor['T_W3_C']) + 273.15
    #     T_glass = float(df_view_factor['T_glass_C']) + 273.15
    #     T_IRW = float(df_view_factor['T_IRW_C']) + 273.15
    #
    #     W1 = float(df_view_factor['W1'])
    #     W2 = float(df_view_factor['W2'])
    #     W3 = float(df_view_factor['W3'])
    #     L_G = float(df_view_factor['L_G'])
    #     R_chamber = float(df_view_factor['R_chamber'])
    #     R_IR_I = float(df_view_factor['R_IR_I'])
    #     R_IR_O = R_chamber
    #     R_G_O = R_chamber
    #     L_IR = W2 + W3
    #
    #     A_W2 = 2 * np.pi * R_chamber * W2
    #     A_W3 = 2 * np.pi * R_chamber * W3
    #     A_W1 = 2 * np.pi * R_chamber * W1
    #     A_glass = np.pi * R_G_O ** 2
    #     A_IR_shield = np.pi * (R_IR_O ** 2 - R_IR_I ** 2)
    #
    #     VF_W1_front = calculator_back_W2_ring_VF(rm_array, dr, 1e-4, W1, R_chamber)
    #     VF_W2_back = calculator_back_W2_ring_VF(rm_array, dr, 1e-4, W2, R_chamber)
    #     VF_W3_back = calculator_back_W2_ring_VF(rm_array, dr, W2, W3, R_chamber)
    #     VF_glass = calculator_coaxial_parallel_rings_view_factors(1e-4, R_G_O, rm_array, dr, L_G)
    #     VF_IR_shield = calculator_coaxial_parallel_rings_view_factors(R_IR_I, R_IR_O, rm_array,dr, L_IR)
    #
    #
    #     Cm_front = 0.95*sigma_sb*VF_glass*A_glass*T_glass**4 + 0.95*sigma_sb*VF_W1_front*A_W1*T_W1**4
    #     Cm_back = sigma_sb*VF_W2_back*A_W2*T_W2**4 + sigma_sb*VF_W3_back*A_W3*T_W3**4 + 0.95*sigma_sb*VF_IR_shield*A_IR_shield*T_IRW**4
    #
    #     Cm_front[0] = 1e-8
    #     Cm_back[0] = 1e-8
    #
    #     T_LB_mean_C = T_W1

    return Cm_LB, qm_LB, Cm_back, T_LB_mean_C




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

        if (numerical_simulation_setting['analysis_mode'] != 'sensitivity') and ((numerical_simulation_setting['analysis_mode'] != 'validation_implicit_variable_properties') and (numerical_simulation_setting['analysis_mode'] != 'validation_implicit_const_alpha') and (numerical_simulation_setting['analysis_mode'] != 'validation_explicit_variable_properties') and (numerical_simulation_setting['analysis_mode'] != 'validation_explicit_const_alpha') and (numerical_simulation_setting['analysis_mode'] != 'validation_explicit_const_properties')):

            if sample_name == 'copper':
                df_LB_temp = df_LB_details_all.query("Material == '{}'".format('copper'))

            else:
                df_LB_temp = df_LB_details_all.query("Material == '{}'".format('graphite_poco'))

            T_LB1_C,T_LB2_C,T_LB3_C, T_LB_mean_C = interpolate_LB_temperatures(focal_shift, VDC, df_LB_temp)
            print("Case variable light blocker temperature!")
            T_LB1 = T_LB1_C + 273.15
            T_LB2 = T_LB2_C + 273.15
            T_LB3 = T_LB3_C + 273.15

        else:
            # In sensitivity analysis light blocker temperature is a constant and can be specified
            print("Light blocker temperature is set as constant, the same as Tsur1.")
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


# def select_data_points_radial_average_MA_match_model_grid(x0, y0, N_Rmax, pr, R_sample, theta_n,
#                                                           file_name):  # N_Rmax: total number of computation nodes between center and edge
#     # This method was originally developed by Mosfata, was adapted here for amplitude and phase estimation
#     df_raw = pd.read_csv(file_name, sep=',', header=None, names=list(np.arange(0, 639)))
#     raw_time_string = df_raw.iloc[2, 0][df_raw.iloc[2, 0].find('=') + 2:]  # '073:02:19:35.160000'
#     raw_time_string = raw_time_string[raw_time_string.find(":") + 1:]  # '02:19:35.160000'
#     strip_time = datetime.strptime(raw_time_string, '%H:%M:%S.%f')
#     time_in_seconds = strip_time.hour * 3600 + strip_time.minute * 60 + strip_time.second + strip_time.microsecond / 10 ** 6
#     theta = np.linspace(0, 2 * np.pi, theta_n)  # The angles 1D array (rad)
#     df_temp = df_raw.iloc[5:, :]
#
#     Tr = np.zeros((N_Rmax, theta_n))  # Initializing the radial temperature matrix (T)
#
#     R = np.linspace(0, R_sample, N_Rmax)
#     for i, R_ in enumerate(R):  # Looping through all radial points
#         for j, theta_ in enumerate(theta):  # Looping through all angular points
#             y = R_ / pr * np.sin(theta_) + y0;
#             x = R_ / pr * np.cos(theta_) + x0  # Identifying the spatial 2D cartesian coordinates
#             y1 = int(np.floor(y));
#             y2 = y1 + 1;
#             x1 = int(np.floor(x));
#             x2 = x1 + 1  # Identifying the neighboring 4 points
#             dy1 = (y2 - y) / (y2 - y1);
#             dy2 = (y - y1) / (y2 - y1)  # Identifying the corresponding weights for the y-coordinates
#             dx1 = (x2 - x) / (x2 - x1);
#             dx2 = (x - x1) / (x2 - x1)  # Identifying the corresponding weights for the x-coordinates
#             T11 = df_temp.iloc[y1, x1];
#             T21 = df_temp.iloc[y2, x1]  # Identifying the corresponding temperatures for the y-coordinates
#             T12 = df_temp.iloc[y1, x2];
#             T22 = df_temp.iloc[y2, x2]  # Identifying the corresponding temperatures for the x-coordinates
#             Tr[
#                 i, j] = dx1 * dy1 * T11 + dx1 * dy2 * T21 + dx2 * dy1 * T12 + dx2 * dy2 * T22 + 273.15  # Interpolated angular temperature matrix
#
#     T_interpolate = np.mean(Tr, axis=1)
#
#     return T_interpolate, time_in_seconds
#
#
# def select_data_points_radial_average_HY(x0, y0, Rmax, file_name):
#     df_raw = pd.read_csv(file_name, sep=',', header=None, names=list(np.arange(0, 639)))
#     raw_time_string = df_raw.iloc[2, 0][df_raw.iloc[2, 0].find('=') + 2:]  # '073:02:19:35.160000'
#     raw_time_string = raw_time_string[raw_time_string.find(":") + 1:]  # '02:19:35.160000'
#     strip_time = datetime.strptime(raw_time_string, '%H:%M:%S.%f')
#     time_in_seconds = strip_time.hour * 3600 + strip_time.minute * 60 + strip_time.second + strip_time.microsecond / 10 ** 6
#
#     df_temp = df_raw.iloc[5:, :]
#     x = []
#     y = []
#     angle = []
#     r_pixel = []
#     T_interpolate = np.zeros(Rmax)
#
#     for r in range(Rmax):
#         if r == 0:
#             x.append(x0)
#             y.append(y0)
#             angle.append(0)
#             r_pixel.append(r)
#             T_interpolate[r] = df_temp.iloc[y0, x0]
#
#         else:
#             temp = []
#             for i in np.arange(x0 - r - 2, x0 + r + 2, 1):
#                 for j in np.arange(y0 - r - 2, y0 + r + 2, 1):
#                     d = ((i - x0) ** 2 + (j - y0) ** 2) ** (0.5)
#                     if d >= r and d < r + 1:
#                         x.append(i)
#                         y.append(j)
#                         r_pixel.append(r)
#                         temp.append(
#                             (df_temp.iloc[j, i] - T_interpolate[r - 1]) / (d - r + 1) + T_interpolate[r - 1] + 273.15)
#
#             T_interpolate[r] = np.mean(temp)
#
#     return T_interpolate, time_in_seconds
#
#

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

            theta_n = 180  # default theta_n=100, however, if R increased significantly theta_n should also increase
                # joblib_output = Parallel(n_jobs=num_cores)(delayed(select_data_points_radial_average_MA)(x0,y0,Rmax,theta_n,file_name) for file_name in tqdm(file_names))
                #joblib_output = Parallel(n_jobs=num_cores)(delayed(select_data_points_radial_average_MA)(x0, y0, N_Rmax, theta_range, file_name) for file_name in tqdm(file_names))
            joblib_output = Parallel(n_jobs=num_cores)(delayed(select_data_points_radial_average_MA_vectorized)(x0, y0, N_Rmax, theta_range, file_name) for file_name in tqdm(file_names))

            # else:
            #     joblib_output = Parallel(n_jobs=num_cores)(
            #         delayed(select_data_points_radial_average_HY)(x0, y0, N_Rmax, file_name) for file_name in
            #         tqdm(file_names))

            pickle.dump(joblib_output, open(dump_file_path, "wb"))  # create a dump file

            e_time = time.time()
            print('Time to process {} is {}'.format(rec_name, (e_time - s_time)))

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

        fitting_params_initial = {'amplitude': 10, 'phase': 0.1, 'bias': 500}

        n_col = df_temperature.shape[1]
        tmin = min(df_temperature['reltime'])
        time = df_temperature['reltime'] - tmin

        A1 = df_temperature[index[0]]
        A2 = df_temperature[index[1]]

        x0 = np.array([10, 0.1, 500])  # amplitude,phase,bias

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
    sigma_s = sigma
    # if ((numerical_simulation_setting['analysis_mode']) == 'regression' or (numerical_simulation_setting['analysis_mode']) =='mcmc') and ((numerical_simulation_setting['regression_parameter'] == 'sigma_s') or (numerical_simulation_setting['regression_parameter'] == 'sigma_s_absorptivity')):
    #     sigma_s = sigma
    #
    # elif (numerical_simulation_setting['analysis_mode']) =='mcmc' and (numerical_simulation_setting['regression_parameter'] == 'alpha_r'):
    #     sigma_s = sigma
    #
    # else:
    #
    #     focal_shift_sigma_calib = sigma_df['focal_shift']
    #     sigma_s_calib = sigma_df['sigma_s']
    #
    #     f_sigma_cubic = interp1d(focal_shift_sigma_calib, sigma_s_calib, kind='linear')
    #     # focal_shift = vacuum_chamber_setting['focal_shift']
    #     sigma_s = float(f_sigma_cubic(focal_shift))

    V_calib = np.array(df_solar_simulator_VQ['V_DC'])
    focal_shift_calib = np.array(df_solar_simulator_VQ['focal_shift'])

    f_V_Q = interp2d(V_calib, focal_shift_calib, np.array(df_solar_simulator_VQ['A_solar']), kind='linear')

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

    N_Rs = vacuum_chamber_setting['N_Rs_node']
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




def finite_difference_implicit_variable_properties_old(sample_information, vacuum_chamber_setting,
                                                          solar_simulator_settings,
                                                          light_source_property, numerical_simulation_setting,
                                                          df_solar_simulator_VQ,
                                                          sigma_df, code_directory, df_view_factor, df_LB_details_all):
    Max_iter = 100000

    T_LBW = float(df_view_factor['T_LBW_C'][0]) + 273.15
    e_LB = float(df_view_factor['e_LB'][0])

    R = sample_information['R']  # sample radius
    Nr = numerical_simulation_setting['Nr_node']  # number of discretization along radial direction
    t_z = sample_information['t_z']  # sample thickness

    dr = R / (Nr - 1)
    r = np.arange(Nr)
    rm = r * dr

    rho = sample_information['rho']
    cp_0 = sample_information['cp_const']  # Cp = cp_const + cp_c1*T + cp_c2/T + cp_c3/T**2, T unit in K
    cp_1 = sample_information['cp_c1']
    cp_2 = sample_information['cp_c2']
    cp_3 = sample_information['cp_c3']

    R0 = vacuum_chamber_setting['R0_node']

    alpha_r_A = float(sample_information['alpha_r_A'])
    alpha_r_B = float(sample_information['alpha_r_B'])

    T_initial = sample_information['T_initial']  # unit in K
    # k = alpha*rho*cp

    f_heating = solar_simulator_settings['f_heating']  # periodic heating frequency
    N_cycle = numerical_simulation_setting['N_cycle']
    N_stable_cycle_output = numerical_simulation_setting['N_stable_cycle_output']

    simulated_amp_phase_extraction_method = numerical_simulation_setting['simulated_amp_phase_extraction_method']

    emissivity_front = sample_information['emissivity_front']  # assumed to be constant
    emissivity_back = sample_information['emissivity_back']  # assumed to be constant

    absorptivity_solar = sample_information['absorptivity_solar']

    absorptivity_front = sample_information['absorptivity_front']  # assumed to be constant
    absorptivity_back = sample_information['absorptivity_back']  # assumed to be constant

    sigma_sb = 5.6703744 * 10 ** (-8)  # stefan-Boltzmann constant

    Cm_front, Cm_back, T_LB_mean_C = radiation_absorption_view_factor_calculations(code_directory, rm, dr,
                                                                                   sample_information,
                                                                                   solar_simulator_settings,
                                                                                   vacuum_chamber_setting,
                                                                                   numerical_simulation_setting,
                                                                                   df_view_factor, df_LB_details_all)


    dt = 1 / f_heating / numerical_simulation_setting['simulated_num_data_per_cycle']

    t_total = 1 / f_heating * N_cycle  # total simulation time

    Nt = int(t_total / dt)  # total number of simulation time step
    time_simulation = dt * np.arange(Nt)

    T = T_initial * np.ones((Nt, Nr + 1))
    dz = t_z
    q_solar = absorptivity_solar*light_source_intensity_Amax_fV_vecterize(np.arange(Nr) * dr, np.arange(Nt) * dt,
                                                       solar_simulator_settings, vacuum_chamber_setting,
                                                       light_source_property,
                                                       df_solar_simulator_VQ, sigma_df)

    T_temp = np.zeros(Nr)
    N_steady_count = 0
    time_index = Nt - 1
    N_one_cycle = int(Nt / N_cycle)

    C_surr = Cm_front + Cm_back

    Jac = np.zeros((Nr, Nr))
    F_matrix = np.zeros(Nr)

    for p in tqdm(range(Nt - 1)):  # p indicate time step

        c_p_1 = cp_0 + cp_1 * T[p, :] + cp_2 / T[p, :] + cp_3 / T[p, :] ** 2
        alpha_p_1 = 1 / (alpha_r_A * T[p, :] + alpha_r_B)
        k_p_1 = c_p_1 * alpha_p_1 * rho

        for iter in range(Max_iter):
            # print(c_p_1)
            alpha_ba_0 = (k_p_1[0] + k_p_1[1]) / (2 * rho * c_p_1[0])
            alpha_0 = k_p_1[0] / (rho * c_p_1[0])
            A0 = -alpha_ba_0 * dt / dr ** 2 + dt * alpha_0 / (2 * rm[1] * dr)

            alpha_ce_0 = (2 * k_p_1[0] + 2 * k_p_1[1]) / (4 * rho * c_p_1[0])
            B0 = 2 * dt / dr ** 2 * alpha_ce_0 + 1

            alpha_fo_0 = (k_p_1[0] + k_p_1[1]) / (2 * rho * c_p_1[0])
            C0 = -dt / dr ** 2 * alpha_fo_0 - dt * alpha_0 / (2 * rm[1] * dr)
            D0 = (emissivity_front - emissivity_front*(1-e_LB)*absorptivity_front + emissivity_back) * sigma_sb * dt / (dz * rho * c_p_1[0])
            #D0 = (emissivity_front + emissivity_back) * sigma_sb * dt / (dz * rho * c_p_1[0])
            E0 = -q_solar[p, 0] * dt / (dz * rho * c_p_1[0])
            F0 = -C_surr[0] / (2 * np.pi * rm[1] * dr * dz) * dt / (rho * c_p_1[0])

            Jac[0, 0] = B0 + 4 * D0 * T[p + 1, 0] ** 3
            Jac[0, 1] = A0 + C0
            F_matrix[0] = - T[p, 0] + (A0 + C0) * T[p + 1, 1] + B0 * T[p + 1, 0] + D0 * T[p + 1, 0] ** 4 + E0 + F0

            # T[p+1,0] = -(A0+C0)/B0*T[p+1,1]- D0/B0*T[p+1,0]**4-E0/B0 - F0/B0 + 1/B0*T[p,0]

            alpha_ba = (k_p_1[1:-2] + k_p_1[0:-3]) / (2 * rho * c_p_1[1:-2])
            alpha = k_p_1[1:-2] / (rho * c_p_1[1:-2])
            Am = -alpha_ba * dt / dr ** 2 + dt * alpha / (2 * rm[1:-1] * dr)

            alpha_ce = (k_p_1[0:-3] + 2 * k_p_1[1:-2] + k_p_1[2:-1]) / (4 * rho * c_p_1[1:-2])
            Bm = 2 * dt / dr ** 2 * alpha_ce + 1

            alpha_fo = (k_p_1[1:-2] + k_p_1[2:-1]) / (2 * rho * c_p_1[1:-2])
            Cm = -dt / dr ** 2 * alpha_fo - dt * alpha / (2 * rm[1:-1] * dr)
            Dm = (emissivity_front - emissivity_front*(1-e_LB)*absorptivity_front + emissivity_back) * sigma_sb * dt / (dz * rho * c_p_1[1:-2])
            #Dm = (emissivity_front + emissivity_back) * sigma_sb * dt / (dz * rho * c_p_1[1:-2])
            Em = -q_solar[p, 1:-1] * dt / (dz * rho * c_p_1[1:-2])
            Fm = -C_surr[1:-1] / (2 * np.pi * rm[1:-1] * dr * dz) * dt / (rho * c_p_1[1:-2])

            for idx in range(len(alpha)):
                Jac[idx + 1, idx] = Am[idx]
                Jac[idx + 1, idx + 1] = Bm[idx] + 4 * Dm[idx] * T[p + 1, idx + 1] ** 3
                Jac[idx + 1, idx + 2] = Cm[idx]

            F_matrix[1:-1] = -T[p, 1:-2] + Am * T[p + 1, 0:-3] + Bm * T[p + 1, 1:-2] + Cm * T[p + 1, 2:-1] + Dm * T[p + 1,1:-2] ** 4 + Em + Fm

            # T[p+1,1:-2] = -Am/Bm*T[p+1,0:-3] - Cm/Bm*T[p+1,2:-1]-Dm/Bm*T[p+1,1:-2]**4 - Em/Bm - Fm/Bm + 1/Bm*T[p,1:-2]

            # Be ware of the properties at node N+1
            alpha_ba_N = (k_p_1[-2] + k_p_1[-3]) / (2 * rho * c_p_1[-2])
            alpha_N = k_p_1[-2] / (rho * c_p_1[-2])
            AN = -alpha_ba_N * dt / dr ** 2 + dt * alpha_N / (2 * rm[-1] * dr)

            alpha_ce_N = (k_p_1[-3] + 2 * k_p_1[-2] + k_p_1[-1]) / (4 * rho * c_p_1[-2])
            BN = 2 * dt / dr ** 2 * alpha_ce_N + 1

            alpha_fo_N = (k_p_1[-2] + k_p_1[-1]) / (2 * rho * c_p_1[-2])
            CN = -dt / dr ** 2 * alpha_fo_N - dt * alpha_N / (2 * rm[-1] * dr)
            DN = (emissivity_front - emissivity_front*(1-e_LB)*absorptivity_front + emissivity_back) * sigma_sb * dt / (dz * rho * c_p_1[-2])
            #DN = (emissivity_front + emissivity_back) * sigma_sb * dt / (dz * rho * c_p_1[-2])
            EN = -q_solar[p, -1] * dt / (dz * rho * c_p_1[-2])
            FN = -C_surr[-1] / (2 * np.pi * rm[-1] * dr * dz) * dt / (rho * c_p_1[-2])

            Jac[-1, -2] = AN + CN
            Jac[-1, -1] = BN + 4 * (DN - 2 * CN * dr * sigma_sb * emissivity_back) / k_p_1[-2] * T[p + 1, -2] ** 3

            F_matrix[-1] = -T[p, -2] + (AN + CN) * T[p + 1, -3] + BN * T[p + 1, -2] + (
                        DN - 2 * CN * dr * sigma_sb * emissivity_back / k_p_1[-2]) * T[
                               p + 1, -2] ** 4 + EN + FN + 2 * dr * sigma_sb * absorptivity_back * T_LBW ** 4 * CN / \
                           k_p_1[-2]

            # T[p+1,-2] = -(AN+CN)/BN*T[p+1,-3]-(DN - 2*CN*dr*sigma_sb*emissivity_back/k_p_1[-2])/BN*T[p+1,-2]**4 - EN/BN - FN/BN - 2*dr*sigma_sb*absorptivity_back*T_LBW**4*CN/BN/k_p_1[-2] + 1/BN*T[p,-2]

            # delta_T = -np.dot(inv(Jac),F_matrix)
            delta_T = np.linalg.solve(Jac, -F_matrix)

            T[p + 1, :-1] = T[p + 1, :-1] + delta_T

            T[p + 1, -1] = 2 * dr * sigma_sb / k_p_1[-2] * (
                        absorptivity_back * T_LBW ** 4 - emissivity_back * T[p + 1, -2] ** 4) + T[p + 1, -3]

            err = np.max(abs(delta_T))

            if err < 1e-11:
                # if p == 10:
                #     print("stable time reach at iter = {:.0f}, with error = {:.2E} and first node temperature = {:.1E}".format(iter, err, T[p + 1, 0]))
                break
            if iter == Max_iter - 1:
                print('Convergence has not yet achieved! Increase iterations! Tempearture = {:.1E}'.format(T[p + 1, 1]))

        if (p > 0) & (p % N_one_cycle == 0) & (simulated_amp_phase_extraction_method != 'fft'):
            A_max = np.max(T[p - N_one_cycle:p, :-1], axis=0)
            A_min = np.min(T[p - N_one_cycle:p, :-1], axis=0)
            if np.max(np.abs((T_temp[:] - T[p+1, :-1]) / (A_max - A_min))) < 1e-3:
                N_steady_count += 1
                if N_steady_count == N_stable_cycle_output:  # only need 2 periods to calculate amplitude and phase
                    time_index = p
                    print("stable temperature profile has been obtained @ iteration N= {}!".format(
                        int(p / N_one_cycle)))
                    break

            T_temp[:] = T[p+1, :-1]

        if (p == Nt - 2) and (N_steady_count < N_stable_cycle_output):
            time_index = p
            print("Error! No stable temperature profile was obtained!")

    print(
        'sigma_s = {:.2E}, Amax = {}, f_heating = {}, focal plane = {}, Rec = {}, T_LB_mean = {:.1f}, alpha_r_A = {:.2e}, alpha_r_B = {:.2e}.'.format(
            light_source_property['sigma_s'], light_source_property['Amax'], f_heating,
            vacuum_chamber_setting['focal_shift'],
            sample_information['rec_name'], T_LB_mean_C,alpha_r_A,alpha_r_B))



    return T[:time_index,:], time_simulation[:time_index], r, N_one_cycle, q_solar[:time_index,:]









def finite_difference_implicit_variable_properties(sample_information, vacuum_chamber_setting,
                                                          solar_simulator_settings,
                                                          light_source_property, numerical_simulation_setting,
                                                          df_solar_simulator_VQ,
                                                          sigma_df, code_directory, df_view_factor, df_LB_details_all):

    Max_iter = 10000

    T_LBW = float(df_view_factor['T_LBW_C'][0]) + 273.15

    R = sample_information['R']  # sample radius
    Nr = numerical_simulation_setting['Nr_node']  # number of discretization along radial direction
    t_z = sample_information['t_z']  # sample thickness

    dr = R / (Nr - 1)
    r = np.arange(Nr)
    rm = r * dr

    rho = sample_information['rho']
    cp_0 = sample_information['cp_const']  # Cp = cp_const + cp_c1*T + cp_c2/T + cp_c3/T**2, T unit in K
    cp_1 = sample_information['cp_c1']
    cp_2 = sample_information['cp_c2']
    cp_3 = sample_information['cp_c3']

    R0 = vacuum_chamber_setting['R0_node']

    alpha_r_A = float(sample_information['alpha_r_A'])
    alpha_r_B = float(sample_information['alpha_r_B'])

    T_initial = sample_information['T_initial']  # unit in K
    # k = alpha*rho*cp

    f_heating = solar_simulator_settings['f_heating']  # periodic heating frequency
    N_cycle = numerical_simulation_setting['N_cycle']
    N_stable_cycle_output = numerical_simulation_setting['N_stable_cycle_output']

    simulated_amp_phase_extraction_method = numerical_simulation_setting['simulated_amp_phase_extraction_method']

    emissivity_front = sample_information['emissivity_front']  # assumed to be constant
    emissivity_back = sample_information['emissivity_back']  # assumed to be constant

    absorptivity_solar = sample_information['absorptivity_solar']

    absorptivity_front = sample_information['absorptivity_front']  # assumed to be constant
    absorptivity_back = sample_information['absorptivity_back']  # assumed to be constant

    sigma_sb = 5.6703744 * 10 ** (-8)  # stefan-Boltzmann constant

    Cm_LB, qm_LB, Cm_back, T_LB_mean_C = radiation_absorption_view_factor_calculations_network(code_directory, rm, dr,
                                                                                   sample_information,
                                                                                   solar_simulator_settings,
                                                                                   vacuum_chamber_setting,
                                                                                   numerical_simulation_setting,
                                                                                   df_view_factor, df_LB_details_all)


    dt = 1 / f_heating / numerical_simulation_setting['simulated_num_data_per_cycle']

    t_total = 1 / f_heating * N_cycle  # total simulation time

    Nt = int(t_total / dt)  # total number of simulation time step
    time_simulation = dt * np.arange(Nt)

    T = T_initial * np.ones((Nt, Nr + 1))
    dz = t_z
    q_solar = absorptivity_solar*light_source_intensity_Amax_fV_vecterize(np.arange(Nr) * dr, np.arange(Nt) * dt,
                                                       solar_simulator_settings, vacuum_chamber_setting,
                                                       light_source_property,
                                                       df_solar_simulator_VQ, sigma_df)

    #r_lb_window,lb_window = pd.read_csv('LB_window_function.csv')
    #f_lb_window = interp1d(r_lb_window,lb_window)
    #lb_window_function = f_lb_window(np.arange(Nr))
    #q_solar = q_solar*lb_window_function

    T_temp = np.zeros(Nr)
    N_steady_count = 0
    time_index = Nt - 1
    N_one_cycle = int(Nt / N_cycle)

    C_surr = qm_LB + Cm_back

    Jac = np.zeros((Nr, Nr))
    F_matrix = np.zeros(Nr)

    for p in tqdm(range(Nt - 1)):  # p indicate time step

        c_p_1 = cp_0 + cp_1 * T[p, :] + cp_2 / T[p, :] + cp_3 / T[p, :] ** 2
        alpha_p_1 = 1 / (alpha_r_A * T[p, :] + alpha_r_B)
        k_p_1 = c_p_1 * alpha_p_1 * rho

        for iter in range(Max_iter):
            # print(c_p_1)
            alpha_ba_0 = (k_p_1[0] + k_p_1[1]) / (2 * rho * c_p_1[0])
            alpha_0 = k_p_1[0] / (rho * c_p_1[0])
            A0 = -alpha_ba_0 * dt / dr ** 2 + dt * alpha_0 / (2 * rm[1] * dr)

            alpha_ce_0 = (2 * k_p_1[0] + 2 * k_p_1[1]) / (4 * rho * c_p_1[0])
            B0 = 2 * dt / dr ** 2 * alpha_ce_0 + 1

            alpha_fo_0 = (k_p_1[0] + k_p_1[1]) / (2 * rho * c_p_1[0])
            C0 = -dt / dr ** 2 * alpha_fo_0 - dt * alpha_0 / (2 * rm[1] * dr)
            D0 = (Cm_LB[0]/(2*np.pi*dr/2*dr) + emissivity_back) * sigma_sb * dt / (dz * rho * c_p_1[0])
            E0 = -q_solar[p, 0] * dt / (dz * rho * c_p_1[0])
            F0 = -C_surr[0] / (2 * np.pi * rm[1] * dr * dz) * dt / (rho * c_p_1[0])

            Jac[0, 0] = B0 + 4 * D0 * T[p + 1, 0] ** 3
            Jac[0, 1] = A0 + C0
            F_matrix[0] = - T[p, 0] + (A0 + C0) * T[p + 1, 1] + B0 * T[p + 1, 0] + D0 * T[p + 1, 0] ** 4 + E0 + F0

            # T[p+1,0] = -(A0+C0)/B0*T[p+1,1]- D0/B0*T[p+1,0]**4-E0/B0 - F0/B0 + 1/B0*T[p,0]

            alpha_ba = (k_p_1[1:-2] + k_p_1[0:-3]) / (2 * rho * c_p_1[1:-2])
            alpha = k_p_1[1:-2] / (rho * c_p_1[1:-2])
            Am = -alpha_ba * dt / dr ** 2 + dt * alpha / (2 * rm[1:-1] * dr)

            alpha_ce = (k_p_1[0:-3] + 2 * k_p_1[1:-2] + k_p_1[2:-1]) / (4 * rho * c_p_1[1:-2])
            Bm = 2 * dt / dr ** 2 * alpha_ce + 1

            alpha_fo = (k_p_1[1:-2] + k_p_1[2:-1]) / (2 * rho * c_p_1[1:-2])
            Cm = -dt / dr ** 2 * alpha_fo - dt * alpha / (2 * rm[1:-1] * dr)
            Dm = (Cm_LB[1:-1]/(2*np.pi*rm[1:-1]*dr) + emissivity_back) * sigma_sb * dt / (dz * rho * c_p_1[1:-2])
            Em = -q_solar[p, 1:-1] * dt / (dz * rho * c_p_1[1:-2])
            Fm = -C_surr[1:-1] / (2 * np.pi * rm[1:-1] * dr * dz) * dt / (rho * c_p_1[1:-2])

            for idx in range(len(alpha)):
                Jac[idx + 1, idx] = Am[idx]
                Jac[idx + 1, idx + 1] = Bm[idx] + 4 * Dm[idx] * T[p + 1, idx + 1] ** 3
                Jac[idx + 1, idx + 2] = Cm[idx]

            F_matrix[1:-1] = -T[p, 1:-2] + Am * T[p + 1, 0:-3] + Bm * T[p + 1, 1:-2] + Cm * T[p + 1, 2:-1] + Dm * T[p + 1,1:-2] ** 4 + Em + Fm

            # T[p+1,1:-2] = -Am/Bm*T[p+1,0:-3] - Cm/Bm*T[p+1,2:-1]-Dm/Bm*T[p+1,1:-2]**4 - Em/Bm - Fm/Bm + 1/Bm*T[p,1:-2]

            # Be ware of the properties at node N+1
            alpha_ba_N = (k_p_1[-2] + k_p_1[-3]) / (2 * rho * c_p_1[-2])
            alpha_N = k_p_1[-2] / (rho * c_p_1[-2])
            AN = -alpha_ba_N * dt / dr ** 2 + dt * alpha_N / (2 * rm[-1] * dr)

            alpha_ce_N = (k_p_1[-3] + 2 * k_p_1[-2] + k_p_1[-1]) / (4 * rho * c_p_1[-2])
            BN = 2 * dt / dr ** 2 * alpha_ce_N + 1

            alpha_fo_N = (k_p_1[-2] + k_p_1[-1]) / (2 * rho * c_p_1[-2])
            CN = -dt / dr ** 2 * alpha_fo_N - dt * alpha_N / (2 * rm[-1] * dr)
            DN = (Cm_LB[-1]/(2*np.pi*rm[-1]*dr) + emissivity_back) * sigma_sb * dt / (dz * rho * c_p_1[-2])
            EN = -q_solar[p, -1] * dt / (dz * rho * c_p_1[-2])
            FN = -C_surr[-1] / (2 * np.pi * rm[-1] * dr * dz) * dt / (rho * c_p_1[-2])

            Jac[-1, -2] = AN + CN
            Jac[-1, -1] = BN + 4 * (DN - 2 * CN * dr * sigma_sb * emissivity_back) / k_p_1[-2] * T[p + 1, -2] ** 3

            F_matrix[-1] = -T[p, -2] + (AN + CN) * T[p + 1, -3] + BN * T[p + 1, -2] + (
                        DN - 2 * CN * dr * sigma_sb * emissivity_back / k_p_1[-2]) * T[
                               p + 1, -2] ** 4 + EN + FN + 2 * dr * sigma_sb * absorptivity_back * T_LBW ** 4 * CN / \
                           k_p_1[-2]
            if p==0:
                df_Jac = pd.DataFrame(data = Jac)
                df_Jac.to_csv('Jacobian_diagnostic_1D.csv')
                df_F_matrix = pd.DataFrame(data = F_matrix)
                df_F_matrix.to_csv('F_matrix_diagonstic_1D.csv')
            # T[p+1,-2] = -(AN+CN)/BN*T[p+1,-3]-(DN - 2*CN*dr*sigma_sb*emissivity_back/k_p_1[-2])/BN*T[p+1,-2]**4 - EN/BN - FN/BN - 2*dr*sigma_sb*absorptivity_back*T_LBW**4*CN/BN/k_p_1[-2] + 1/BN*T[p,-2]

            # delta_T = -np.dot(inv(Jac),F_matrix)
            delta_T = np.linalg.solve(Jac, -F_matrix)

            T[p + 1, :-1] = T[p + 1, :-1] + delta_T

            T[p + 1, -1] = 2 * dr * sigma_sb / k_p_1[-2] * (
                        absorptivity_back * T_LBW ** 4 - emissivity_back * T[p + 1, -2] ** 4) + T[p + 1, -3]

            err = np.max(abs(delta_T))

            if err < 1e-11:
                # if p == 10:
                #     print("stable time reach at iter = {:.0f}, with error = {:.2E} and first node temperature = {:.1E}".format(iter, err, T[p + 1, 0]))
                break
            if iter == Max_iter - 1:
                print('Convergence has not yet achieved! Increase iterations! Tempearture = {:.1E}'.format(T[p + 1, 1]))

        if (p > 0) & (p % N_one_cycle == 0) & (simulated_amp_phase_extraction_method != 'fft'):
            A_max = np.max(T[p - N_one_cycle:p, :-1], axis=0)
            A_min = np.min(T[p - N_one_cycle:p, :-1], axis=0)
            if np.max(np.abs((T_temp[:] - T[p+1, :-1]) / (A_max - A_min))) < 1e-3:
                N_steady_count += 1
                if N_steady_count == N_stable_cycle_output:  # only need 2 periods to calculate amplitude and phase
                    time_index = p
                    print("stable temperature profile has been obtained @ iteration N= {}!".format(
                        int(p / N_one_cycle)))
                    break

            T_temp[:] = T[p+1, :-1]

        if (p == Nt - 2) and (N_steady_count < N_stable_cycle_output):
            time_index = p
            print("Error! No stable temperature profile was obtained!")

    print(
        'sigma_s = {:.2E}, Amax = {}, f_heating = {}, focal plane = {}, Rec = {}, T_LB_mean = {:.1f}, alpha_r_A = {:.2e}, alpha_r_B = {:.2e}.'.format(
            light_source_property['sigma_s'], light_source_property['Amax'], f_heating,
            vacuum_chamber_setting['focal_shift'],
            sample_information['rec_name'], T_LB_mean_C,alpha_r_A,alpha_r_B))


    return T[:time_index,:], time_simulation[:time_index], r, N_one_cycle, q_solar[:time_index,:]




def finite_difference_2D_nonvectorized_implicit_variable_properties(sample_information, vacuum_chamber_setting,
                                                          solar_simulator_settings,
                                                          light_source_property, numerical_simulation_setting,
                                                          df_solar_simulator_VQ,
                                                          sigma_df, code_directory, df_view_factor, df_LB_details_all):

    Max_iter = 1000

    T_LBW = float(df_view_factor['T_LBW_C'][0]) + 273.15

    R = sample_information['R']  # sample radius
    Nr = int(numerical_simulation_setting['Nr_node'])  # number of discretization along radial direction
    Nz = int(numerical_simulation_setting['Nz_node']) # number of discretization along radial direction

    t_z = sample_information['t_z']  # sample thickness

    dr = R / (Nr - 1)
    dz = t_z/(Nz-1)

    r = np.arange(Nr)
    rm = r * dr

    rho = sample_information['rho']
    cp_0 = sample_information['cp_const']  # Cp = cp_const + cp_c1*T + cp_c2/T + cp_c3/T**2, T unit in K
    cp_1 = sample_information['cp_c1']
    cp_2 = sample_information['cp_c2']
    cp_3 = sample_information['cp_c3']

    R0 = vacuum_chamber_setting['R0_node']

    alpha_r_A = float(sample_information['alpha_r_A'])
    alpha_r_B = float(sample_information['alpha_r_B'])

    T_initial = sample_information['T_initial']  # unit in K
    # k = alpha*rho*cp

    f_heating = solar_simulator_settings['f_heating']  # periodic heating frequency
    N_cycle = numerical_simulation_setting['N_cycle']
    N_stable_cycle_output = numerical_simulation_setting['N_stable_cycle_output']

    simulated_amp_phase_extraction_method = numerical_simulation_setting['simulated_amp_phase_extraction_method']

    emissivity_front = sample_information['emissivity_front']  # assumed to be constant
    emissivity_back = sample_information['emissivity_back']  # assumed to be constant

    absorptivity_solar = sample_information['absorptivity_solar']

    absorptivity_front = sample_information['absorptivity_front']  # assumed to be constant
    absorptivity_back = sample_information['absorptivity_back']  # assumed to be constant

    sigma_sb = 5.6703744 * 10 ** (-8)  # stefan-Boltzmann constant

    Cm_LB, qm_LB, Cm_back, T_LB_mean_C = radiation_absorption_view_factor_calculations_network(code_directory, rm, dr,
                                                                                   sample_information,
                                                                                   solar_simulator_settings,
                                                                                   vacuum_chamber_setting,
                                                                                   numerical_simulation_setting,
                                                                                   df_view_factor, df_LB_details_all)


    dt = 1 / f_heating / numerical_simulation_setting['simulated_num_data_per_cycle']

    t_total = 1 / f_heating * N_cycle  # total simulation time

    Nt = int(t_total / dt)  # total number of simulation time step
    time_simulation = dt * np.arange(Nt)

    T = T_initial * np.ones((Nt, Nr*Nz))

    q_solar = absorptivity_solar*light_source_intensity_Amax_fV_vecterize(np.arange(Nr) * dr, np.arange(Nt) * dt,
                                                       solar_simulator_settings, vacuum_chamber_setting,
                                                       light_source_property,
                                                       df_solar_simulator_VQ, sigma_df)

    #r_lb_window,lb_window = pd.read_csv('LB_window_function.csv')
    #f_lb_window = interp1d(r_lb_window,lb_window)
    #lb_window_function = f_lb_window(np.arange(Nr))
    #q_solar = q_solar*lb_window_function

    T_temp = np.zeros(Nr)
    N_steady_count = 0
    time_index = Nt - 1
    N_one_cycle = int(Nt / N_cycle)

    C_surr = qm_LB + Cm_back

    Jac = np.zeros((Nr*Nz, Nr*Nz))
    F_matrix = np.zeros(Nr*Nz)

    for p in tqdm(range(Nt - 1)):  # p indicate time step

        c_p_1 = cp_0 + cp_1 * T[p, :] + cp_2 / T[p, :] + cp_3 / T[p, :] ** 2
        alpha_p_1 = 1 / (alpha_r_A * T[p, :] + alpha_r_B)
        k_p_1 = c_p_1 * alpha_p_1 * rho

        for iter in range(Max_iter):
            # print(c_p_1)

            # top left corner -- consistent with derivation
            A0_0 = dt/(2*dr**2*rho*c_p_1[0])*(k_p_1[0]+k_p_1[0]) - dt/(2*dr*dr*rho*c_p_1[0])*k_p_1[0]
            B0_0 = -(dt/(2*dr**2*rho*c_p_1[0])*(k_p_1[1]+2*k_p_1[0]+k_p_1[0])+dt/(2*dz**2*rho*c_p_1[0])*(k_p_1[0]+2*k_p_1[0]+k_p_1[Nr])+1)
            C0_0 = dt*k_p_1[0]/(2*dr*dr*rho*c_p_1[0]) + dt/(2*dr**2*rho*c_p_1[0])*(k_p_1[0]+k_p_1[0])
            D0_0 = dt/(2*dz**2*rho*c_p_1[0])*(k_p_1[0]+k_p_1[0])
            E0_0 = dt/(2*dz**2*rho*c_p_1[0])*(k_p_1[0]+k_p_1[Nr])

            Jac[0,0] = B0_0 - 8*dz*Cm_LB[0]*sigma_sb*D0_0/(k_p_1[0]*2*np.pi*dr*dr)*T[p+1,0]**3
            Jac[0,1] = A0_0 + C0_0
            Jac[0,Nr] = D0_0 + E0_0

            F_matrix[0] = T[p,0] + (A0_0+C0_0)*T[p+1,1] + B0_0*T[p+1,0] - 2*dz*Cm_LB[0]*sigma_sb*D0_0/(k_p_1[0]*2*np.pi*dr*dr)*T[p+1,0]**4 + (D0_0+E0_0)*T[p+1,Nr] +\
                          (2*dz/(k_p_1[0]*2*np.pi*dr*dr)*qm_LB[0] + 2*dz*q_solar[p,0]/k_p_1[0])*D0_0

            #top middle region -- consistent with derivation
            for i in np.arange(1,Nr-1,1):
                Am_0 = dt/(2*dr**2*rho*c_p_1[i])*(k_p_1[i-1]+k_p_1[i]) - dt/(2*rm[i]*dr*rho*c_p_1[i])*k_p_1[i]

                T_p_ghost = T[p+1,i+Nr] - 2*dz*Cm_LB[i]*sigma_sb/(k_p_1[i]*2*np.pi*rm[i]*dr)*T[p+1,i]**4 + 2*dz/(k_p_1[i]*2*np.pi*rm[i]*dr)*qm_LB[i] + 2*dz/k_p_1[i]*q_solar[p,i]
                k_p_1_ghost = rho*(cp_0 + cp_1 * T_p_ghost + cp_2 / T_p_ghost + cp_3 / T_p_ghost ** 2)*(1 / (alpha_r_A * T_p_ghost + alpha_r_B))
                Bm_0 = -(dt/(2*dr**2*rho*c_p_1[i])*(k_p_1[i+1]+2*k_p_1[i]+k_p_1[i-1])+dt/(2*dz**2*rho*c_p_1[i])*(k_p_1_ghost+2*k_p_1[i]+k_p_1[Nr+i])+1)
                Cm_0 = dt * k_p_1[i] / (2 * rm[i] * dr * rho * c_p_1[i]) + dt / (2 * dr ** 2 * rho * c_p_1[i]) * (k_p_1[i-1] + k_p_1[i])
                Dm_0 = dt / (2 * dz ** 2 * rho * c_p_1[i]) * (k_p_1_ghost + k_p_1[i])
                Em_0 = dt/(2*dz**2*rho*c_p_1[i])*(k_p_1[i]+k_p_1[Nr+i])

                Jac[i,i-1] = Am_0
                Jac[i,i] = Bm_0 - 8*dz*Cm_LB[i]*sigma_sb/(k_p_1[i]*2*np.pi*rm[i]*dr)*Dm_0*T[p+1,i]**3
                Jac[i,i+1] = Cm_0
                Jac[i, Nr+i] = Dm_0 + Em_0

                F_matrix[i] = T[p,i] + T[p+1,i-1]*Am_0 + T[p+1,i]*Bm_0 - 2*dz*Cm_LB[i]*sigma_sb/(k_p_1[i]*2*np.pi*rm[i]*dr)*Dm_0*T[p+1,i]**4 + \
                              Cm_0*T[p+1,i+1]+ T[p+1,Nr+i]*(Dm_0+Em_0) + (2*dz/(k_p_1[i]*2*np.pi*rm[i]*dr)*qm_LB[i] + 2*dz/(k_p_1[i])*q_solar[p,i])*Dm_0

            #top right corner --consistent with derivation
            AM_0 = dt/(2*dr**2*rho*c_p_1[Nr-1])*(k_p_1[Nr-1-1]+k_p_1[Nr-1]) - dt/(2*dr*dr*rho*c_p_1[Nr-1])*k_p_1[Nr-1]
            BM_0 = -(dt/(2*dr**2*rho*c_p_1[Nr-1])*(k_p_1[Nr-1]+2*k_p_1[Nr-1]+k_p_1[Nr-1-1])+dt/(2*dz**2*rho*c_p_1[Nr-1])*(k_p_1[Nr-1+Nr]+2*k_p_1[Nr-1]+k_p_1[Nr-1])+1)
            CM_0 = dt*k_p_1[Nr-1]/(2*rm[Nr-1]*dr*rho*c_p_1[Nr-1]) + dt/(2*dr**2*rho*c_p_1[Nr-1])*(k_p_1[Nr-1-1]+k_p_1[Nr-1])
            DM_0 = dt/(2*dz**2*rho*c_p_1[Nr-1])*(k_p_1[Nr-1]+k_p_1[Nr-1])
            EM_0 = dt/(2*dz**2*rho*c_p_1[Nr-1])*(k_p_1[Nr-1]+k_p_1[Nr-1+Nr])

            Jac[Nr-1,Nr-1-1] = AM_0 + CM_0
            Jac[Nr-1,Nr-1] = BM_0 - 8*dz*Cm_LB[Nr-1]*sigma_sb*DM_0/(k_p_1[Nr-1]*2*np.pi*rm[Nr-1]*dr)*T[p+1,Nr-1]**3 - 8*dr*sigma_sb*emissivity_front/k_p_1[Nr-1]*CM_0*T[p+1,Nr-1]**3
            Jac[Nr-1,2*Nr-1] = DM_0 + EM_0

            F_matrix[Nr-1] = T[p,Nr-1] + T[p+1,Nr-1-1]*(AM_0 + CM_0) + T[p+1,Nr-1]*BM_0 - 2*dr*sigma_sb*emissivity_front/k_p_1[Nr-1]*CM_0*T[p+1,Nr-1]**4 -2*dz*Cm_LB[Nr-1]*sigma_sb/(
                    k_p_1[Nr-1]*rm[Nr-1]*dr*2*np.pi)*DM_0*T[p+1,Nr-1]**4 + T[p+1,2*Nr-1]*(EM_0+DM_0) + 2*dz/(k_p_1[Nr-1]*rm[Nr-1]*dr*2*np.pi)*qm_LB[Nr-1] + 2*dz/k_p_1[Nr-1]*q_solar[p,Nr-1]

            #left center edge--consistent with derivation
            for n in np.arange(1,Nz-1,1):
                A0_n = dt/(2*dr**2*rho*c_p_1[Nr*n])*(k_p_1[Nr*n]+k_p_1[Nr*n]) - dt/(2*dr*dr*rho*c_p_1[Nr*n])*k_p_1[Nr*n]
                B0_n = -(dt/(2*dr**2*rho*c_p_1[Nr*n])*(k_p_1[Nr*n+1]+2*k_p_1[Nr*n]+k_p_1[Nr*n])+dt/(2*dz**2*rho*c_p_1[Nr*n])*(k_p_1[Nr*(n-1)]+2*k_p_1[Nr*n]+k_p_1[Nr*(n+1)])+1)
                C0_n = dt*k_p_1[Nr*n]/(2*dr*dr*rho*c_p_1[Nr*n]) + dt/(2*dr**2*rho*c_p_1[Nr*n])*(k_p_1[Nr*n]+k_p_1[Nr*n])
                D0_n = dt/(2*dz**2*rho*c_p_1[Nr*n])*(k_p_1[Nr*(n-1)]+k_p_1[Nr*n])
                E0_n = dt/(2*dz**2*rho*c_p_1[Nr*n])*(k_p_1[Nr*n]+k_p_1[Nr*(n+1)])

                Jac[Nr*n,Nr*n] = B0_n
                Jac[Nr*n,Nr*n+1] = A0_n + C0_n
                Jac[Nr*n,Nr*(n-1)] = D0_n
                Jac[Nr*n,Nr*(n+1)] = E0_n

                F_matrix[Nr*n] = T[p,Nr*n] + T[p+1,Nr*n]*B0_n + T[p+1,Nr*n+1]*(C0_n+A0_n) + T[p+1,Nr*(n-1)]*D0_n + T[p+1,Nr*(n+1)]*E0_n

            #considering the body area -- consistent with derivation
            for n in np.arange(1,Nz-1,1): # row, indicating n in thickness direction
                for m in np.arange(1,Nr-1,1): # column, indicating m in radial direction
                    Am_n = dt / (2 * dr ** 2 * rho * c_p_1[Nr * n+m]) * (k_p_1[Nr * n+m-1] + k_p_1[Nr * n+m]) - dt / (
                                2 * rm[m] * dr * rho * c_p_1[Nr * n+m]) * k_p_1[Nr * n+m]
                    Bm_n = -(dt / (2 * dr ** 2 * rho * c_p_1[Nr * n+m]) * (
                                k_p_1[Nr * n+m+1] + 2 * k_p_1[Nr * n+m] + k_p_1[Nr * n+m-1]) + dt / (
                                         2 * dz ** 2 * rho * c_p_1[Nr * n+m]) * (
                                         k_p_1[Nr * (n+1)+m] + 2 * k_p_1[Nr * n+m] + k_p_1[Nr * (n-1)+m]) + 1)
                    Cm_n = dt * k_p_1[Nr * n+m] / (2 * rm[m] * dr * rho * c_p_1[Nr * n+m]) + dt / (
                                2 * dr ** 2 * rho * c_p_1[Nr * n+m]) * (k_p_1[Nr * n+m-1] + k_p_1[Nr * n+m])
                    Dm_n = dt / (2 * dz ** 2 * rho * c_p_1[Nr * n+m]) * (k_p_1[Nr * (n-1)+m] + k_p_1[Nr * n+m])
                    Em_n = dt / (2 * dz ** 2 * rho * c_p_1[Nr * n+m]) * (k_p_1[Nr * n+m] + k_p_1[Nr * (n+1)+m])

                    Jac[Nr * n+m, Nr * n+m-1] = Am_n
                    Jac[Nr * n+m, Nr * n+m] = Bm_n
                    Jac[Nr * n+m, Nr * n+m+1] = Cm_n
                    Jac[Nr * n+m, Nr * (n-1)+m] = Dm_n
                    Jac[Nr * n + m, Nr * (n + 1) + m] = Em_n

                    F_matrix[Nr * n + m] = T[p,Nr * n + m] + T[p+1,Nr * n + m-1]*Am_n + T[p+1,Nr * n + m]*Bm_n + T[p+1,Nr * n + m+1]*Cm_n + T[p+1,Nr * (n-1) + m]*Dm_n + T[p+1,Nr * (n+1) + m]*Em_n

            #right edge--consistent with derivation
            for n in np.arange(1,Nz-1,1):
                AM_n = dt / (2 * dr ** 2 * rho * c_p_1[Nr * n+Nr-1]) * (k_p_1[Nr * n+Nr-1-1] + k_p_1[Nr * n+Nr-1]) - dt / (
                                2 * rm[Nr-1] * dr * rho * c_p_1[Nr * n+Nr-1]) * k_p_1[Nr * n+Nr-1]
                BM_n = -(dt / (2 * dr ** 2 * rho * c_p_1[Nr * n+Nr-1]) * (
                                k_p_1[Nr * n+Nr-1] + 2 * k_p_1[Nr * n+Nr-1] + k_p_1[Nr * n+Nr-1-1]) + dt / (
                                         2 * dz ** 2 * rho * c_p_1[Nr * n+Nr-1]) * (k_p_1[Nr * (n-1)+Nr-1] + 2 * k_p_1[Nr * n+Nr-1] + k_p_1[Nr * (n+1)+Nr-1]) + 1)
                CM_n = dt * k_p_1[Nr * n + Nr-1] / (2 * rm[Nr-1] * dr * rho * c_p_1[Nr * n + Nr-1]) + dt / (
                        2 * dr ** 2 * rho * c_p_1[Nr * n + Nr-1]) * (k_p_1[Nr * n + Nr-1 - 1] + k_p_1[Nr * n + Nr-1])
                DM_n = dt / (2 * dz ** 2 * rho * c_p_1[Nr * n+Nr-1]) * (k_p_1[Nr * (n-1)+Nr-1] + k_p_1[Nr * n+Nr-1])
                EM_n = dt / (2 * dz ** 2 * rho * c_p_1[Nr * n+Nr-1]) * (k_p_1[Nr * n+Nr-1] + k_p_1[Nr * (n+1)+Nr-1])

                Jac[Nr * n+Nr-1,Nr * n+Nr-1-1] = AM_n + CM_n
                Jac[Nr * n+Nr-1,Nr * n+Nr-1] = BM_n - 8*sigma_sb*dr*emissivity_front*CM_n/k_p_1[Nr * n+Nr-1]*T[p+1,Nr * n+Nr-1]**3
                Jac[Nr * n+Nr-1,Nr * (n-1)+Nr-1] = DM_n
                Jac[Nr * n+Nr-1,Nr * (n+1)+Nr-1] = Em_n

                F_matrix[Nr * n+Nr-1] = T[p,Nr * n+Nr-1] + T[p+1,Nr * n+Nr-1-1]*(AM_n+CM_n) + T[p+1,Nr * n+Nr-1]*BM_n - 2*sigma_sb*dr*emissivity_front*CM_n/k_p_1[Nr * n+Nr-1]*T[p+1,Nr * n+Nr-1]**4 + \
                    T[p+1,Nr * (n-1)+Nr-1]*DM_n + T[p+1,Nr * (n+1)+Nr-1]*EM_n + 2*sigma_sb*dr*emissivity_front*CM_n/k_p_1[Nr * n+Nr-1]*T_LBW**4


            #bottom left corner-consistent with derivation
            A0_N = dt/(2*dr**2*rho*c_p_1[Nr*(Nz-1)])*(k_p_1[Nr*(Nz-1)]+k_p_1[Nr*(Nz-1)]) - dt/(2*dr*dr*rho*c_p_1[Nr*(Nz-1)])*k_p_1[Nr*(Nz-1)]
            B0_N = -(dt/(2*dr**2*rho*c_p_1[Nr*(Nz-1)])*(k_p_1[Nr*(Nz-1)+1]+2*k_p_1[Nr*(Nz-1)]+k_p_1[Nr*(Nz-1)])+dt/(2*dz**2*rho*c_p_1[Nr*(Nz-1)])*(k_p_1[Nr*(Nz-1)]+2*k_p_1[Nr*(Nz-1)]+k_p_1[Nr*(Nz-1-1)])+1)
            C0_N = dt*k_p_1[Nr*(Nz-1)]/(2*dr*dr*rho*c_p_1[Nr*(Nz-1)]) + dt/(2*dr**2*rho*c_p_1[Nr*(Nz-1)])*(k_p_1[Nr*(Nz-1)-1]+k_p_1[Nr*(Nz-1)])
            D0_N = dt/(2*dz**2*rho*c_p_1[Nr*(Nz-1)])*(k_p_1[Nr*(Nz-1-1)]+k_p_1[Nr*(Nz-1)])
            E0_N = dt/(2*dz**2*rho*c_p_1[Nr*(Nz-1)])*(k_p_1[Nr*(Nz-1)]+k_p_1[Nr*(Nz-1)])

            Jac[Nr*(Nz-1),Nr*(Nz-1)] = B0_N - 8*dz*sigma_sb*emissivity_back*E0_N/k_p_1[Nr*(Nz-1)]*T[p+1,Nr*(Nz-1)]**3
            Jac[Nr*(Nz-1),Nr*(Nz-1)+1] = A0_N + C0_N
            Jac[Nr*(Nz-1),Nr*(Nz-1-1)] = D0_N + E0_N

            F_matrix[Nr*(Nz-1)] = T[p,Nr*(Nz-1)] + T[p+1,Nr*(Nz-1)+1]*A0_N + T[p+1,Nr*(Nz-1)]*B0_N -2*dz*sigma_sb*emissivity_back*E0_N/k_p_1[Nr*(Nz-1)]*T[p+1,Nr*(Nz-1)]**4 + \
                T[p+1,Nr*(Nz-1)+1]*C0_N + T[p+1,Nr*(Nz-1-1)]*(D0_N + E0_N) + 2*dz*Cm_back[0]*E0_N/(k_p_1[Nr*(Nz-1)]*dr*dr*np.pi*2)

            #bottom center region - consistent with derivation
            for i in np.arange(1,Nr-1,1):
                Am_N = dt/(2*dr**2*rho*c_p_1[i + (Nz-1)*Nr])*(k_p_1[i + (Nz-1)*Nr-1]+k_p_1[i + (Nz-1)*Nr]) - dt/(2*rm[i]*dr*rho*c_p_1[i + (Nz-1)*Nr])*k_p_1[i + (Nz-1)*Nr]

                T_p_ghost = T[p+1,i + (Nz-1-1)*Nr] - 2*dz*emissivity_back*sigma_sb/k_p_1[i + (Nz-1)*Nr]*T[p+1,i + (Nz-1)*Nr]**4 + 2*dz/(rm[i]*dr*np.pi*2*k_p_1[i + (Nz-1)*Nr])*Cm_back[i]
                k_p_1_ghost = rho*(cp_0 + cp_1 * T_p_ghost + cp_2 / T_p_ghost + cp_3 / T_p_ghost ** 2)*(1 / (alpha_r_A * T_p_ghost + alpha_r_B))

                Bm_N = -(dt/(2*dr**2*rho*c_p_1[i + (Nz-1)*Nr])*(k_p_1[i + (Nz-1)*Nr+1]+2*k_p_1[i + (Nz-1)*Nr]+k_p_1[i + (Nz-1)*Nr-1])+dt/(2*dz**2*rho*c_p_1[i + (Nz-1)*Nr])*(k_p_1_ghost+2*k_p_1[i + (Nz-1)*Nr]+k_p_1[i + (Nz-1-1)*Nr])+1)
                Cm_N = dt * k_p_1[i + (Nz-1)*Nr] / (2 * rm[i] * dr * rho * c_p_1[i + (Nz-1)*Nr]) + dt / (2 * dr ** 2 * rho * c_p_1[i + (Nz-1)*Nr]) * (k_p_1[i + (Nz-1)*Nr-1] + k_p_1[i + (Nz-1)*Nr])
                Dm_N = dt / (2 * dz ** 2 * rho * c_p_1[i + (Nz-1)*Nr]) * (k_p_1[i + (Nz-1-1)*Nr] + k_p_1[i + (Nz-1)*Nr])
                Em_N = dt/(2*dz**2*rho*c_p_1[i + (Nz-1)*Nr])*(k_p_1[i + (Nz-1)*Nr]+k_p_1_ghost)

                Jac[i + (Nz-1)*Nr,i + (Nz-1)*Nr-1] = Am_N
                Jac[i + (Nz-1)*Nr,i + (Nz-1)*Nr] = Bm_N - 8*dz*sigma_sb*emissivity_back*Em_N/k_p_1[i + (Nz-1)*Nr]*T[p+1,i + (Nz-1)*Nr]**3
                Jac[i + (Nz-1)*Nr, i + (Nz-1)*Nr+1] = Cm_N
                Jac[i + (Nz-1)*Nr, i + (Nz-1-1)*Nr] = Dm_N + Em_N

                F_matrix[i + (Nz-1)*Nr] = T[p,i + (Nz-1)*Nr] + T[p+1,i + (Nz-1)*Nr-1]*Am_N + T[p+1,i + (Nz-1)*Nr]*Bm_N - 2*dz*emissivity_back*sigma_sb/k_p_1[i + (Nz-1)*Nr]*Em_N*T[p+1,i + (Nz-1)*Nr]**4 + T[p+1, i + (Nz-1)*Nr+1]*Cm_N +\
                    T[p+1,i + (Nz-1-1)*Nr]*(Dm_N + Em_N) + 2*dz*Cm_back[i]*Em_N/(rm[i]*dr*2*np.pi*k_p_1[i + (Nz-1)*Nr])

            #bottom right corner
            AM_N = dt/(2*dr**2*rho*c_p_1[Nr*Nz-1])*(k_p_1[Nr*Nz-1-1]+k_p_1[Nr*Nz-1]) - dt/(2*rm[Nr-1]*dr*rho*c_p_1[Nr*Nz-1])*k_p_1[Nr*Nz-1]
            BM_N = -(dt/(2*dr**2*rho*c_p_1[Nr*Nz-1])*(k_p_1[Nr*Nz-1]+2*k_p_1[Nr*Nz-1]+k_p_1[Nr*Nz-1-1])+dt/(2*dz**2*rho*c_p_1[Nr*Nz-1])*(k_p_1[Nr*Nz-1]+2*k_p_1[Nr*Nz-1]+k_p_1[Nr*(Nz-1)-1])+1)
            CM_N = dt*k_p_1[Nr*Nz-1]/(2*rm[Nr-1]*dr*rho*c_p_1[Nr*Nz-1]) + dt/(2*dr**2*rho*c_p_1[Nr*Nz-1])*(k_p_1[Nr*Nz-1-1]+k_p_1[Nr*Nz-1])
            DM_N = dt/(2*dz**2*rho*c_p_1[Nr*Nz-1])*(k_p_1[Nr*(Nz-1)-1]+k_p_1[Nr*Nz-1])
            EM_N = dt/(2*dz**2*rho*c_p_1[Nr*Nz-1])*(k_p_1[Nr*Nz-1]+k_p_1[Nr*Nz-1])

            Jac[Nr*Nz-1,Nr*Nz-1-1] = AM_N + CM_N
            Jac[Nr*Nz-1,Nr*Nz-1] = BM_N - 8*dr*emissivity_back*sigma_sb*CM_N/k_p_1[Nr*Nz-1]*T[p+1,Nr*Nz-1]**3 -8*dz*emissivity_back*sigma_sb*EM_N/k_p_1[Nr*Nz-1]*T[p+1,Nr*Nz-1]**3
            Jac[Nr*Nz-1,Nr*(Nz-1)-1] = DM_N + EM_N

            F_matrix[Nr*Nz-1] = T[p,Nr*Nz-1] + T[p+1,Nr*Nz-1-1]*(AM_N+CM_N) + T[p+1,Nr*Nz-1]*BM_N - 2*dr*emissivity_back*sigma_sb*CM_N/k_p_1[Nr*Nz-1]*T[p+1,Nr*Nz-1]**4 -2*dz*emissivity_back*sigma_sb*EM_N/k_p_1[Nr*Nz-1]*T[p+1,Nr*Nz-1]**4 +\
                T[p+1,Nr*(Nz-1)-1]*(DM_N + EM_N) + 2*Cm_back[Nr-1]*dz/(rm[Nr-1]*2*np.pi*dr*k_p_1[Nr*Nz-1])*EM_N

            if p==0:
                df_Jac = pd.DataFrame(data = Jac)
                df_Jac.to_csv('Jacobian_diagnostic.csv')
            delta_T = np.linalg.solve(Jac, -F_matrix)
            T[p + 1, :] = T[p + 1, :] + delta_T


            err = np.max(abs(delta_T))

            if err < 1e-11:
                if p%20==0:
                    pass
                    #rint("stable time reach at iter = {:.0f}, with error = {:.2E} and first node temperature = {:.1E}".format(iter, err, T[p + 1, 0]))
                break
            if iter == Max_iter - 1:
                print('Convergence has not yet achieved! Increase iterations! Tempearture = {:.1E}'.format(T[p + 1, 1]))

        if (p > 0) & (p % N_one_cycle == 0) & (simulated_amp_phase_extraction_method != 'fft'):
            A_max = np.max(T[p - N_one_cycle:p, (Nz-1)*Nr:], axis=0)
            A_min = np.min(T[p - N_one_cycle:p, (Nz-1)*Nr:], axis=0)
            if np.max(np.abs((T_temp[:] - T[p+1, (Nz-1)*Nr:]) / (A_max - A_min))) < 1e-3:
                N_steady_count += 1
                if N_steady_count == N_stable_cycle_output:  # only need 2 periods to calculate amplitude and phase
                    time_index = p
                    print("stable temperature profile has been obtained @ iteration N= {}!".format(
                        int(p / N_one_cycle)))
                    break

            T_temp[:] = T[p+1, (Nz-1)*Nr:]

        if (p == Nt - 2) and (N_steady_count < N_stable_cycle_output):
            time_index = p
            print("Error! No stable temperature profile was obtained!")

    print(
        'sigma_s = {:.2E}, Amax = {}, f_heating = {}, focal plane = {}, Rec = {}, T_LB_mean = {:.1f}, alpha_r_A = {:.2e}, alpha_r_B = {:.2e}.'.format(
            light_source_property['sigma_s'], light_source_property['Amax'], f_heating,
            vacuum_chamber_setting['focal_shift'],
            sample_information['rec_name'], T_LB_mean_C,alpha_r_A,alpha_r_B))


    return T[:time_index,:], time_simulation[:time_index], r, N_one_cycle, q_solar[:time_index,:]


def finite_difference_2D_implicit_variable_properties(sample_information, vacuum_chamber_setting,
                                                          solar_simulator_settings,
                                                          light_source_property, numerical_simulation_setting,
                                                          df_solar_simulator_VQ,
                                                          sigma_df, code_directory, df_view_factor, df_LB_details_all):

    Max_iter = 100000

    T_LBW = float(df_view_factor['T_LBW_C'][0]) + 273.15

    R = sample_information['R']  # sample radius
    Nr = int(numerical_simulation_setting['Nr_node'])  # number of discretization along radial direction
    Nz = int(numerical_simulation_setting['Nz_node']) # number of discretization along radial direction

    t_z = sample_information['t_z']  # sample thickness

    dr = R / (Nr - 1)
    dz = t_z/(Nz-1)

    r = np.arange(Nr)
    rm = r * dr

    rho = sample_information['rho']
    cp_0 = sample_information['cp_const']  # Cp = cp_const + cp_c1*T + cp_c2/T + cp_c3/T**2, T unit in K
    cp_1 = sample_information['cp_c1']
    cp_2 = sample_information['cp_c2']
    cp_3 = sample_information['cp_c3']

    R0 = vacuum_chamber_setting['R0_node']

    alpha_r_A = float(sample_information['alpha_r_A'])
    alpha_r_B = float(sample_information['alpha_r_B'])

    T_initial = sample_information['T_initial']  # unit in K
    # k = alpha*rho*cp

    f_heating = solar_simulator_settings['f_heating']  # periodic heating frequency
    N_cycle = numerical_simulation_setting['N_cycle']
    N_stable_cycle_output = numerical_simulation_setting['N_stable_cycle_output']

    simulated_amp_phase_extraction_method = numerical_simulation_setting['simulated_amp_phase_extraction_method']

    emissivity_front = sample_information['emissivity_front']  # assumed to be constant
    emissivity_back = sample_information['emissivity_back']  # assumed to be constant

    absorptivity_solar = sample_information['absorptivity_solar']

    absorptivity_front = sample_information['absorptivity_front']  # assumed to be constant
    absorptivity_back = sample_information['absorptivity_back']  # assumed to be constant

    sigma_sb = 5.6703744 * 10 ** (-8)  # stefan-Boltzmann constant

    Cm_LB, qm_LB, Cm_back, T_LB_mean_C = radiation_absorption_view_factor_calculations_network(code_directory, rm, dr,
                                                                                   sample_information,
                                                                                   solar_simulator_settings,
                                                                                   vacuum_chamber_setting,
                                                                                   numerical_simulation_setting,
                                                                                   df_view_factor, df_LB_details_all)


    dt = 1 / f_heating / numerical_simulation_setting['simulated_num_data_per_cycle']

    t_total = 1 / f_heating * N_cycle  # total simulation time

    Nt = int(t_total / dt)  # total number of simulation time step
    time_simulation = dt * np.arange(Nt)

    T = T_initial * np.ones((Nt, Nr*Nz))

    q_solar = absorptivity_solar*light_source_intensity_Amax_fV_vecterize(np.arange(Nr) * dr, np.arange(Nt) * dt,
                                                       solar_simulator_settings, vacuum_chamber_setting,
                                                       light_source_property,
                                                       df_solar_simulator_VQ, sigma_df)

    #r_lb_window,lb_window = pd.read_csv('LB_window_function.csv')
    #f_lb_window = interp1d(r_lb_window,lb_window)
    #lb_window_function = f_lb_window(np.arange(Nr))
    #q_solar = q_solar*lb_window_function

    T_temp = np.zeros(Nr)
    N_steady_count = 0
    time_index = Nt - 1
    N_one_cycle = int(Nt / N_cycle)

    C_surr = qm_LB + Cm_back

    Jac = np.zeros((Nr*Nz, Nr*Nz))
    F_matrix = np.zeros(Nr*Nz)

    for p in tqdm(range(Nt - 1)):  # p indicate time step

        c_p_1 = cp_0 + cp_1 * T[p, :] + cp_2 / T[p, :] + cp_3 / T[p, :] ** 2
        alpha_p_1 = 1 / (alpha_r_A * T[p, :] + alpha_r_B)
        k_p_1 = c_p_1 * alpha_p_1 * rho

        for iter in range(Max_iter):
            # print(c_p_1)

            # top left corner -- consistent with derivation, checked non-vectorized code
            A0_0 = dt/(2*dr**2*rho*c_p_1[0])*(k_p_1[0]+k_p_1[0]) - dt/(2*dr*dr*rho*c_p_1[0])*k_p_1[0]
            B0_0 = -(dt/(2*dr**2*rho*c_p_1[0])*(k_p_1[1]+2*k_p_1[0]+k_p_1[0])+dt/(2*dz**2*rho*c_p_1[0])*(k_p_1[0]+2*k_p_1[0]+k_p_1[Nr])+1)
            C0_0 = dt*k_p_1[0]/(2*dr*dr*rho*c_p_1[0]) + dt/(2*dr**2*rho*c_p_1[0])*(k_p_1[0]+k_p_1[0])
            D0_0 = dt/(2*dz**2*rho*c_p_1[0])*(k_p_1[0]+k_p_1[0])
            E0_0 = dt/(2*dz**2*rho*c_p_1[0])*(k_p_1[0]+k_p_1[Nr])

            Jac[0,0] = B0_0 - 8*dz*Cm_LB[0]*sigma_sb*D0_0/(k_p_1[0]*2*np.pi*dr*dr)*T[p+1,0]**3
            Jac[0,1] = A0_0 + C0_0
            Jac[0,Nr] = D0_0 + E0_0

            F_matrix[0] = T[p,0] + (A0_0+C0_0)*T[p+1,1] + B0_0*T[p+1,0] - 2*dz*Cm_LB[0]*sigma_sb*D0_0/(k_p_1[0]*2*np.pi*dr*dr)*T[p+1,0]**4 + (D0_0+E0_0)*T[p+1,Nr] +\
                          (2*dz/(k_p_1[0]*2*np.pi*dr*dr)*qm_LB[0] + 2*dz*q_solar[p,0]/k_p_1[0])*D0_0

            #top middle region -- consistent with derivation
            Am_0 = dt/(2*dr**2*rho*c_p_1[1:Nr-1])*(k_p_1[0:Nr-2]+k_p_1[1:Nr-1]) - dt/(2*rm[1:Nr-1]*dr*rho*c_p_1[1:Nr-1])*k_p_1[1:Nr-1]
            T_p_ghost = T[p+1,1+Nr:Nr+Nr-1] - 2*dz*Cm_LB[1:Nr-1]*sigma_sb/(k_p_1[1:Nr-1]*2*np.pi*rm[1:Nr-1]*dr)*T[p+1,1:Nr-1]**4 + 2*dz/(k_p_1[1:Nr-1]*2*np.pi*rm[1:Nr-1]*dr)*qm_LB[1:Nr-1] + 2*dz/k_p_1[1:Nr-1]*q_solar[p,1:Nr-1]
            k_p_1_ghost = rho*(cp_0 + cp_1 * T_p_ghost + cp_2 / T_p_ghost + cp_3 / T_p_ghost ** 2)*(1 / (alpha_r_A * T_p_ghost + alpha_r_B))
            Bm_0 = -(dt/(2*dr**2*rho*c_p_1[1:Nr-1])*(k_p_1[1+1:Nr-1+1]+2*k_p_1[1:Nr-1]+k_p_1[1-1:Nr-1-1])+dt/(2*dz**2*rho*c_p_1[1:Nr-1])*(k_p_1_ghost+2*k_p_1[1:Nr-1]+k_p_1[1+Nr:Nr-1+Nr])+1)

            Cm_0 = dt * k_p_1[1:Nr-1] / (2 * rm[1:Nr-1] * dr * rho * c_p_1[1:Nr-1]) + dt / (2 * dr ** 2 * rho * c_p_1[1:Nr-1]) * (k_p_1[1-1:Nr-1-1] + k_p_1[1:Nr-1])

            Dm_0 = dt / (2 * dz ** 2 * rho * c_p_1[1:Nr-1]) * (k_p_1_ghost + k_p_1[1:Nr-1])
            Em_0 = dt/(2*dz**2*rho*c_p_1[1:Nr-1])*(k_p_1[1:Nr-1]+k_p_1[1+Nr:Nr-1+Nr])

            for m in np.arange(1,Nr-1,1):
                Jac[m, m - 1] = Am_0[m-1]
                Jac[m, m] = Bm_0[m-1] - 8 * dz * Cm_LB[m] * sigma_sb / (k_p_1[m] * 2 * np.pi * rm[m] * dr) * Dm_0[m-1] * T[p + 1, m] ** 3
                Jac[m, m + 1] = Cm_0[m-1]
                Jac[m, Nr + m] = Dm_0[m-1]+ Em_0[m-1]


            F_matrix[1:Nr-1] = T[p,1:Nr-1] + T[p+1,1-1:Nr-1-1]*Am_0 + T[p+1,1:Nr-1]*Bm_0 - 2*dz*Cm_LB[1:Nr-1]*sigma_sb/(k_p_1[1:Nr-1]*2*np.pi*rm[1:Nr-1]*dr)*Dm_0*T[p+1,1:Nr-1]**4 + \
                              Cm_0*T[p+1,1+1:Nr-1+1]+ T[p+1,1+Nr:Nr-1+Nr]*(Dm_0+Em_0) + (2*dz/(k_p_1[1:Nr-1]*2*np.pi*rm[1:Nr-1]*dr)*qm_LB[1:Nr-1] + 2*dz/(k_p_1[1:Nr-1])*q_solar[p,1:Nr-1])*Dm_0


            #top right corner --consistent with derivation
            AM_0 = dt/(2*dr**2*rho*c_p_1[Nr-1])*(k_p_1[Nr-1-1]+k_p_1[Nr-1]) - dt/(2*dr*dr*rho*c_p_1[Nr-1])*k_p_1[Nr-1]
            BM_0 = -(dt/(2*dr**2*rho*c_p_1[Nr-1])*(k_p_1[Nr-1]+2*k_p_1[Nr-1]+k_p_1[Nr-1-1])+dt/(2*dz**2*rho*c_p_1[Nr-1])*(k_p_1[Nr-1+Nr]+2*k_p_1[Nr-1]+k_p_1[Nr-1])+1)
            CM_0 = dt*k_p_1[Nr-1]/(2*rm[Nr-1]*dr*rho*c_p_1[Nr-1]) + dt/(2*dr**2*rho*c_p_1[Nr-1])*(k_p_1[Nr-1-1]+k_p_1[Nr-1])
            DM_0 = dt/(2*dz**2*rho*c_p_1[Nr-1])*(k_p_1[Nr-1]+k_p_1[Nr-1])
            EM_0 = dt/(2*dz**2*rho*c_p_1[Nr-1])*(k_p_1[Nr-1]+k_p_1[Nr-1+Nr])

            Jac[Nr-1,Nr-1-1] = AM_0 + CM_0
            Jac[Nr-1,Nr-1] = BM_0 - 8*dz*Cm_LB[Nr-1]*sigma_sb*DM_0/(k_p_1[Nr-1]*2*np.pi*rm[Nr-1]*dr)*T[p+1,Nr-1]**3 - 8*dr*sigma_sb*emissivity_front/k_p_1[Nr-1]*CM_0*T[p+1,Nr-1]**3
            Jac[Nr-1,2*Nr-1] = DM_0 + EM_0

            F_matrix[Nr-1] = T[p,Nr-1] + T[p+1,Nr-1-1]*(AM_0 + CM_0) + T[p+1,Nr-1]*BM_0 - 2*dr*sigma_sb*emissivity_front/k_p_1[Nr-1]*CM_0*T[p+1,Nr-1]**4 -2*dz*Cm_LB[Nr-1]*sigma_sb/(
                    k_p_1[Nr-1]*rm[Nr-1]*dr*2*np.pi)*DM_0*T[p+1,Nr-1]**4 + T[p+1,2*Nr-1]*(EM_0+DM_0) + 2*dz/(k_p_1[Nr-1]*rm[Nr-1]*dr*2*np.pi)*qm_LB[Nr-1] + 2*dz/k_p_1[Nr-1]*q_solar[p,Nr-1]

            #left center edge--consistent with derivation
            for n in np.arange(1,Nz-1,1):
                A0_n = dt/(2*dr**2*rho*c_p_1[Nr*n])*(k_p_1[Nr*n]+k_p_1[Nr*n]) - dt/(2*dr*dr*rho*c_p_1[Nr*n])*k_p_1[Nr*n]
                B0_n = -(dt/(2*dr**2*rho*c_p_1[Nr*n])*(k_p_1[Nr*n+1]+2*k_p_1[Nr*n]+k_p_1[Nr*n])+dt/(2*dz**2*rho*c_p_1[Nr*n])*(k_p_1[Nr*(n-1)]+2*k_p_1[Nr*n]+k_p_1[Nr*(n+1)])+1)
                C0_n = dt*k_p_1[Nr*n]/(2*dr*dr*rho*c_p_1[Nr*n]) + dt/(2*dr**2*rho*c_p_1[Nr*n])*(k_p_1[Nr*n]+k_p_1[Nr*n])
                D0_n = dt/(2*dz**2*rho*c_p_1[Nr*n])*(k_p_1[Nr*(n-1)]+k_p_1[Nr*n])
                E0_n = dt/(2*dz**2*rho*c_p_1[Nr*n])*(k_p_1[Nr*n]+k_p_1[Nr*(n+1)])

                Jac[Nr*n,Nr*n] = B0_n
                Jac[Nr*n,Nr*n+1] = A0_n + C0_n
                Jac[Nr*n,Nr*(n-1)] = D0_n
                Jac[Nr*n,Nr*(n+1)] = E0_n

                F_matrix[Nr*n] = T[p,Nr*n] + T[p+1,Nr*n]*B0_n + T[p+1,Nr*n+1]*(C0_n+A0_n) + T[p+1,Nr*(n-1)]*D0_n + T[p+1,Nr*(n+1)]*E0_n

            #considering the body area -- consistent with derivation
            for n in np.arange(1,Nz-1,1): # row, indicating n in thickness direction
                Am_n = dt / (2 * dr ** 2 * rho * c_p_1[1 + Nr * n:Nr-1 + Nr * n]) * (k_p_1[1-1 + Nr * n:Nr-1 -1+ Nr * n] + k_p_1[1 + Nr * n:Nr-1 + Nr * n]) - dt / (
                                2 * rm[1:Nr-1] * dr * rho * c_p_1[1 + Nr * n:Nr-1 + Nr * n]) * k_p_1[1 + Nr * n:Nr-1 + Nr * n]

                Bm_n = -(dt / (2 * dr ** 2 * rho * c_p_1[1 + Nr * n:Nr-1 + Nr * n]) * (
                                k_p_1[2 + Nr * n:Nr + Nr * n] + 2 * k_p_1[1 + Nr * n:Nr-1 + Nr * n] + k_p_1[Nr * n:Nr-2 + Nr * n]) + dt / (
                                         2 * dz ** 2 * rho * c_p_1[1 + Nr * n:Nr-1 + Nr * n]) * (
                                         k_p_1[1+Nr * (n+1):Nr-1+Nr * (n+1)] + 2 * k_p_1[1 + Nr * n:Nr-1 + Nr * n] + k_p_1[1+Nr * (n-1):Nr-1+Nr * (n-1)]) + 1)

                Cm_n = dt * k_p_1[1 + Nr * n:Nr-1 + Nr * n] / (2 * rm[1:Nr-1] * dr * rho * c_p_1[1 + Nr * n:Nr-1 + Nr * n]) + dt / (
                                2 * dr ** 2 * rho * c_p_1[1 + Nr * n:Nr-1 + Nr * n]) * (k_p_1[Nr * n:Nr-2 + Nr * n] + k_p_1[1 + Nr * n:Nr-1 + Nr * n])
                Dm_n = dt / (2 * dz ** 2 * rho * c_p_1[1 + Nr * n:Nr-1 + Nr * n]) * (k_p_1[1+Nr * (n-1):Nr-1+Nr * (n-1)] + k_p_1[1 + Nr * n:Nr-1 + Nr * n])
                Em_n = dt / (2 * dz ** 2 * rho * c_p_1[1 + Nr * n:Nr-1 + Nr * n]) * (k_p_1[1 + Nr * n:Nr-1 + Nr * n] + k_p_1[1+Nr * (n+1):Nr-1+Nr * (n+1)])

                for m in np.arange(1,Nr-1,1):
                    Jac[Nr * n+m, Nr * n+m-1] = Am_n[m-1]
                    Jac[Nr * n+m, Nr * n+m] = Bm_n[m-1]
                    Jac[Nr * n+m, Nr * n+m+1] = Cm_n[m-1]
                    Jac[Nr * n+m, Nr * (n-1)+m] = Dm_n[m-1]
                    Jac[Nr * n + m, Nr * (n + 1) + m] = Em_n[m-1]

                F_matrix[1+Nr * n:Nr-1+Nr * n] = T[p,1+Nr * n:Nr-1+Nr * n] + T[p+1,Nr * n:Nr-2+Nr * n]*Am_n + T[p+1,1+Nr * n:Nr-1+Nr * n]*Bm_n + \
                                                 T[p+1,2+Nr * n:Nr+Nr * n]*Cm_n + T[p+1,1+Nr * (n-1):Nr-1+Nr * (n-1)]*Dm_n + T[p+1,1+Nr * (n+1):Nr-1+Nr * (n+1)]*Em_n

            #right edge--consistent with derivation
            for n in np.arange(1,Nz-1,1):
                AM_n = dt / (2 * dr ** 2 * rho * c_p_1[Nr * n+Nr-1]) * (k_p_1[Nr * n+Nr-1-1] + k_p_1[Nr * n+Nr-1]) - dt / (
                                2 * rm[Nr-1] * dr * rho * c_p_1[Nr * n+Nr-1]) * k_p_1[Nr * n+Nr-1]
                BM_n = -(dt / (2 * dr ** 2 * rho * c_p_1[Nr * n+Nr-1]) * (
                                k_p_1[Nr * n+Nr-1] + 2 * k_p_1[Nr * n+Nr-1] + k_p_1[Nr * n+Nr-1-1]) + dt / (
                                         2 * dz ** 2 * rho * c_p_1[Nr * n+Nr-1]) * (k_p_1[Nr * (n-1)+Nr-1] + 2 * k_p_1[Nr * n+Nr-1] + k_p_1[Nr * (n+1)+Nr-1]) + 1)
                CM_n = dt * k_p_1[Nr * n + Nr-1] / (2 * rm[Nr-1] * dr * rho * c_p_1[Nr * n + Nr-1]) + dt / (
                        2 * dr ** 2 * rho * c_p_1[Nr * n + Nr-1]) * (k_p_1[Nr * n + Nr-1 - 1] + k_p_1[Nr * n + Nr-1])
                DM_n = dt / (2 * dz ** 2 * rho * c_p_1[Nr * n+Nr-1]) * (k_p_1[Nr * (n-1)+Nr-1] + k_p_1[Nr * n+Nr-1])
                EM_n = dt / (2 * dz ** 2 * rho * c_p_1[Nr * n+Nr-1]) * (k_p_1[Nr * n+Nr-1] + k_p_1[Nr * (n+1)+Nr-1])

                Jac[Nr * n+Nr-1,Nr * n+Nr-1-1] = AM_n + CM_n
                Jac[Nr * n+Nr-1,Nr * n+Nr-1] = BM_n - 8*sigma_sb*dr*emissivity_front*CM_n/k_p_1[Nr * n+Nr-1]*T[p+1,Nr * n+Nr-1]**3
                Jac[Nr * n+Nr-1,Nr * (n-1)+Nr-1] = DM_n
                Jac[Nr * n+Nr-1,Nr * (n+1)+Nr-1] = Em_n

                F_matrix[Nr * n+Nr-1] = T[p,Nr * n+Nr-1] + T[p+1,Nr * n+Nr-1-1]*(AM_n+CM_n) + T[p+1,Nr * n+Nr-1]*BM_n - 2*sigma_sb*dr*emissivity_front*CM_n/k_p_1[Nr * n+Nr-1]*T[p+1,Nr * n+Nr-1]**4 + \
                    T[p+1,Nr * (n-1)+Nr-1]*DM_n + T[p+1,Nr * (n+1)+Nr-1]*EM_n + 2*sigma_sb*dr*emissivity_front*CM_n/k_p_1[Nr * n+Nr-1]*T_LBW**4


            #bottom left corner-consistent with derivation
            A0_N = dt/(2*dr**2*rho*c_p_1[Nr*(Nz-1)])*(k_p_1[Nr*(Nz-1)]+k_p_1[Nr*(Nz-1)]) - dt/(2*dr*dr*rho*c_p_1[Nr*(Nz-1)])*k_p_1[Nr*(Nz-1)]
            B0_N = -(dt/(2*dr**2*rho*c_p_1[Nr*(Nz-1)])*(k_p_1[Nr*(Nz-1)+1]+2*k_p_1[Nr*(Nz-1)]+k_p_1[Nr*(Nz-1)])+dt/(2*dz**2*rho*c_p_1[Nr*(Nz-1)])*(k_p_1[Nr*(Nz-1)]+2*k_p_1[Nr*(Nz-1)]+k_p_1[Nr*(Nz-1-1)])+1)
            C0_N = dt*k_p_1[Nr*(Nz-1)]/(2*dr*dr*rho*c_p_1[Nr*(Nz-1)]) + dt/(2*dr**2*rho*c_p_1[Nr*(Nz-1)])*(k_p_1[Nr*(Nz-1)-1]+k_p_1[Nr*(Nz-1)])
            D0_N = dt/(2*dz**2*rho*c_p_1[Nr*(Nz-1)])*(k_p_1[Nr*(Nz-1-1)]+k_p_1[Nr*(Nz-1)])
            E0_N = dt/(2*dz**2*rho*c_p_1[Nr*(Nz-1)])*(k_p_1[Nr*(Nz-1)]+k_p_1[Nr*(Nz-1)])

            Jac[Nr*(Nz-1),Nr*(Nz-1)] = B0_N - 8*dz*sigma_sb*emissivity_back*E0_N/k_p_1[Nr*(Nz-1)]*T[p+1,Nr*(Nz-1)]**3
            Jac[Nr*(Nz-1),Nr*(Nz-1)+1] = A0_N + C0_N
            Jac[Nr*(Nz-1),Nr*(Nz-1-1)] = D0_N + E0_N

            F_matrix[Nr*(Nz-1)] = T[p,Nr*(Nz-1)] + T[p+1,Nr*(Nz-1)+1]*A0_N + T[p+1,Nr*(Nz-1)]*B0_N -2*dz*sigma_sb*emissivity_back*E0_N/k_p_1[Nr*(Nz-1)]*T[p+1,Nr*(Nz-1)]**4 + \
                T[p+1,Nr*(Nz-1)+1]*C0_N + T[p+1,Nr*(Nz-1-1)]*(D0_N + E0_N) + 2*dz*Cm_back[0]*E0_N/(k_p_1[Nr*(Nz-1)]*dr*dr*np.pi*2)

            #bottom center region - consistent with derivation
            #for i in np.arange(1,Nr-1,1):
            Am_N = dt/(2*dr**2*rho*c_p_1[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr])*(k_p_1[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr]+k_p_1[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr]) - dt/(2*rm[1:Nr-1]*dr*rho*c_p_1[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr])*k_p_1[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr]

            T_p_ghost = T[p+1,1+(Nz-1-1)*Nr:Nr-1+(Nz-1-1)*Nr] - 2*dz*emissivity_back*sigma_sb/k_p_1[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr]*T[p+1,1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr]**4 + 2*dz/(rm[1:Nr]*dr*np.pi*2*k_p_1[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr])*Cm_back[1:Nr]
            k_p_1_ghost = rho*(cp_0 + cp_1 * T_p_ghost + cp_2 / T_p_ghost + cp_3 / T_p_ghost ** 2)*(1 / (alpha_r_A * T_p_ghost + alpha_r_B))

            Bm_N = -(dt/(2*dr**2*rho*c_p_1[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr])*(k_p_1[2+(Nz-1)*Nr:Nr+(Nz-1)*Nr]+2*k_p_1[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr]+k_p_1[(Nz-1)*Nr:Nr-2+(Nz-1)*Nr])+dt/(2*dz**2*rho*c_p_1[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr])*(k_p_1_ghost+2*k_p_1[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr]+k_p_1[1+(Nz-1-1)*Nr:Nr-1+(Nz-1-1)*Nr])+1)
            Cm_N = dt * k_p_1[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr] / (2 * rm[1:Nr-1] * dr * rho * c_p_1[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr]) + dt / (2 * dr ** 2 * rho * c_p_1[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr]) * (k_p_1[(Nz-1)*Nr:Nr-2+(Nz-1)*Nr] + k_p_1[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr])
            Dm_N = dt / (2 * dz ** 2 * rho * c_p_1[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr]) * (k_p_1[1+(Nz-1-1)*Nr:Nr-1+(Nz-1-1)*Nr] + k_p_1[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr])
            Em_N = dt/(2*dz**2*rho*c_p_1[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr])*(k_p_1[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr]+k_p_1_ghost)

            for m in np.arange(1,Nr-1,1):
                Jac[m + (Nz-1)*Nr, m + (Nz-1)*Nr-1] = Am_N[m-1]
                Jac[m + (Nz-1)*Nr, m + (Nz-1)*Nr] = Bm_N[m-1] - 8*dz*sigma_sb*emissivity_back*Em_N[m-1]/k_p_1[m + (Nz-1)*Nr]*T[p+1,m + (Nz-1)*Nr]**3
                Jac[m + (Nz-1)*Nr, m + (Nz-1)*Nr+1] = Cm_N[m-1]
                Jac[m + (Nz-1)*Nr, m + (Nz-1-1)*Nr] = Dm_N[m-1] + Em_N[m-1]

            F_matrix[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr] = T[p,1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr] + T[p+1,(Nz-1)*Nr:Nr-2+(Nz-1)*Nr]*Am_N + T[p+1,1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr]*Bm_N - 2*dz*emissivity_back*sigma_sb/k_p_1[1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr]*Em_N*T[p+1,1+(Nz-1)*Nr:Nr-1+(Nz-1)*Nr]**4 + T[p+1, 2+(Nz-1)*Nr:Nr+(Nz-1)*Nr]*Cm_N +\
                    T[p+1,1+(Nz-1-1)*Nr:Nr-1+(Nz-1-1)*Nr]*(Dm_N + Em_N) + 2*dz*Cm_back[1:Nr-1]*Em_N/(rm[1:Nr-1]*dr*2*np.pi*k_p_1[1+(Nz-1-1)*Nr:Nr-1+(Nz-1-1)*Nr])

            #bottom right corner
            AM_N = dt/(2*dr**2*rho*c_p_1[Nr*Nz-1])*(k_p_1[Nr*Nz-1-1]+k_p_1[Nr*Nz-1]) - dt/(2*rm[Nr-1]*dr*rho*c_p_1[Nr*Nz-1])*k_p_1[Nr*Nz-1]
            BM_N = -(dt/(2*dr**2*rho*c_p_1[Nr*Nz-1])*(k_p_1[Nr*Nz-1]+2*k_p_1[Nr*Nz-1]+k_p_1[Nr*Nz-1-1])+dt/(2*dz**2*rho*c_p_1[Nr*Nz-1])*(k_p_1[Nr*Nz-1]+2*k_p_1[Nr*Nz-1]+k_p_1[Nr*(Nz-1)-1])+1)
            CM_N = dt*k_p_1[Nr*Nz-1]/(2*rm[Nr-1]*dr*rho*c_p_1[Nr*Nz-1]) + dt/(2*dr**2*rho*c_p_1[Nr*Nz-1])*(k_p_1[Nr*Nz-1-1]+k_p_1[Nr*Nz-1])
            DM_N = dt/(2*dz**2*rho*c_p_1[Nr*Nz-1])*(k_p_1[Nr*(Nz-1)-1]+k_p_1[Nr*Nz-1])
            EM_N = dt/(2*dz**2*rho*c_p_1[Nr*Nz-1])*(k_p_1[Nr*Nz-1]+k_p_1[Nr*Nz-1])

            Jac[Nr*Nz-1,Nr*Nz-1-1] = AM_N + CM_N
            Jac[Nr*Nz-1,Nr*Nz-1] = BM_N - 8*dr*emissivity_back*sigma_sb*CM_N/k_p_1[Nr*Nz-1]*T[p+1,Nr*Nz-1]**3 -8*dz*emissivity_back*sigma_sb*EM_N/k_p_1[Nr*Nz-1]*T[p+1,Nr*Nz-1]**3
            Jac[Nr*Nz-1,Nr*(Nz-1)-1] = DM_N + EM_N

            F_matrix[Nr*Nz-1] = T[p,Nr*Nz-1] + T[p+1,Nr*Nz-1-1]*(AM_N+CM_N) + T[p+1,Nr*Nz-1]*BM_N - 2*dr*emissivity_back*sigma_sb*CM_N/k_p_1[Nr*Nz-1]*T[p+1,Nr*Nz-1]**4 -2*dz*emissivity_back*sigma_sb*EM_N/k_p_1[Nr*Nz-1]*T[p+1,Nr*Nz-1]**4 +\
                T[p+1,Nr*(Nz-1)-1]*(DM_N + EM_N) + 2*Cm_back[Nr-1]*dz/(rm[Nr-1]*2*np.pi*dr*k_p_1[Nr*Nz-1])*EM_N

            # T[p+1,-2] = -(AN+CN)/BN*T[p+1,-3]-(DN - 2*CN*dr*sigma_sb*emissivity_back/k_p_1[-2])/BN*T[p+1,-2]**4 - EN/BN - FN/BN - 2*dr*sigma_sb*absorptivity_back*T_LBW**4*CN/BN/k_p_1[-2] + 1/BN*T[p,-2]

            # delta_T = -np.dot(inv(Jac),F_matrix)

            # if p==0:
            #     df_Jac = pd.DataFrame(data = Jac)
            #     df_Jac.to_csv('Jacobian_diagnostic.csv')
            delta_T = np.linalg.solve(Jac, -F_matrix)
            T[p + 1, :] = T[p + 1, :] + delta_T

            # T[p + 1, :-1] = T[p + 1, :-1] + delta_T
            # T[p + 1, -1] = 2 * dr * sigma_sb / k_p_1[-2] * (
            #             absorptivity_back * T_LBW ** 4 - emissivity_back * T[p + 1, -2] ** 4) + T[p + 1, -3]

            err = np.max(abs(delta_T))

            if err < 1e-11:
                # if p == 10:
                #     print("stable time reach at iter = {:.0f}, with error = {:.2E} and first node temperature = {:.1E}".format(iter, err, T[p + 1, 0]))
                break
            if iter == Max_iter - 1:
                print('Convergence has not yet achieved! Increase iterations! Tempearture = {:.1E}'.format(T[p + 1, 1]))

        if (p > 0) & (p % N_one_cycle == 0) & (simulated_amp_phase_extraction_method != 'fft'):
            A_max = np.max(T[p - N_one_cycle:p, (Nz-1)*Nr:], axis=0)
            A_min = np.min(T[p - N_one_cycle:p, (Nz-1)*Nr:], axis=0)
            if np.max(np.abs((T_temp[:] - T[p+1, (Nz-1)*Nr:]) / (A_max - A_min))) < 1e-3:
                N_steady_count += 1
                if N_steady_count == N_stable_cycle_output:  # only need 2 periods to calculate amplitude and phase
                    time_index = p
                    print("stable temperature profile has been obtained @ iteration N= {}!".format(
                        int(p / N_one_cycle)))
                    break

            T_temp[:] = T[p+1, (Nz-1)*Nr:]

        if (p == Nt - 2) and (N_steady_count < N_stable_cycle_output):
            time_index = p
            print("Error! No stable temperature profile was obtained!")

    print(
        'sigma_s = {:.2E}, Amax = {}, f_heating = {}, focal plane = {}, Rec = {}, T_LB_mean = {:.1f}, alpha_r_A = {:.2e}, alpha_r_B = {:.2e}.'.format(
            light_source_property['sigma_s'], light_source_property['Amax'], f_heating,
            vacuum_chamber_setting['focal_shift'],
            sample_information['rec_name'], T_LB_mean_C,alpha_r_A,alpha_r_B))


    return T[:time_index,:], time_simulation[:time_index], r, N_one_cycle, q_solar[:time_index,:]


def simulation_result_amplitude_phase_extraction(sample_information,
                                                 vacuum_chamber_setting, solar_simulator_settings,
                                                 light_source_property, numerical_simulation_setting, code_directory,
                                                 df_solar_simulator_VQ, sigma_df, df_view_factor, df_LB_details_all):
    R0 = vacuum_chamber_setting['R0_node']
    R_analysis = vacuum_chamber_setting['R_analysis_node']
    simulated_amp_phase_extraction_method = numerical_simulation_setting['simulated_amp_phase_extraction_method']
    N_stable_cycle_output = numerical_simulation_setting['N_stable_cycle_output']
    N_div = numerical_simulation_setting['N_div']
    # print(sample_information)
    T_temp, time_T_, r_, N_one_cycle, q_solar = finite_difference_implicit_variable_properties(sample_information,vacuum_chamber_setting,solar_simulator_settings,
                                                                                               light_source_property,numerical_simulation_setting,df_solar_simulator_VQ,sigma_df, code_directory,
                                                                                               df_view_factor,df_LB_details_all)

    f_heating = solar_simulator_settings['f_heating']
    gap = numerical_simulation_setting['gap_node']

    # I want max 50 samples per period
    N_skip_time = max(int(N_one_cycle / numerical_simulation_setting['N_sample_to_keep_each_cycle']),
                      1)  # avoid N_skip to be zero

    if numerical_simulation_setting['axial_conduction'] == True:
        T_ = T_temp[:, :, -1]  # only take temperature profiles at the last node facing IR
    else:
        T_ = T_temp  # 1D heat conduction keep the temperature profile as is

    df_temperature_simulation_steady_oscillation_ = pd.DataFrame(
        data=T_[-N_stable_cycle_output * N_one_cycle::N_skip_time,:])  # return a dataframe containing radial averaged temperature and relative time
    df_temperature_simulation_steady_oscillation_['reltime'] = time_T_[-N_stable_cycle_output * N_one_cycle::N_skip_time]

    df_temperature_transient_ = pd.DataFrame(data=T_[::N_skip_time, :])
    df_temperature_transient_['reltime'] = time_T_[::N_skip_time]

    df_solar_simulator_heat_flux_ = pd.DataFrame(data=q_solar[-N_stable_cycle_output * N_one_cycle::N_skip_time, :])
    df_solar_simulator_heat_flux_['reltime'] = time_T_[-N_stable_cycle_output * N_one_cycle::N_skip_time]
    df_amp_phase_simulated_ = batch_process_horizontal_lines(df_temperature_simulation_steady_oscillation_, f_heating, R0,
                                                            gap, R_analysis,
                                                            simulated_amp_phase_extraction_method)  # The default frequency analysis for simulated temperature profile is sine



    # convert node back to pixels
    r_pixel = np.array(df_amp_phase_simulated_[N_div-1::N_div]['r'])/N_div
    amp_ratio = df_amp_phase_simulated_[N_div-1::N_div]['amp_ratio']
    phase_diff = df_amp_phase_simulated_[N_div-1::N_div]['phase_diff']
    r_ref = np.array(df_amp_phase_simulated_[N_div-1::N_div]['r_ref'])/N_div

    df_amp_phase_simulated = pd.DataFrame(data = {'r_pixels':r_pixel,'r_ref_pixels':r_ref,'amp_ratio':amp_ratio,'phase_diff':phase_diff}) # This outputs pixels

    df_temperature_simulation_steady_oscillation = pd.DataFrame(data =T_[-N_stable_cycle_output * N_one_cycle::N_skip_time,::N_div])
    df_temperature_simulation_steady_oscillation['reltime'] = time_T_[-N_stable_cycle_output * N_one_cycle::N_skip_time]

    df_temperature_transient = pd.DataFrame(data = T_[::N_skip_time, ::N_div])
    df_temperature_transient['reltime'] = time_T_[::N_skip_time]

    df_solar_simulator_heat_flux = pd.DataFrame(data=q_solar[-N_stable_cycle_output * N_one_cycle::N_skip_time, ::N_div])
    df_solar_simulator_heat_flux['reltime'] = time_T_[-N_stable_cycle_output * N_one_cycle::N_skip_time]

    return df_amp_phase_simulated, df_temperature_simulation_steady_oscillation, df_solar_simulator_heat_flux, df_temperature_transient



def simulation_result_amplitude_phase_extraction_old(sample_information,
                                                 vacuum_chamber_setting, solar_simulator_settings,
                                                 light_source_property, numerical_simulation_setting,code_directory,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all):
    R0 = vacuum_chamber_setting['R0']
    R_analysis = vacuum_chamber_setting['R_analysis']
    simulated_amp_phase_extraction_method = numerical_simulation_setting['simulated_amp_phase_extraction_method']
    N_stable_cycle_output = numerical_simulation_setting['N_stable_cycle_output']

    #print(sample_information)
    T_temp, time_T_, r_,N_one_cycle, q_solar = finite_difference_implicit_variable_properties(sample_information, vacuum_chamber_setting, solar_simulator_settings,
                       light_source_property, numerical_simulation_setting,df_solar_simulator_VQ, sigma_df,code_directory,df_view_factor,df_LB_details_all)

    f_heating = solar_simulator_settings['f_heating']
    gap = numerical_simulation_setting['gap']


    # I want max 50 samples per period
    N_skip_time = max(int(N_one_cycle /numerical_simulation_setting['N_sample_to_keep_each_cycle']),1) # avoid N_skip to be zero

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
    plt.scatter(df_amplitude_phase_measurement['r_pixels'], df_amplitude_phase_measurement['amp_ratio'], facecolors='none',
                edgecolors='r', label='measurement results')
    plt.scatter(df_amp_phase_simulated['r_pixels'], df_amp_phase_simulated['amp_ratio'], marker='+',
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
    N_inner = int(df_amplitude_phase_measurement['r_pixels'].min())
    N_outer = int(df_amplitude_phase_measurement['r_pixels'].max())
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

    #Nr = numerical_simulation_setting['Nr_node']
    #N_pixels = int((Nr+1)/2)
    #dr = R / (N_pixels-1)

    T_average = np.sum(
        [2 * np.pi * m_ * np.mean(df_temperature_.iloc[:, m_]) for m_ in np.arange(N_inner, N_outer, 1)]) / (
                        ((N_outer) ** 2 - (N_inner) ** 2) * np.pi)
    plt.title('Tmin:{:.0f}K, Tmax:{:.0f}K, Tmean:{:.0f}K'.format(np.mean(df_temperature_[N_outer]),
                                                                 np.mean(df_temperature_[N_inner]), T_average),
              fontsize=11, fontweight='bold')

    plt.tight_layout()

   # plt.show()

    return fig, T_average, amp_residual_mean, phase_residual_mean,total_residual_mean


def result_visulization_one_case(df_exp_condition_i, code_directory, data_directory, df_result_i,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all,df_sample_cp_rho_alpha_all):

    fig = plt.figure(figsize=(17, 16))
    colors = ['red', 'black', 'green', 'blue', 'orange', 'magenta', 'brown', 'yellow', 'purple', 'cornflowerblue']

    x0 = int(df_exp_condition_i['x0_pixels'])
    y0 = int(df_exp_condition_i['y0_pixels'])
    pr = float(df_exp_condition_i['pr'])
    Rmax = int(df_exp_condition_i['Nr_pixels'])

    rec_name = df_exp_condition_i['rec_name']
    simulated_amp_phase_extraction_method = df_exp_condition_i['simulated_amp_phase_extraction_method']
    f_heating = float(df_exp_condition_i['f_heating'])

    R0 = int(df_exp_condition_i['R0_pixels'])
    gap = int(df_exp_condition_i['gap_pixels'])
    R_analysis = int(df_exp_condition_i['R_analysis_pixels'])

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

    T_average = float(df_result_i['T_average(K)'])
    T_min = float(df_result_i['T_min(K)'])



    sample_information = {'R': df_exp_condition_i['sample_radius(m)'], 't_z': df_exp_condition_i['sample_thickness(m)'],
                          'rho': float(df_sample_cp_rho_alpha['rho']),
                          'cp_const': float(df_sample_cp_rho_alpha['cp_const']), 'cp_c1':
                              float(df_sample_cp_rho_alpha['cp_c1']), 'cp_c2': float(df_sample_cp_rho_alpha['cp_c2']),
                          'cp_c3': float(df_sample_cp_rho_alpha['cp_c3']), 'alpha_r': df_result_i['alpha_r'],
                          'alpha_z': float(df_sample_cp_rho_alpha['alpha_z']), 'T_initial': T_average,
                          'emissivity_front': float(df_exp_condition_i['emissivity_front']),
                          'absorptivity_front': float(df_exp_condition_i['absorptivity_front']),
                          'emissivity_back': float(df_exp_condition_i['emissivity_back']),
                          'absorptivity_back': float(df_exp_condition_i['absorptivity_back']),
                          'absorptivity_solar':float(df_exp_condition_i['absorptivity_solar']),
                          'sample_name': sample_name,'T_min':T_min,'alpha_r_A':float(df_sample_cp_rho_alpha['alpha_r_A']),'alpha_r_B':float(df_sample_cp_rho_alpha['alpha_r_B']),'alpha_z_A':float(df_sample_cp_rho_alpha['alpha_z_A']),'alpha_z_B':float(df_sample_cp_rho_alpha['alpha_z_B']),'rec_name': rec_name}

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
                                    'analysis_mode':df_exp_condition_i['analysis_mode'],'N_stable_cycle_output':int(2),'N_sample_to_keep_each_cycle':50,'simulated_num_data_per_cycle':400}



    # the code is executed using vectorized approach by default

    # numerical_simulation_setting
    solar_simulator_settings = {'f_heating': float(df_exp_condition_i['f_heating']),
                                'V_amplitude': float(df_exp_condition_i['V_amplitude']),
                                'V_DC': V_DC}

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


    print(sample_information)

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

    #R = sample_information['R']
    #Nr = numerical_simulation_setting['Nr_node']

    #dr = R / Nr

    T_average = np.sum(
        [2 * np.pi * m_ * np.mean(df_temperature_.iloc[:, m_]) for m_ in np.arange(N_inner, N_outer, 1)]) / (
                        ((N_outer) ** 2 - (N_inner) ** 2) * np.pi)
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


def high_T_Angstrom_execute_one_case_regression(df_exp_condition, data_directory, code_directory, df_amplitude_phase_measurement,df_temperature,df_sample_cp_rho_alpha, df_solar_simulator_VQ,sigma_df,df_view_factor):


    sample_name = df_exp_condition['sample_name']
    #df_sample_cp_rho_alpha = df_sample_cp_rho_alpha_all.query("sample_name=='{}'".format(sample_name))

    # this function read a row from an excel spread sheet and execute
    LB_file_name = df_exp_condition['LB_file_name']
    df_LB_details_all = pd.read_csv(code_directory+"sample specifications//"+ LB_file_name+ ".csv")

    rec_name = df_exp_condition['rec_name']

    N_sample_to_keep_each_cycle = 50

    focal_shift = float(df_exp_condition['focal_shift'])
    VDC = float(df_exp_condition['V_DC'])

    # We need to evaluate the sample's average temperature in the region of analysis, and feed in a thermal diffusivity value from a reference source, this will be important for sigma measurement

    R0 = int(df_exp_condition['R0_pixels'])
    R_analysis  = int(df_exp_condition['R_analysis_pixels'])
    gap = int(df_exp_condition['gap_pixels'])
    emissivity_front = float(df_exp_condition['emissivity_front'])
    emissivity_back = float(df_exp_condition['emissivity_back'])
    absorptivity_front = float(df_exp_condition['absorptivity_front'])
    absorptivity_back = float(df_exp_condition['absorptivity_back'])
    absorptivity_solar = float(df_exp_condition['absorptivity_solar'])



    Nr = int(df_exp_condition['Nr_pixels'])
    N_Rs = int(df_exp_condition['N_Rs_pixels'])
    x0 = int(df_exp_condition['x0_pixels'])
    y0 = int(df_exp_condition['y0_pixels'])
    N_div = int(df_exp_condition['N_div'])

    # alpha_r_A = float(df_sample_cp_rho_alpha['alpha_r_A'])
    # alpha_r_B = float(df_sample_cp_rho_alpha['alpha_r_B'])
    # alpha_z_A = float(df_sample_cp_rho_alpha['alpha_z_A'])
    # alpha_z_B = float(df_sample_cp_rho_alpha['alpha_z_B'])

    N_stable_cycle_output = 2 # by default, we only analyze 2 cycle using sine fitting method

    T_average = np.sum(
        [2 * np.pi *  m_ *  np.mean(df_temperature.iloc[:, m_]) for m_ in np.arange(R0, R_analysis+R0, 1)]) / (
                        ((R_analysis+R0) ** 2 - (R0) ** 2) * np.pi) # unit in K

    sample_information = {'R': float(df_exp_condition['sample_radius(m)']), 't_z': float(df_exp_condition['sample_thickness(m)']),
                          'rho': float(df_sample_cp_rho_alpha['rho']),
                          'cp_const': float(df_sample_cp_rho_alpha['cp_const']), 'cp_c1':
                              float(df_sample_cp_rho_alpha['cp_c1']), 'cp_c2': float(df_sample_cp_rho_alpha['cp_c2']),
                          'cp_c3': float(df_sample_cp_rho_alpha['cp_c3']), 'T_initial': T_average,
                          'emissivity_front': emissivity_front,
                          'absorptivity_front': absorptivity_front,
                          'emissivity_back': emissivity_back,
                          'absorptivity_back': absorptivity_back,
                          'absorptivity_solar': absorptivity_solar,
                          'sample_name':sample_name,'rec_name': rec_name,'T_bias':0}

    # sample_information
    # Note that T_sur1 is read in degree C, must be converted to K.
    # Indicate where light_blocker is used or not, option here: True, False

    vacuum_chamber_setting = {'N_Rs_node': N_div*N_Rs, 'R0_node': N_div*R0, 'focal_shift':focal_shift,'R_analysis_node':N_div*R_analysis-2*N_div+2,'light_blocker':df_exp_condition['light_blocker']}
    # vacuum_chamber_setting

    numerical_simulation_setting = {'Nr_node': N_div*Nr-N_div+1,
                                    'N_cycle': int(df_exp_condition['N_cycle']),
                                    'N_div':N_div,
                                    'simulated_amp_phase_extraction_method': df_exp_condition['simulated_amp_phase_extraction_method'],
                                    'gap_node': gap,
                                    'regression_method': df_exp_condition['regression_method'],
                                    'regression_parameter': df_exp_condition['regression_parameter'],
                                    'regression_residual_converging_criteria': df_exp_condition[
                                        'regression_residual_converging_criteria'],
                                    'axial_conduction':df_exp_condition['axial_conduction'],
                                    'analysis_mode':df_exp_condition['analysis_mode'],'N_stable_cycle_output':N_stable_cycle_output,'N_sample_to_keep_each_cycle':N_sample_to_keep_each_cycle,'simulated_num_data_per_cycle':400}

    # numerical_simulation_setting
    solar_simulator_settings = {'f_heating': float(df_exp_condition['f_heating']),
                                'V_amplitude': float(df_exp_condition['V_amplitude']),
                                'V_DC': VDC}


    regression_result = None

    if df_exp_condition['regression_parameter'] == 'alpha_r':

        params = Parameters()
        params.add('alpha_r_A', value=float(df_exp_condition['alpha_r_A_initial']), min=0.5,max = 100)
        params.add('alpha_r_B', value=float(df_exp_condition['alpha_r_B_initial']), min=-1000, max = 20000)
        params.add('sigma_s', value=float(df_exp_condition['sigma_s_initial']), min=4e-3,max = 18e-3)
        params.add('T_bias', value=float(df_exp_condition['T_bias_initial']), min=-100,max = 100)


        out = lmfit.minimize(residual_update, params,
                             args=(df_amplitude_phase_measurement,df_temperature, sample_information, vacuum_chamber_setting,
                                   solar_simulator_settings, numerical_simulation_setting,
                                   code_directory, df_solar_simulator_VQ, sigma_df, df_view_factor, df_LB_details_all),
                             xtol=df_exp_condition['regression_residual_converging_criteria'])

        regression_result = np.array([out.params['sigma_s'].value, out.params['alpha_r_A'].value, out.params['alpha_r_B'].value,out.params['T_bias'].value])



    print("recording {} completed.".format(rec_name))
    return regression_result, T_average


def high_T_Angstrom_execute_one_case_show_simulation(df_exp_condition, data_directory, code_directory, df_amplitude_phase_measurement,df_temperature,df_sample_cp_rho_alpha, df_solar_simulator_VQ,sigma_df,df_view_factor):


    sample_name = df_exp_condition['sample_name']
    #df_sample_cp_rho_alpha = df_sample_cp_rho_alpha_all.query("sample_name=='{}'".format(sample_name))

    # this function read a row from an excel spread sheet and execute
    LB_file_name = df_exp_condition['LB_file_name']
    df_LB_details_all = pd.read_csv(code_directory+"sample specifications//"+ LB_file_name+ ".csv")

    rec_name = df_exp_condition['rec_name']

    N_sample_to_keep_each_cycle = 50

    focal_shift = float(df_exp_condition['focal_shift'])
    VDC = float(df_exp_condition['V_DC'])

    # We need to evaluate the sample's average temperature in the region of analysis, and feed in a thermal diffusivity value from a reference source, this will be important for sigma measurement

    R0 = int(df_exp_condition['R0_pixels'])
    R_analysis  = int(df_exp_condition['R_analysis_pixels'])
    gap = int(df_exp_condition['gap_pixels'])
    emissivity_front = float(df_exp_condition['emissivity_front'])
    emissivity_back = float(df_exp_condition['emissivity_back'])
    absorptivity_front = float(df_exp_condition['absorptivity_front'])
    absorptivity_back = float(df_exp_condition['absorptivity_back'])
    absorptivity_solar = float(df_exp_condition['absorptivity_solar'])



    Nr = int(df_exp_condition['Nr_pixels'])
    N_Rs = int(df_exp_condition['N_Rs_pixels'])
    x0 = int(df_exp_condition['x0_pixels'])
    y0 = int(df_exp_condition['y0_pixels'])
    N_div = int(df_exp_condition['N_div'])

    # alpha_r_A = float(df_sample_cp_rho_alpha['alpha_r_A'])
    # alpha_r_B = float(df_sample_cp_rho_alpha['alpha_r_B'])
    # alpha_z_A = float(df_sample_cp_rho_alpha['alpha_z_A'])
    # alpha_z_B = float(df_sample_cp_rho_alpha['alpha_z_B'])

    N_stable_cycle_output = 2 # by default, we only analyze 2 cycle using sine fitting method

    T_average = np.sum(
        [2 * np.pi *  m_ *  np.mean(df_temperature.iloc[:, m_]) for m_ in np.arange(R0, R_analysis+R0, 1)]) / (
                        ((R_analysis+R0) ** 2 - (R0) ** 2) * np.pi) # unit in K


    sample_information = {'R': float(df_exp_condition['sample_radius(m)']), 't_z': float(df_exp_condition['sample_thickness(m)']),
                          'rho': float(df_sample_cp_rho_alpha['rho']),
                          'cp_const': float(df_sample_cp_rho_alpha['cp_const']), 'cp_c1':
                              float(df_sample_cp_rho_alpha['cp_c1']), 'cp_c2': float(df_sample_cp_rho_alpha['cp_c2']),
                          'cp_c3': float(df_sample_cp_rho_alpha['cp_c3']), 'T_initial': T_average,
                          'emissivity_front': emissivity_front,
                          'absorptivity_front': absorptivity_front,
                          'emissivity_back': emissivity_back,
                          'absorptivity_back': absorptivity_back,
                          'absorptivity_solar': absorptivity_solar,
                          'sample_name':sample_name,
                          'rec_name': rec_name,
                          'alpha_r_A':float(df_exp_condition['alpha_r_A']),
                          'alpha_r_B':float(df_exp_condition['alpha_r_B']),'T_bias':float(df_exp_condition['T_bias'])}


    # sample_information
    # Note that T_sur1 is read in degree C, must be converted to K.
    # Indicate where light_blocker is used or not, option here: True, False

    vacuum_chamber_setting = {'N_Rs_node': N_div*N_Rs, 'R0_node': N_div*R0, 'focal_shift':focal_shift,'R_analysis_node':N_div*R_analysis-2*N_div+2,'light_blocker':df_exp_condition['light_blocker']}
    # vacuum_chamber_setting

    numerical_simulation_setting = {'Nr_node': N_div*Nr-N_div+1,
                                    'N_div':N_div,
                                    'N_cycle': int(df_exp_condition['N_cycle']),
                                    'simulated_amp_phase_extraction_method': df_exp_condition['simulated_amp_phase_extraction_method'],
                                    'gap_node': gap*N_div,
                                    'regression_method': df_exp_condition['regression_method'],
                                    'regression_parameter': df_exp_condition['regression_parameter'],
                                    'regression_residual_converging_criteria': df_exp_condition[
                                        'regression_residual_converging_criteria'],
                                    'axial_conduction':df_exp_condition['axial_conduction'],
                                    'analysis_mode':df_exp_condition['analysis_mode'],'N_stable_cycle_output':N_stable_cycle_output,'N_sample_to_keep_each_cycle':N_sample_to_keep_each_cycle,'simulated_num_data_per_cycle':400}

    # numerical_simulation_setting
    solar_simulator_settings = {'f_heating': float(df_exp_condition['f_heating']),
                                'V_amplitude': float(df_exp_condition['V_amplitude']),
                                'V_DC': VDC}

    sigma_s_ = float(df_exp_condition['sigma_s'])


    Amax, sigma_s, kvd, bvd = interpolate_light_source_characteristic(sigma_df, df_solar_simulator_VQ, sigma_s_,
                                                                      numerical_simulation_setting,
                                                                      vacuum_chamber_setting)

    light_source_property = {'Amax':Amax,'sigma_s':sigma_s,'kvd':kvd,'bvd':bvd}

    df_amp_phase_simulated,df_temperature_simulation,df_light_source,df_temperature_transient = simulation_result_amplitude_phase_extraction(sample_information,
                                                 vacuum_chamber_setting, solar_simulator_settings,
                                                 light_source_property, numerical_simulation_setting,code_directory,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all)



    return df_amp_phase_simulated,df_temperature_simulation,df_light_source,df_temperature_transient



def residual_update(params, df_amplitude_phase_measurement, df_temperature_measurement, sample_information, vacuum_chamber_setting,
                    solar_simulator_settings, numerical_simulation_setting, code_directory,
                    df_solar_simulator_VQ, sigma_df, df_view_factor, df_LB_details_all):
    error = None

    regression_method = numerical_simulation_setting['regression_method']

    sigma_s_ = params['sigma_s'].value
    sample_information['alpha_r_A'] = params['alpha_r_A'].value
    sample_information['alpha_r_B'] = params['alpha_r_B'].value
    sample_information['T_bias'] = params['T_bias'].value


    R0 = vacuum_chamber_setting['R0_node']
    R_analysis = vacuum_chamber_setting['R_analysis_node']

    Amax, sigma_s, kvd, bvd = interpolate_light_source_characteristic(sigma_df, df_solar_simulator_VQ, sigma_s_,
                                                                      numerical_simulation_setting,
                                                                      vacuum_chamber_setting)

    light_source_property = {'Amax':Amax,'sigma_s':sigma_s,'kvd':kvd,'bvd':bvd}

    df_amp_phase_simulated,df_temperature_simulation,df_light_source,df_temperature_transient = simulation_result_amplitude_phase_extraction(sample_information,
                                                 vacuum_chamber_setting, solar_simulator_settings,
                                                 light_source_property, numerical_simulation_setting,code_directory,df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all)

    phase_relative_error = np.array([((measure - calculate) / dP)**2 for measure, calculate, dP in
                                     zip(df_amplitude_phase_measurement['phase_diff'],
                                         df_amp_phase_simulated['phase_diff'], df_amplitude_phase_measurement['dP'])])

    amplitude_relative_error = np.array([((measure - calculate) / dA)**2 for measure, calculate, dA in
                                         zip(df_amplitude_phase_measurement['amp_ratio'],
                                             df_amp_phase_simulated['amp_ratio'], df_amplitude_phase_measurement['dA'])])

    temp_relative_error = ((df_temperature_measurement.iloc[:,R0].mean() - df_temperature_simulation.iloc[:,R0].mean())**2 + \
                          (df_temperature_measurement.iloc[:,R0+R_analysis].mean() - df_temperature_simulation.iloc[:,R0+R_analysis].mean())**2)/9

    if regression_method == 1:
        error = amplitude_relative_error + temp_relative_error
    elif regression_method == 2:
        error = phase_relative_error + temp_relative_error
    elif regression_method == 0:
        error = amplitude_relative_error + amplitude_relative_error + temp_relative_error

    return error


def parallel_regression_batch_experimental_results_regression(df_exp_condition_spreadsheet_filename, data_directory,
                                                        num_cores, code_directory, df_sample_cp_rho_alpha,
                                                        df_amplitude_phase_measurement_list, df_temperature_list,
                                                        df_solar_simulator_VQ,sigma_df, df_view_factor):
    # df_exp_condition_spreadsheet = pd.read_excel(code_directory+"batch process information//" + df_exp_condition_spreadsheet_filename)
    df_exp_condition_spreadsheet = pd.read_csv(code_directory + "batch process information//" + df_exp_condition_spreadsheet_filename)

    joblib_output = Parallel(n_jobs=num_cores, verbose=0)(delayed(high_T_Angstrom_execute_one_case_regression)(df_exp_condition_spreadsheet.iloc[i, :], data_directory,
                                                                                                               code_directory, df_amplitude_phase_measurement_list[i], df_temperature_list[i],
                                                                                                               df_sample_cp_rho_alpha, df_solar_simulator_VQ, sigma_df, df_view_factor) for i in tqdm(range(len(df_exp_condition_spreadsheet))))

    return joblib_output


def parallel_regression_batch_experimental_results_show(df_exp_condition_spreadsheet_filename, data_directory,
                                                        num_cores, code_directory, df_sample_cp_rho_alpha,
                                                        df_amplitude_phase_measurement_list, df_temperature_list,
                                                        df_solar_simulator_VQ,sigma_df, df_view_factor):
    # df_exp_condition_spreadsheet = pd.read_excel(code_directory+"batch process information//" + df_exp_condition_spreadsheet_filename)
    df_exp_condition_spreadsheet = pd.read_csv(code_directory + "batch process information//" + df_exp_condition_spreadsheet_filename)

    joblib_output = Parallel(n_jobs=num_cores, verbose=0)(delayed(high_T_Angstrom_execute_one_case_show_simulation)(df_exp_condition_spreadsheet.iloc[i, :], data_directory,
                                                                                                               code_directory, df_amplitude_phase_measurement_list[i], df_temperature_list[i],
                                                                                                               df_sample_cp_rho_alpha, df_solar_simulator_VQ, sigma_df, df_view_factor) for i in tqdm(range(len(df_exp_condition_spreadsheet))))

    return joblib_output


# Now we train the surrogate model
def evaluate_PC_node(node_values,code_directory, sample_information,solar_simulator_settings, numerical_simulation_setting,vacuum_chamber_setting,sigma_df,df_solar_simulator_VQ, df_view_factor,df_LB_details_all):

    sigma_s = node_values[0]
    sample_information['alpha_r_A'] = node_values[1]
    sample_information['alpha_r_B'] = node_values[2]

    Amax, sigma_s, kvd, bvd = interpolate_light_source_characteristic(sigma_df, df_solar_simulator_VQ, sigma_s,
                                                                          numerical_simulation_setting,
                                                                          vacuum_chamber_setting)

    light_source_property = {'Amax': Amax, 'sigma_s': sigma_s, 'kvd': kvd, 'bvd': bvd}

    df_amp_phase_simulated, df_temperature_simulation, df_light_source, df_temperature_transient = simulation_result_amplitude_phase_extraction(sample_information,
                                                                                                                                                vacuum_chamber_setting, solar_simulator_settings,light_source_property, numerical_simulation_setting, code_directory, df_solar_simulator_VQ, sigma_df,df_view_factor, df_LB_details_all)

    return np.array(df_temperature_simulation.iloc[:, :numerical_simulation_setting['Nr_node']]), np.array(df_amp_phase_simulated['amp_ratio']), np.array(df_amp_phase_simulated['phase_diff'])




def evaluate_PC_node_P4(node_values,code_directory, sample_information,solar_simulator_settings, numerical_simulation_setting,vacuum_chamber_setting,sigma_df,df_solar_simulator_VQ, df_view_factor,df_LB_details_all):

    sigma_s = node_values[0]
    sample_information['alpha_r_A'] = node_values[1]
    sample_information['alpha_r_B'] = node_values[2]
    #T_bias = node_values[3]
    sample_information['T_bias'] = node_values[3]

    Amax, sigma_s, kvd, bvd = interpolate_light_source_characteristic(sigma_df, df_solar_simulator_VQ, sigma_s,
                                                                          numerical_simulation_setting,
                                                                          vacuum_chamber_setting)

    light_source_property = {'Amax': Amax, 'sigma_s': sigma_s, 'kvd': kvd, 'bvd': bvd}

    df_amp_phase_simulated, df_temperature_simulation, df_light_source, df_temperature_transient = simulation_result_amplitude_phase_extraction(sample_information,
                                                                                                                                                vacuum_chamber_setting, solar_simulator_settings,light_source_property, numerical_simulation_setting, code_directory, df_solar_simulator_VQ, sigma_df,df_view_factor, df_LB_details_all)

    return np.array(df_temperature_simulation.iloc[:, :numerical_simulation_setting['Nr_node']]), np.array(df_amp_phase_simulated['amp_ratio']), np.array(df_amp_phase_simulated['phase_diff'])



def high_T_Angstrom_execute_one_case_mcmc_train_surrogate(df_exp_condition, code_directory,df_temperature,df_sample_cp_rho_alpha, df_solar_simulator_VQ,sigma_df,df_view_factor,mcmc_other_setting):


    sample_name = df_exp_condition['sample_name']
    LB_file_name = df_exp_condition['LB_file_name']
    df_LB_details_all = pd.read_csv(code_directory+"sample specifications//"+ LB_file_name+ ".csv")
    #df_sample_cp_rho_alpha = df_sample_cp_rho_alpha_all.query("sample_name=='{}'".format(sample_name))

    # this function read a row from an excel spread sheet and execute

    rec_name = df_exp_condition['rec_name']
    N_div = int(df_exp_condition['N_div'])
    #view_factor_setting = df_exp_condition['view_factor_setting']

    #regression_module = df_exp_condition['regression_module']
    N_sample_to_keep_each_cycle = 50

    focal_shift = float(df_exp_condition['focal_shift'])
    VDC = float(df_exp_condition['V_DC'])

    # We need to evaluate the sample's average temperature in the region of analysis, and feed in a thermal diffusivity value from a reference source, this will be important for sigma measurement

    R0 = int(df_exp_condition['R0_pixels'])
    R_analysis  = int(df_exp_condition['R_analysis_pixels'])
    gap = int(df_exp_condition['gap_pixels'])
    emissivity_front = float(df_exp_condition['emissivity_front'])
    emissivity_back = float(df_exp_condition['emissivity_back'])
    absorptivity_front = float(df_exp_condition['absorptivity_front'])
    absorptivity_back = float(df_exp_condition['absorptivity_back'])
    absorptivity_solar = float(df_exp_condition['absorptivity_solar'])

    Nr = int(df_exp_condition['Nr_pixels'])
    N_Rs = int(df_exp_condition['N_Rs_pixels'])
    x0 = int(df_exp_condition['x0_pixels'])
    y0 = int(df_exp_condition['y0_pixels'])


    N_stable_cycle_output = 2 # by default, we only analyze 2 cycle using sine fitting method

    T_average = np.sum(
        [2 * np.pi *  m_ *  np.mean(df_temperature.iloc[:, m_]) for m_ in np.arange(R0, R_analysis+R0, 1)]) / (
                        ((R_analysis+R0) ** 2 - (R0) ** 2) * np.pi) # unit in K

    sample_information = {'R': float(df_exp_condition['sample_radius(m)']), 't_z': float(df_exp_condition['sample_thickness(m)']),
                          'rho': float(df_sample_cp_rho_alpha['rho']),
                          'cp_const': float(df_sample_cp_rho_alpha['cp_const']), 'cp_c1':
                              float(df_sample_cp_rho_alpha['cp_c1']), 'cp_c2': float(df_sample_cp_rho_alpha['cp_c2']),
                          'cp_c3': float(df_sample_cp_rho_alpha['cp_c3']), 'T_initial': T_average,
                          'emissivity_front': emissivity_front,
                          'absorptivity_front': absorptivity_front,
                          'emissivity_back': emissivity_back,
                          'absorptivity_back': absorptivity_back,
                          'absorptivity_solar': absorptivity_solar,
                          'sample_name':sample_name,'rec_name': rec_name,'T_bias':0}

    # sample_information
    # Note that T_sur1 is read in degree C, must be converted to K.
    # Indicate where light_blocker is used or not, option here: True, False

    vacuum_chamber_setting = {'N_Rs_node': N_div*N_Rs, 'R0_node': N_div*R0, 'focal_shift':focal_shift,'R_analysis_node':N_div*R_analysis-2*N_div+2,'light_blocker':df_exp_condition['light_blocker']}
    # vacuum_chamber_setting

    #note that #node = 2*# pixels
    numerical_simulation_setting = {'Nr_node': N_div*Nr-N_div+1,
                                    'N_div':N_div,
                                    'N_cycle': int(df_exp_condition['N_cycle']),
                                    'simulated_amp_phase_extraction_method': df_exp_condition['simulated_amp_phase_extraction_method'],
                                    'gap_node': gap,
                                    'regression_module': df_exp_condition['regression_module'],
                                    'regression_method': df_exp_condition['regression_method'],
                                    'regression_parameter': df_exp_condition['regression_parameter'],
                                    'regression_residual_converging_criteria': df_exp_condition[
                                        'regression_residual_converging_criteria'],
                                    'axial_conduction':df_exp_condition['axial_conduction'],
                                    'analysis_mode':df_exp_condition['analysis_mode'],'N_stable_cycle_output':N_stable_cycle_output,'N_sample_to_keep_each_cycle':N_sample_to_keep_each_cycle,'simulated_num_data_per_cycle':400}

    # numerical_simulation_setting
    solar_simulator_settings = {'f_heating': float(df_exp_condition['f_heating']),
                                'V_amplitude': float(df_exp_condition['V_amplitude']),
                                'V_DC': VDC}

    mcmc_setting = {'alpha_A_prior_range':[float(df_exp_condition['alpha_A_LL']),float(df_exp_condition['alpha_A_UL'])],'alpha_B_prior_range':[float(df_exp_condition['alpha_B_LL']),float(df_exp_condition['alpha_B_UL'])],'sigma_s_prior_range':[float(df_exp_condition['sigma_s_LL']),float(df_exp_condition['sigma_s_UL'])],
                    'step_size':mcmc_other_setting['step_size'],'p_initial':mcmc_other_setting['p_initial'],'N_total_mcmc_samples':mcmc_other_setting['N_total_mcmc_samples'],'PC_order':mcmc_other_setting['PC_order'],'PC_training_core_num':mcmc_other_setting['PC_training_core_num'],
                    'T_regularization':mcmc_other_setting['T_regularization'],'chain_num':mcmc_other_setting['chain_num']}


    dump_file_path_surrogate = code_directory + "surrogate_dump//" + df_exp_condition[
        'rec_name'] + '_R0{:}Ra{:}gap{:}_{:}{:}{:}{:}{:}_NRs{:}ord{:}_x0{:}y0{:}{:}_{:}{:}_{:}{:}_{:}{:}_P3{:}'.format(
        int(R0), int(R_analysis), int(gap), int(emissivity_front * 100), int(emissivity_back * 100),
        int(absorptivity_front * 100), int(absorptivity_back * 100), int(absorptivity_solar * 100),
        int(N_Rs),mcmc_setting['PC_order'],x0,y0,LB_file_name,int(1000*mcmc_setting['sigma_s_prior_range'][0]),
        int(1000*mcmc_setting['sigma_s_prior_range'][1]),int(mcmc_setting['alpha_A_prior_range'][0]),
        int(mcmc_setting['alpha_A_prior_range'][1]),int(mcmc_setting['alpha_B_prior_range'][0]/100),
        int(mcmc_setting['alpha_B_prior_range'][1]/100),N_div)

    if (os.path.isfile(dump_file_path_surrogate)):  # First check if a dump file exist:
        print('Found previous dump file :' + dump_file_path_surrogate)
        #temp_dump = pickle.load(open(dump_file_path, 'rb'))
    else:

        if numerical_simulation_setting['regression_parameter'] == 'sigma_s' and numerical_simulation_setting['analysis_mode'] == 'mcmc':
            pass # This code needs further development


        elif numerical_simulation_setting['regression_parameter'] == 'alpha_r' and (numerical_simulation_setting['analysis_mode'] == 'NUTs' or numerical_simulation_setting['analysis_mode'] == 'RW_Metropolis'):

            # Note alpha = 1/(alpha_A * T + alpha_B)
            alpha_A_LL = mcmc_setting['alpha_A_prior_range'][0] #lower limit for alpha_A
            alpha_A_UL = mcmc_setting['alpha_A_prior_range'][1] #upper limit for alpha_B

            alpha_B_LL = mcmc_setting['alpha_B_prior_range'][0] #lower limit for alpha_A
            alpha_B_UL = mcmc_setting['alpha_B_prior_range'][1] #upper limit for alpha_B


            sigma_s_LL = mcmc_setting['sigma_s_prior_range'][0] #lower limit for alpha_A
            sigma_s_UL = mcmc_setting['sigma_s_prior_range'][1] #upper limit for alpha_B

            cp_p_sigmas = cp.Uniform(sigma_s_LL, sigma_s_UL)
            cp_P_AT = cp.Uniform(alpha_A_LL, alpha_A_UL)
            cp_P_BT = cp.Uniform(alpha_B_LL, alpha_B_UL)
            cp_distribution = cp.J(cp_p_sigmas, cp_P_AT, cp_P_BT)

            #distribution = cp.J(cp_p_alpha_r)
            order = mcmc_setting['PC_order']
            nodes, weights = cp.generate_quadrature(order, cp_distribution, rule='Gaussian')

            num_cores = mcmc_setting['PC_training_core_num']

            # evaluate_PC_node(node_values, code_directory, sample_information, solar_simulator_settings,
            #                  numerical_simulation_setting, vacuum_chamber_setting, sigma_df, df_solar_simulator_VQ,
            #                  df_view_factor, df_LB_details_all)

            joblib_output = Parallel(n_jobs=num_cores, verbose=0)(
                delayed(evaluate_PC_node)(node,code_directory,sample_information,solar_simulator_settings,
                             numerical_simulation_setting, vacuum_chamber_setting, sigma_df, df_solar_simulator_VQ,
                             df_view_factor, df_LB_details_all) for node in
                tqdm(nodes.T))

            amp_ratio_list = []
            phase_diff_list = []
            temp_steady_list = []

            for item in joblib_output:
                temp_steady_list.append(item[0])
                amp_ratio_list.append(item[1])
                phase_diff_list.append(item[2])


            amp_ratio_list = np.array(amp_ratio_list)
            phase_diff_list = np.array(phase_diff_list)
            temp_steady_list = np.array(temp_steady_list)

            polynomials_amp = cp.orth_ttr(order, dist=cp_distribution)
            amp_ratio_approx = cp.fit_quadrature(polynomials_amp, nodes, weights, amp_ratio_list)

            polynomials_phase = cp.orth_ttr(order, dist=cp_distribution)
            phase_diff_approx = cp.fit_quadrature(polynomials_phase, nodes, weights, phase_diff_list)

            polynomials_ss_temp = cp.orth_ttr(order, dist=cp_distribution)
            ss_temp_approx = cp.fit_quadrature(polynomials_ss_temp, nodes, weights, temp_steady_list)


            pickle.dump((amp_ratio_approx, phase_diff_approx, ss_temp_approx), open(dump_file_path_surrogate, "wb"))



def high_T_Angstrom_execute_one_case_mcmc_train_surrogate_P4(df_exp_condition, code_directory,df_temperature,df_sample_cp_rho_alpha, df_solar_simulator_VQ,sigma_df,df_view_factor,mcmc_other_setting):


    sample_name = df_exp_condition['sample_name']
    LB_file_name = df_exp_condition['LB_file_name']
    df_LB_details_all = pd.read_csv(code_directory+"sample specifications//"+ LB_file_name+ ".csv")
    #df_sample_cp_rho_alpha = df_sample_cp_rho_alpha_all.query("sample_name=='{}'".format(sample_name))

    # this function read a row from an excel spread sheet and execute

    rec_name = df_exp_condition['rec_name']
    N_div = int(df_exp_condition['N_div'])
    #view_factor_setting = df_exp_condition['view_factor_setting']

    #regression_module = df_exp_condition['regression_module']
    N_sample_to_keep_each_cycle = 50

    focal_shift = float(df_exp_condition['focal_shift'])
    VDC = float(df_exp_condition['V_DC'])

    # We need to evaluate the sample's average temperature in the region of analysis, and feed in a thermal diffusivity value from a reference source, this will be important for sigma measurement

    R0 = int(df_exp_condition['R0_pixels'])
    R_analysis  = int(df_exp_condition['R_analysis_pixels'])
    gap = int(df_exp_condition['gap_pixels'])
    emissivity_front = float(df_exp_condition['emissivity_front'])
    emissivity_back = float(df_exp_condition['emissivity_back'])
    absorptivity_front = float(df_exp_condition['absorptivity_front'])
    absorptivity_back = float(df_exp_condition['absorptivity_back'])
    absorptivity_solar = float(df_exp_condition['absorptivity_solar'])

    Nr = int(df_exp_condition['Nr_pixels'])
    N_Rs = int(df_exp_condition['N_Rs_pixels'])
    x0 = int(df_exp_condition['x0_pixels'])
    y0 = int(df_exp_condition['y0_pixels'])
    N_div = int(df_exp_condition['N_div'])

    N_stable_cycle_output = 2 # by default, we only analyze 2 cycle using sine fitting method

    T_average = np.sum(
        [2 * np.pi *  m_ *  np.mean(df_temperature.iloc[:, m_]) for m_ in np.arange(R0, R_analysis+R0, 1)]) / (
                        ((R_analysis+R0) ** 2 - (R0) ** 2) * np.pi) # unit in K

    sample_information = {'R': float(df_exp_condition['sample_radius(m)']), 't_z': float(df_exp_condition['sample_thickness(m)']),
                          'rho': float(df_sample_cp_rho_alpha['rho']),
                          'cp_const': float(df_sample_cp_rho_alpha['cp_const']), 'cp_c1':
                              float(df_sample_cp_rho_alpha['cp_c1']), 'cp_c2': float(df_sample_cp_rho_alpha['cp_c2']),
                          'cp_c3': float(df_sample_cp_rho_alpha['cp_c3']), 'T_initial': T_average,
                          'emissivity_front': emissivity_front,
                          'absorptivity_front': absorptivity_front,
                          'emissivity_back': emissivity_back,
                          'absorptivity_back': absorptivity_back,
                          'absorptivity_solar': absorptivity_solar,
                          'sample_name':sample_name,'rec_name': rec_name,'T_bias':0}

    # sample_information
    # Note that T_sur1 is read in degree C, must be converted to K.
    # Indicate where light_blocker is used or not, option here: True, False

    vacuum_chamber_setting = {'N_Rs_node': N_div*N_Rs, 'R0_node': N_div*R0, 'focal_shift':focal_shift,'R_analysis_node':N_div*R_analysis-2*N_div+2,'light_blocker':df_exp_condition['light_blocker']}
    # vacuum_chamber_setting

    #note that #node = 2*# pixels
    numerical_simulation_setting = {'Nr_node': N_div*Nr-N_div+1,
                                    'N_div':N_div,
                                    'N_cycle': int(df_exp_condition['N_cycle']),
                                    'simulated_amp_phase_extraction_method': df_exp_condition['simulated_amp_phase_extraction_method'],
                                    'gap_node': gap,
                                    'regression_module': df_exp_condition['regression_module'],
                                    'regression_method': df_exp_condition['regression_method'],
                                    'regression_parameter': df_exp_condition['regression_parameter'],
                                    'regression_residual_converging_criteria': df_exp_condition[
                                        'regression_residual_converging_criteria'],
                                    'axial_conduction':df_exp_condition['axial_conduction'],
                                    'analysis_mode':df_exp_condition['analysis_mode'],'N_stable_cycle_output':N_stable_cycle_output,'N_sample_to_keep_each_cycle':N_sample_to_keep_each_cycle,'simulated_num_data_per_cycle':400}

    # numerical_simulation_setting
    solar_simulator_settings = {'f_heating': float(df_exp_condition['f_heating']),
                                'V_amplitude': float(df_exp_condition['V_amplitude']),
                                'V_DC': VDC}

    mcmc_setting = {'alpha_A_prior_range':[float(df_exp_condition['alpha_A_LL']),float(df_exp_condition['alpha_A_UL'])],'alpha_B_prior_range':[float(df_exp_condition['alpha_B_LL']),float(df_exp_condition['alpha_B_UL'])],
                    'sigma_s_prior_range':[float(df_exp_condition['sigma_s_LL']),float(df_exp_condition['sigma_s_UL'])],'T_bias_prior_range':[float(df_exp_condition['T_bias_LL']),float(df_exp_condition['T_bias_UL'])],
                    'step_size':mcmc_other_setting['step_size'],'p_initial':mcmc_other_setting['p_initial'],'N_total_mcmc_samples':mcmc_other_setting['N_total_mcmc_samples'],'PC_order':mcmc_other_setting['PC_order'],
                    'PC_training_core_num':mcmc_other_setting['PC_training_core_num'],'T_regularization':mcmc_other_setting['T_regularization'],'chain_num':mcmc_other_setting['chain_num']}



    dump_file_path_surrogate = code_directory + "surrogate_dump//" + df_exp_condition[
        'rec_name'] + '_R0{:}Ra{:}gap{:}_{:}{:}{:}{:}{:}_NRs{:}ord{:}_x0{:}y0{:}{:}_{:}{:}_{:}{:}_{:}{:}_P4D{:}_T{:}{:}'.format(
        int(R0), int(R_analysis), int(gap), int(emissivity_front * 100), int(emissivity_back * 100),
        int(absorptivity_front * 100), int(absorptivity_back * 100), int(absorptivity_solar * 100),
        int(N_Rs),mcmc_setting['PC_order'],x0,y0,LB_file_name,int(1000*mcmc_setting['sigma_s_prior_range'][0]),
        int(1000*mcmc_setting['sigma_s_prior_range'][1]),int(mcmc_setting['alpha_A_prior_range'][0]),
        int(mcmc_setting['alpha_A_prior_range'][1]),int(mcmc_setting['alpha_B_prior_range'][0]/100),
        int(mcmc_setting['alpha_B_prior_range'][1]/100),N_div,int(mcmc_setting['T_bias_prior_range'][0]/10),int(mcmc_setting['T_bias_prior_range'][1]/10))

    if (os.path.isfile(dump_file_path_surrogate)):  # First check if a dump file exist:
        print('Found previous dump file :' + dump_file_path_surrogate)
        #temp_dump = pickle.load(open(dump_file_path, 'rb'))
    else:
        print("Start training surrogate model!")
        if numerical_simulation_setting['regression_parameter'] == 'sigma_s' and numerical_simulation_setting['analysis_mode'] == 'mcmc':
            pass # This code needs further development


        elif numerical_simulation_setting['regression_parameter'] == 'alpha_r' and (numerical_simulation_setting['analysis_mode'] == 'NUTs_P4' or numerical_simulation_setting['analysis_mode'] == 'RW_Metropolis_P4'):

            # Note alpha = 1/(alpha_A * T + alpha_B)
            alpha_A_LL = mcmc_setting['alpha_A_prior_range'][0] #lower limit for alpha_A
            alpha_A_UL = mcmc_setting['alpha_A_prior_range'][1] #upper limit for alpha_B

            alpha_B_LL = mcmc_setting['alpha_B_prior_range'][0] #lower limit for alpha_A
            alpha_B_UL = mcmc_setting['alpha_B_prior_range'][1] #upper limit for alpha_B


            sigma_s_LL = mcmc_setting['sigma_s_prior_range'][0] #lower limit for alpha_A
            sigma_s_UL = mcmc_setting['sigma_s_prior_range'][1] #upper limit for alpha_B

            T_bias_LL = mcmc_setting['T_bias_prior_range'][0] #lower limit for alpha_A
            T_bias_UL = mcmc_setting['T_bias_prior_range'][1]  # upper limit for alpha_A

            cp_p_sigmas = cp.Uniform(sigma_s_LL, sigma_s_UL)
            cp_P_AT = cp.Uniform(alpha_A_LL, alpha_A_UL)
            cp_P_BT = cp.Uniform(alpha_B_LL, alpha_B_UL)
            cp_P_T_bias = cp.Uniform(T_bias_LL,T_bias_UL)
            cp_distribution = cp.J(cp_p_sigmas, cp_P_AT, cp_P_BT,cp_P_T_bias)

            #distribution = cp.J(cp_p_alpha_r)
            order = mcmc_setting['PC_order']
            nodes, weights = cp.generate_quadrature(order, cp_distribution, rule='Gaussian')

            num_cores = mcmc_setting['PC_training_core_num']

            # evaluate_PC_node(node_values, code_directory, sample_information, solar_simulator_settings,
            #                  numerical_simulation_setting, vacuum_chamber_setting, sigma_df, df_solar_simulator_VQ,
            #                  df_view_factor, df_LB_details_all)

            joblib_output = Parallel(n_jobs=num_cores, verbose=0)(
                delayed(evaluate_PC_node_P4)(node,code_directory,sample_information,solar_simulator_settings,
                             numerical_simulation_setting, vacuum_chamber_setting, sigma_df, df_solar_simulator_VQ,
                             df_view_factor, df_LB_details_all) for node in
                tqdm(nodes.T))

            amp_ratio_list = []
            phase_diff_list = []
            temp_steady_list = []

            for item in joblib_output:
                temp_steady_list.append(item[0])
                amp_ratio_list.append(item[1])
                phase_diff_list.append(item[2])


            amp_ratio_list = np.array(amp_ratio_list)
            phase_diff_list = np.array(phase_diff_list)
            temp_steady_list = np.array(temp_steady_list)

            polynomials_amp = cp.orth_ttr(order, dist=cp_distribution)
            amp_ratio_approx = cp.fit_quadrature(polynomials_amp, nodes, weights, amp_ratio_list)

            polynomials_phase = cp.orth_ttr(order, dist=cp_distribution)
            phase_diff_approx = cp.fit_quadrature(polynomials_phase, nodes, weights, phase_diff_list)

            polynomials_ss_temp = cp.orth_ttr(order, dist=cp_distribution)
            ss_temp_approx = cp.fit_quadrature(polynomials_ss_temp, nodes, weights, temp_steady_list)


            pickle.dump((amp_ratio_approx, phase_diff_approx, ss_temp_approx), open(dump_file_path_surrogate, "wb"))





def gradients(vals, func, releps=1e-3, abseps=None, mineps=1e-9, reltol=1e-3,
              epsscale=0.5):
    """
    Calculate the partial derivatives of a function at a set of values. The
    derivatives are calculated using the central difference, using an iterative
    method to check that the values converge as step size decreases.

    Parameters
    ----------
    vals: array_like
        A set of values, that are passed to a function, at which to calculate
        the gradient of that function
    func:
        A function that takes in an array of values.
    releps: float, array_like, 1e-3
        The initial relative step size for calculating the derivative.
    abseps: float, array_like, None
        The initial absolute step size for calculating the derivative.
        This overrides `releps` if set.
        `releps` is set then that is used.
    mineps: float, 1e-9
        The minimum relative step size at which to stop iterations if no
        convergence is achieved.
    epsscale: float, 0.5
        The factor by which releps if scaled in each iteration.

    Returns
    -------
    grads: array_like
        An array of gradients for each non-fixed value.
    """

    grads = np.zeros(len(vals))

    # maximum number of times the gradient can change sign
    flipflopmax = 10.

    # set steps
    if abseps is None:
        if isinstance(releps, float):
            eps = np.abs(vals)*releps
            eps[eps == 0.] = releps  # if any values are zero set eps to releps
            teps = releps*np.ones(len(vals))
        elif isinstance(releps, (list, np.ndarray)):
            if len(releps) != len(vals):
                raise ValueError("Problem with input relative step sizes")
            eps = np.multiply(np.abs(vals), releps)
            eps[eps == 0.] = np.array(releps)[eps == 0.]
            teps = releps
        else:
            raise RuntimeError("Relative step sizes are not a recognised type!")
    else:
        if isinstance(abseps, float):
            eps = abseps*np.ones(len(vals))
        elif isinstance(abseps, (list, np.ndarray)):
            if len(abseps) != len(vals):
                raise ValueError("Problem with input absolute step sizes")
            eps = np.array(abseps)
        else:
            raise RuntimeError("Absolute step sizes are not a recognised type!")
        teps = eps

    # for each value in vals calculate the gradient
    count = 0
    for i in range(len(vals)):
        # initial parameter diffs
        leps = eps[i]
        cureps = teps[i]

        flipflop = 0

        # get central finite difference
        fvals = np.copy(vals)
        bvals = np.copy(vals)

        # central difference
        fvals[i] += 0.5*leps  # change forwards distance to half eps
        bvals[i] -= 0.5*leps  # change backwards distance to half eps
        cdiff = (func(fvals)-func(bvals))/leps

        while 1:
            fvals[i] -= 0.5*leps  # remove old step
            bvals[i] += 0.5*leps

            # change the difference by a factor of two
            cureps *= epsscale
            if cureps < mineps or flipflop > flipflopmax:
                # if no convergence set flat derivative (TODO: check if there is a better thing to do instead)
                print("Derivative calculation did not converge: setting flat derivative.")
                grads[count] = 0.
                break
            leps *= epsscale

            # central difference
            fvals[i] += 0.5*leps  # change forwards distance to half eps
            bvals[i] -= 0.5*leps  # change backwards distance to half eps
            cdiffnew = (func(fvals)-func(bvals))/leps

            if cdiffnew == cdiff:
                grads[count] = cdiff
                break

            # check whether previous diff and current diff are the same within reltol
            rat = (cdiff/cdiffnew)
            if np.isfinite(rat) and rat > 0.:
                # gradient has not changed sign
                if np.abs(1.-rat) < reltol:
                    grads[count] = cdiffnew
                    break
                else:
                    cdiff = cdiffnew
                    continue
            else:
                cdiff = cdiffnew
                flipflop += 1
                continue

        count += 1

    return grads


# define a theano Op for our likelihood function
class LogLikeWithGrad(tt.Op):

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, df_amp_phase_measurement,df_temperature_measurement,T_regularization, vacuum_chamber_setting):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.df_amp_phase_measurement = df_amp_phase_measurement
        self.df_temperature_measurement = df_temperature_measurement
        self.T_regularization = T_regularization
        self.vacuum_chamber_setting = vacuum_chamber_setting

        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.likelihood,self.df_amp_phase_measurement,self.df_temperature_measurement,self.T_regularization,self.vacuum_chamber_setting)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta,self.df_amp_phase_measurement,self.df_temperature_measurement,self.T_regularization,self.vacuum_chamber_setting)

        outputs[0][0] = np.array(logl)  # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs  # our parameters
        return [g[0] * self.logpgrad(theta)]


class LogLikeGrad(tt.Op):

    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """

    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, loglike, df_amp_phase_measurement,df_temperature_measurement,T_regularization,vacuum_chamber_setting):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.df_amp_phase_measurement = df_amp_phase_measurement
        self.df_temperature_measurement = df_temperature_measurement
        self.T_regularization = T_regularization
        self.vacuum_chamber_setting = vacuum_chamber_setting

    def perform(self, node, inputs, outputs):
        (theta,) = inputs

        # define version of likelihood function to pass to derivative function
        def lnlike(values):
            return self.likelihood(values,self.df_amp_phase_measurement,self.df_temperature_measurement, self.T_regularization, self.vacuum_chamber_setting)

        # calculate gradients
        grads = gradients(theta, lnlike)

        outputs[0][0] = grads


def high_T_Angstrom_execute_one_case_NUTs(df_exp_condition, data_directory, code_directory,df_sample_cp_rho_alpha, df_amplitude_phase_measurement,df_temperature,mcmc_other_setting):

    focal_shift = float(df_exp_condition['focal_shift'])
    LB_file_name = df_exp_condition['LB_file_name']
    rec_name = df_exp_condition['rec_name']
    R0 = int(df_exp_condition['R0_pixels'])
    R_analysis  = int(df_exp_condition['R_analysis_pixels'])
    gap = int(df_exp_condition['gap_pixels'])
    emissivity_front = float(df_exp_condition['emissivity_front'])
    emissivity_back = float(df_exp_condition['emissivity_back'])
    absorptivity_front = float(df_exp_condition['absorptivity_front'])
    absorptivity_back = float(df_exp_condition['absorptivity_back'])
    absorptivity_solar = float(df_exp_condition['absorptivity_solar'])

    x0 = int(df_exp_condition['x0_pixels'])
    y0 = int(df_exp_condition['y0_pixels'])
    N_div = int(df_exp_condition['N_div'])
    #alpha_r_A_ref = float(df_sample_cp_rho_alpha['alpha_r_A'])
    #alpha_r_B_ref = float(df_sample_cp_rho_alpha['alpha_r_B'])

    N_Rs = int(df_exp_condition['N_Rs_pixels'])
    mcmc_mode = df_exp_condition['analysis_mode']

    vacuum_chamber_setting = {'N_Rs_pixels': N_Rs, 'R0_pixels': R0,'focal_shift':focal_shift,'R_analysis_pixels':R_analysis,'light_blocker':df_exp_condition['light_blocker']}

    mcmc_setting = {'alpha_A_prior_range':[float(df_exp_condition['alpha_A_LL']),float(df_exp_condition['alpha_A_UL'])],'alpha_B_prior_range':[float(df_exp_condition['alpha_B_LL']),float(df_exp_condition['alpha_B_UL'])],
                    'sigma_s_prior_range':[float(df_exp_condition['sigma_s_LL']),float(df_exp_condition['sigma_s_UL'])],'T_bias_prior_range':[float(df_exp_condition['T_bias_LL']),float(df_exp_condition['T_bias_UL'])],
                    'step_size':mcmc_other_setting['step_size'],'p_initial':mcmc_other_setting['p_initial'],'N_total_mcmc_samples':mcmc_other_setting['N_total_mcmc_samples'],'PC_order':mcmc_other_setting['PC_order'],
                    'PC_training_core_num':mcmc_other_setting['PC_training_core_num'],'T_regularization':mcmc_other_setting['T_regularization'],'chain_num':mcmc_other_setting['chain_num']}


    dump_file_path_surrogate = code_directory + "surrogate_dump//" + df_exp_condition[
        'rec_name'] + '_R0{:}Ra{:}gap{:}_{:}{:}{:}{:}{:}_NRs{:}ord{:}_x0{:}y0{:}{:}_{:}{:}_{:}{:}_{:}{:}_P4D{:}_T{:}{:}'.format(
        int(R0), int(R_analysis), int(gap), int(emissivity_front * 100), int(emissivity_back * 100),
        int(absorptivity_front * 100), int(absorptivity_back * 100), int(absorptivity_solar * 100),
        int(N_Rs),mcmc_setting['PC_order'],x0,y0,LB_file_name,int(1000*mcmc_setting['sigma_s_prior_range'][0]),
        int(1000*mcmc_setting['sigma_s_prior_range'][1]),int(mcmc_setting['alpha_A_prior_range'][0]),
        int(mcmc_setting['alpha_A_prior_range'][1]),int(mcmc_setting['alpha_B_prior_range'][0]/100),
        int(mcmc_setting['alpha_B_prior_range'][1]/100),N_div,int(mcmc_setting['T_bias_prior_range'][0]/10),int(mcmc_setting['T_bias_prior_range'][1]/10))

    amp_ratio_approx, phase_diff_approx, ss_temp_approx = pickle.load(open(dump_file_path_surrogate, 'rb'))

    def physical_model_PC(parameter):

        # print(parameter)
        A = amp_ratio_approx(parameter[0], parameter[1], parameter[2])
        P = phase_diff_approx(parameter[0], parameter[1], parameter[2])
        SS_T = ss_temp_approx(parameter[0], parameter[1], parameter[2])
        return A, P, SS_T


    def log_likelihood_AP(parameter, df_amp_phase_measurement,df_temperature_measurement,T_regularization,vacuum_chamber_setting):
        simulation_A, simulation_P, simulation_SS_T = physical_model_PC(parameter)
        observation_A = np.array(df_amp_phase_measurement['amp_ratio'])
        observation_P = np.array(df_amp_phase_measurement['phase_diff'])
        sigma_noise_A = np.array(df_amp_phase_measurement['dA'])
        sigma_noise_P = np.array(df_amp_phase_measurement['dP'])

        R0 = vacuum_chamber_setting['R0_pixels']
        #R_analysis = vacuum_chamber_setting['R_analysis']

        # p_T = T_regularization*(SS_T[:,R0:R0+R_analysis].mean(axis = 0) - np.array(df_temperature_measurement.iloc[:,R0:R0+R_analysis]))**2

        p_T = T_regularization * np.sum((simulation_SS_T[:, R0:R0 + 1].mean(axis=0) - np.array(
            df_temperature_measurement.iloc[:, R0:R0 + 1].mean(axis=0))) ** 2)

        if df_exp_condition['regression_method'] == 0:
            return -np.sum((simulation_A - observation_A) ** 2 / sigma_noise_A ** 2) - np.sum(
            (simulation_P - observation_P) ** 2 / sigma_noise_P ** 2) - p_T  # calculate the sum of the natural log probability
        elif df_exp_condition['regression_method'] == 1:
            return -np.sum((simulation_A - observation_A) ** 2 / sigma_noise_A ** 2) - p_T
        elif df_exp_condition['regression_method'] == 2:
            return - np.sum((simulation_P - observation_P) ** 2 / sigma_noise_P ** 2) - p_T


    #step = mcmc_setting['step_size']
    parameter_initial = mcmc_setting['p_initial']
    N_total_samples = mcmc_setting['N_total_mcmc_samples']
    T_regularization = mcmc_setting['T_regularization']
    sigma_s_prior_range = mcmc_setting['sigma_s_prior_range']
    alpha_A_prior_range = mcmc_setting['alpha_A_prior_range']
    alpha_B_prior_range = mcmc_setting['alpha_B_prior_range']
    chain_num = mcmc_setting['chain_num']

    dump_file_path_mcmc_results = code_directory + "mcmc_results_dump//" + df_exp_condition[
        'rec_name'] + '_R0{:}{:}{:}_{:}{:}{:}{:}{:}{:}{:}_Tr{:}_{:}{:}_{:}c{:}_{:}{:}{:}{:}{:}{:}{:}_P4D{:}_T{:}{:}'.format(
        int(R0), int(R_analysis), int(gap), int(emissivity_front * 100), int(emissivity_back * 100),
        int(absorptivity_front * 100), int(absorptivity_back * 100), int(absorptivity_solar * 100),
        N_Rs,int(mcmc_setting['N_total_mcmc_samples']),int(mcmc_setting['T_regularization']*1000),
        x0,y0,mcmc_mode,mcmc_setting['chain_num'],LB_file_name,int(1000*mcmc_setting['sigma_s_prior_range'][0]),
        int(1000*mcmc_setting['sigma_s_prior_range'][1]),int(mcmc_setting['alpha_A_prior_range'][0]),
        int(mcmc_setting['alpha_A_prior_range'][1]),int(mcmc_setting['alpha_B_prior_range'][0]/100),
        int(mcmc_setting['alpha_B_prior_range'][1]/100),N_div,int(mcmc_setting['T_bias_prior_range'][0]/10),int(mcmc_setting['T_bias_prior_range'][1]/10))

    if (os.path.isfile(dump_file_path_mcmc_results)):  # First check if a dump file exist:
        print('Found previous dump file for mcmc results:' + dump_file_path_mcmc_results)
        accepted_samples_array = pickle.load(open(dump_file_path_mcmc_results, 'rb'))

    else:

        #ndraws = 500
        nburn = 25
        logl = LogLikeWithGrad(log_likelihood_AP, df_amplitude_phase_measurement, df_temperature, T_regularization, vacuum_chamber_setting)



        with pm.Model():
            # uniform priors on m and c
            sigma_s = pm.Uniform("sigma_s", lower=sigma_s_prior_range[0], upper=sigma_s_prior_range[1])
            alpha_A = pm.Uniform("alpha_A", lower=alpha_A_prior_range[0], upper=alpha_A_prior_range[1])
            alpha_B = pm.Uniform("alpha_B", lower=alpha_B_prior_range[0], upper=alpha_B_prior_range[1])

            theta = tt.as_tensor_variable([sigma_s, alpha_A, alpha_B])

            # use a DensityDist (use a lamdba function to "call" the Op)
            pm.DensityDist("likelihood", lambda v: logl(v), observed={"v": theta})
            # step = pm.Metropolis()
            trace = pm.sample(N_total_samples, tune=nburn, discard_tuned_samples=True, chains=1)

        accepted_samples_array = np.array(
            [trace.get_values('sigma_s'), trace.get_values('alpha_A'), trace.get_values('alpha_B')]).T

    print("recording {} completed.".format(rec_name))

    if (os.path.isfile(dump_file_path_mcmc_results) == False):  # First check if a dump file exist:
        pickle.dump(accepted_samples_array, open(dump_file_path_mcmc_results, "wb"))  # create a dump file



def high_T_Angstrom_execute_one_case_NUTs_P4(df_exp_condition, data_directory, code_directory,df_sample_cp_rho_alpha, df_amplitude_phase_measurement,df_temperature, mcmc_other_setting):

    focal_shift = float(df_exp_condition['focal_shift'])
    LB_file_name = df_exp_condition['LB_file_name']
    rec_name = df_exp_condition['rec_name']
    R0 = int(df_exp_condition['R0_pixels'])
    R_analysis  = int(df_exp_condition['R_analysis_pixels'])
    gap = int(df_exp_condition['gap_pixels'])
    emissivity_front = float(df_exp_condition['emissivity_front'])
    emissivity_back = float(df_exp_condition['emissivity_back'])
    absorptivity_front = float(df_exp_condition['absorptivity_front'])
    absorptivity_back = float(df_exp_condition['absorptivity_back'])
    absorptivity_solar = float(df_exp_condition['absorptivity_solar'])

    x0 = int(df_exp_condition['x0_pixels'])
    y0 = int(df_exp_condition['y0_pixels'])
    N_div = int(df_exp_condition['N_div'])
    #alpha_r_A_ref = float(df_sample_cp_rho_alpha['alpha_r_A'])
    #alpha_r_B_ref = float(df_sample_cp_rho_alpha['alpha_r_B'])

    N_Rs = int(df_exp_condition['N_Rs_pixels'])
    mcmc_mode = df_exp_condition['analysis_mode']

    vacuum_chamber_setting = {'N_Rs_pixels': N_Rs, 'R0_pixels': R0,'focal_shift':focal_shift,'R_analysis_pixels':R_analysis,'light_blocker':df_exp_condition['light_blocker']}

    mcmc_setting = {'alpha_A_prior_range':[float(df_exp_condition['alpha_A_LL']),float(df_exp_condition['alpha_A_UL'])],'alpha_B_prior_range':[float(df_exp_condition['alpha_B_LL']),float(df_exp_condition['alpha_B_UL'])],
                    'sigma_s_prior_range':[float(df_exp_condition['sigma_s_LL']),float(df_exp_condition['sigma_s_UL'])],'T_bias_prior_range':[float(df_exp_condition['T_bias_LL']),float(df_exp_condition['T_bias_UL'])],
                    'step_size':mcmc_other_setting['step_size'],'p_initial':mcmc_other_setting['p_initial'],'N_total_mcmc_samples':mcmc_other_setting['N_total_mcmc_samples'],'PC_order':mcmc_other_setting['PC_order'],
                    'PC_training_core_num':mcmc_other_setting['PC_training_core_num'],'T_regularization':mcmc_other_setting['T_regularization'],'chain_num':mcmc_other_setting['chain_num']}

    dump_file_path_surrogate = code_directory + "surrogate_dump//" + df_exp_condition[
        'rec_name'] + '_R0{:}Ra{:}gap{:}_{:}{:}{:}{:}{:}_NRs{:}ord{:}_x0{:}y0{:}{:}_{:}{:}_{:}{:}_{:}{:}_P4D{:}_T{:}{:}'.format(
        int(R0), int(R_analysis), int(gap), int(emissivity_front * 100), int(emissivity_back * 100),
        int(absorptivity_front * 100), int(absorptivity_back * 100), int(absorptivity_solar * 100),
        int(N_Rs),mcmc_setting['PC_order'],x0,y0,LB_file_name,int(1000*mcmc_setting['sigma_s_prior_range'][0]),
        int(1000*mcmc_setting['sigma_s_prior_range'][1]),int(mcmc_setting['alpha_A_prior_range'][0]),
        int(mcmc_setting['alpha_A_prior_range'][1]),int(mcmc_setting['alpha_B_prior_range'][0]/100),
        int(mcmc_setting['alpha_B_prior_range'][1]/100),N_div,int(mcmc_setting['T_bias_prior_range'][0]/10),int(mcmc_setting['T_bias_prior_range'][1]/10))

    amp_ratio_approx, phase_diff_approx, ss_temp_approx = pickle.load(open(dump_file_path_surrogate, 'rb'))

    def physical_model_PC(parameter):

        # print(parameter)
        A = amp_ratio_approx(parameter[0], parameter[1], parameter[2], parameter[3])
        P = phase_diff_approx(parameter[0], parameter[1], parameter[2], parameter[3])
        SS_T = ss_temp_approx(parameter[0], parameter[1], parameter[2], parameter[3])
        return A, P, SS_T


    def log_likelihood_AP(parameter, df_amp_phase_measurement,df_temperature_measurement,T_regularization,vacuum_chamber_setting):
        simulation_A, simulation_P, simulation_SS_T = physical_model_PC(parameter)
        observation_A = np.array(df_amp_phase_measurement['amp_ratio'])
        observation_P = np.array(df_amp_phase_measurement['phase_diff'])
        sigma_noise_A = np.array(df_amp_phase_measurement['dA'])
        sigma_noise_P = np.array(df_amp_phase_measurement['dP'])

        R0 = vacuum_chamber_setting['R0_pixels']
        #R_analysis = vacuum_chamber_setting['R_analysis']

        # p_T = T_regularization*(SS_T[:,R0:R0+R_analysis].mean(axis = 0) - np.array(df_temperature_measurement.iloc[:,R0:R0+R_analysis]))**2

        p_T = T_regularization * np.sum((simulation_SS_T[:, R0:R0 + 1].mean(axis=0) - np.array(
            df_temperature_measurement.iloc[:, R0:R0 + 1].mean(axis=0))) ** 2) + T_regularization * np.sum((simulation_SS_T[:, R0+R_analysis:R0+R_analysis + 1].mean(axis=0) - np.array(
            df_temperature_measurement.iloc[:, R0+R_analysis:R0+R_analysis + 1].mean(axis=0))) ** 2)

        if df_exp_condition['regression_method'] == 0:
            return -np.sum((simulation_A - observation_A) ** 2 / sigma_noise_A ** 2) - np.sum(
            (simulation_P - observation_P) ** 2 / sigma_noise_P ** 2) - p_T  # calculate the sum of the natural log probability
        elif df_exp_condition['regression_method'] == 1:
            return -np.sum((simulation_A - observation_A) ** 2 / sigma_noise_A ** 2) - p_T
        elif df_exp_condition['regression_method'] == 2:
            return - np.sum((simulation_P - observation_P) ** 2 / sigma_noise_P ** 2) - p_T


    #step = mcmc_setting['step_size']
    parameter_initial = mcmc_setting['p_initial']
    N_total_samples = mcmc_setting['N_total_mcmc_samples']
    T_regularization = mcmc_setting['T_regularization']
    sigma_s_prior_range = mcmc_setting['sigma_s_prior_range']
    alpha_A_prior_range = mcmc_setting['alpha_A_prior_range']
    alpha_B_prior_range = mcmc_setting['alpha_B_prior_range']
    T_bias_prior_range = mcmc_setting['T_bias_prior_range']

    chain_num = mcmc_setting['chain_num']

    dump_file_path_mcmc_results = code_directory + "mcmc_results_dump//" + df_exp_condition[
        'rec_name'] + '_R0{:}{:}{:}_{:}{:}{:}{:}{:}{:}{:}_Tr{:}_{:}{:}_{:}c{:}_{:}{:}{:}{:}{:}{:}{:}_P4D{:}_T{:}{:}'.format(
        int(R0), int(R_analysis), int(gap), int(emissivity_front * 100), int(emissivity_back * 100),
        int(absorptivity_front * 100), int(absorptivity_back * 100), int(absorptivity_solar * 100),
        N_Rs,int(mcmc_setting['N_total_mcmc_samples']),int(mcmc_setting['T_regularization']*1000),
        x0,y0,mcmc_mode,mcmc_setting['chain_num'],LB_file_name,int(1000*mcmc_setting['sigma_s_prior_range'][0]),
        int(1000*mcmc_setting['sigma_s_prior_range'][1]),int(mcmc_setting['alpha_A_prior_range'][0]),
        int(mcmc_setting['alpha_A_prior_range'][1]),int(mcmc_setting['alpha_B_prior_range'][0]/100),
        int(mcmc_setting['alpha_B_prior_range'][1]/100),N_div,int(mcmc_setting['T_bias_prior_range'][0]/10),int(mcmc_setting['T_bias_prior_range'][1]/10))



    if (os.path.isfile(dump_file_path_mcmc_results)):  # First check if a dump file exist:
        print('Found previous dump file for mcmc results:' + dump_file_path_mcmc_results)
        accepted_samples_array = pickle.load(open(dump_file_path_mcmc_results, 'rb'))

    else:

        #ndraws = 500
        nburn = 25
        logl = LogLikeWithGrad(log_likelihood_AP, df_amplitude_phase_measurement, df_temperature, T_regularization, vacuum_chamber_setting)



        with pm.Model():
            # uniform priors on m and c
            sigma_s = pm.Uniform("sigma_s", lower=sigma_s_prior_range[0], upper=sigma_s_prior_range[1])
            alpha_A = pm.Uniform("alpha_A", lower=alpha_A_prior_range[0], upper=alpha_A_prior_range[1])
            alpha_B = pm.Uniform("alpha_B", lower=alpha_B_prior_range[0], upper=alpha_B_prior_range[1])
            T_bias = pm.Uniform("T_bias", lower=T_bias_prior_range[0], upper=T_bias_prior_range[1])

            theta = tt.as_tensor_variable([sigma_s, alpha_A, alpha_B,T_bias])

            # use a DensityDist (use a lamdba function to "call" the Op)
            pm.DensityDist("likelihood", lambda v: logl(v), observed={"v": theta})
            # step = pm.Metropolis()
            trace = pm.sample(N_total_samples, tune=nburn, discard_tuned_samples=True, chains=1)

        accepted_samples_array = np.array(
            [trace.get_values('sigma_s'), trace.get_values('alpha_A'), trace.get_values('alpha_B'),trace.get_values('T_bias')]).T

    print("recording {} completed.".format(rec_name))

    if (os.path.isfile(dump_file_path_mcmc_results) == False):  # First check if a dump file exist:
        pickle.dump(accepted_samples_array, open(dump_file_path_mcmc_results, "wb"))  # create a dump file



def high_T_Angstrom_execute_one_case_rw_mcmc(df_exp_condition, data_directory, code_directory,df_sample_cp_rho_alpha, df_amplitude_phase_measurement,df_temperature, mcmc_other_setting):

    focal_shift = float(df_exp_condition['focal_shift'])
    LB_file_name = df_exp_condition['LB_file_name']

    rec_name = df_exp_condition['rec_name']
    R0 = int(df_exp_condition['R0_pixels'])
    R_analysis  = int(df_exp_condition['R_analysis_pixels'])
    gap = int(df_exp_condition['gap_pixels'])
    emissivity_front = float(df_exp_condition['emissivity_front'])
    emissivity_back = float(df_exp_condition['emissivity_back'])
    absorptivity_front = float(df_exp_condition['absorptivity_front'])
    absorptivity_back = float(df_exp_condition['absorptivity_back'])
    absorptivity_solar = float(df_exp_condition['absorptivity_solar'])
    mcmc_mode = df_exp_condition['analysis_mode']

    x0 = int(df_exp_condition['x0_pixels'])
    y0 = int(df_exp_condition['y0_pixels'])

    #alpha_r_A_ref = float(df_sample_cp_rho_alpha['alpha_r_A'])
    #alpha_r_B_ref = float(df_sample_cp_rho_alpha['alpha_r_B'])

    N_Rs = int(df_exp_condition['N_Rs_pixels'])
    N_div = int(df_exp_condition['N_div'])

    vacuum_chamber_setting = {'N_Rs_pixels': N_Rs, 'R0_pixels': R0,'focal_shift':focal_shift,'R_analysis_pixels':R_analysis,'light_blocker':df_exp_condition['light_blocker']}
    mcmc_setting = {
        'alpha_A_prior_range': [float(df_exp_condition['alpha_A_LL']), float(df_exp_condition['alpha_A_UL'])],
        'alpha_B_prior_range': [float(df_exp_condition['alpha_B_LL']), float(df_exp_condition['alpha_B_UL'])],
        'sigma_s_prior_range': [float(df_exp_condition['sigma_s_LL']), float(df_exp_condition['sigma_s_UL'])],
        'T_bias_prior_range': [float(df_exp_condition['T_bias_LL']), float(df_exp_condition['T_bias_UL'])],
        'step_size': mcmc_other_setting['step_size'], 'p_initial': mcmc_other_setting['p_initial'],
        'N_total_mcmc_samples': mcmc_other_setting['N_total_mcmc_samples'], 'PC_order': mcmc_other_setting['PC_order'],
        'PC_training_core_num': mcmc_other_setting['PC_training_core_num'],
        'T_regularization': mcmc_other_setting['T_regularization'], 'chain_num': mcmc_other_setting['chain_num']}

    dump_file_path_surrogate = code_directory + "surrogate_dump//" + df_exp_condition[
        'rec_name'] + '_R0{:}Ra{:}gap{:}_{:}{:}{:}{:}{:}_NRs{:}ord{:}_x0{:}y0{:}{:}_{:}{:}_{:}{:}_{:}{:}_P3{:}'.format(
        int(R0), int(R_analysis), int(gap), int(emissivity_front * 100), int(emissivity_back * 100),
        int(absorptivity_front * 100), int(absorptivity_back * 100), int(absorptivity_solar * 100),
        int(N_Rs),mcmc_setting['PC_order'],x0,y0,LB_file_name,int(1000*mcmc_setting['sigma_s_prior_range'][0]),
        int(1000*mcmc_setting['sigma_s_prior_range'][1]),int(mcmc_setting['alpha_A_prior_range'][0]),
        int(mcmc_setting['alpha_A_prior_range'][1]),int(mcmc_setting['alpha_B_prior_range'][0]/100),
        int(mcmc_setting['alpha_B_prior_range'][1]/100),N_div)

    amp_ratio_approx, phase_diff_approx, ss_temp_approx = pickle.load(open(dump_file_path_surrogate, 'rb'))


    def physical_model_PC(parameter):

        # print(parameter)
        A = amp_ratio_approx(parameter[0], parameter[1], parameter[2])
        P = phase_diff_approx(parameter[0], parameter[1], parameter[2])
        SS_T = ss_temp_approx(parameter[0], parameter[1], parameter[2])
        return A, P, SS_T


    def log_likelihood_AP(parameter, df_amp_phase_measurement,df_temperature_measurement,T_regularization,vacuum_chamber_setting):
        simulation_A, simulation_P, simulation_SS_T = physical_model_PC(parameter)
        observation_A = np.array(df_amp_phase_measurement['amp_ratio'])
        observation_P = np.array(df_amp_phase_measurement['phase_diff'])
        sigma_noise_A = np.array(df_amp_phase_measurement['dA'])
        sigma_noise_P = np.array(df_amp_phase_measurement['dP'])

        R0 = vacuum_chamber_setting['R0_pixels']
        R_analysis = vacuum_chamber_setting['R_analysis_pixels']

        p_A = norm.pdf(simulation_A, observation_A, sigma_noise_A)
        p_P = norm.pdf(simulation_P, observation_P, sigma_noise_P)
        # p_T = T_regularization*(SS_T[:,R0:R0+R_analysis].mean(axis = 0) - np.array(df_temperature_measurement.iloc[:,R0:R0+R_analysis]))**2

        p_T = T_regularization * np.sum((simulation_SS_T[:, R0:R0 + 1].mean(axis=0) - np.array(
            df_temperature_measurement.iloc[:, R0:R0 + 1].mean(axis=0))) ** 2)

        for i in range(len(p_A)):
            if p_A[i] < 10 ** (-256):
                p_A[i] = 10 ** (-256)  # avoid getting zero for log
                # print("Warning! A")
            if p_P[i] < 10 ** (-256):
                p_P[i] = 10 ** (-256)  # avoid getting zero for log

        if df_exp_condition['regression_method'] == 0:
            return np.sum(np.log(p_A)) + np.sum(np.log(p_P)) - p_T  # calculate the sum of the natural log probability
        elif df_exp_condition['regression_method'] == 1:
            return np.sum(np.log(p_A)) - p_T
        elif df_exp_condition['regression_method'] == 2:
            return np.sum(np.log(p_P)) - p_T


    def log_prior(parameter, sigma_s_prior_range,alpha_A_prior_range,alpha_B_prior_range):
        #     p1 = norm.pdf(parameter[0],1e-2,3e-3)
        #     p2 = norm.pdf(parameter[1],5e-6,2e-6)
        #     p3 = norm.pdf(parameter[2],0.03,0.01)
        p1 = uniform.pdf(parameter[0], sigma_s_prior_range[0], sigma_s_prior_range[1]-sigma_s_prior_range[0])
        p2 = uniform.pdf(parameter[1], alpha_A_prior_range[0], alpha_A_prior_range[1]-alpha_A_prior_range[0])
        p3 = uniform.pdf(parameter[2], alpha_B_prior_range[0], alpha_B_prior_range[1]-alpha_B_prior_range[0])

        if p1 * p2 * p3 == 0:
            return -10 ** 9
        else:
            return np.log(p1) + np.log(p2) + np.log(p3)


    def acceptance(log_likelihood_current, log_likelihood_new, log_prior_current, log_prior_new):
        C = random.uniform(0, 1)
        A = min(np.exp(log_likelihood_new + log_prior_new - log_likelihood_current - log_prior_current), 1)
        if C <= A:
            return True  # accept tht proposal, move on
        else:
            return False  # reject the proposal, say at current

    def proposal_new_parameter(parameter, step):
        parameter_new = np.array([norm.rvs(parameter, step)])[0]
        return parameter_new


    def rw_Metropolis_Hasting(parameter_initial, step, df_amp_phase_measurement, df_temperature_measurement,
                              T_regularization, vacuum_chamber_setting, N_total_samples, sigma_s_prior_range,alpha_A_prior_range,alpha_B_prior_range):

        parameter = parameter_initial
        log_likelihood_current = log_likelihood_AP(parameter, df_amp_phase_measurement, df_temperature_measurement,
                                                   T_regularization, vacuum_chamber_setting)
        log_prior_current = log_prior(parameter, sigma_s_prior_range,alpha_A_prior_range,alpha_B_prior_range)

        N_accepted_sample = 0
        N_rejected_sample = 0
        accepted_samples = []

        while (N_accepted_sample < N_total_samples):
            i_iter = N_rejected_sample + N_accepted_sample

            parameter_new = proposal_new_parameter(parameter, step)
            log_likelihood_new = log_likelihood_AP(parameter_new, df_amp_phase_measurement, df_temperature_measurement,
                                                   T_regularization, vacuum_chamber_setting)
            log_prior_new = log_prior(parameter_new, sigma_s_prior_range,alpha_A_prior_range,alpha_B_prior_range)

            if acceptance(log_likelihood_current, log_likelihood_new, log_prior_current, log_prior_new):
                parameter = parameter_new
                log_likelihood_current = log_likelihood_new
                log_prior_current = log_prior_new
                N_accepted_sample += 1
                accepted_samples.append(parameter)
                if i_iter % 100 == 0:
                    print(
                        "For Rec {}. At iteration {},The accepted sigma is {:.2e}, alpha_A is {:.2e}, alpha_B is {:.2e}, acceptance rate is {:.2f}.".format(
                            rec_name, i_iter, parameter[0], parameter[1], parameter[2],
                            N_accepted_sample / (N_accepted_sample + N_rejected_sample)))
            else:
                N_rejected_sample += 1

            if i_iter % 10000 == 0:
                print("Iteration {}!".format(i_iter))
        return accepted_samples

    step = mcmc_setting['step_size']
    parameter_initial = mcmc_setting['p_initial']
    N_total_samples = mcmc_setting['N_total_mcmc_samples']
    T_regularization = mcmc_setting['T_regularization']
    sigma_s_prior_range = mcmc_setting['sigma_s_prior_range']
    alpha_A_prior_range = mcmc_setting['alpha_A_prior_range']
    alpha_B_prior_range = mcmc_setting['alpha_B_prior_range']
    chain_num = mcmc_setting['chain_num']
    #df_amplitude_phase_measurement, df_temperature

    dump_file_path_mcmc_results = code_directory + "mcmc_results_dump//" + df_exp_condition[
        'rec_name'] + '_R0{:}{:}{:}_{:}{:}{:}{:}{:}{:}{:}_Tr{:}_{:}{:}_{:}c{:}_{:}{:}{:}{:}{:}{:}{:}_P3{:}'.format(
        int(R0), int(R_analysis), int(gap), int(emissivity_front * 100), int(emissivity_back * 100),
        int(absorptivity_front * 100), int(absorptivity_back * 100), int(absorptivity_solar * 100),
        N_Rs,int(mcmc_setting['N_total_mcmc_samples']),int(mcmc_setting['T_regularization']*1000),
        x0,y0,mcmc_mode,mcmc_setting['chain_num'],LB_file_name,int(1000*mcmc_setting['sigma_s_prior_range'][0]),
        int(1000*mcmc_setting['sigma_s_prior_range'][1]),int(mcmc_setting['alpha_A_prior_range'][0]),
        int(mcmc_setting['alpha_A_prior_range'][1]),int(mcmc_setting['alpha_B_prior_range'][0]/100),
        int(mcmc_setting['alpha_B_prior_range'][1]/100),N_div)

    if (os.path.isfile(dump_file_path_mcmc_results)):  # First check if a dump file exist:
        print('Found previous dump file for mcmc results:' + dump_file_path_mcmc_results)
        accepted_samples_array = pickle.load(open(dump_file_path_mcmc_results, 'rb'))

    else:
        accepted_samples = rw_Metropolis_Hasting(parameter_initial, step, df_amplitude_phase_measurement,
                                                 df_temperature, T_regularization, vacuum_chamber_setting,
                                                 N_total_samples,sigma_s_prior_range,alpha_A_prior_range,alpha_B_prior_range)
        accepted_samples_array = np.array(accepted_samples)

    print("recording {} completed.".format(rec_name))

    if (os.path.isfile(dump_file_path_mcmc_results) == False):  # First check if a dump file exist:
        pickle.dump(accepted_samples_array, open(dump_file_path_mcmc_results, "wb"))  # create a dump file



def high_T_Angstrom_execute_one_case_rw_mcmc_P4(df_exp_condition, data_directory, code_directory,df_sample_cp_rho_alpha, df_amplitude_phase_measurement,df_temperature, mcmc_other_setting):

    focal_shift = float(df_exp_condition['focal_shift'])
    LB_file_name = df_exp_condition['LB_file_name']

    rec_name = df_exp_condition['rec_name']
    R0 = int(df_exp_condition['R0_pixels'])
    R_analysis  = int(df_exp_condition['R_analysis_pixels'])
    gap = int(df_exp_condition['gap_pixels'])
    emissivity_front = float(df_exp_condition['emissivity_front'])
    emissivity_back = float(df_exp_condition['emissivity_back'])
    absorptivity_front = float(df_exp_condition['absorptivity_front'])
    absorptivity_back = float(df_exp_condition['absorptivity_back'])
    absorptivity_solar = float(df_exp_condition['absorptivity_solar'])
    mcmc_mode = df_exp_condition['analysis_mode']

    x0 = int(df_exp_condition['x0_pixels'])
    y0 = int(df_exp_condition['y0_pixels'])

    #alpha_r_A_ref = float(df_sample_cp_rho_alpha['alpha_r_A'])
    #alpha_r_B_ref = float(df_sample_cp_rho_alpha['alpha_r_B'])

    N_Rs = int(df_exp_condition['N_Rs_pixels'])
    N_div = int(df_exp_condition['N_div'])

    vacuum_chamber_setting = {'N_Rs_pixels': N_Rs, 'R0_pixels': R0,'focal_shift':focal_shift,'R_analysis_pixels':R_analysis,'light_blocker':df_exp_condition['light_blocker']}
    mcmc_setting = {
        'alpha_A_prior_range': [float(df_exp_condition['alpha_A_LL']), float(df_exp_condition['alpha_A_UL'])],
        'alpha_B_prior_range': [float(df_exp_condition['alpha_B_LL']), float(df_exp_condition['alpha_B_UL'])],
        'sigma_s_prior_range': [float(df_exp_condition['sigma_s_LL']), float(df_exp_condition['sigma_s_UL'])],
        'T_bias_prior_range': [float(df_exp_condition['T_bias_LL']), float(df_exp_condition['T_bias_UL'])],
        'step_size': mcmc_other_setting['step_size'], 'p_initial': mcmc_other_setting['p_initial'],
        'N_total_mcmc_samples': mcmc_other_setting['N_total_mcmc_samples'], 'PC_order': mcmc_other_setting['PC_order'],
        'PC_training_core_num': mcmc_other_setting['PC_training_core_num'],
        'T_regularization': mcmc_other_setting['T_regularization'], 'chain_num': mcmc_other_setting['chain_num']}

    dump_file_path_surrogate = code_directory + "surrogate_dump//" + df_exp_condition[
        'rec_name'] + '_R0{:}Ra{:}gap{:}_{:}{:}{:}{:}{:}_NRs{:}ord{:}_x0{:}y0{:}{:}_{:}{:}_{:}{:}_{:}{:}_P4D{:}_T{:}{:}'.format(
        int(R0), int(R_analysis), int(gap), int(emissivity_front * 100), int(emissivity_back * 100),
        int(absorptivity_front * 100), int(absorptivity_back * 100), int(absorptivity_solar * 100),
        int(N_Rs),mcmc_setting['PC_order'],x0,y0,LB_file_name,int(1000*mcmc_setting['sigma_s_prior_range'][0]),
        int(1000*mcmc_setting['sigma_s_prior_range'][1]),int(mcmc_setting['alpha_A_prior_range'][0]),
        int(mcmc_setting['alpha_A_prior_range'][1]),int(mcmc_setting['alpha_B_prior_range'][0]/100),
        int(mcmc_setting['alpha_B_prior_range'][1]/100),N_div,int(mcmc_setting['T_bias_prior_range'][0]/10),int(mcmc_setting['T_bias_prior_range'][1]/10))

    amp_ratio_approx, phase_diff_approx, ss_temp_approx = pickle.load(open(dump_file_path_surrogate, 'rb'))



    def physical_model_PC(parameter):

        # print(parameter)
        A = amp_ratio_approx(parameter[0], parameter[1], parameter[2],parameter[3])
        P = phase_diff_approx(parameter[0], parameter[1], parameter[2],parameter[3])
        SS_T = ss_temp_approx(parameter[0], parameter[1], parameter[2],parameter[3])
        return A, P, SS_T


    def log_likelihood_AP(parameter, df_amp_phase_measurement,df_temperature_measurement,T_regularization,vacuum_chamber_setting):
        simulation_A, simulation_P, simulation_SS_T = physical_model_PC(parameter)
        observation_A = np.array(df_amp_phase_measurement['amp_ratio'])
        observation_P = np.array(df_amp_phase_measurement['phase_diff'])
        sigma_noise_A = np.array(df_amp_phase_measurement['dA'])
        sigma_noise_P = np.array(df_amp_phase_measurement['dP'])

        R0 = vacuum_chamber_setting['R0_pixels']
        R_analysis = vacuum_chamber_setting['R_analysis_pixels']

        p_A = norm.pdf(simulation_A, observation_A, sigma_noise_A)
        p_P = norm.pdf(simulation_P, observation_P, sigma_noise_P)
        # p_T = T_regularization*(SS_T[:,R0:R0+R_analysis].mean(axis = 0) - np.array(df_temperature_measurement.iloc[:,R0:R0+R_analysis]))**2

        p_T = T_regularization * np.sum((simulation_SS_T[:, R0:R0 + 1].mean(axis=0) - np.array(
            df_temperature_measurement.iloc[:, R0:R0 + 1].mean(axis=0))) ** 2) + T_regularization/2 * np.sum((simulation_SS_T[:, R0+R_analysis:R0+R_analysis + 1].mean(axis=0) - np.array(
            df_temperature_measurement.iloc[:, R0+R_analysis:R0+R_analysis + 1].mean(axis=0))) ** 2)

        for i in range(len(p_A)):
            if p_A[i] < 10 ** (-256):
                p_A[i] = 10 ** (-256)  # avoid getting zero for log
                # print("Warning! A")
            if p_P[i] < 10 ** (-256):
                p_P[i] = 10 ** (-256)  # avoid getting zero for log

        if df_exp_condition['regression_method'] == 0:
            return np.sum(np.log(p_A)) + np.sum(np.log(p_P)) - p_T  # calculate the sum of the natural log probability
        elif df_exp_condition['regression_method'] == 1:
            return np.sum(np.log(p_A)) - p_T
        elif df_exp_condition['regression_method'] == 2:
            return np.sum(np.log(p_P)) - p_T


    def log_prior(parameter, sigma_s_prior_range,alpha_A_prior_range,alpha_B_prior_range,T_bias_prior_range):
        #     p1 = norm.pdf(parameter[0],1e-2,3e-3)
        #     p2 = norm.pdf(parameter[1],5e-6,2e-6)
        #     p3 = norm.pdf(parameter[2],0.03,0.01)
        p1 = uniform.pdf(parameter[0], sigma_s_prior_range[0], sigma_s_prior_range[1]-sigma_s_prior_range[0])
        p2 = uniform.pdf(parameter[1], alpha_A_prior_range[0], alpha_A_prior_range[1]-alpha_A_prior_range[0])
        p3 = uniform.pdf(parameter[2], alpha_B_prior_range[0], alpha_B_prior_range[1]-alpha_B_prior_range[0])
        p4 = uniform.pdf(parameter[3], T_bias_prior_range[0], T_bias_prior_range[1]-T_bias_prior_range[0])

        if p1 * p2 * p3 * p4 == 0:
            return -10 ** 9
        else:
            return np.log(p1) + np.log(p2) + np.log(p3) + np.log(p4)


    def acceptance(log_likelihood_current, log_likelihood_new, log_prior_current, log_prior_new):
        C = random.uniform(0, 1)
        A = min(np.exp(log_likelihood_new + log_prior_new - log_likelihood_current - log_prior_current), 1)
        if C <= A:
            return True  # accept tht proposal, move on
        else:
            return False  # reject the proposal, say at current

    def proposal_new_parameter(parameter, step):
        parameter_new = np.array([norm.rvs(parameter, step)])[0]
        return parameter_new


    def rw_Metropolis_Hasting(parameter_initial, step, df_amp_phase_measurement, df_temperature_measurement,
                              T_regularization, vacuum_chamber_setting, N_total_samples, sigma_s_prior_range,alpha_A_prior_range,alpha_B_prior_range,T_bias_prior_range):

        parameter = parameter_initial
        log_likelihood_current = log_likelihood_AP(parameter, df_amp_phase_measurement, df_temperature_measurement,
                                                   T_regularization, vacuum_chamber_setting)
        log_prior_current = log_prior(parameter, sigma_s_prior_range,alpha_A_prior_range,alpha_B_prior_range,T_bias_prior_range)

        N_accepted_sample = 0
        N_rejected_sample = 0
        accepted_samples = []

        while (N_accepted_sample < N_total_samples):
            i_iter = N_rejected_sample + N_accepted_sample

            parameter_new = proposal_new_parameter(parameter, step)
            log_likelihood_new = log_likelihood_AP(parameter_new, df_amp_phase_measurement, df_temperature_measurement,
                                                   T_regularization, vacuum_chamber_setting)
            log_prior_new = log_prior(parameter_new, sigma_s_prior_range,alpha_A_prior_range,alpha_B_prior_range,T_bias_prior_range)

            if acceptance(log_likelihood_current, log_likelihood_new, log_prior_current, log_prior_new):
                parameter = parameter_new
                log_likelihood_current = log_likelihood_new
                log_prior_current = log_prior_new
                N_accepted_sample += 1
                accepted_samples.append(parameter)
                if i_iter % 100 == 0:
                    print(
                        "For Rec {}. At iteration {},The accepted sigma is {:.2e}, alpha_A is {:.2e}, alpha_B is {:.2e}, T_bias is {:.2e}, acceptance rate is {:.2f}.".format(
                            rec_name, i_iter, parameter[0], parameter[1], parameter[2], parameter[3],
                            N_accepted_sample / (N_accepted_sample + N_rejected_sample)))
            else:
                N_rejected_sample += 1

            if i_iter % 10000 == 0:
                print("Iteration {}!".format(i_iter))
        return accepted_samples

    step = mcmc_setting['step_size']
    parameter_initial = mcmc_setting['p_initial']
    N_total_samples = mcmc_setting['N_total_mcmc_samples']
    T_regularization = mcmc_setting['T_regularization']
    sigma_s_prior_range = mcmc_setting['sigma_s_prior_range']
    alpha_A_prior_range = mcmc_setting['alpha_A_prior_range']
    alpha_B_prior_range = mcmc_setting['alpha_B_prior_range']
    T_bias_prior_range = mcmc_setting['T_bias_prior_range']
    chain_num = mcmc_setting['chain_num']
    #df_amplitude_phase_measurement, df_temperature

    dump_file_path_mcmc_results = code_directory + "mcmc_results_dump//" + df_exp_condition[
        'rec_name'] + '_R0{:}{:}{:}_{:}{:}{:}{:}{:}{:}{:}_Tr{:}_{:}{:}_{:}c{:}_{:}{:}{:}{:}{:}{:}{:}_P4D{:}_T{:}{:}'.format(
        int(R0), int(R_analysis), int(gap), int(emissivity_front * 100), int(emissivity_back * 100),
        int(absorptivity_front * 100), int(absorptivity_back * 100), int(absorptivity_solar * 100),
        N_Rs,int(mcmc_setting['N_total_mcmc_samples']),int(mcmc_setting['T_regularization']*1000),
        x0,y0,mcmc_mode,mcmc_setting['chain_num'],LB_file_name,int(1000*mcmc_setting['sigma_s_prior_range'][0]),
        int(1000*mcmc_setting['sigma_s_prior_range'][1]),int(mcmc_setting['alpha_A_prior_range'][0]),
        int(mcmc_setting['alpha_A_prior_range'][1]),int(mcmc_setting['alpha_B_prior_range'][0]/100),
        int(mcmc_setting['alpha_B_prior_range'][1]/100),N_div,int(mcmc_setting['T_bias_prior_range'][0]/10),int(mcmc_setting['T_bias_prior_range'][1]/10))

    if (os.path.isfile(dump_file_path_mcmc_results)):  # First check if a dump file exist:
        print('Found previous dump file for mcmc results:' + dump_file_path_mcmc_results)
        accepted_samples_array = pickle.load(open(dump_file_path_mcmc_results, 'rb'))

    else:
        accepted_samples = rw_Metropolis_Hasting(parameter_initial, step, df_amplitude_phase_measurement,
                                                 df_temperature, T_regularization, vacuum_chamber_setting,
                                                 N_total_samples,sigma_s_prior_range,alpha_A_prior_range,alpha_B_prior_range,T_bias_prior_range)
        accepted_samples_array = np.array(accepted_samples)

    print("recording {} completed.".format(rec_name))

    if (os.path.isfile(dump_file_path_mcmc_results) == False):  # First check if a dump file exist:
        pickle.dump(accepted_samples_array, open(dump_file_path_mcmc_results, "wb"))  # create a dump file


def parallel_regression_batch_experimental_results_mcmc(df_exp_condition_spreadsheet_filename, data_directory,
                                                        num_cores, code_directory, df_sample_cp_rho_alpha,
                                                        df_amplitude_phase_measurement_list, df_temperature_list,
                                                        mcmc_setting):
    # df_exp_condition_spreadsheet = pd.read_excel(code_directory+"batch process information//" + df_exp_condition_spreadsheet_filename)
    df_exp_condition_spreadsheet = pd.read_csv(
        code_directory + "batch process information//" + df_exp_condition_spreadsheet_filename)

    mcmc_mode = df_exp_condition_spreadsheet.iloc[0,:]['analysis_mode']
    print(mcmc_mode)
    # df_temperature_list, df_amplitude_phase_measurement_list = parallel_temperature_average_batch_experimental_results(df_exp_condition_spreadsheet_filename, data_directory, num_cores,code_directory)
    if mcmc_mode == 'RW_Metropolis':
        Parallel(n_jobs=num_cores, verbose=0)(
            delayed(high_T_Angstrom_execute_one_case_rw_mcmc)(df_exp_condition_spreadsheet.iloc[i, :], data_directory,
                                                              code_directory, df_sample_cp_rho_alpha,
                                                              df_amplitude_phase_measurement_list[i],
                                                              df_temperature_list[i], mcmc_setting) for i in
            tqdm(range(len(df_exp_condition_spreadsheet))))

    elif mcmc_mode == 'RW_Metropolis_P4':
        Parallel(n_jobs=num_cores, verbose=0)(
            delayed(high_T_Angstrom_execute_one_case_rw_mcmc_P4)(df_exp_condition_spreadsheet.iloc[i, :], data_directory,
                                                           code_directory, df_sample_cp_rho_alpha,
                                                           df_amplitude_phase_measurement_list[i],
                                                           df_temperature_list[i], mcmc_setting) for i in
            tqdm(range(len(df_exp_condition_spreadsheet))))


    elif mcmc_mode == 'NUTs':
        Parallel(n_jobs=num_cores, verbose=0)(
            delayed(high_T_Angstrom_execute_one_case_NUTs)(df_exp_condition_spreadsheet.iloc[i, :], data_directory,
                                                           code_directory, df_sample_cp_rho_alpha,
                                                           df_amplitude_phase_measurement_list[i],
                                                           df_temperature_list[i], mcmc_setting) for i in
            tqdm(range(len(df_exp_condition_spreadsheet))))

    elif mcmc_mode == 'NUTs_P4':
        Parallel(n_jobs=num_cores, verbose=0)(
            delayed(high_T_Angstrom_execute_one_case_NUTs_P4)(df_exp_condition_spreadsheet.iloc[i, :], data_directory,
                                                           code_directory, df_sample_cp_rho_alpha,
                                                           df_amplitude_phase_measurement_list[i],
                                                           df_temperature_list[i], mcmc_setting) for i in
            tqdm(range(len(df_exp_condition_spreadsheet))))

    # pickle.dump(joblib_output,open(code_directory+"result cache dump//mcmc_results_" + df_exp_condition_spreadsheet_filename, "wb"))


def show_mcmc_results_one_case(df_exp_condition, code_directory, df_temperature, df_amplitude_phase_measurement,
                               df_sample_cp_rho_alpha, mcmc_other_setting):
    # df_exp_condition_spreadsheet = pd.read_excel(code_directory + "batch process information//" + df_exp_condition_spreadsheet_filename)
    # df_exp_condition_spreadsheet = pd.read_csv(code_directory+"batch process information//" + df_exp_condition_spreadsheet_filename)

    mcmc_setting = {'alpha_A_prior_range':[float(df_exp_condition['alpha_A_LL']),float(df_exp_condition['alpha_A_UL'])],'alpha_B_prior_range':[float(df_exp_condition['alpha_B_LL']),float(df_exp_condition['alpha_B_UL'])],
                    'sigma_s_prior_range':[float(df_exp_condition['sigma_s_LL']),float(df_exp_condition['sigma_s_UL'])],'T_bias_prior_range':[float(df_exp_condition['T_bias_LL']),float(df_exp_condition['T_bias_UL'])],
                    'step_size':mcmc_other_setting['step_size'],'p_initial':mcmc_other_setting['p_initial'],'N_total_mcmc_samples':mcmc_other_setting['N_total_mcmc_samples'],'PC_order':mcmc_other_setting['PC_order'],
                    'PC_training_core_num':mcmc_other_setting['PC_training_core_num'],'T_regularization':mcmc_other_setting['T_regularization'],'chain_num':mcmc_other_setting['chain_num']}


    parameter_initial = mcmc_setting['p_initial']
    LB_file_name = df_exp_condition['LB_file_name']

    N_total_samples = mcmc_setting['N_total_mcmc_samples']
    T_regularization = mcmc_setting['T_regularization']
    chain_num = mcmc_setting['chain_num']

    rec_name = df_exp_condition['rec_name']
    R0 = int(df_exp_condition['R0_pixels'])
    R_analysis = int(df_exp_condition['R_analysis_pixels'])
    gap = int(df_exp_condition['gap_pixels'])
    emissivity_front = float(df_exp_condition['emissivity_front'])
    emissivity_back = float(df_exp_condition['emissivity_back'])
    absorptivity_front = float(df_exp_condition['absorptivity_front'])
    absorptivity_back = float(df_exp_condition['absorptivity_back'])
    absorptivity_solar = float(df_exp_condition['absorptivity_solar'])
    x0 = int(df_exp_condition['x0_pixels'])
    y0 = int(df_exp_condition['y0_pixels'])
    N_div = int(df_exp_condition['N_div'])
    N_Rs = int(df_exp_condition['N_Rs_pixels'])

    alpha_r_A_ref = float(df_sample_cp_rho_alpha['alpha_r_A'])

    alpha_r_B_ref = float(df_sample_cp_rho_alpha['alpha_r_B'])

    mcmc_mode = df_exp_condition['analysis_mode']

    f_heating = float(df_exp_condition['f_heating'])

    dump_file_path_mcmc_results = code_directory + "mcmc_results_dump//" + df_exp_condition[
        'rec_name'] + '_R0{:}{:}{:}_{:}{:}{:}{:}{:}{:}{:}_Tr{:}_{:}{:}_{:}c{:}_{:}{:}{:}{:}{:}{:}{:}_P3{:}'.format(
        int(R0), int(R_analysis), int(gap), int(emissivity_front * 100), int(emissivity_back * 100),
        int(absorptivity_front * 100), int(absorptivity_back * 100), int(absorptivity_solar * 100),
        N_Rs,int(mcmc_setting['N_total_mcmc_samples']),int(mcmc_setting['T_regularization']*1000),
        x0,y0,mcmc_mode,mcmc_setting['chain_num'],LB_file_name,int(1000*mcmc_setting['sigma_s_prior_range'][0]),
        int(1000*mcmc_setting['sigma_s_prior_range'][1]),int(mcmc_setting['alpha_A_prior_range'][0]),
        int(mcmc_setting['alpha_A_prior_range'][1]),int(mcmc_setting['alpha_B_prior_range'][0]/100),
        int(mcmc_setting['alpha_B_prior_range'][1]/100),N_div)

    accepted_samples_array = pickle.load(open(dump_file_path_mcmc_results, 'rb'))
    if mcmc_mode == 'RW_Metropolis':
        n_burn = int(0.4*len(accepted_samples_array.T[0]))
    else:
        n_burn = int(0.05 * len(accepted_samples_array.T[0]))

    accepted_samples_trim = accepted_samples_array[n_burn:]

    label_font_size = 12

    #f, axes = plt.subplots(4, 3, figsize=(26, 24))
    f, axes = plt.subplots(4, 3, figsize=(26, 30))
    axes[0, 0].hist(accepted_samples_trim.T[0], bins=30)
    axes[0, 0].set_xlabel(r'$\sigma_{solar}$(m)', fontsize=label_font_size,fontweight = 'bold')


    axes[0, 1].hist(accepted_samples_trim.T[1], bins=30)
    axes[0, 1].set_xlabel(r'$A_{\alpha}$ (s/m$^2$K)', fontsize=label_font_size,fontweight = 'bold')

    axes[0, 2].hist(accepted_samples_trim.T[2], bins=30)
    axes[0, 2].set_xlabel(r'$B_{\alpha}$ (s/m$^2$)', fontsize=label_font_size,fontweight = 'bold')

    axes[1, 0].plot(accepted_samples_trim[:, 0])
    axes[1, 0].set_ylabel(r'$\sigma_{solar}$(m)', fontsize=label_font_size,fontweight = 'bold')
    axes[1, 0].set_xlabel('sample num', fontsize=14,fontweight = 'bold')

    axes[1, 1].plot(accepted_samples_trim[:, 1])
    axes[1, 1].set_ylabel(r'$A_{\alpha}$ (s/m$^2$K)', fontsize=label_font_size,fontweight = 'bold')
    axes[1, 1].set_xlabel('sample num', fontsize=14,fontweight = 'bold')

    axes[1, 2].plot(accepted_samples_trim[:, 2])
    axes[1, 2].set_ylabel(r'$B_{\alpha}$ (s/m$^2$)', fontsize=label_font_size,fontweight = 'bold')
    axes[1, 2].set_xlabel('sample num', fontsize=14,fontweight = 'bold')

    T_array = np.linspace(df_temperature.iloc[:, R0 + R_analysis].mean(), df_temperature.iloc[:, R0].mean(), 50)
    alpha_T_array = np.zeros((len(T_array), len(accepted_samples_trim)))

    for i in range(len(accepted_samples_trim)):
        # alpha_T_array[:,i] = 1/(accepted_samples_trim[i][1]*T_array+accepted_samples_trim[i][2])
        alpha_T_array[:, i] = 1 / (accepted_samples_trim[i][1] * T_array + accepted_samples_trim[i][2])

    alpha_T_array_mean = alpha_T_array.mean(axis=1)
    alpha_T_array_std = alpha_T_array.std(axis=1)
    alpha_std_to_mean_avg = np.mean(alpha_T_array_std/alpha_T_array_mean)

    alpha_reference = 1 / (alpha_r_A_ref * T_array + alpha_r_B_ref)

    #f_posterior_mean_vs_reference = plt.figure(figsize=(7, 5))
    axes[2, 0].fill_between(T_array, alpha_T_array_mean - 3 * alpha_T_array_std,
                            alpha_T_array_mean + 3 * alpha_T_array_std,
                            alpha=0.2)
    axes[2, 0].plot(T_array, alpha_T_array_mean, color='k', label='Bayesian mean')
    #axes[2, 0].plot(T_array, alpha_reference, color='r', label='reference')
    #axes[2, 0].set_ylim([0,2.5e-5])
    axes[2, 0].set_xlabel('Temperature (K)',fontsize=label_font_size,fontweight = 'bold')
    axes[2, 0].set_ylabel(r'Thermal diffusivity (m$^2$/s)', fontsize=label_font_size, fontweight='bold')
    axes[2, 0].legend(prop={'weight': 'bold', 'size': 14})
    alpha_T_array_mean = alpha_T_array.mean(axis=1)

    def residual(params, x, data):
        alpha_A = params['alpha_A']
        alpha_B = params['alpha_B']

        model = 1 / (alpha_A * x + alpha_B)

        return model - data

    params = Parameters()
    params.add('alpha_A', value=1e-7)
    params.add('alpha_B', value=2e-2)

    out = minimize(residual, params, args=(T_array, alpha_T_array_mean))

    alpha_A_posterior_mean = out.params['alpha_A'].value
    alpha_B_posterior_mean = out.params['alpha_B'].value

    sigma_s_posterior_mean = accepted_samples_trim.T[0].mean()

    dump_file_path_surrogate = code_directory + "surrogate_dump//" + df_exp_condition[
        'rec_name'] + '_R0{:}Ra{:}gap{:}_{:}{:}{:}{:}{:}_NRs{:}ord{:}_x0{:}y0{:}{:}_{:}{:}_{:}{:}_{:}{:}_P3{:}'.format(
        int(R0), int(R_analysis), int(gap), int(emissivity_front * 100), int(emissivity_back * 100),
        int(absorptivity_front * 100), int(absorptivity_back * 100), int(absorptivity_solar * 100),
        int(N_Rs),mcmc_setting['PC_order'],x0,y0,LB_file_name,int(1000*mcmc_setting['sigma_s_prior_range'][0]),
        int(1000*mcmc_setting['sigma_s_prior_range'][1]),int(mcmc_setting['alpha_A_prior_range'][0]),
        int(mcmc_setting['alpha_A_prior_range'][1]),int(mcmc_setting['alpha_B_prior_range'][0]/100),
        int(mcmc_setting['alpha_B_prior_range'][1]/100),N_div)

    amp_ratio_approx, phase_diff_approx, ss_temp_approx = pickle.load(open(dump_file_path_surrogate, 'rb'))

    axes[2, 1].errorbar(df_amplitude_phase_measurement['r_pixels'], df_amplitude_phase_measurement['amp_ratio'],yerr = df_amplitude_phase_measurement['dA']*3,
                    label='measurement',capsize=5, elinewidth=2,color = 'red',fmt= 'o')
    axes[2, 1].plot(df_amplitude_phase_measurement['r_pixels'],
                       amp_ratio_approx(sigma_s_posterior_mean, alpha_A_posterior_mean, alpha_B_posterior_mean),
                       label='Bayesian fitting',color = 'black')
    axes[2, 1].set_xlabel('R (node num)',fontsize=label_font_size,fontweight = 'bold')
    axes[2, 1].set_ylabel('Amplitude ratio',fontsize=label_font_size,fontweight = 'bold')
    axes[2, 1].legend(prop={'weight': 'bold', 'size': 14})

    axes[2, 2].errorbar(df_amplitude_phase_measurement['r_pixels'], df_amplitude_phase_measurement['phase_diff'],yerr = df_amplitude_phase_measurement['dP']*3,
                    label='measurement',capsize=5, elinewidth=2,color = 'red',fmt= 'o')
    axes[2, 2].plot(df_amplitude_phase_measurement['r_pixels'],
                       phase_diff_approx(sigma_s_posterior_mean, alpha_A_posterior_mean, alpha_B_posterior_mean),
                       label='Bayesian fitting',color = 'black')
    axes[2, 2].set_xlabel('R (node num)',fontsize=label_font_size,fontweight = 'bold')
    axes[2, 2].set_ylabel('Phase difference',fontsize=label_font_size,fontweight = 'bold')
    axes[2, 2].legend(prop={'weight': 'bold', 'size': 14})

    T_ss_RO = ss_temp_approx(sigma_s_posterior_mean, alpha_A_posterior_mean, alpha_B_posterior_mean)[:, R0]
    time_ss_RO = np.linspace(0, 2 / f_heating, len(T_ss_RO))
    axes[3, 0].plot(time_ss_RO, T_ss_RO, label='simulation R = {:}'.format(R0))

    time_ss_measured_R0 = df_temperature.query('reltime<{:}'.format(2 / f_heating))['reltime']
    T_ss_measured_R0 = df_temperature.iloc[:len(time_ss_measured_R0), R0]
    axes[3, 0].plot(time_ss_measured_R0, T_ss_measured_R0, label='measurement R = {:}'.format(R0))

    T_ss_RN = ss_temp_approx(sigma_s_posterior_mean, alpha_A_posterior_mean, alpha_B_posterior_mean)[:, R0 + R_analysis]
    time_ss_RN = np.linspace(0,2/f_heating,len(T_ss_RN))
    axes[3, 0].plot(time_ss_RN,T_ss_RN,label='simulation R = {:}'.format(R0 + R_analysis))

    time_ss_measured_RN = time_ss_measured_R0
    T_ss_measured_RN = df_temperature.iloc[:len(time_ss_measured_RN), R0+R_analysis]
    axes[3, 0].plot(time_ss_measured_RN, T_ss_measured_RN, label='measurement R = {:}'.format(R0+R_analysis))

    axes[3, 0].set_ylabel('Temperature (K)', fontsize=label_font_size, fontweight='bold')
    axes[3, 0].set_xlabel('Time (s)', fontsize=label_font_size, fontweight='bold')
    axes[3, 0].legend(prop={'weight': 'bold', 'size': 14})


    def acf(x, length):
        return np.array([1] + [np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, length)])

    def auto_correlation_function(trace, lags):
        autocorr_trace = acf(trace, lags)
        return autocorr_trace

    # def plot_auto_correlation(trace, lags):

    lags = 100

    autocorr_trace_alpha_A = auto_correlation_function(accepted_samples_trim[:, 1], lags)
    axes[3, 1].plot(autocorr_trace_alpha_A)
    axes[3, 1].set_xlabel('lags N', fontsize=label_font_size,fontweight = 'bold')
    axes[3, 1].set_ylabel('alpha_A', fontsize=label_font_size,fontweight = 'bold')

    autocorr_trace_alpha_B = auto_correlation_function(accepted_samples_trim[:, 2], lags)
    axes[3, 2].plot(autocorr_trace_alpha_B)
    axes[3, 2].set_xlabel('lags N', fontsize=label_font_size,fontweight = 'bold')
    axes[3, 2].set_ylabel('alpha_B', fontsize=label_font_size,fontweight = 'bold')

    for i in range(4):
        for j in range(3):
            for tick in axes[i, j].xaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize=12)
                tick.label.set_fontweight('bold')
            for tick in axes[i, j].yaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize=12)
                tick.label.set_fontweight('bold')

    f.suptitle("{},f_heating={:.2e}, VDC={}, R0={}, R_a={}, e_f={}, e_b={}, a_solar={}, x0={}, y0={}".format(rec_name,float(df_exp_condition['f_heating']),float(df_exp_condition['V_DC']),R0,R_analysis,int(100 * float(df_exp_condition['emissivity_front'])),
                                                                                                             int(100 * float( df_exp_condition['emissivity_back'])), int(100 * float(df_exp_condition['absorptivity_solar'])),x0, y0),y=0.91, fontsize=16)
    # plt.show()

    # print("------------------------------------Happy 2021!------------------------------------")
    print(rec_name + " is shown here, and with mean sigma_s = {:.2e}, alpha_A = {:.2e}, alpha_B = {:.2e}".format(
        accepted_samples_array.T[0].mean(), accepted_samples_array.T[1].mean(), accepted_samples_array.T[2].mean()))
    # return f
    return accepted_samples_trim, alpha_A_posterior_mean, alpha_B_posterior_mean, sigma_s_posterior_mean, alpha_std_to_mean_avg, np.min(T_array), np.max(T_array), amp_ratio_approx, phase_diff_approx, ss_temp_approx



class MathTextSciFormatter(mticker.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)



def show_mcmc_results_one_case_P4(df_exp_condition, code_directory,data_directory, df_temperature, df_amplitude_phase_measurement,
                               df_sample_cp_rho_alpha, mcmc_other_setting):
    # df_exp_condition_spreadsheet = pd.read_excel(code_directory + "batch process information//" + df_exp_condition_spreadsheet_filename)
    # df_exp_condition_spreadsheet = pd.read_csv(code_directory+"batch process information//" + df_exp_condition_spreadsheet_filename)

    mcmc_setting = {'alpha_A_prior_range':[float(df_exp_condition['alpha_A_LL']),float(df_exp_condition['alpha_A_UL'])],'alpha_B_prior_range':[float(df_exp_condition['alpha_B_LL']),float(df_exp_condition['alpha_B_UL'])],
                    'sigma_s_prior_range':[float(df_exp_condition['sigma_s_LL']),float(df_exp_condition['sigma_s_UL'])],'T_bias_prior_range':[float(df_exp_condition['T_bias_LL']),float(df_exp_condition['T_bias_UL'])],
                    'step_size':mcmc_other_setting['step_size'],'p_initial':mcmc_other_setting['p_initial'],'N_total_mcmc_samples':mcmc_other_setting['N_total_mcmc_samples'],'PC_order':mcmc_other_setting['PC_order'],
                    'PC_training_core_num':mcmc_other_setting['PC_training_core_num'],'T_regularization':mcmc_other_setting['T_regularization'],'chain_num':mcmc_other_setting['chain_num']}


    parameter_initial = mcmc_setting['p_initial']
    LB_file_name = df_exp_condition['LB_file_name']

    N_total_samples = mcmc_setting['N_total_mcmc_samples']
    T_regularization = mcmc_setting['T_regularization']
    chain_num = mcmc_setting['chain_num']

    rec_name = df_exp_condition['rec_name']
    R0 = int(df_exp_condition['R0_pixels'])
    R_analysis = int(df_exp_condition['R_analysis_pixels'])

    gap = int(df_exp_condition['gap_pixels'])
    emissivity_front = float(df_exp_condition['emissivity_front'])
    emissivity_back = float(df_exp_condition['emissivity_back'])
    absorptivity_front = float(df_exp_condition['absorptivity_front'])
    absorptivity_back = float(df_exp_condition['absorptivity_back'])
    absorptivity_solar = float(df_exp_condition['absorptivity_solar'])
    x0 = int(df_exp_condition['x0_pixels'])
    y0 = int(df_exp_condition['y0_pixels'])
    N_div = int(df_exp_condition['N_div'])
    N_Rs = int(df_exp_condition['N_Rs_pixels'])


    alpha_r_A_ref = float(df_sample_cp_rho_alpha['alpha_r_A'])

    alpha_r_B_ref = float(df_sample_cp_rho_alpha['alpha_r_B'])

    mcmc_mode = df_exp_condition['analysis_mode']

    dump_file_path_mcmc_results = code_directory + "mcmc_results_dump//" + df_exp_condition[
        'rec_name'] + '_R0{:}{:}{:}_{:}{:}{:}{:}{:}{:}{:}_Tr{:}_{:}{:}_{:}c{:}_{:}{:}{:}{:}{:}{:}{:}_P4D{:}_T{:}{:}'.format(
        int(R0), int(R_analysis), int(gap), int(emissivity_front * 100), int(emissivity_back * 100),
        int(absorptivity_front * 100), int(absorptivity_back * 100), int(absorptivity_solar * 100),
        N_Rs,int(mcmc_setting['N_total_mcmc_samples']),int(mcmc_setting['T_regularization']*1000),
        x0,y0,mcmc_mode,mcmc_setting['chain_num'],LB_file_name,int(1000*mcmc_setting['sigma_s_prior_range'][0]),
        int(1000*mcmc_setting['sigma_s_prior_range'][1]),int(mcmc_setting['alpha_A_prior_range'][0]),
        int(mcmc_setting['alpha_A_prior_range'][1]),int(mcmc_setting['alpha_B_prior_range'][0]/100),
        int(mcmc_setting['alpha_B_prior_range'][1]/100),N_div,int(mcmc_setting['T_bias_prior_range'][0]/10),int(mcmc_setting['T_bias_prior_range'][1]/10))

    accepted_samples_array = pickle.load(open(dump_file_path_mcmc_results, 'rb'))
    if mcmc_mode == 'RW_Metropolis' or mcmc_mode=='RW_Metropolis_P4':
        n_burn = int(0.3*len(accepted_samples_array.T[0]))
    else:
        n_burn = int(0.05 * len(accepted_samples_array.T[0]))

    accepted_samples_trim = accepted_samples_array[n_burn:]


    Nr = int(df_exp_condition['Nr_pixels'])
    pr = float(df_exp_condition['sample_radius(m)'])/(Nr-1)
    f_heating = float(df_exp_condition['f_heating'])
    exp_amp_phase_extraction_method = df_exp_condition['exp_amp_phase_extraction_method']

    colors = ['red', 'black', 'green', 'blue', 'orange', 'magenta', 'brown', 'yellow', 'purple', 'cornflowerblue']

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

    output_name = rec_name
    path = data_directory + str(rec_name) + "//"
    df_temperature_list_all_ranges, df_temperature = radial_temperature_average_disk_sample_several_ranges(x0, y0, Nr,angle_range,pr, path,rec_name,output_name,'MA', 2, code_directory)




    #f, axes = plt.subplots(4, 3, figsize=(26, 24))
    f, axes = plt.subplots(5, 4, figsize=(30, 38))

    label_font_size = 18

    axes[0, 0].hist(accepted_samples_trim.T[0], bins=15)
    axes[0, 0].set_xlabel(r'$\sigma_{solar}$(m)', fontsize=label_font_size,fontweight = 'bold')

    axes[0, 1].hist(accepted_samples_trim.T[1], bins=15)
    axes[0, 1].set_xlabel(r'$A_{\alpha}$ (s/m$^2$K)', fontsize=label_font_size,fontweight = 'bold')

    axes[0, 2].hist(accepted_samples_trim.T[2], bins=15)
    axes[0, 2].set_xlabel(r'$B_{\alpha}$ (s/m$^2$)', fontsize=label_font_size,fontweight = 'bold')

    axes[0, 3].hist(accepted_samples_trim.T[3], bins=15)
    axes[0, 3].set_xlabel('$T_{bias}$ (K)', fontsize=label_font_size,fontweight = 'bold')

    axes[1, 0].plot(accepted_samples_trim[:, 0])
    axes[1, 0].set_ylabel(r'$\sigma_{solar}$(m)', fontsize=label_font_size,fontweight = 'bold')
    axes[1, 0].set_xlabel('sample num', fontsize=label_font_size,fontweight = 'bold')

    axes[1, 1].plot(accepted_samples_trim[:, 1])
    axes[1, 1].set_ylabel(r'$A_{\alpha}$ (s/m$^2$K)', fontsize=label_font_size,fontweight = 'bold')
    axes[1, 1].set_xlabel('sample num', fontsize=label_font_size,fontweight = 'bold')

    axes[1, 2].plot(accepted_samples_trim[:, 2])
    axes[1, 2].set_ylabel(r'$B_{\alpha}$ (s/m$^2$)', fontsize=label_font_size,fontweight = 'bold')
    axes[1, 2].set_xlabel('sample num', fontsize=label_font_size,fontweight = 'bold')

    axes[1, 3].plot(accepted_samples_trim[:, 3])
    axes[1, 3].set_ylabel('$T_{bias}$ (K)', fontsize=label_font_size,fontweight = 'bold')
    axes[1, 3].set_xlabel('sample num', fontsize=label_font_size,fontweight = 'bold')


    T_array = np.linspace(df_temperature.iloc[:, R0 + R_analysis].mean(), df_temperature.iloc[:, R0].mean(), 4)
    alpha_T_array = np.zeros((len(T_array), len(accepted_samples_trim)))

    for i in range(len(accepted_samples_trim)):
        # alpha_T_array[:,i] = 1/(accepted_samples_trim[i][1]*T_array+accepted_samples_trim[i][2])
        alpha_T_array[:, i] = 1 / (accepted_samples_trim[i][1] * T_array + accepted_samples_trim[i][2])

    alpha_T_array_mean = alpha_T_array.mean(axis=1)
    alpha_T_array_std = alpha_T_array.std(axis=1)
    alpha_std_to_mean_avg = np.mean(alpha_T_array_std/alpha_T_array_mean)


    alpha_reference = 1 / (alpha_r_A_ref * T_array + alpha_r_B_ref)

    #f_posterior_mean_vs_reference = plt.figure(figsize=(7, 5))
    axes[2, 0].fill_between(T_array, alpha_T_array_mean - 3 * alpha_T_array_std,
                            alpha_T_array_mean + 3 * alpha_T_array_std,
                            alpha=0.2)
    axes[2, 0].plot(T_array, alpha_T_array_mean, color='k', label='Bayesian posterior distribution')
    axes[2, 0].errorbar(T_array, alpha_reference,yerr = alpha_reference*0.05,
                    label='reference value',capsize=5, elinewidth=2,color = 'green',fmt= 'o')

        #plot(T_array, alpha_reference, color='r', label='reference value')
    #axes[2, 0].set_ylim([0,2.1e-5])
    axes[2, 0].set_xlabel('Temperature (K)',fontsize=label_font_size, fontweight='bold')
    axes[2, 0].set_ylabel(r'Thermal diffusivity (m$^2$/s)', fontsize=label_font_size, fontweight='bold')
    #axes[2, 0].yaxis.set_major_formatter(MathTextSciFormatter("%1.2e"))
    #axes[2, 0].yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    axes[2, 0].legend(prop={'weight': 'bold', 'size': 14})
    alpha_T_array_mean = alpha_T_array.mean(axis=1)

    def residual(params, x, data):
        alpha_A = params['alpha_A']
        alpha_B = params['alpha_B']

        model = 1 / (alpha_A * x + alpha_B)

        return model - data

    params = Parameters()
    params.add('alpha_A', value=1e-7)
    params.add('alpha_B', value=2e-2)

    out = minimize(residual, params, args=(T_array, alpha_T_array_mean))

    alpha_A_posterior_mean = out.params['alpha_A'].value
    alpha_B_posterior_mean = out.params['alpha_B'].value

    sigma_s_posterior_mean = accepted_samples_trim.T[0].mean()
    T_bias_posterior_mean = accepted_samples_trim.T[3].mean()

    dump_file_path_surrogate = code_directory + "surrogate_dump//" + df_exp_condition[
        'rec_name'] + '_R0{:}Ra{:}gap{:}_{:}{:}{:}{:}{:}_NRs{:}ord{:}_x0{:}y0{:}{:}_{:}{:}_{:}{:}_{:}{:}_P4D{:}_T{:}{:}'.format(
        int(R0), int(R_analysis), int(gap), int(emissivity_front * 100), int(emissivity_back * 100),
        int(absorptivity_front * 100), int(absorptivity_back * 100), int(absorptivity_solar * 100),
        int(N_Rs),mcmc_setting['PC_order'],x0,y0,LB_file_name,int(1000*mcmc_setting['sigma_s_prior_range'][0]),
        int(1000*mcmc_setting['sigma_s_prior_range'][1]),int(mcmc_setting['alpha_A_prior_range'][0]),
        int(mcmc_setting['alpha_A_prior_range'][1]),int(mcmc_setting['alpha_B_prior_range'][0]/100),
        int(mcmc_setting['alpha_B_prior_range'][1]/100),N_div,int(mcmc_setting['T_bias_prior_range'][0]/10),int(mcmc_setting['T_bias_prior_range'][1]/10))

    amp_ratio_approx, phase_diff_approx, ss_temp_approx = pickle.load(open(dump_file_path_surrogate, 'rb'))

    axes[2, 1].errorbar(df_amplitude_phase_measurement['r_pixels'], df_amplitude_phase_measurement['amp_ratio'],yerr = df_amplitude_phase_measurement['dA']*3,
                    label='measurement',capsize=5, elinewidth=2,color = 'red',fmt= 'o')
    axes[2, 1].plot(df_amplitude_phase_measurement['r_pixels'],
                       amp_ratio_approx(sigma_s_posterior_mean, alpha_A_posterior_mean, alpha_B_posterior_mean,T_bias_posterior_mean),
                       label='simulation',color = 'black')
    axes[2, 1].set_xlabel('R (node num)', fontsize=label_font_size, fontweight='bold')
    axes[2, 1].set_ylabel('Amplitude ratio', fontsize=label_font_size, fontweight='bold')
    #axes[2, 1].set_ylim([0,2.1e-5])
    axes[2, 1].legend(prop={'weight': 'bold', 'size': 14})

    # axes[2, 2].scatter(df_amplitude_phase_measurement['r_pixels'], df_amplitude_phase_measurement['phase_diff'],
    #                 label='measurement',color = 'red')
    axes[2, 2].errorbar(df_amplitude_phase_measurement['r_pixels'], df_amplitude_phase_measurement['phase_diff'],yerr = df_amplitude_phase_measurement['dP']*3,
                    label='measurement',capsize=5, elinewidth=2,color = 'red',fmt= 'o')
    axes[2, 2].plot(df_amplitude_phase_measurement['r_pixels'],
                       phase_diff_approx(sigma_s_posterior_mean, alpha_A_posterior_mean, alpha_B_posterior_mean,T_bias_posterior_mean),
                       label='simulation',color = 'black')
    axes[2, 2].set_xlabel('R (node num)', fontsize=label_font_size, fontweight='bold')
    axes[2, 2].set_ylabel('Phase difference', fontsize=label_font_size, fontweight='bold')
    axes[2, 2].legend(prop={'weight': 'bold', 'size': 14})



    T_ss_RO = ss_temp_approx(sigma_s_posterior_mean, alpha_A_posterior_mean, alpha_B_posterior_mean, T_bias_posterior_mean)[:, R0]
    time_ss_RO = np.linspace(0,2/f_heating,len(T_ss_RO))
    axes[2, 3].plot(time_ss_RO, T_ss_RO,label='simulation R = {:}'.format(R0))


    time_ss_measured_R0 = df_temperature.query('reltime<{:}'.format(2 / f_heating))['reltime']
    T_ss_measured_R0 = df_temperature.iloc[:len(time_ss_measured_R0), R0]

    # time_ss_measured_R0 = df_temperature.query('reltime<{:}'.format(2/f_heating))['reltime']
    # T_ss_measured_R0 = df_temperature.iloc[:len(time_ss_measured_R0), R0]
    axes[2, 3].plot(time_ss_measured_R0, T_ss_measured_R0, label='measurement R = {:}'.format(R0))


    T_ss_RN = ss_temp_approx(sigma_s_posterior_mean, alpha_A_posterior_mean, alpha_B_posterior_mean,T_bias_posterior_mean)[:, R0 + R_analysis]
    time_ss_RN = np.linspace(0,2/f_heating,len(T_ss_RN))
    axes[2, 3].plot(time_ss_RN,T_ss_RN,label='simulation R = {:}'.format(R0 + R_analysis))

    time_ss_measured_RN = time_ss_measured_R0
    T_ss_measured_RN = df_temperature.iloc[:len(time_ss_measured_RN), R0+R_analysis]
    # time_ss_measured_RN = df_temperature.query('reltime<{:}'.format(2/f_heating))['reltime']
    # T_ss_measured_RN = df_temperature.iloc[:len(time_ss_measured_RN), R0+R_analysis]
    axes[2, 3].plot(time_ss_measured_RN, T_ss_measured_RN, label='measurement R = {:}'.format(R0+R_analysis))

    #axes[2, 3].plot(df_temperature.iloc[:, R0 + R_analysis], label='measured R = {:}'.format(R0 + R_analysis))
    axes[2, 3].set_ylabel('Temperature (K)', fontsize=label_font_size, fontweight='bold')
    axes[2, 3].set_xlabel('Time (s)', fontsize=label_font_size, fontweight='bold')
    axes[2, 3].legend(prop={'weight': 'bold', 'size': 14})

    def acf(x, length):
        return np.array([1] + [np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, length)])

    def auto_correlation_function(trace, lags):
        autocorr_trace = acf(trace, lags)
        return autocorr_trace

    # def plot_auto_correlation(trace, lags):

    lags = 100

    autocorr_trace_sigma = auto_correlation_function(accepted_samples_trim[:, 0], lags)
    axes[3, 0].plot(autocorr_trace_sigma)
    axes[3, 0].set_xlabel('lags N', fontsize=label_font_size)
    axes[3, 0].set_ylabel('sigma_s', fontsize=label_font_size)

    autocorr_trace_alpha_A = auto_correlation_function(accepted_samples_trim[:, 1], lags)
    axes[3, 1].plot(autocorr_trace_alpha_A)
    axes[3, 1].set_xlabel('lags N', fontsize=label_font_size)
    axes[3, 1].set_ylabel('alpha_A', fontsize=label_font_size)

    autocorr_trace_alpha_B = auto_correlation_function(accepted_samples_trim[:, 2], lags)
    axes[3, 2].plot(autocorr_trace_alpha_B)
    axes[3, 2].set_xlabel('lags N', fontsize=label_font_size)
    axes[3, 2].set_ylabel('alpha_B', fontsize=label_font_size)

    autocorr_T_bias = auto_correlation_function(accepted_samples_trim[:, 3], lags)
    axes[3, 3].plot(autocorr_T_bias)
    axes[3, 3].set_xlabel('lags N', fontsize=label_font_size)
    axes[3, 3].set_ylabel('T_bias', fontsize=label_font_size)

    df_temperature_angle_list = []
    df_amp_phase_angle_list = []
    for j, angle in enumerate(angle_range):
        # note radial_temperature_average_disk_sample automatically checks if a dump file exist
        df_temperature_ = df_temperature_list_all_ranges[j]
        df_amplitude_phase_measurement_ = batch_process_horizontal_lines(df_temperature_, f_heating, R0, gap, R_analysis,
                                                                        exp_amp_phase_extraction_method)
        df_temperature_angle_list.append(df_temperature_)
        df_amp_phase_angle_list.append(df_amplitude_phase_measurement_)

        axes[4, 2].scatter(df_amplitude_phase_measurement_['r'],
                    df_amplitude_phase_measurement_['amp_ratio'], facecolors='none',
                    s=60, linewidths=2, edgecolor=colors[j], label=str(angle[0]) + ' to ' + str(angle[1]) + ' Degs')


    axes[4, 2].set_xlabel('R (pixels)', fontsize=label_font_size, fontweight='bold')
    axes[4, 2].set_ylabel('Amplitude Ratio', fontsize=label_font_size, fontweight='bold')
    axes[4, 2].legend(prop={'weight': 'bold', 'size': 12})


    for j, angle in enumerate(angle_range):
        df_temperature_ = df_temperature_angle_list[j]
        df_amplitude_phase_measurement_ = df_amp_phase_angle_list[j]
        axes[4, 3].scatter(df_amplitude_phase_measurement_['r'],
                    df_amplitude_phase_measurement_['phase_diff'], facecolors='none',
                    s=60, linewidths=2, edgecolor=colors[j], label=str(angle[0]) + ' to ' + str(angle[1]) + ' Degs')

    axes[4, 3].set_xlabel('R (pixels)', fontsize=label_font_size, fontweight='bold')
    axes[4, 3].set_ylabel('Phase difference (rad)', fontsize=label_font_size, fontweight='bold')
    axes[4, 3].legend(prop={'weight': 'bold', 'size': 12})


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
    #ax = plt.gca()
    CS = axes[4, 0].contour(X, Y, Z, 18)
    axes[4, 0].plot([xmin, xmax], [y0, y0], ls='-.', color='k', lw=2)  # add a horizontal line cross x0,y0
    axes[4, 0].plot([x0, x0], [ymin, ymax], ls='-.', color='k', lw=2)  # add a vertical line cross x0,y0

    R_angle_show = R0 + R_analysis

    circle1 = plt.Circle((x0, y0), R0, edgecolor='r', fill=False, linewidth=3, linestyle='-.')
    circle2 = plt.Circle((x0, y0), R0 + R_analysis, edgecolor='k', fill=False, linewidth=3, linestyle='-.')

    circle3 = plt.Circle((x0, y0), int(0.01 / pr), edgecolor='black', fill=False, linewidth=3, linestyle='dotted')

    axes[4, 0].invert_yaxis()
    axes[4, 0].clabel(CS, inline=1, fontsize=12, manual=manual_locations)
    axes[4, 0].add_artist(circle1)
    axes[4, 0].add_artist(circle2)
    axes[4, 0].add_artist(circle3)

    axes[4, 0].set_xlabel('x (pixels)', fontsize=label_font_size, fontweight='bold')
    axes[4, 0].set_ylabel('y (pixels)', fontsize=label_font_size, fontweight='bold')
    #axes[4, 2].set_title('x0 = {}, y0 = {}, R0 = {}'.format(x0, y0, R0), fontsize=12, fontweight='bold')

    for j, angle in enumerate(angle_range):
        axes[4, 0].plot([x0, x0 + R_angle_show * np.cos(angle[0] * np.pi / 180)], [y0, y0 + R_angle_show * np.sin(angle[0] * np.pi / 180)], ls='-.',color='blue', lw=2)



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
    #ax = plt.gca()
    CS = axes[4, 1].contour(X, Y, Z, 18)
    axes[4, 1].plot([xmin, xmax], [y0, y0], ls='-.', color='k', lw=2)  # add a horizontal line cross x0,y0
    axes[4, 1].plot([x0, x0], [ymin, ymax], ls='-.', color='k', lw=2)  # add a vertical line cross x0,y0

    R_angle_show = R0 + R_analysis

    circle1 = plt.Circle((x0, y0), R0, edgecolor='r', fill=False, linewidth=3, linestyle='-.')
    circle2 = plt.Circle((x0, y0), R0 + R_analysis, edgecolor='k', fill=False, linewidth=3, linestyle='-.')

    circle3 = plt.Circle((x0, y0), int(0.01 / pr), edgecolor='black', fill=False, linewidth=3, linestyle='dotted')

    axes[4, 1].invert_yaxis()
    axes[4, 1].clabel(CS, inline=1, fontsize=12, manual=manual_locations)
    axes[4, 1].add_artist(circle1)
    axes[4, 1].add_artist(circle2)
    axes[4, 1].add_artist(circle3)

    axes[4, 1].set_xlabel('x (pixels)', fontsize=label_font_size, fontweight='bold')
    axes[4, 1].set_ylabel('y (pixels)', fontsize=label_font_size, fontweight='bold')
    #axes[4, 3].set_title('x0 = {}, y0 = {}, R0 = {}'.format(x0, y0, R0), fontsize=12, fontweight='bold')

    for j, angle in enumerate(angle_range):
        axes[4, 1].plot([x0, x0 + R_angle_show * np.cos(angle[0] * np.pi / 180)], [y0, y0 + R_angle_show * np.sin(angle[0] * np.pi / 180)], ls='-.',color='blue', lw=2)


    for i in range(5):
        for j in range(4):
            for tick in axes[i, j].xaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize=12)
                tick.label.set_fontweight('bold')
            for tick in axes[i, j].yaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize=12)
                tick.label.set_fontweight('bold')
    f.suptitle("{},f_heating={:.2e}, VDC={}, R0={}, R_a={}, e_f={}, e_b={}, a_solar={}, x0={}, y0={}".format(rec_name,float(df_exp_condition['f_heating']),float(df_exp_condition['V_DC']),R0,R_analysis,int(100 * float(df_exp_condition['emissivity_front'])),
                                                                                                             int(100 * float( df_exp_condition['emissivity_back'])), int(100 * float(df_exp_condition['absorptivity_solar'])),x0, y0),y=0.91, fontsize=16)
    # plt.show()

    # print("------------------------------------Happy 2021!------------------------------------")
    print(rec_name + " is shown here, and with mean sigma_s = {:.2e}, alpha_A = {:.2e}, alpha_B = {:.2e}, T_bias = {:.2e}".format(
        accepted_samples_array.T[0].mean(), accepted_samples_array.T[1].mean(), accepted_samples_array.T[2].mean(), accepted_samples_array.T[3].mean()))
    # return f


    return accepted_samples_trim, alpha_A_posterior_mean, alpha_B_posterior_mean, sigma_s_posterior_mean, T_bias_posterior_mean, alpha_std_to_mean_avg, np.min(T_array), np.max(T_array), amp_ratio_approx, phase_diff_approx, ss_temp_approx





def parallel_batch_show_mcmc_results_P4(df_exp_condition_spreadsheet_filename, code_directory,data_directory, df_temperature_list,
                                     df_amplitude_phase_measurement_list, df_sample_cp_rho_alpha, mcmc_other_setting):

    df_exp_condition_spreadsheet = pd.read_csv(
        code_directory + "batch process information//" + df_exp_condition_spreadsheet_filename)
    T_min_list = []
    T_max_list = []
    accepted_samples_trim_list = []
    alpha_A_posterior_mean_list = []
    alpha_B_posterior_mean_list = []
    sigma_s_posterior_mean_list = []
    T_bias_posterior_mean_list = []
    alpha_std_to_mean_list = []
    amp_ratio_approx_list = []
    phase_diff_approx_list = []
    ss_temp_approx_list = []

    for i in tqdm(range(len(df_exp_condition_spreadsheet))):
        df_exp_condition = df_exp_condition_spreadsheet.iloc[i, :]

        mcmc_setting = {
            'alpha_A_prior_range': [float(df_exp_condition['alpha_A_LL']), float(df_exp_condition['alpha_A_UL'])],
            'alpha_B_prior_range': [float(df_exp_condition['alpha_B_LL']), float(df_exp_condition['alpha_B_UL'])],
            'sigma_s_prior_range': [float(df_exp_condition['sigma_s_LL']), float(df_exp_condition['sigma_s_UL'])],
            'T_bias_prior_range': [float(df_exp_condition['T_bias_LL']), float(df_exp_condition['T_bias_UL'])],
            'step_size': mcmc_other_setting['step_size'], 'p_initial': mcmc_other_setting['p_initial'],
            'N_total_mcmc_samples': mcmc_other_setting['N_total_mcmc_samples'],
            'PC_order': mcmc_other_setting['PC_order'],
            'PC_training_core_num': mcmc_other_setting['PC_training_core_num'],
            'T_regularization': mcmc_other_setting['T_regularization'], 'chain_num': mcmc_other_setting['chain_num']}

        accepted_samples_trim, alpha_A_posterior_mean, alpha_B_posterior_mean, sigma_s_posterior_mean, T_bias_posterior_mean, alpha_std_to_mean, \
        T_exp_min, T_exp_max, amp_ratio_approx, phase_diff_approx, ss_temp_approx = show_mcmc_results_one_case_P4(
            df_exp_condition_spreadsheet.iloc[i, :], code_directory,data_directory,
            df_temperature_list[i], df_amplitude_phase_measurement_list[i], df_sample_cp_rho_alpha, mcmc_setting)

        T_min_list.append(T_exp_min)
        T_max_list.append(T_exp_max)
        accepted_samples_trim_list.append(accepted_samples_trim)
        alpha_A_posterior_mean_list.append(alpha_A_posterior_mean)
        alpha_B_posterior_mean_list.append(alpha_B_posterior_mean)
        sigma_s_posterior_mean_list.append(sigma_s_posterior_mean)
        T_bias_posterior_mean_list.append(T_bias_posterior_mean)
        alpha_std_to_mean_list.append(alpha_std_to_mean)
        amp_ratio_approx_list.append(amp_ratio_approx)
        phase_diff_approx_list.append(phase_diff_approx)
        ss_temp_approx_list.append(ss_temp_approx)

    df_result_summary = pd.DataFrame(
        data={'T_min': T_min_list, 'T_max': T_max_list, 'sigma_s': sigma_s_posterior_mean_list,
              'alpha_A': alpha_A_posterior_mean_list,
              'alpha_B': alpha_B_posterior_mean_list, 'rec_name': df_exp_condition_spreadsheet['rec_name'],
              'R0_pixels': df_exp_condition_spreadsheet['R0_pixels'],
              'V_DC': df_exp_condition_spreadsheet['V_DC'], 'f_heating': df_exp_condition_spreadsheet['f_heating'],
              'T_bias':T_bias_posterior_mean_list,'alpha_std_to_mean':alpha_std_to_mean_list,'V_AC':df_exp_condition_spreadsheet['V_amplitude']})

    return df_result_summary, accepted_samples_trim,amp_ratio_approx_list, phase_diff_approx_list, ss_temp_approx_list



def parallel_batch_show_mcmc_results(df_exp_condition_spreadsheet_filename, code_directory, df_temperature_list,
                                     df_amplitude_phase_measurement_list, df_sample_cp_rho_alpha, mcmc_other_setting):
    df_exp_condition_spreadsheet = pd.read_csv(
        code_directory + "batch process information//" + df_exp_condition_spreadsheet_filename)
    T_min_list = []
    T_max_list = []
    accepted_samples_trim_list = []
    alpha_A_posterior_mean_list = []
    alpha_B_posterior_mean_list = []
    sigma_s_posterior_mean_list = []
    alpha_std_to_mean_list = []

    for i in tqdm(range(len(df_exp_condition_spreadsheet))):

        df_exp_condition = df_exp_condition_spreadsheet.iloc[i, :]

        mcmc_setting = {
            'alpha_A_prior_range': [float(df_exp_condition['alpha_A_LL']), float(df_exp_condition['alpha_A_UL'])],
            'alpha_B_prior_range': [float(df_exp_condition['alpha_B_LL']), float(df_exp_condition['alpha_B_UL'])],
            'sigma_s_prior_range': [float(df_exp_condition['sigma_s_LL']), float(df_exp_condition['sigma_s_UL'])],
            'T_bias_prior_range': [float(df_exp_condition['T_bias_LL']), float(df_exp_condition['T_bias_UL'])],
            'step_size': mcmc_other_setting['step_size'], 'p_initial': mcmc_other_setting['p_initial'],
            'N_total_mcmc_samples': mcmc_other_setting['N_total_mcmc_samples'],
            'PC_order': mcmc_other_setting['PC_order'],
            'PC_training_core_num': mcmc_other_setting['PC_training_core_num'],
            'T_regularization': mcmc_other_setting['T_regularization'], 'chain_num': mcmc_other_setting['chain_num']}

        accepted_samples_trim, alpha_A_posterior_mean, alpha_B_posterior_mean, sigma_s_posterior_mean,alpha_std_to_mean, \
        T_exp_min, T_exp_max, amp_ratio_approx, phase_diff_approx, ss_temp_approx = show_mcmc_results_one_case(df_exp_condition_spreadsheet.iloc[i, :], code_directory,
                                   df_temperature_list[i], df_amplitude_phase_measurement_list[i], df_sample_cp_rho_alpha, mcmc_setting)

        T_min_list.append(T_exp_min)
        T_max_list.append(T_exp_max)
        accepted_samples_trim_list.append(accepted_samples_trim)
        alpha_A_posterior_mean_list.append(alpha_A_posterior_mean)
        alpha_B_posterior_mean_list.append(alpha_B_posterior_mean)
        sigma_s_posterior_mean_list.append(sigma_s_posterior_mean)
        alpha_std_to_mean_list.append(alpha_std_to_mean)

    df_result_summary = pd.DataFrame(data = {'T_min':T_min_list,'T_max':T_max_list,'sigma_s':sigma_s_posterior_mean_list,'alpha_A':alpha_A_posterior_mean_list,
                                             'alpha_B':alpha_B_posterior_mean_list,'rec_name':df_exp_condition_spreadsheet['rec_name'],'R0_pixels':df_exp_condition_spreadsheet['R0_pixels'],
                                             'V_DC':df_exp_condition_spreadsheet['V_DC'],'f_heating':df_exp_condition_spreadsheet['f_heating'],'alpha_std_to_mean':alpha_std_to_mean_list})

    return df_result_summary, accepted_samples_trim


def show_mcmc_results_one_case_P4_match_phase(df_exp_condition, code_directory, data_directory, df_temperature,
                                  df_amplitude_phase_measurement,
                                  df_sample_cp_rho_alpha, mcmc_other_setting):
    # df_exp_condition_spreadsheet = pd.read_excel(code_directory + "batch process information//" + df_exp_condition_spreadsheet_filename)
    # df_exp_condition_spreadsheet = pd.read_csv(code_directory+"batch process information//" + df_exp_condition_spreadsheet_filename)

    mcmc_setting = {
        'alpha_A_prior_range': [float(df_exp_condition['alpha_A_LL']), float(df_exp_condition['alpha_A_UL'])],
        'alpha_B_prior_range': [float(df_exp_condition['alpha_B_LL']), float(df_exp_condition['alpha_B_UL'])],
        'sigma_s_prior_range': [float(df_exp_condition['sigma_s_LL']), float(df_exp_condition['sigma_s_UL'])],
        'T_bias_prior_range': [float(df_exp_condition['T_bias_LL']), float(df_exp_condition['T_bias_UL'])],
        'step_size': mcmc_other_setting['step_size'], 'p_initial': mcmc_other_setting['p_initial'],
        'N_total_mcmc_samples': mcmc_other_setting['N_total_mcmc_samples'], 'PC_order': mcmc_other_setting['PC_order'],
        'PC_training_core_num': mcmc_other_setting['PC_training_core_num'],
        'T_regularization': mcmc_other_setting['T_regularization'], 'chain_num': mcmc_other_setting['chain_num']}

    parameter_initial = mcmc_setting['p_initial']
    LB_file_name = df_exp_condition['LB_file_name']

    N_total_samples = mcmc_setting['N_total_mcmc_samples']
    T_regularization = mcmc_setting['T_regularization']
    chain_num = mcmc_setting['chain_num']

    rec_name = df_exp_condition['rec_name']
    R0 = int(df_exp_condition['R0_pixels'])
    R_analysis = int(df_exp_condition['R_analysis_pixels'])

    gap = int(df_exp_condition['gap_pixels'])
    emissivity_front = float(df_exp_condition['emissivity_front'])
    emissivity_back = float(df_exp_condition['emissivity_back'])
    absorptivity_front = float(df_exp_condition['absorptivity_front'])
    absorptivity_back = float(df_exp_condition['absorptivity_back'])
    absorptivity_solar = float(df_exp_condition['absorptivity_solar'])
    x0 = int(df_exp_condition['x0_pixels'])
    y0 = int(df_exp_condition['y0_pixels'])
    N_div = int(df_exp_condition['N_div'])
    N_Rs = int(df_exp_condition['N_Rs_pixels'])

    alpha_r_A_ref = float(df_sample_cp_rho_alpha['alpha_r_A'])

    alpha_r_B_ref = float(df_sample_cp_rho_alpha['alpha_r_B'])

    mcmc_mode = df_exp_condition['analysis_mode']

    dump_file_path_mcmc_results = code_directory + "mcmc_results_dump//" + df_exp_condition[
        'rec_name'] + '_R0{:}{:}{:}_{:}{:}{:}{:}{:}{:}{:}_Tr{:}_{:}{:}_{:}c{:}_{:}{:}{:}{:}{:}{:}{:}_P4D{:}_T{:}{:}'.format(
        int(R0), int(R_analysis), int(gap), int(emissivity_front * 100), int(emissivity_back * 100),
        int(absorptivity_front * 100), int(absorptivity_back * 100), int(absorptivity_solar * 100),
        N_Rs, int(mcmc_setting['N_total_mcmc_samples']), int(mcmc_setting['T_regularization'] * 1000),
        x0, y0, mcmc_mode, mcmc_setting['chain_num'], LB_file_name, int(1000 * mcmc_setting['sigma_s_prior_range'][0]),
        int(1000 * mcmc_setting['sigma_s_prior_range'][1]), int(mcmc_setting['alpha_A_prior_range'][0]),
        int(mcmc_setting['alpha_A_prior_range'][1]), int(mcmc_setting['alpha_B_prior_range'][0] / 100),
        int(mcmc_setting['alpha_B_prior_range'][1] / 100), N_div, int(mcmc_setting['T_bias_prior_range'][0] / 10),
        int(mcmc_setting['T_bias_prior_range'][1] / 10))

    accepted_samples_array = pickle.load(open(dump_file_path_mcmc_results, 'rb'))
    if mcmc_mode == 'RW_Metropolis' or mcmc_mode == 'RW_Metropolis_P4':
        n_burn = int(0.3 * len(accepted_samples_array.T[0]))
    else:
        n_burn = int(0.05 * len(accepted_samples_array.T[0]))

    accepted_samples_trim = accepted_samples_array[n_burn:]

    Nr = int(df_exp_condition['Nr_pixels'])
    pr = float(df_exp_condition['sample_radius(m)']) / (Nr - 1)
    f_heating = float(df_exp_condition['f_heating'])
    exp_amp_phase_extraction_method = df_exp_condition['exp_amp_phase_extraction_method']

    colors = ['red', 'black', 'green', 'blue', 'orange', 'magenta', 'brown', 'yellow', 'purple', 'cornflowerblue']

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

    output_name = rec_name
    path = data_directory + str(rec_name) + "//"
    df_temperature_list_all_ranges, df_temperature = radial_temperature_average_disk_sample_several_ranges(x0, y0, Nr,
                                                                                                           angle_range,
                                                                                                           pr, path,
                                                                                                           rec_name,
                                                                                                           output_name,
                                                                                                           'MA', 2,
                                                                                                           code_directory)

    # f, axes = plt.subplots(4, 3, figsize=(26, 24))
    f, axes = plt.subplots(5, 4, figsize=(30, 38))

    label_font_size = 18

    axes[0, 0].hist(accepted_samples_trim.T[0], bins=15)
    axes[0, 0].set_xlabel(r'$\sigma_{solar}$(m)', fontsize=label_font_size, fontweight='bold')

    axes[0, 1].hist(accepted_samples_trim.T[1], bins=15)
    axes[0, 1].set_xlabel(r'$A_{\alpha}$ (s/m$^2$K)', fontsize=label_font_size, fontweight='bold')

    axes[0, 2].hist(accepted_samples_trim.T[2], bins=15)
    axes[0, 2].set_xlabel(r'$B_{\alpha}$ (s/m$^2$)', fontsize=label_font_size, fontweight='bold')

    axes[0, 3].hist(accepted_samples_trim.T[3], bins=15)
    axes[0, 3].set_xlabel('$T_{bias}$ (K)', fontsize=label_font_size, fontweight='bold')

    axes[1, 0].plot(accepted_samples_trim[:, 0])
    axes[1, 0].set_ylabel(r'$\sigma_{solar}$(m)', fontsize=label_font_size, fontweight='bold')
    axes[1, 0].set_xlabel('sample num', fontsize=label_font_size, fontweight='bold')

    axes[1, 1].plot(accepted_samples_trim[:, 1])
    axes[1, 1].set_ylabel(r'$A_{\alpha}$ (s/m$^2$K)', fontsize=label_font_size, fontweight='bold')
    axes[1, 1].set_xlabel('sample num', fontsize=label_font_size, fontweight='bold')

    axes[1, 2].plot(accepted_samples_trim[:, 2])
    axes[1, 2].set_ylabel(r'$B_{\alpha}$ (s/m$^2$)', fontsize=label_font_size, fontweight='bold')
    axes[1, 2].set_xlabel('sample num', fontsize=label_font_size, fontweight='bold')

    axes[1, 3].plot(accepted_samples_trim[:, 3])
    axes[1, 3].set_ylabel('$T_{bias}$ (K)', fontsize=label_font_size, fontweight='bold')
    axes[1, 3].set_xlabel('sample num', fontsize=label_font_size, fontweight='bold')

    T_array = np.linspace(df_temperature.iloc[:, R0 + R_analysis].mean(), df_temperature.iloc[:, R0].mean(), 4)
    alpha_T_array = np.zeros((len(T_array), len(accepted_samples_trim)))

    for i in range(len(accepted_samples_trim)):
        # alpha_T_array[:,i] = 1/(accepted_samples_trim[i][1]*T_array+accepted_samples_trim[i][2])
        alpha_T_array[:, i] = 1 / (accepted_samples_trim[i][1] * T_array + accepted_samples_trim[i][2])

    alpha_T_array_mean = alpha_T_array.mean(axis=1)
    alpha_T_array_std = alpha_T_array.std(axis=1)
    alpha_std_to_mean_avg = np.mean(alpha_T_array_std / alpha_T_array_mean)

    alpha_reference = 1 / (alpha_r_A_ref * T_array + alpha_r_B_ref)

    # f_posterior_mean_vs_reference = plt.figure(figsize=(7, 5))
    axes[2, 0].fill_between(T_array, alpha_T_array_mean - 3 * alpha_T_array_std,
                            alpha_T_array_mean + 3 * alpha_T_array_std,
                            alpha=0.2)
    axes[2, 0].plot(T_array, alpha_T_array_mean, color='k', label='Bayesian posterior distribution')
    axes[2, 0].errorbar(T_array, alpha_reference, yerr=alpha_reference * 0.05,
                        label='reference value', capsize=5, elinewidth=2, color='green', fmt='o')

    # plot(T_array, alpha_reference, color='r', label='reference value')
    # axes[2, 0].set_ylim([0,2.1e-5])
    axes[2, 0].set_xlabel('Temperature (K)', fontsize=label_font_size, fontweight='bold')
    axes[2, 0].set_ylabel(r'Thermal diffusivity (m$^2$/s)', fontsize=label_font_size, fontweight='bold')
    # axes[2, 0].yaxis.set_major_formatter(MathTextSciFormatter("%1.2e"))
    # axes[2, 0].yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    axes[2, 0].legend(prop={'weight': 'bold', 'size': 14})
    alpha_T_array_mean = alpha_T_array.mean(axis=1)

    def residual(params, x, data):
        alpha_A = params['alpha_A']
        alpha_B = params['alpha_B']

        model = 1 / (alpha_A * x + alpha_B)

        return model - data

    params = Parameters()
    params.add('alpha_A', value=1e-7)
    params.add('alpha_B', value=2e-2)

    out = minimize(residual, params, args=(T_array, alpha_T_array_mean))

    alpha_A_posterior_mean = out.params['alpha_A'].value
    alpha_B_posterior_mean = out.params['alpha_B'].value

    sigma_s_posterior_mean = accepted_samples_trim.T[0].mean()
    T_bias_posterior_mean = accepted_samples_trim.T[3].mean()

    dump_file_path_surrogate = code_directory + "surrogate_dump//" + df_exp_condition[
        'rec_name'] + '_R0{:}Ra{:}gap{:}_{:}{:}{:}{:}{:}_NRs{:}ord{:}_x0{:}y0{:}{:}_{:}{:}_{:}{:}_{:}{:}_P4D{:}_T{:}{:}'.format(
        int(R0), int(R_analysis), int(gap), int(emissivity_front * 100), int(emissivity_back * 100),
        int(absorptivity_front * 100), int(absorptivity_back * 100), int(absorptivity_solar * 100),
        int(N_Rs), mcmc_setting['PC_order'], x0, y0, LB_file_name, int(1000 * mcmc_setting['sigma_s_prior_range'][0]),
        int(1000 * mcmc_setting['sigma_s_prior_range'][1]), int(mcmc_setting['alpha_A_prior_range'][0]),
        int(mcmc_setting['alpha_A_prior_range'][1]), int(mcmc_setting['alpha_B_prior_range'][0] / 100),
        int(mcmc_setting['alpha_B_prior_range'][1] / 100), N_div, int(mcmc_setting['T_bias_prior_range'][0] / 10),
        int(mcmc_setting['T_bias_prior_range'][1] / 10))

    amp_ratio_approx, phase_diff_approx, ss_temp_approx = pickle.load(open(dump_file_path_surrogate, 'rb'))

    axes[2, 1].errorbar(df_amplitude_phase_measurement['r_pixels'], df_amplitude_phase_measurement['amp_ratio'],
                        yerr=df_amplitude_phase_measurement['dA'] * 3,
                        label='measurement', capsize=5, elinewidth=2, color='red', fmt='o')
    axes[2, 1].plot(df_amplitude_phase_measurement['r_pixels'],
                    amp_ratio_approx(sigma_s_posterior_mean, alpha_A_posterior_mean, alpha_B_posterior_mean,
                                     T_bias_posterior_mean),
                    label='Bayesian fitting', color='black')
    axes[2, 1].set_xlabel('R (node num)', fontsize=label_font_size, fontweight='bold')
    axes[2, 1].set_ylabel('Amplitude ratio', fontsize=label_font_size, fontweight='bold')
    # axes[2, 1].set_ylim([0,2.1e-5])
    axes[2, 1].legend(prop={'weight': 'bold', 'size': 14})

    # axes[2, 2].scatter(df_amplitude_phase_measurement['r_pixels'], df_amplitude_phase_measurement['phase_diff'],
    #                 label='measurement',color = 'red')
    axes[2, 2].errorbar(df_amplitude_phase_measurement['r_pixels'], df_amplitude_phase_measurement['phase_diff'],
                        yerr=df_amplitude_phase_measurement['dP'] * 3,
                        label='measurement', capsize=5, elinewidth=2, color='red', fmt='o')
    axes[2, 2].plot(df_amplitude_phase_measurement['r_pixels'],
                    phase_diff_approx(sigma_s_posterior_mean, alpha_A_posterior_mean, alpha_B_posterior_mean,
                                      T_bias_posterior_mean),
                    label='Bayesian fitting', color='black')
    axes[2, 2].set_xlabel('R (node num)', fontsize=label_font_size, fontweight='bold')
    axes[2, 2].set_ylabel('Phase difference', fontsize=label_font_size, fontweight='bold')
    axes[2, 2].legend(prop={'weight': 'bold', 'size': 14})

    def residual_temp_shift(params, f_heating, df_temperature, data):

        phi_shift = params[0]
        #print(df_temperature.shape)
        temp_shift = df_temperature.query('reltime>{:} and reltime<{:}'.format(phi_shift, phi_shift + 2 / f_heating))
        #print(temp_shift)
        T_ss_measured_R0 = temp_shift.iloc[:, R0] - temp_shift.iloc[:, R0].mean()
        time_measurement = temp_shift['reltime'] - temp_shift['reltime'].min()
        data_ = data - np.mean(data)
        #print(time_measurement)
        f_interp_measurements = interp1d(time_measurement, T_ss_measured_R0, kind='cubic')
        # x_new = np.range(len(data))
        time_ss_simulation = np.linspace(0, max(time_measurement), len(data))
        T_ss_measurement_interpolated = f_interp_measurements(time_ss_simulation)

        # print(-np.sum(T_ss_measurement_interpolated*data_))
        return -np.sum(T_ss_measurement_interpolated * data_)

    T_ss_RO = ss_temp_approx(sigma_s_posterior_mean, alpha_A_posterior_mean, alpha_B_posterior_mean,
                             T_bias_posterior_mean)[:, R0]
    time_ss_RO = np.linspace(0, 2 / f_heating, len(T_ss_RO))
    # print(T_ss_RO)
    out_phi_shift = minimize2(residual_temp_shift, [30], args=(f_heating, df_temperature, T_ss_RO), method='Nelder-Mead')

    phi_shift = out_phi_shift.x[0]
    #     params = Parameters()
    #     params.add('phi_shift',value = 20,min = 0, max = 100)
    #     out_phi_shift = minimize(residual_temp_shift, params, args=(f_heating,T_ss_RO))
    #     phi_shift = out_phi_shift.params['phi_shift'].value

    #print('The optimized result is {:}'.format(phi_shift))

    # T_ss_RO = ss_temp_approx(sigma_s_posterior_mean, alpha_A_posterior_mean, alpha_B_posterior_mean, T_bias_posterior_mean)[:, R0]
    # time_ss_RO = np.linspace(0,2/f_heating,len(T_ss_RO))
    axes[2, 3].plot(time_ss_RO, T_ss_RO, label='simulation R = {:}'.format(R0))

    temp_shift = df_temperature.query('reltime>{:} and reltime<{:}'.format(phi_shift, phi_shift + 2 / f_heating))
    time_ss_measured_R0 = temp_shift['reltime'] - temp_shift['reltime'].min()

    T_ss_measured_R0 = temp_shift.iloc[:, R0]

    # time_ss_measured_R0 = df_temperature.query('reltime<{:}'.format(2/f_heating))['reltime']
    # T_ss_measured_R0 = df_temperature.iloc[:len(time_ss_measured_R0), R0]
    axes[2, 3].plot(time_ss_measured_R0, T_ss_measured_R0, label='measurement R = {:}'.format(R0))

    T_ss_RN = ss_temp_approx(sigma_s_posterior_mean, alpha_A_posterior_mean, alpha_B_posterior_mean,
                             T_bias_posterior_mean)[:, R0 + R_analysis]
    time_ss_RN = np.linspace(0, 2 / f_heating, len(T_ss_RN))
    axes[2, 3].plot(time_ss_RN, T_ss_RN, label='simulation R = {:}'.format(R0 + R_analysis))

    time_ss_measured_RN = time_ss_measured_R0
    T_ss_measured_RN = temp_shift.iloc[:, R0 + R_analysis]
    # time_ss_measured_RN = df_temperature.query('reltime<{:}'.format(2/f_heating))['reltime']
    # T_ss_measured_RN = df_temperature.iloc[:len(time_ss_measured_RN), R0+R_analysis]
    axes[2, 3].plot(time_ss_measured_RN, T_ss_measured_RN, label='measurement R = {:}'.format(R0 + R_analysis))

    # axes[2, 3].plot(df_temperature.iloc[:, R0 + R_analysis], label='measured R = {:}'.format(R0 + R_analysis))
    axes[2, 3].set_ylabel('Temperature (K)', fontsize=label_font_size, fontweight='bold')
    axes[2, 3].set_xlabel('Time (s)', fontsize=label_font_size, fontweight='bold')
    axes[2, 3].legend(prop={'weight': 'bold', 'size': 14})

    def acf(x, length):
        return np.array([1] + [np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, length)])

    def auto_correlation_function(trace, lags):
        autocorr_trace = acf(trace, lags)
        return autocorr_trace

    # def plot_auto_correlation(trace, lags):

    lags = 100

    autocorr_trace_sigma = auto_correlation_function(accepted_samples_trim[:, 0], lags)
    axes[3, 0].plot(autocorr_trace_sigma)
    axes[3, 0].set_xlabel('lags N', fontsize=label_font_size)
    axes[3, 0].set_ylabel('sigma_s', fontsize=label_font_size)

    autocorr_trace_alpha_A = auto_correlation_function(accepted_samples_trim[:, 1], lags)
    axes[3, 1].plot(autocorr_trace_alpha_A)
    axes[3, 1].set_xlabel('lags N', fontsize=label_font_size)
    axes[3, 1].set_ylabel('alpha_A', fontsize=label_font_size)

    autocorr_trace_alpha_B = auto_correlation_function(accepted_samples_trim[:, 2], lags)
    axes[3, 2].plot(autocorr_trace_alpha_B)
    axes[3, 2].set_xlabel('lags N', fontsize=label_font_size)
    axes[3, 2].set_ylabel('alpha_B', fontsize=label_font_size)

    autocorr_T_bias = auto_correlation_function(accepted_samples_trim[:, 3], lags)
    axes[3, 3].plot(autocorr_T_bias)
    axes[3, 3].set_xlabel('lags N', fontsize=label_font_size)
    axes[3, 3].set_ylabel('T_bias', fontsize=label_font_size)

    df_temperature_angle_list = []
    df_amp_phase_angle_list = []
    for j, angle in enumerate(angle_range):
        # note radial_temperature_average_disk_sample automatically checks if a dump file exist
        df_temperature_ = df_temperature_list_all_ranges[j]
        df_amplitude_phase_measurement_ = batch_process_horizontal_lines(df_temperature_, f_heating, R0, gap,
                                                                         R_analysis,
                                                                         exp_amp_phase_extraction_method)
        df_temperature_angle_list.append(df_temperature_)
        df_amp_phase_angle_list.append(df_amplitude_phase_measurement_)

        axes[4, 2].scatter(df_amplitude_phase_measurement_['r'],
                           df_amplitude_phase_measurement_['amp_ratio'], facecolors='none',
                           s=60, linewidths=2, edgecolor=colors[j],
                           label=str(angle[0]) + ' to ' + str(angle[1]) + ' Degs')

    axes[4, 2].set_xlabel('R (pixels)', fontsize=label_font_size, fontweight='bold')
    axes[4, 2].set_ylabel('Amplitude Ratio', fontsize=label_font_size, fontweight='bold')
    axes[4, 2].legend(prop={'weight': 'bold', 'size': 12})

    for j, angle in enumerate(angle_range):
        df_temperature_ = df_temperature_angle_list[j]
        df_amplitude_phase_measurement_ = df_amp_phase_angle_list[j]
        axes[4, 3].scatter(df_amplitude_phase_measurement_['r'],
                           df_amplitude_phase_measurement_['phase_diff'], facecolors='none',
                           s=60, linewidths=2, edgecolor=colors[j],
                           label=str(angle[0]) + ' to ' + str(angle[1]) + ' Degs')

    axes[4, 3].set_xlabel('R (pixels)', fontsize=label_font_size, fontweight='bold')
    axes[4, 3].set_ylabel('Phase difference (rad)', fontsize=label_font_size, fontweight='bold')
    axes[4, 3].legend(prop={'weight': 'bold', 'size': 12})

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
        # N_mid = 20
        file_name_1 = [path + x for x in os.listdir(path)][N_mid]
        n2 = file_name_1.rfind('//')
        n3 = file_name_1.rfind('.csv')
        frame_num_mid = file_name_1[n2 + 2:n3]

        df_mid_frame = pd.read_csv(file_name_1, skiprows=5, header=None)

        temp_dump = [df_first_frame, df_mid_frame, frame_num_first, frame_num_mid]

        pickle.dump(temp_dump, open(rep_csv_dump_path, "wb"))

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
    # ax = plt.gca()
    CS = axes[4, 0].contour(X, Y, Z, 18)
    axes[4, 0].plot([xmin, xmax], [y0, y0], ls='-.', color='k', lw=2)  # add a horizontal line cross x0,y0
    axes[4, 0].plot([x0, x0], [ymin, ymax], ls='-.', color='k', lw=2)  # add a vertical line cross x0,y0

    R_angle_show = R0 + R_analysis

    circle1 = plt.Circle((x0, y0), R0, edgecolor='r', fill=False, linewidth=3, linestyle='-.')
    circle2 = plt.Circle((x0, y0), R0 + R_analysis, edgecolor='k', fill=False, linewidth=3, linestyle='-.')

    circle3 = plt.Circle((x0, y0), int(0.01 / pr), edgecolor='black', fill=False, linewidth=3, linestyle='dotted')

    axes[4, 0].invert_yaxis()
    axes[4, 0].clabel(CS, inline=1, fontsize=12, manual=manual_locations)
    axes[4, 0].add_artist(circle1)
    axes[4, 0].add_artist(circle2)
    axes[4, 0].add_artist(circle3)

    axes[4, 0].set_xlabel('x (pixels)', fontsize=label_font_size, fontweight='bold')
    axes[4, 0].set_ylabel('y (pixels)', fontsize=label_font_size, fontweight='bold')
    # axes[4, 2].set_title('x0 = {}, y0 = {}, R0 = {}'.format(x0, y0, R0), fontsize=12, fontweight='bold')

    for j, angle in enumerate(angle_range):
        axes[4, 0].plot([x0, x0 + R_angle_show * np.cos(angle[0] * np.pi / 180)],
                        [y0, y0 + R_angle_show * np.sin(angle[0] * np.pi / 180)], ls='-.', color='blue', lw=2)

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
    # ax = plt.gca()
    CS = axes[4, 1].contour(X, Y, Z, 18)
    axes[4, 1].plot([xmin, xmax], [y0, y0], ls='-.', color='k', lw=2)  # add a horizontal line cross x0,y0
    axes[4, 1].plot([x0, x0], [ymin, ymax], ls='-.', color='k', lw=2)  # add a vertical line cross x0,y0

    R_angle_show = R0 + R_analysis

    circle1 = plt.Circle((x0, y0), R0, edgecolor='r', fill=False, linewidth=3, linestyle='-.')
    circle2 = plt.Circle((x0, y0), R0 + R_analysis, edgecolor='k', fill=False, linewidth=3, linestyle='-.')

    circle3 = plt.Circle((x0, y0), int(0.01 / pr), edgecolor='black', fill=False, linewidth=3, linestyle='dotted')

    axes[4, 1].invert_yaxis()
    axes[4, 1].clabel(CS, inline=1, fontsize=12, manual=manual_locations)
    axes[4, 1].add_artist(circle1)
    axes[4, 1].add_artist(circle2)
    axes[4, 1].add_artist(circle3)

    axes[4, 1].set_xlabel('x (pixels)', fontsize=label_font_size, fontweight='bold')
    axes[4, 1].set_ylabel('y (pixels)', fontsize=label_font_size, fontweight='bold')
    # axes[4, 3].set_title('x0 = {}, y0 = {}, R0 = {}'.format(x0, y0, R0), fontsize=12, fontweight='bold')

    for j, angle in enumerate(angle_range):
        axes[4, 1].plot([x0, x0 + R_angle_show * np.cos(angle[0] * np.pi / 180)],
                        [y0, y0 + R_angle_show * np.sin(angle[0] * np.pi / 180)], ls='-.', color='blue', lw=2)

    for i in range(5):
        for j in range(4):
            for tick in axes[i, j].xaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize=12)
                tick.label.set_fontweight('bold')
            for tick in axes[i, j].yaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize=12)
                tick.label.set_fontweight('bold')
    f.suptitle("{},f_heating={:.2e}, VDC={}, R0={}, R_a={}, e_f={}, e_b={}, a_solar={}, x0={}, y0={}".format(rec_name,
                                                                                                             float(
                                                                                                                 df_exp_condition[
                                                                                                                     'f_heating']),
                                                                                                             float(
                                                                                                                 df_exp_condition[
                                                                                                                     'V_DC']),
                                                                                                             R0,
                                                                                                             R_analysis,
                                                                                                             int(
                                                                                                                 100 * float(
                                                                                                                     df_exp_condition[
                                                                                                                         'emissivity_front'])),
                                                                                                             int(
                                                                                                                 100 * float(
                                                                                                                     df_exp_condition[
                                                                                                                         'emissivity_back'])),
                                                                                                             int(
                                                                                                                 100 * float(
                                                                                                                     df_exp_condition[
                                                                                                                         'absorptivity_solar'])),
                                                                                                             x0, y0),
               y=0.91, fontsize=16)
    # plt.show()

    # print("------------------------------------Happy 2021!------------------------------------")
    print(
        rec_name + " is shown here, and with mean sigma_s = {:.2e}, alpha_A = {:.2e}, alpha_B = {:.2e}, T_bias = {:.2e}".format(
            accepted_samples_array.T[0].mean(), accepted_samples_array.T[1].mean(), accepted_samples_array.T[2].mean(),
            accepted_samples_array.T[3].mean()))
    # return f

    return accepted_samples_trim, alpha_A_posterior_mean, alpha_B_posterior_mean, sigma_s_posterior_mean, T_bias_posterior_mean, alpha_std_to_mean_avg, np.min(
        T_array), np.max(T_array), amp_ratio_approx, phase_diff_approx, ss_temp_approx


def mcmc_result_vs_reference_visualization(df_result_summary, f_heating, R0_max, mcmc_method, alpha_A, alpha_B,ylim):
    df_result_summary_ = df_result_summary.query("f_heating =={:} and R0_pixels<{:}".format(f_heating, R0_max))

    plt.figure(figsize=(8, 6))
    T_list = []
    alpha_list = []
    for index, item in df_result_summary_.iterrows():
        # print(item)
        T = np.linspace(item['T_min'], item['T_max'], 3)
        alpha = 1 / (T * item['alpha_A'] + item['alpha_B'])
        plt.fill_between(T, alpha - alpha * item['alpha_std_to_mean'] * 3,
                         alpha + alpha * item['alpha_std_to_mean'] * 3,
                         label='R0 = {:}, V = {:}'.format(item['R0_pixels'], item['V_DC']), alpha=0.5)
        T_list = T_list + list(T)

    T_ref = np.linspace(np.sort(np.array(T_list))[0], np.sort(np.array(T_list))[-1], 10)
    plt.errorbar(T_ref, 1 / (T_ref * alpha_A + alpha_B), yerr=1 / (T_ref * alpha_A + alpha_B) * 0.05, label='Reference',
                 capsize=5, elinewidth=2)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=10)
        tick.label.set_fontweight('bold')

    plt.grid()
    plt.legend(prop={'weight': 'bold', 'size': 12})
    plt.ylim(ylim)
    plt.xlabel('Temperature(K)', fontsize=14, fontweight='bold')
    plt.ylabel('Thermal diffusivity (m2/s)', fontsize=14, fontweight='bold')
    plt.title("f_heating = {:} Hz, {:}, with T_bias".format(f_heating, mcmc_method), fontsize=12, fontweight='bold')
    plt.show()



# def parallel_batch_show_mcmc_results_P4(df_exp_condition_spreadsheet_filename, code_directory, df_temperature_list,
#                                      df_amplitude_phase_measurement_list, df_sample_cp_rho_alpha, mcmc_setting):
#     df_exp_condition_spreadsheet = pd.read_csv(
#         code_directory + "batch process information//" + df_exp_condition_spreadsheet_filename)
#     for i in tqdm(range(len(df_exp_condition_spreadsheet))):
#         show_mcmc_results_one_case_P4(df_exp_condition_spreadsheet.iloc[i, :], code_directory,
#                                    df_temperature_list[i], df_amplitude_phase_measurement_list[i],
#                                    df_sample_cp_rho_alpha, mcmc_setting)

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
    df_exp_condition_spreadsheet = pd.read_csv(
        code_directory + "batch process information//" + df_exp_condition_spreadsheet_filename)

    steady_state_figure_list = []
    df_temperature_list = []

    for i in range(len(df_exp_condition_spreadsheet)):

        df_exp_condition = df_exp_condition_spreadsheet.iloc[i, :]

        rec_name = df_exp_condition['rec_name']
        path = data_directory + str(rec_name) + "//"

        output_name = rec_name

        method = "MA"  # default uses Mosfata's code

        x0 = df_exp_condition['x0_pixels']  # in pixels
        y0 = df_exp_condition['y0_pixels']  # in pixels
        Rmax = df_exp_condition['Nr_pixels']  # in pixels
        # x0,y0,N_Rmax,pr,path,rec_name,output_name,method,num_cores
        pr = df_exp_condition['pr']
        # df_temperature = radial_temperature_average_disk_sample_several_ranges(x0,y0,Rmax,[[0,np.pi/3],[2*np.pi/3,np.pi],[4*np.pi/3,5*np.pi/3],[5*np.pi/3,2*np.pi]],pr,path,rec_name,output_name,method,num_cores)

        # After obtaining temperature profile, next we obtain amplitude and phase
        f_heating = df_exp_condition['f_heating']
        # 1cm ->35
        R0 = df_exp_condition['R0_pixels']
        gap = df_exp_condition['gap_pixels']
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
    #df_exp_condition_spreadsheet = pd.read_excel(code_directory+"batch process information//" + df_exp_condition_spreadsheet_filename)
    df_exp_condition_spreadsheet = pd.read_csv(code_directory + "batch process information//" + df_exp_condition_spreadsheet_filename)
    diagnostic_figure_list = []
    df_temperature_list = []
    #df_amplitude_phase_measurement_list = []
    df_amplitude_phase_measurement_averaged_list = []

    for i in range(len(df_exp_condition_spreadsheet)):

        df_exp_condition = df_exp_condition_spreadsheet.iloc[i, :]

        rec_name = df_exp_condition['rec_name']
        path = data_directory + str(rec_name) + "//"

        output_name = rec_name
        # num_cores = df_exp_condition['num_cores']

        method = "MA"  # default uses Mosfata's code
        # print(method)

        x0 = int(df_exp_condition['x0_pixels'])  # in pixels
        y0 = int(df_exp_condition['y0_pixels'])  # in pixels
        Rmax = int(df_exp_condition['Nr_pixels'])  # in pixels
        # x0,y0,N_Rmax,pr,path,rec_name,output_name,method,num_cores
        pr = df_exp_condition['pr']
        # df_temperature = radial_temperature_average_disk_sample_several_ranges(x0,y0,Rmax,[[0,np.pi/3],[2*np.pi/3,np.pi],[4*np.pi/3,5*np.pi/3],[5*np.pi/3,2*np.pi]],pr,path,rec_name,output_name,method,num_cores)

        # After obtaining temperature profile, next we obtain amplitude and phase
        f_heating = df_exp_condition['f_heating']
        # 1cm ->35
        R0 = int(df_exp_condition['R0_pixels'])
        gap = int(df_exp_condition['gap_pixels'])
        # Rmax = 125
        R_analysis = int(df_exp_condition['R_analysis_pixels'])
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
            angle_range.append([int(float(element_before_comma)),int(float(element_after_comma))])

        # sum_std, diagnostic_figure = check_angular_uniformity(x0, y0, Rmax, pr, path, rec_name, output_name, method,
        #                                                       num_cores, f_heating, R0, gap, R_analysis,angle_range,focal_plane_location, VDC, exp_amp_phase_extraction_method,code_directory)
        #
        # diagnostic_figure_list.append(diagnostic_figure)

        df_temperature_list_all_ranges, df_temperature = radial_temperature_average_disk_sample_several_ranges(x0, y0, Rmax, angle_range, pr, path,
                                                                               rec_name, output_name, method, num_cores,code_directory)

        df_amplitude_phase_angle_range_list = []

        for i in range(len(angle_range)):
            df_amplitude_phase_measurement_temp = batch_process_horizontal_lines(df_temperature_list_all_ranges[i], f_heating, R0, gap,
                                                                            R_analysis,
                                                                            exp_amp_phase_extraction_method)
            df_amplitude_phase_angle_range_list.append(df_amplitude_phase_measurement_temp)

        d_A = np.std(np.array(
            [df_amplitude_phase_measurement_list_['amp_ratio'] for df_amplitude_phase_measurement_list_ in
             df_amplitude_phase_angle_range_list]), axis=0)
        d_P = np.std(np.array(
            [df_amplitude_phase_measurement_list_['phase_diff'] for df_amplitude_phase_measurement_list_ in
             df_amplitude_phase_angle_range_list]), axis=0)

        amp_ratio = np.mean(np.array(
            [df_amplitude_phase_measurement_list_['amp_ratio'] for df_amplitude_phase_measurement_list_ in
             df_amplitude_phase_angle_range_list]), axis=0)
        phase_diff = np.mean(np.array(
            [df_amplitude_phase_measurement_list_['phase_diff'] for df_amplitude_phase_measurement_list_ in
             df_amplitude_phase_angle_range_list]), axis=0)


        df_amplitude_phase_measurement_averaged = pd.DataFrame({'r_pixels':df_amplitude_phase_measurement_temp['r'],
                                                                'r_ref_pixels':df_amplitude_phase_measurement_temp['r_ref'],'amp_ratio':amp_ratio,'phase_diff':phase_diff,
                                                                'dA':d_A,'dP':d_P})

        # df_amplitude_phase_measurement = batch_process_horizontal_lines(df_temperature, f_heating, R0, gap, R_analysis,
        #                                                                 exp_amp_phase_extraction_method)
        df_temperature_list.append(df_temperature)
        # df_amplitude_phase_measurement_list.append(df_amplitude_phase_measurement)

        df_amplitude_phase_measurement_averaged_list.append(df_amplitude_phase_measurement_averaged)

    return df_temperature_list, df_amplitude_phase_measurement_averaged_list

#
# def parallel_regression_batch_experimental_results(df_exp_condition_spreadsheet_filename, data_directory, num_cores,code_directory,df_sample_cp_rho_alpha_all,df_thermal_diffusivity_temperature_all, df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all):
#     df_exp_condition_spreadsheet = pd.csv(code_directory+"batch process information//" + df_exp_condition_spreadsheet_filename)
#
#     df_temperature_list, df_amplitude_phase_measurement_list = parallel_temperature_average_batch_experimental_results(df_exp_condition_spreadsheet_filename, data_directory, num_cores,code_directory)
#
#     joblib_output = Parallel(n_jobs=num_cores, verbose=0)(
#         delayed(high_T_Angstrom_execute_one_case)(df_exp_condition_spreadsheet.iloc[i, :], data_directory,code_directory, df_amplitude_phase_measurement_list[i], df_temperature_list[i],df_sample_cp_rho_alpha_all,df_thermal_diffusivity_temperature_all, df_solar_simulator_VQ,sigma_df,df_view_factor,df_LB_details_all) for i in
#         tqdm(range(len(df_exp_condition_spreadsheet))))
#
#     pickle.dump(joblib_output,open(code_directory+"result cache dump//regression_results_" + df_exp_condition_spreadsheet_filename, "wb"))
#


def parallel_batch_experimental_results_mcmc_implicit_train_surrogate(df_exp_condition_spreadsheet_filename, code_directory,df_temperature_list,df_sample_cp_rho_alpha, df_solar_simulator_VQ,sigma_df,df_view_factor,mcmc_setting):

    df_exp_condition_spreadsheet = pd.read_csv(code_directory+"batch process information//" + df_exp_condition_spreadsheet_filename)


    # df_temperature_list, df_amplitude_phase_measurement_list = parallel_temperature_average_batch_experimental_results(df_exp_condition_spreadsheet_filename, data_directory, num_cores,code_directory)
    for i in tqdm(range(len(df_exp_condition_spreadsheet))):
        high_T_Angstrom_execute_one_case_mcmc_train_surrogate(df_exp_condition_spreadsheet.iloc[i, :], code_directory,
                                                              df_temperature_list[i], df_sample_cp_rho_alpha,
                                                              df_solar_simulator_VQ, sigma_df, df_view_factor,
                                                              mcmc_setting)


def parallel_batch_experimental_results_mcmc_implicit_train_surrogate_P4(df_exp_condition_spreadsheet_filename,
                                                                      code_directory, df_temperature_list,
                                                                      df_sample_cp_rho_alpha, df_solar_simulator_VQ,
                                                                      sigma_df, df_view_factor, mcmc_setting):
    df_exp_condition_spreadsheet = pd.read_csv(
        code_directory + "batch process information//" + df_exp_condition_spreadsheet_filename)

    # df_temperature_list, df_amplitude_phase_measurement_list = parallel_temperature_average_batch_experimental_results(df_exp_condition_spreadsheet_filename, data_directory, num_cores,code_directory)
    for i in tqdm(range(len(df_exp_condition_spreadsheet))):
        high_T_Angstrom_execute_one_case_mcmc_train_surrogate_P4(df_exp_condition_spreadsheet.iloc[i, :], code_directory,
                                                              df_temperature_list[i], df_sample_cp_rho_alpha,
                                                              df_solar_simulator_VQ, sigma_df, df_view_factor,
                                                              mcmc_setting)


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

def display_high_dimensional_regression_results_one_row_mcmc(x_name, y_name, column_name, series_name, df_results_all, ylim):
    column_items = np.unique(df_results_all[column_name])
    series_items = np.unique(df_results_all[series_name])
    f, axes = plt.subplots(1, len(column_items),
                           figsize=(int(len(column_items) * 5), 5),sharex=True, sharey=True)
    for j, column in enumerate(column_items):
        df_results_all_ = df_results_all.query("{} == {}".format(column_name, column))

        for series in series_items:
            if type(series) == str:
                df_ = df_results_all_.query("{}=='{}'".format(series_name, series))
                #axes[j].scatter(df_[x_name], df_[y_name], label="{} = '{}'".format(series_name, series))
                axes[j].errorbar(df_[x_name], df_[y_name], 2 * df_['parameter_std'],
                             label="'{}' = '{}'".format(series_name, series), fmt='.')
            else:
                df_ = df_results_all_.query("{}=={}".format(series_name, series))
                #axes[j].scatter(df_[x_name], df_[y_name], label="{} = {:.1E}".format(series_name, series))
                axes[j].errorbar(df_[x_name], df_[y_name], 2 * df_['parameter_std'],
                             label="{} = {:.1E}".format(series_name, series), fmt='.')

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

def display_high_dimensional_regression_results_one_row_one_column_mcmc(x_name, y_name, series_name, df_results_all, ylim):
    # column_items = np.unique(df_results_all[column_name])
    series_items = np.unique(df_results_all[series_name])
    plt.figure(figsize=(8, 6))

    for series in series_items:

        if type(series) == str:
            df_ = df_results_all.query("{}=='{}'".format(series_name, series))
            plt.errorbar(df_[x_name], df_[y_name],2*df_['parameter_std'], label="'{}' = '{}'".format(series_name, series),fmt='.')
        else:
            df_ = df_results_all.query("{}=={}".format(series_name, series))
            plt.errorbar(df_[x_name], df_[y_name],2*df_['parameter_std'], label="{} = {:.1E}".format(series_name, series),fmt='.')

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


def calculate_total_absorption(R_sample, pr, absorptivity_front, absorptivity_back,T_W1):

    rm_array = np.arange(0,R_sample,pr)
    dr = pr
    W1 = 4e-2 #back side graphite foil length
    R_chamber = 48.65e-3
    A_W1 = np.pi*R_chamber*2*W1
    e_W1 = 0.93
    sigma_sb = 5.67e-8

    Wf = 1e-2 # front side graphite foil length
    A_Wf = np.pi*R_chamber*2*Wf

    A_back = calculator_back_W2_ring_VF(rm_array, dr, 1e-4, W1,
                               R_chamber) * A_W1 * e_W1 * sigma_sb * T_W1 ** 4 * absorptivity_back

    A_front = calculator_back_W2_ring_VF(rm_array, dr, 1e-4, Wf,
                               R_chamber) * A_Wf * e_W1 * sigma_sb * T_W1 ** 4 * absorptivity_front

    A_total = np.sum(A_front) + np.sum(A_back)

    return A_total


def batch_process_steady_state_emission_spread_sheet(data_directory, code_directory,
                                                     df_exp_condition_spreadsheet_filename, num_cores):
    df_exp_condition_spreadsheet = pd.read_csv(
        code_directory + "batch process information//steady emission//" + df_exp_condition_spreadsheet_filename)
    emission_list = []
    surr_absorption_list = []

    for i in range(len(df_exp_condition_spreadsheet)):

        df_exp_condition = df_exp_condition_spreadsheet.iloc[i, :]

        rec_name = df_exp_condition['rec_name']
        print("{} is being processed!".format(rec_name))

        path = data_directory + str(rec_name) + "//"

        output_name = rec_name
        x0 = int(df_exp_condition['x0_pixels'])  # in pixels
        y0 = int(df_exp_condition['y0_pixels'])  # in pixels
        Rmax = int(df_exp_condition['Nr_pixels'])  # in pixels

        # x0,y0,N_Rmax,pr,path,rec_name,output_name,method,num_cores
        pr = float(df_exp_condition['pr'])

        R_sample = Rmax*pr
        # df_temperature = radial_temperature_average_disk_sample_several_ranges(x0,y0,Rmax,[[0,np.pi/3],[2*np.pi/3,np.pi],[4*np.pi/3,5*np.pi/3],[5*np.pi/3,2*np.pi]],pr,path,rec_name,output_name,method,num_cores)

        emissivity_front = float(df_exp_condition['emissivity_front'])
        emissivity_back = float(df_exp_condition['emissivity_back'])
        absorptivity_front = float(df_exp_condition['absorptivity_front'])
        absorptivity_back = float(df_exp_condition['absorptivity_back'])

        T_W1 = float(df_exp_condition['TW1_C'])+273.15

        file_names = [path + x for x in os.listdir(path)]

        joblib_output = Parallel(n_jobs=num_cores)(
            delayed(calculate_total_emission)(file_name, x0, y0, Rmax, pr, emissivity_front, emissivity_back) for
            file_name in tqdm(file_names))

        E_mean = np.mean(joblib_output)
        emission_list.append(E_mean)

        surr_absorption_mean = calculate_total_absorption(R_sample, pr, absorptivity_front, absorptivity_back,T_W1)

        surr_absorption_list.append(surr_absorption_mean)

    df_DC_results = pd.DataFrame({'rec_name': df_exp_condition_spreadsheet['rec_name'],
                                  'focal_shift': df_exp_condition_spreadsheet['focal_shift_cm'],
                                  'V_DC': df_exp_condition_spreadsheet['V_DC'], 'E_total': emission_list,
                                  'emissivity_front': df_exp_condition_spreadsheet['emissivity_front'],
                                  'emissivity_back': df_exp_condition_spreadsheet['emissivity_back'],
                                  'absorptivity_front': df_exp_condition_spreadsheet['absorptivity_front'],
                                  'absorptivity_back': df_exp_condition_spreadsheet['absorptivity_back'],
                                  'A_total':surr_absorption_list,'A_solar':np.array(emission_list)-np.array(surr_absorption_list)})

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
            if parameter_name == 'emissivity_front':
                sample_information['absorptivity_solar'] = DOE_parameter_value
                sample_information['absorptivity_front'] = DOE_parameter_value

        elif parameter_name in vacuum_chamber_setting.keys():
            vacuum_chamber_setting[parameter_name] = DOE_parameter_value
        elif parameter_name in numerical_simulation_setting.keys():
            numerical_simulation_setting[parameter_name] = DOE_parameter_value
        elif parameter_name in solar_simulator_settings.keys():
            solar_simulator_settings[parameter_name] = DOE_parameter_value
        elif parameter_name in light_source_property.keys():
            light_source_property[parameter_name] = DOE_parameter_value
            if parameter_name == 'sigma_s':
                Amax, sigma_s, kvd, bvd = interpolate_light_source_characteristic(sigma_df, df_solar_simulator_VQ,
                                                                                  DOE_parameter_value, numerical_simulation_setting,
                                                                                  vacuum_chamber_setting)
                light_source_property['Amax'] = Amax
                light_source_property['sigma_s'] = sigma_s
                light_source_property['kvd'] = kvd
                light_source_property['bvd'] = bvd


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

    df_results_amp_only = pd.DataFrame(columns=joblib_output[0]['r_pixels'], data = amp_ratio_results)
    df_results_phase_only = pd.DataFrame(columns=joblib_output[0]['r_pixels'], data = phase_diff_results)

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
            if parameter_name == 'alpha_r_B':
                legend_temp = 'Thermal diffusivity: '+ r'$\alpha$'
            elif parameter_name == 'Amax':
                legend_temp = 'Peak nominal intensity: '+ r'$A_{max}$'
            elif parameter_name == 'sigma_s':
                legend_temp = 'Intensity distribution: '+r'$\sigma_{solar}$'
            elif parameter_name == 'emissivity_front':
                legend_temp = 'Front emissivity: '+ r'$\epsilon$'
            elif parameter_name == 'absorptivity_front':
                legend_temp = 'Front absorptivity: '+r'$\eta$'
            elif parameter_name == 'T_sur1':
                legend_temp = 'Front temperature: '+r'$T_0$'

            axes[0, i].scatter(df_main_effect['r'], df_main_effect[parameter_name], label=legend_temp)
            axes[1, i].scatter(df_main_effect_relative['r'], df_main_effect_relative[parameter_name],
                               label=legend_temp)

        axes[0, i].set_xlabel('R (pixels)', fontsize=12, fontweight='bold')
        axes[0, i].set_title("heating frequency = {} Hz".format(f_heating), fontsize=12, fontweight='bold')
        axes[0, i].set_xlim([61,73])

        axes[1, i].set_xlabel('R (pixels)', fontsize=12, fontweight='bold')
        axes[1, i].set_title("heating frequency = {} Hz".format(f_heating), fontsize=12, fontweight='bold')

        if i == 0:
            if y_axis_label =='Amplitude ratio':
                y_axis_label_1 = 'Amplitude ratio main effect'
                y_axis_label_2 = 'Amplitude ratio main effect percentage'

            if y_axis_label =='Phase difference':
                y_axis_label_1 = 'Phase difference main effect'
                y_axis_label_2 = 'Phase difference main effect percentage'
            axes[0, i].set_ylabel('{}'.format(y_axis_label_1), fontsize=12, fontweight='bold')
            axes[1, i].set_ylabel('{}'.format(y_axis_label_2), fontsize=12, fontweight='bold')
            axes[1, i].set_xlim([61, 73])


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

        axes[0, 0].legend(prop={'weight': 'bold', 'size': 12})
        axes[1, 0].legend(prop={'weight': 'bold', 'size': 12})

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






# class MCMC_sampler:
#
#     def __init__(self, df_temperature,df_amplitude_phase_measurement,
#                  sample_information, vacuum_chamber_setting, numerical_simulation_setting,
#                  solar_simulator_settings, light_source_property, prior_mu, prior_sigma,
#                  transition_sigma, result_name, N_sample):
#
#         self.sample_information = sample_information
#         self.vacuum_chamber_setting = vacuum_chamber_setting
#         self.numerical_simulation_setting = numerical_simulation_setting
#         self.solar_simulator_settings = solar_simulator_settings
#         self.light_source_property = light_source_property
#
#         self.prior_mu = prior_mu
#         self.prior_sigma = prior_sigma
#         self.transition_sigma = transition_sigma
#         self.df_temperature = df_temperature
#         self.result_name = result_name
#
#         self.df_amplitude_phase_measurement = df_amplitude_phase_measurement
#
#         self.N_sample = N_sample
#
#     def ln_prior(self, params):
#         p_ln_alpha = norm.pdf(params['ln_alpha'], loc=self.prior_mu['ln_alpha'],
#                               scale=self.prior_sigma['ln_alpha_sigma'])
#         # p_ln_h = norm.pdf(params['ln_h'], loc=self.prior_mu['ln_h'], scale=self.prior_sigma['ln_h'])
#         p_ln_sigma_dA = norm.pdf(params['ln_sigma_dA'], loc=self.prior_mu['ln_sigma_dA'],
#                                  scale=self.prior_sigma['ln_sigma_dA_sigma'])
#         p_ln_sigma_dP = norm.pdf(params['ln_sigma_dP'], loc=self.prior_mu['ln_sigma_dP'],
#                                  scale=self.prior_sigma['ln_sigma_dP_sigma'])
#         return np.log(p_ln_alpha) + np.log(p_ln_sigma_dA) + np.log(p_ln_sigma_dP)
#
#     def ln_transformation_jacobian(self, params):
#         jac_alpha = 1 / (np.exp(params['ln_alpha']))
#         jac_sigma_dA = 1 / (np.exp(params['ln_sigma_dA']))
#         jac_sigma_dP = 1 / (np.exp(params['ln_sigma_dP']))
#         jac_rho = (1 + np.exp(2 * params['z'])) / (4 * np.exp(2 * params['z']))
#         return np.log(jac_alpha) + np.log(jac_sigma_dA) + np.log(jac_sigma_dP) + np.log(jac_rho)
#
#     def ln_likelihood(self, params):
#         self.sample_information['alpha_r'] = np.exp(params['ln_alpha'])
#         df_amp_phase_simulated, df_temperature_simulation,df_light_source, df_temperature_transient = simulation_result_amplitude_phase_extraction(self.df_temperature,
#             self.df_amplitude_phase_measurement, self.sample_information, self.vacuum_chamber_setting,
#             self.solar_simulator_settings, self.light_source_property, self.numerical_simulation_setting,self.code_directory)
#         mean_measured = np.array(
#             [self.df_amplitude_phase_measurement['amp_ratio'], self.df_amplitude_phase_measurement['phase_diff']])
#         mean_theoretical = np.array([df_amp_phase_simulated['amp_ratio'], df_amp_phase_simulated['phase_diff']])
#
#         sigma_dA = np.exp(params['ln_sigma_dA'])
#         sigma_dP = np.exp(params['ln_sigma_dP'])
#         rho_dA_dP = np.tanh(params['z'])
#
#         cov_errs = [[sigma_dA ** 2, sigma_dA * sigma_dP * rho_dA_dP], [sigma_dA * sigma_dP * rho_dA_dP, sigma_dP ** 2]]
#
#         return np.sum([np.log(multivariate_normal.pdf(mean_measured_, mean_theoretical_, cov_errs)) for
#                        (mean_measured_, mean_theoretical_) in zip(mean_measured.T, mean_theoretical.T)])
#
#     def rw_proposal(self, params):
#
#         ln_sigma_dA = params['ln_sigma_dA']
#         ln_sigma_dP = params['ln_sigma_dP']
#         ln_alpha = params['ln_alpha']
#         z = params['z']
#
#         ln_alpha, ln_sigma_dA, ln_sigma_dP, z = np.random.normal(
#             [ln_alpha, ln_sigma_dA, ln_sigma_dP, z], scale=self.transition_sigma)
#
#         params_star = {'ln_alpha': ln_alpha, 'ln_sigma_dA': ln_sigma_dA, 'ln_sigma_dP': ln_sigma_dP, 'z': z}
#         return params_star
#
#     def rw_metropolis(self):
#
#         n_accepted = 0
#         n_rejected = 0
#
#         params = self.prior_mu
#         accepted = []
#         posterior = np.exp(self.ln_likelihood(params) + self.ln_transformation_jacobian(params) + self.ln_prior(params))
#
#         while (n_accepted < self.N_sample):
#             params_star = self.rw_proposal(params)
#             posterior_star = np.exp(
#                 self.ln_likelihood(params_star) + self.ln_transformation_jacobian(params_star) + self.ln_prior(
#                     params_star))
#
#             accept_ratio = min(1, posterior_star / posterior)
#             u = np.random.rand()
#             if u <= accept_ratio:  # accept the new state
#                 params = params_star
#                 posterior = posterior_star
#                 n_accepted += 1
#                 accepted.append(params)
#
#                 print('Accepted sample is {} and acceptance rate is {:.3f}.'.format(n_accepted, n_accepted / (
#                             n_accepted + n_rejected)))
#
#             else:  # reject the new state
#                 n_rejected += 1
#
#         accepted = np.array(accepted)
#
#         return accepted