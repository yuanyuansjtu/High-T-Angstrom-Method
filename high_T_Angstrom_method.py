import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
from numba import jit
import operator

def select_data_points_radial_average_MA(x0,y0,Rmax,theta_n,file_name):
    # This method was originally developed by Mosfata, was adapted by HY to use for amplitude and phase estimation
    df_raw = pd.read_csv(file_name, sep = ',',header = None,names=list(np.arange(0,639)))
    raw_time_string = df_raw.iloc[2,0][df_raw.iloc[2,0].find('=')+2:] #'073:02:19:35.160000'
    raw_time_string = raw_time_string[raw_time_string.find(":")+1:] #'02:19:35.160000'
    strip_time = datetime.strptime(raw_time_string,'%H:%M:%S.%f') 
    time_in_seconds = strip_time.hour*3600+strip_time.minute*60+strip_time.second+strip_time.microsecond/10**6
    theta = np.linspace(0,2*np.pi,theta_n) # The angles 1D array (rad)
    df_temp = df_raw.iloc[5:,:]
    
    Tr = np.zeros((Rmax,theta_n)) # Initializing the radial temperature matrix (T)
    
    for i in range(Rmax): # Looping through all radial points
        for j,theta_ in enumerate(theta): # Looping through all angular points
            y = i*np.sin(theta_) + y0; x = i*np.cos(theta_) + x0 # Identifying the spatial 2D cartesian coordinates
            y1 = int(np.floor(y)); y2 = y1+1; x1 = int(np.floor(x)); x2 = x1+1 # Identifying the neighboring 4 points
            dy1 = (y2-y)/(y2-y1); dy2 = (y-y1)/(y2-y1) # Identifying the corresponding weights for the y-coordinates
            dx1 = (x2-x)/(x2-x1); dx2 = (x-x1)/(x2-x1) # Identifying the corresponding weights for the x-coordinates
            T11 = df_temp.iloc[y1,x1]; T21 = df_temp.iloc[y2,x1] # Identifying the corresponding temperatures for the y-coordinates
            T12 = df_temp.iloc[y1,x2]; T22 = df_temp.iloc[y2,x2] # Identifying the corresponding temperatures for the x-coordinates
            Tr[i,j] = dx1*dy1*T11 + dx1*dy2*T21 + dx2*dy1*T12 + dx2*dy2*T22 +273.15 # Interpolated angular temperature matrix

    T_interpolate = np.mean(Tr,axis=1)
    
    return T_interpolate,time_in_seconds

def select_data_points_radial_average_MA_match_model_grid(x0,y0,N_Rmax,pr,R_sample,theta_n,file_name): #N_Rmax: total number of computation nodes between center and edge
    # This method was originally developed by Mosfata, was adapted by HY to use for amplitude and phase estimation
    df_raw = pd.read_csv(file_name, sep = ',',header = None,names=list(np.arange(0,639)))
    raw_time_string = df_raw.iloc[2,0][df_raw.iloc[2,0].find('=')+2:] #'073:02:19:35.160000'
    raw_time_string = raw_time_string[raw_time_string.find(":")+1:] #'02:19:35.160000'
    strip_time = datetime.strptime(raw_time_string,'%H:%M:%S.%f') 
    time_in_seconds = strip_time.hour*3600+strip_time.minute*60+strip_time.second+strip_time.microsecond/10**6
    theta = np.linspace(0,2*np.pi,theta_n) # The angles 1D array (rad)
    df_temp = df_raw.iloc[5:,:]
    
    Tr = np.zeros((N_Rmax,theta_n)) # Initializing the radial temperature matrix (T)
    
    R = np.linspace(0,R_sample,N_Rmax)
    for i,R_ in enumerate(R): # Looping through all radial points
        for j,theta_ in enumerate(theta): # Looping through all angular points
            y = R_/pr*np.sin(theta_) + y0; x = R_/pr*np.cos(theta_) + x0 # Identifying the spatial 2D cartesian coordinates
            y1 = int(np.floor(y)); y2 = y1+1; x1 = int(np.floor(x)); x2 = x1+1 # Identifying the neighboring 4 points
            dy1 = (y2-y)/(y2-y1); dy2 = (y-y1)/(y2-y1) # Identifying the corresponding weights for the y-coordinates
            dx1 = (x2-x)/(x2-x1); dx2 = (x-x1)/(x2-x1) # Identifying the corresponding weights for the x-coordinates
            T11 = df_temp.iloc[y1,x1]; T21 = df_temp.iloc[y2,x1] # Identifying the corresponding temperatures for the y-coordinates
            T12 = df_temp.iloc[y1,x2]; T22 = df_temp.iloc[y2,x2] # Identifying the corresponding temperatures for the x-coordinates
            Tr[i,j] = dx1*dy1*T11 + dx1*dy2*T21 + dx2*dy1*T12 + dx2*dy2*T22 +273.15 # Interpolated angular temperature matrix

    T_interpolate = np.mean(Tr,axis=1)
    
    return T_interpolate,time_in_seconds

def select_data_points_radial_average_HY(x0,y0,Rmax,file_name):
    
    df_raw = pd.read_csv(file_name, sep = ',',header = None,names=list(np.arange(0,639)))
    raw_time_string = df_raw.iloc[2,0][df_raw.iloc[2,0].find('=')+2:] #'073:02:19:35.160000'
    raw_time_string = raw_time_string[raw_time_string.find(":")+1:] #'02:19:35.160000'
    strip_time = datetime.strptime(raw_time_string,'%H:%M:%S.%f') 
    time_in_seconds = strip_time.hour*3600+strip_time.minute*60+strip_time.second+strip_time.microsecond/10**6
    
    df_temp = df_raw.iloc[5:,:]
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
            T_interpolate[r]=df_temp.iloc[y0,x0]
            
        else:
            temp = []
            for i in np.arange(x0-r-2,x0+r+2,1):
                for j in np.arange(y0-r-2,y0+r+2,1):
                    d = ((i-x0)**2+(j-y0)**2)**(0.5)
                    if d>= r and d<r+1:
                        x.append(i)
                        y.append(j)
                        r_pixel.append(r)                 
                        temp.append((df_temp.iloc[j,i]-T_interpolate[r-1])/(d-r+1)+T_interpolate[r-1]+273.15)
                        
            T_interpolate[r] = np.mean(temp)
            
    return T_interpolate,time_in_seconds



def radial_temperature_average_disk_sample(x0,y0,N_Rmax,pr,path,rec_name,output_name,method,num_cores): #unit in K
    #path= "C://Users//NTRG lab//Desktop//yuan//"
    #rec_name = "Rec-000011_e63", this is the folder which contains all csv data files
    
    
    dump_file_path = output_name+'_x0_{}_y0_{}_Rmax_{}_method_{}'.format(x0,y0,Rmax,method)
    
    if (os.path.isfile(dump_file_path)):  # First check if a dump file exist:
        print('Found previous dump file :'+dump_file_path)
        temp_dump = pickle.load(open(dump_file_path, 'rb'))
    
    else: #If not we obtain the dump file, note the dump file is averaged radial temperature
        
        file_names = [path+x for x in os.listdir(path)]
        s_time = time.time()
        
        if method == 'MA': # default method, this one is much faster
            theta_n = 100 # default theta_n=100, however, if R increased significantly theta_n should also increase
            #joblib_output = Parallel(n_jobs=num_cores)(delayed(select_data_points_radial_average_MA)(x0,y0,Rmax,theta_n,file_name) for file_name in tqdm(file_names))
            joblib_output = Parallel(n_jobs=num_cores)(delayed(select_data_points_radial_average_MA)(x0,y0,Rmax,theta_n,file_name) for file_name in tqdm(file_names))
            
        else:
            joblib_output = Parallel(n_jobs=num_cores)(delayed(select_data_points_radial_average_HY)(x0,y0,Rmax,theta_n,file_name) for file_name in tqdm(file_names))

        pickle.dump(joblib_output, open(dump_file_path, "wb")) # create a dump file

        e_time = time.time()
        print('Time to process the entire dataset is {}'.format((e_time-s_time)))
        
        temp_dump = pickle.load(open(dump_file_path, 'rb'))
        
    temp_dump = np.array(temp_dump); temp_dump = temp_dump[temp_dump[:,1].argsort()] # sort based on the second column, which is relative time

    temp_profile = []; time_series = []
    for item in temp_dump:
        temp_profile.append(item[0])
        time_series.append(item[1])
    time_series = np.array(time_series)
    temp_data = np.array(temp_profile)
    time_data = time_series-min(time_series) # such that time start from zero
    
    df_temperature = pd.DataFrame(data = temp_data) # return a dataframe containing radial averaged temperature and relative time
    df_temperature['reltime'] = time_data 
    
    return df_temperature #note the column i of the df_temperature indicate the temperature in pixel i



def amp_phase_one_pair(index, df_temperature, f_heating, gap):


    n_col = df_temperature.shape[1]
    tmin = min(df_temperature['reltime'])
    time = df_temperature['reltime'] - tmin

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

    return L, phase_diff, amp_ratio # note L is in the unit of pixel



def fit_amp_phase_one_batch(df_temperature,R0,gap):

    N_lines = df_temperature.shape[1] - 1
    r_list = np.zeros(N_lines - 1)
    phase_diff_list = np.zeros(N_lines - 1)
    amp_ratio_list = np.zeros(N_lines - 1)
    r_ref_list = np.zeros(N_lines - 1)


    R_ring_inner_edge = R0 # R0 is the refernce radius, it should be minimumly equal to the radius of the light blocker


    for i in range(N_lines):

        if i > 0:
            index = [0, i]
            r_list[i - 1], phase_diff_list[i - 1], amp_ratio_list[i - 1] = amp_phase_one_pair(index, df_temperature, f_heating, gap)
            r_list[i - 1] = r_list[i - 1] + R_ring_inner_edge
            r_ref_list[i-1] = R_ring_inner_edge

    return r_list,r_ref_list, phase_diff_list, amp_ratio_list



def batch_process_horizontal_lines(df_temperature,f_heating,R0,gap,R_analysis):

    #N_line_groups = gap - 1
    N_line_groups = gap
    N_line_each_group = int(R_analysis/gap)-1 # Don't have to analyze entire Rmax pixels
    r_list_all = []
    r_ref_list_all = []
    phase_diff_list_all = []
    amp_ratio_list_all = []
    N_frame_keep = df_temperature.shape[0]
    
    T_group = np.zeros((N_line_groups, N_line_each_group, N_frame_keep))
    
    for k in range(N_frame_keep):
        for j in range(N_line_groups):
            for i in range(N_line_each_group):
                T_group[j, i, k] = df_temperature.iloc[k,R0+j+gap*i]
    
    
    for j in range(N_line_groups):

        df = pd.DataFrame(T_group[j, :, :].T)
        df['reltime'] = df_temperature['reltime']
        r_list, r_ref_list, phase_diff_list, amp_ratio_list = fit_amp_phase_one_batch(df,R0+j,gap)
        #print(df.shape[1]-1)

        r_list_all = r_list_all + list(r_list)
        r_ref_list_all = r_ref_list_all + list(r_ref_list)

        phase_diff_list_all = phase_diff_list_all + list(phase_diff_list)
        amp_ratio_list_all = amp_ratio_list_all + list(amp_ratio_list)

    df_result_IR = pd.DataFrame(data={'r': r_list_all, 'r_ref':r_ref_list_all,'amp_ratio': amp_ratio_list_all, 'phase_diff': phase_diff_list_all})

    return df_result_IR


def light_source_intensity(r,t,solar_simulator_settings, light_source_property):
    
    f_heating = solar_simulator_settings['f_heating']
    As = solar_simulator_settings['V_amplitude']
    V_DC = solar_simulator_settings['V_DC']
    
    ks = light_source_property['ks'] #solar simulator setting->current
    bs = light_source_property['bs']
    
    ka = light_source_property['ka'] #currect -> heat flux
    ba = light_source_property['ba']
    
    Amax = light_source_property['Amax']
    sigma_s = light_source_property['sigma_s']
    I = bs+ks*V_DC+ks*As*np.sin(2*np.pi*f_heating*t) # t: time, r: distance from center
    
    q = Amax*(ka*I+ba)/np.pi*(sigma_s/(r**2+sigma_s**2))
    return q


def radial_1D_explicit(Fo_criteria,sample_information, vacuum_chamber_setting, solar_simulator_settings,light_source_property, numerical_simulation_setting):
    
    
    s_time = time.time()

    R = sample_information['R']
    t_z = sample_information['t_z']
     
    dr = R/Nr
    dz = t_z
    
    rho = sample_information['rho']
    cp_const = sample_information['cp_const'] #Cp = cp_const + cp_c1*T+ cp_c2*T**2+cp_c3*T**3, T unit in K
    cp_c1 = sample_information['cp_c1']
    cp_c2 = sample_information['cp_c2']
    cp_c3 = sample_information['cp_c3']
    alpha_r = sample_information['alpha_r']
    alpha_z = sample_information['alpha_z']
    
    T_initial = sample_information['T_initial'] # unit in K
    #k = alpha*rho*cp
  
    f_heating = solar_simulator_settings['f_heating'] # periodic heating frequency
    N_cycle = numerical_simulation_setting['N_cycle']
    
    
    T_sur1 = vacuum_chamber_setting['T_sur1'] # unit in K
    T_sur2 = vacuum_chamber_setting['T_sur2'] # unit in K
    
    emissivity = sample_information['emissivity'] # assumed to be constant
    absorptivity = sample_information['absorptivity'] # assumed to be constant
    sigma_sb = 5.6703744*10**(-8) # stefan-Boltzmann constant
    
    dt = min(Fo_criteria*(dr**2)/(alpha_r),1/f_heating/15) # assume 15 samples per period, Fo_criteria default = 1/3
    t_total = 1/f_heating*N_cycle # total simulation time
    
    Nt = int(t_total/dt) # total number of simulation time step
    time_simulation = dt*np.arange(Nt)
    r = np.arange(Nr)
    
    Fo_r = alpha_r*dt/dr**2    
    
    Rs = vacuum_chamber_setting['Rs'] # the location where the solar light shines on the sample
    R0 = vacuum_chamber_setting['R0'] # the location of reference ring
    
    N_Rs = int(Rs/R*Nr)
    
    q_solar = np.zeros((Nt,Nr)) #heat flux of the solar simulator
    
    for i in range(N_Rs): #truncated solar light
        for j in range(Nt):
            #q_solar[j,i] = 10**5 #light_source_intensity(i*dr,j*dt,solar_simulator_settings, light_source_property)
            q_solar[j,i] = light_source_intensity(i*dr,j*dt,solar_simulator_settings, light_source_property)
    
    T = T_initial*np.ones((Nt,Nr))
    
    for p in range(Nt-1): #p indicate time step
        for m in range(Nr): # m indicate node along radial direction  
            rm = dr*m
            cp = cp_const + cp_c1*T[p,m]+cp_c2*T[p,m]**2+cp_c3*T[p,m]**3
                                   
            if m == 0:
                # consider V1
                T[p+1,m] = T[p,m]+4*Fo_r*(T[p,m+1]-T[p,m])+ \
                dt/(rho*cp*dz)*(absorptivity*sigma_sb*T_sur1**4+absorptivity*q_solar[p,m]-emissivity*sigma_sb*T[p,m]**4)+ \
                dt/(rho*cp*dz)*(absorptivity*sigma_sb*T_sur2**4-emissivity*sigma_sb*T[p,m]**4)
                   
            elif m == Nr-1:
                # consider V2
                T[p+1,m] = T[p,m]+2*(rm-dr/2)/rm*Fo_r*(T[p,m-1]-T[p,m])+ \
                dt/(rho*cp*dz)*(absorptivity*sigma_sb*T_sur1**4+absorptivity*q_solar[p,m]-emissivity*sigma_sb*T[p,m]**4) \
                + 2*dt/(rho*cp*dr)*(absorptivity*sigma_sb*T_sur2**4-emissivity*sigma_sb*T[p,m]**4) \
                + dt/(rho*cp*dz)*(absorptivity*sigma_sb*T_sur2**4 -emissivity*sigma_sb*T[p,m]**4)
               
            elif m>= 1 and m != Nr-1:
                # Consider E1
                T[p+1,m] = T[p,m,]+Fo_r*(rm-dr/2)/rm*(T[p,m-1]-T[p,m])+Fo_r*(rm+dr/2)/rm*(T[p,m+1]-T[p,m]) \
                + dt/(rho*cp*dz)*(absorptivity*sigma_sb*T_sur1**4+absorptivity*q_solar[p,m]-emissivity*sigma_sb*T[p,m]**4) \
                + dt/(rho*cp*dz)*(absorptivity*sigma_sb*T_sur2**4-emissivity*sigma_sb*T[p,m]**4)
    
    e_time = time.time()
    
    #print('A case has been evaluated!')
    print('alpha_r = {},run time = {} s, N_cycle = {}, dt = {}, Nr = {}, Nt = {}, Fo_r = {}'.format(alpha_r,e_time-s_time ,N_cycle,dt,Nr,Nt,Fo_r))

    return T,time_simulation, r



def simulation_result_amplitude_phase_extraction(df_amplitude_phase_measurement,sample_information, vacuum_chamber_setting, solar_simulator_settings,light_source_property, numerical_simulation_setting):
    
    T_,time_T_, r_ = radial_1D_explicit(1/3,sample_information, vacuum_chamber_setting, solar_simulator_settings,light_source_property, numerical_simulation_setting)
    
    
    df_temperature_simulation = pd.DataFrame(data = T_) # return a dataframe containing radial averaged temperature and relative time
    df_temperature_simulation['reltime'] = time_T_ 
    
    
    N_steady = int(len(df_temperature_simulation)/3)
    df_temperature_simulation = df_temperature_simulation.iloc[N_steady:,:] # now we only use steady part
    
    df_temperature_simulation = pd.DataFrame(data = T_) # return a dataframe containing radial averaged temperature and relative time
    df_temperature_simulation['reltime'] = time_T_ 

    phase_diff_simulation = []
    amplitude_ratio_simulation = []

    for i in range(len(df_amplitude_phase_measurement)):
        L,phase_diff, amp_ratio = amp_phase_one_pair([df_amplitude_phase_measurement.iloc[i,:]['r_ref'],df_amplitude_phase_measurement.iloc[i,:]['r']], df_temperature_simulation, f_heating, gap)
        amplitude_ratio_simulation.append(amp_ratio)
        phase_diff_simulation.append(phase_diff)

    df_amp_phase_simulated = pd.DataFrame(data = {'amp_ratio':amplitude_ratio_simulation,'phase_diff':phase_diff_simulation})

    df_amp_phase_simulated['r'] = df_amplitude_phase_measurement['r']
    df_amp_phase_simulated['r_ref'] = df_amplitude_phase_measurement['r_ref']
    
    return df_amp_phase_simulated
    

def residual(params,df_amplitude_phase_measurement,sample_information, vacuum_chamber_setting, solar_simulator_settings,light_source_property, numerical_simulation_setting):

    sample_information['alpha_r'] = params[0]
    
    df_amp_phase_simulated = simulation_result_amplitude_phase_extraction(df_amplitude_phase_measurement,sample_information, vacuum_chamber_setting, solar_simulator_settings,light_source_property, numerical_simulation_setting)
    
    phase_relative_error = [abs((measure-calculate)/measure) for measure,calculate in zip(df_amplitude_phase_measurement['phase_diff'],df_amp_phase_simulated['phase_diff'])]
    amplitude_relative_error = [abs((measure-calculate)/measure) for measure,calculate in zip(df_amplitude_phase_measurement['amp_ratio'],df_amp_phase_simulated['amp_ratio'])]
    
    
    return np.sum(amplitude_relative_error)+np.sum(phase_relative_error)
    #return (data-model)