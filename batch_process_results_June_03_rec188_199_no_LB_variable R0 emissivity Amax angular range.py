  # some_file.py
import sys
#python_source_code_path = "C://Users//NTRG lab//PycharmProjects//High-T-Angstrom-Method"
python_source_code_path = "C://Users//yuan//PycharmProjects//High-T-Angstrom-Method"

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, python_source_code_path)
from high_T_angstrom_method import parallel_result_summary

from high_T_angstrom_method import radial_temperature_average_disk_sample_several_ranges,batch_process_horizontal_lines,radial_1D_explicit,show_regression_results
# #import high_T_angstrom_method
from high_T_angstrom_method import residual_solar, residual
# from high_T_angstrom_method import MCMC_sampler
from high_T_angstrom_method import check_angular_uniformity
from high_T_angstrom_method import display_high_dimensional_regression_results

from high_T_angstrom_method import parallel_regression_batch_experimental_results

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time

from scipy.optimize import minimize
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed



import time
from datetime import datetime

# you will need to create these folders under the jupyter notebook directory, the data will read and write automatically, don't need to manually do anything, unless a data is wrong and need to delete manually
# (1)result cache dump, results will be saved in this folder
# (2)batch process information, the experimental condition spread sheet will be in this folder
# (3)temperature cache dump, the radial averaged temperature picke file will be in this folder

df_exp_condition_spreadsheet_filename = 'batch_process_results_June_03_rec188_199_no_LB_variable R0 emissivity Amax angular range.xlsx'
data_directory = "C://Users//yuan//Desktop//Amgstrom_method//temperature data//" # this is the directory to save csv files, note each experiment must have its own folder, for example Rec-0000X should be-> data directory + Rec-0000X+ //Rec-0000X_01.csv
code_directory = "G://My Drive//Current Projects//Angstrom Method//High Temperature Simulation//Jupyter Notebook//"

num_cores = 8 # number of cores to run regression in parallel, note there is a num_core in the excel spreadsheet, it indicates the number of cores required to process the radial averaged temperture profile

parallel_regression_batch_experimental_results(df_exp_condition_spreadsheet_filename,data_directory,num_cores,code_directory)

joblib_output = pickle.load(open( code_directory+"result cache dump//regression_results_" + df_exp_condition_spreadsheet_filename, 'rb'))

df_results_all = parallel_result_summary(joblib_output,df_exp_condition_spreadsheet_filename,code_directory)
df_exp_condition_spreadsheet_filename_no_extension = df_exp_condition_spreadsheet_filename[:df_exp_condition_spreadsheet_filename.find(".")]

df_results_all.to_csv(code_directory+"result summary dataframe//"+"result_summary_dataframe_"+df_exp_condition_spreadsheet_filename_no_extension+".csv")