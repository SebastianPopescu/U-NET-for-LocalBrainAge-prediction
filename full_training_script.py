# -*- coding: utf-8 -*-
from itertools import chain
import nibabel as nib
import numpy as np
import tensorflow as tf
from collections import defaultdict
import sys
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import time
import argparse
import os
import math
import random
from sklearn.cluster import  KMeans
from sklearn.preprocessing import StandardScaler
DTYPE=tf.float32
import subprocess
from LocalBrainAge_training import UNET_Dropout_ROI_Context_Enhanced
from loading_data import *
from sklearn.model_selection import KFold

def absoluteFilePaths(dirpath):
   
	### dirpath has to be absolute path ####

	list_realpaths = []
	filenames = os.listdir(dirpath)

	for f in filenames:
		list_realpaths.append(os.path.abspath(os.path.join(dirpath, f)))

	return list_realpaths

if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--num_encoding_layers', type = int, default = 2, help = 'the number of scales in the U-Net architecture')
	parser.add_argument('--num_filters', type = int, default = 64, help = 'the number of filters in the first scale, it gets multiplied by 2 as it goes down in the hierarchy')
	parser.add_argument('--num_subjects', type = int, default = 2, help = 'the number of subjects to sample data points in a given minibatch')
	parser.add_argument('--num_voxels_per_subject', type = int , default = 1, 'the number of data points to sample from each subject in a given minibatch')
	parser.add_argument('--location_metadata', tpye = str, help = 'the absolute location of the dataset metadata')
	parser.add_argument('--dirpath_gm', tpye = str, help = 'the absolute location of the grey matter files')
	parser.add_argument('--dirpath_wm', tpye = str, help = 'the absolute location of the white matter files')
	parser.add_argument('--dataset_name', tpye = str, help = 'dataset name, will influce where the output will be written')
	args = parser.parse_args()

	#### load metadata ####
	info_subjects = pd.read_csv(args.location_metadata)
	### Column names -- ['Subject', 'Age', 'Gender, 'Source']

	print(info_subjects.head)
	
	age_subjects = info_subjects['Age'].values
	gender_subjects = info_subjects['Gender'].values
	name_subjects = info_subjects['Subject'].values
	name_subjects.tolist()

	list_overall_gm_files = absoluteFilePaths(args.dirpath_gm)
	list_overall_wm_files = absoluteFilePaths(args.dirpath_wm)


	kf = KFold(n_splits=5, shuffle=True,random_state=7)
	cv_num = 0
	control = 0

	for train_index, test_index in kf.split(range(len(name_subjects))):

		if control==cv_num:

			list_subjects_training =  parse_string_list(string_list = name_subjects, index = train_index)
			list_subjects_testing =  parse_string_list(string_list = name_subjects, index = test_index)
			
			control+=1

		else:

			control+=1

	dict_X_training, dict_Y_training, dict_gender_training, parsed_gm_training, parsed_wm_training, dict_X_training_names = data_factory_whole_brain_training(list_of_nifty_files_gm = list_overall_gm_files,
		list_of_nifty_files_wm = list_overall_wm_files,  subject_info = info_subjects.copy(), list_extract_subjects = list_subjects_training)

	dict_X_testing, dict_Y_testing, dict_gender_testing, parsed_gm_testing, parsed_wm_testing,  dict_X_testing_names = data_factory_whole_brain_training(list_of_nifty_files_gm = list_overall_gm_files,
		list_of_nifty_files_wm = list_overall_wm_files,  subject_info = info_subjects.copy(), list_extract_subjects = list_subjects_testing)


	########## Sanity check #############################################
	##### save the current training and testing sets into csv files #####
	df_training =  pd.DataFrame(columns=['Subject', 'Age', 'Gender','GM','WM'])
	df_training['Subject'] = dict_X_training_names.values()
	df_training['Age'] = dict_Y_training.values()
	df_training['Gender'] = dict_gender_training.values()
	df_training['GM'] = parsed_gm_training
	df_training['WM'] = parsed_wm_training

	df_training.to_csv('./training_set_subject_information.csv', index=False)

	df_testing =  pd.DataFrame(columns=['Subject', 'Age', 'Gender','GM','WM'])
	df_testing['Subject'] = dict_X_testing_names.values()
	df_testing['Age'] = dict_Y_testing.values()
	df_testing['Gender'] = dict_gender_testing.values()
	df_testing['GM'] = parsed_gm_testing
	df_testing['WM'] = parsed_wm_testing

	df_testing.to_csv('./testing_set_subject_information.csv', index=False)

	Y_training = np.array(list(dict_Y_training.values())).flatten()
	print(Y_training)
	print(Y_training.shape)

	#np.savetxt('./training_age.txt',Y_training)
	for kkt in list(dict_Y_training.values()):
		print('*************************')
		print(kkt)

	mean_age = np.mean(Y_training)

	print('**** mean age of dataset *****')
	print(mean_age)

	for key in dict_Y_training.keys():
		dict_Y_training[key]-=mean_age

	mask_object = nib.load('./data/brain_mask.nii')
	mask_data = mask_object.get_data()

	num_filters = [args.num_filters * (_+1) for _ in range(args.num_encoding_layers)]

	obiect =  UNET_Dropout_ROI_Context_Enhanced(dim_input = 3, dim_output = 1,
		num_iterations = 1000001, num_encoding_layers = args.num_encoding_layers, 
		num_batch = 32, 
		num_filters = num_filters, dim_filter = 3 , num_stride = 1,
		use_epistemic_uncertainty = True,
		size_cube_input = 52, size_cube_output = 12, learning_rate = 1e-5, num_layers_same_scale = 2,  
		import_model = True, iteration_restored = 130000, unet_type = '3D', keep_prob = 0.8, mean_age = mean_age,
		num_averaged_gradients = 12,
		num_subjects = args.num_subjects, num_voxels_per_subject = args.num_voxels_per_subject, testing_time = False)
	obiect.session_TF(dict_X_training, dict_Y_training, dict_gender_training,
		dict_X_testing, dict_Y_testing, dict_gender_testing, mask_data, mask_object.affine, args.dataset_name, dict_X_testing_names)
