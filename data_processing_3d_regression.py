import numpy as np
import tensorflow as tf
from collections import defaultdict
import random
from sklearn import preprocessing
import os

def apply_mask(array, mask):

	##### array --- (dim1, dim2, dim3)
	##### mask --- (dim1, dim2, dim3)
	##### masks the array so that the non-ROI part is zero ####

	array[mask < 1.0] = 0.0

	return array


def check_mask(mask, central_points, semi_block_size_output, semi_block_size_output2):

	### mask in output space, in the sense of masking some predictions of the U-NET as that region is not of interest ####
	### think of voxel-level brain age prediction where you don't want predictions of brain age in regions where there is no brain ###

	current_shape = mask.shape

	control=0
	padding_dimensions=[]
	
	for _ in range(3):

		dim_list = []

		if central_points[_]-semi_block_size_output < 0:			
			dim_list.append(np.abs(central_points[_]-semi_block_size_output))
			control+=1

		else:

			dim_list.append(0)

		if central_points[_]+semi_block_size_output2 > (current_shape[_]-1):
			dim_list.append(np.abs(central_points[_]+semi_block_size_output2 - current_shape[_]+1))
			control+=1
		else:
			dim_list.append(0)

		padding_dimensions.append(tuple(dim_list))

	if control > 0:

		padding_dimensions = tuple(padding_dimensions)
		mask = np.pad(mask, padding_dimensions, mode='constant', constant_values = 0.0)	
		central_points = [central_points[_]+padding_dimensions[_][0] for _ in range(3)]

	correct_mask = mask[central_points[0]-semi_block_size_output:central_points[0]+semi_block_size_output2,
		central_points[1]-semi_block_size_output:central_points[1]+semi_block_size_output2,
		central_points[2]-semi_block_size_output:central_points[2]+semi_block_size_output2]

	return correct_mask


##########################################################################################
####### This is for splitting 3D objects into smaller non-overlapping 3D cubes ###########
##########################################################################################

def cubify(arr, newshape):

	###############################################
	#### non-overlapping cubes from 3D block ######
	###############################################

    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

def uncubify(arr, oldshape):

	###############################################################
	#### gather smaller 3D blocks into original big 3D block ######
	###############################################################

    N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)    
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)

def crop_3D_block(image,central_points,semi_block_size1,semi_block_size2):

	#### basically crops a 3D block using Cnetral coordiantes and semi_sizes of block in both directins across all dimensions ###

	### image -- shape (height, width, depth, channels)
	### central_points -- (c1,c2,c3)
	### semi_block_size -- (l1,l2,l3)

	cropped_image = image[central_points[0]-semi_block_size1:central_points[0]+semi_block_size2,
		central_points[1]-semi_block_size1:central_points[1]+semi_block_size2,
		central_points[2]-semi_block_size1:central_points[2]+semi_block_size2,:]

	return cropped_image

def crop_image(input_image, ROI_end_points):

	#################################################
	### most basic type of cropping in 3D space #####
	#################################################

	output_image = input_image[ROI_end_points[0][0]:ROI_end_points[0][1]+1,
		ROI_end_points[1][0]:ROI_end_points[1][1]+1,
		ROI_end_points[2][0]:ROI_end_points[2][1]+1,:]

	return output_image


def crop_as_much_as_possible_then_pad(input_image, ROI_end_points, diff_semi_block, padding_output_space):

	########################################################################################
	#### to be used at testing time for the input space 3D blocks to be fed into U-NET #####
	########################################################################################

	#################################################################################################################
	### padding_output_space -- [padding_output_space_dim1, padding_output_space_dim2, padding_output_space_dim3] ###

	### because output_image was padded to ensure divisibility by block_output_space, we need to chance the coordinates of the ROI in the lower parts across all dimensions ####

	ROI_end_points[0][0] = ROI_end_points[0][0] - padding_output_space[0]
	ROI_end_points[1][0] = ROI_end_points[1][0] - padding_output_space[1]
	ROI_end_points[2][0] = ROI_end_points[2][0] - padding_output_space[2]

	padding_dimensions = []
	cropping_dimensions = defaultdict()
	control = 0				
	current_shape  = input_image.shape
	#current_shape[0] = current_shape[0]-1
	#current_shape[1] = current_shape[1]-1
	#current_shape[2] = current_shape[2]-1

	for _ in range(3):

		padding_list = []
		cropping_dimensions[_] = defaultdict()

		####################
		#### lower part ####
		####################

		if ROI_end_points[_][0] - diff_semi_block < 0:			
		
			padding_list.append(np.abs(ROI_end_points[_][0] - diff_semi_block))
			cropping_dimensions[_][0] = diff_semi_block - padding_list[-1]
			control+=1

		else:

			padding_list.append(0)
			cropping_dimensions[_][0] = diff_semi_block - padding_list[-1]

		####################
		#### upper part ####
		####################

		if ROI_end_points[_][1] + diff_semi_block > (current_shape[_]-1):

			padding_list.append(np.abs(ROI_end_points[_][1] + diff_semi_block - current_shape[_]+1))
			cropping_dimensions[_][1] = diff_semi_block - padding_list[-1]
			control+=1
		
		else:
		
			padding_list.append(0)
			cropping_dimensions[_][1] = diff_semi_block - padding_list[-1]

		padding_dimensions.append(tuple(padding_list))

	###########################
	###### Cropping first #####
	###########################

	#print('size of input iage before ay transformations')
	#print(input_image.shape)

	#print('******************')
	#print('cropping dimensions')
	#print(cropping_dimensions[0])
	#print(cropping_dimensions[1])
	#print(cropping_dimensions[2])


	input_image = input_image[(ROI_end_points[0][0] - cropping_dimensions[0][0]):(ROI_end_points[0][1] + 1 + cropping_dimensions[0][1]),
		(ROI_end_points[1][0] - cropping_dimensions[1][0]):(ROI_end_points[1][1] + 1 + cropping_dimensions[1][1]),
		(ROI_end_points[2][0] - cropping_dimensions[2][0]):(ROI_end_points[2][1] + 1 + cropping_dimensions[2][1]),:]

	#print('size after cropping')
	#print(input_image.shape)

	#####################################################
	##### Only if padding is needed after cropping ######
	#####################################################

	if control > 0:

		padding_dimensions = tuple(padding_dimensions)
		padding_dimensions_extra = list(padding_dimensions)
		padding_dimensions_extra.append(tuple([0,0]))
		padding_dimensions_extra = tuple(padding_dimensions_extra)
		input_image = np.pad(input_image, padding_dimensions_extra, mode='constant', constant_values = 0.0)	
		#central_points = [central_points[_]+padding_dimensions[_][0] for _ in range(3)]	

	#print('********')
	#print('padding dimensions')
	#print(padding_dimensions_extra)


	return input_image



def check_and_add_zero_padding_regression(input_image, central_points, semi_block_size1, semi_block_size2):

	#### checks if extracting a block need padding or not 
	#### accounts for the case where the central_points are close to the boundary of the brain scan and expands it with 0s

	### input_image -- shape (height, width, depth, channels)
	### central_points -- (c1,c2,c3)
	### semi_block_size -- (l1,l2,l3)

	current_shape = input_image.shape

	min_value_image = np.min(input_image)
	padding_dimensions = []
	control=0				

	for _ in range(3):

		dim_list = []

		if central_points[_]-semi_block_size1 < 0:			
			dim_list.append(np.abs(central_points[_]-semi_block_size1))
			control+=1

		else:

			dim_list.append(0)

		if central_points[_]+semi_block_size2 > (current_shape[_]-1):
			dim_list.append(np.abs(central_points[_]+semi_block_size2 - current_shape[_]+1))
			control+=1
		else:
			dim_list.append(0)

		padding_dimensions.append(tuple(dim_list))

	if control > 0:

		padding_dimensions = tuple(padding_dimensions)
		padding_dimensions_extra = list(padding_dimensions)
		padding_dimensions_extra.append(tuple([0,0]))
		padding_dimensions_extra = tuple(padding_dimensions_extra)
		input_image = np.pad(input_image, padding_dimensions_extra, mode='constant', constant_values = min_value_image)	
		central_points = [central_points[_]+padding_dimensions[_][0] for _ in range(3)]

	return input_image, central_points


#########################################################################################################
####################################### Training time routines ##########################################
#########################################################################################################






#########################################################################################################
####################################### Testing time routines ###########################################
#########################################################################################################




def extract_3D_cubes_input_seg_regression(input_image, output_image, gender_image, semi_block_size_input1, semi_block_size_output1,
	semi_block_size_input2, semi_block_size_output2, dim_output):

	######  Extract non-overlapping 3D cubes in Regression output space ######################################################
	###### also extracts the overlapping bigger 3D cubes in raw input space ##################################################
	###### this is iterating over the whole brain, it cannot control for obtaining 3D blocks just within an ROI ##############


	block_size_output = semi_block_size_output1 + semi_block_size_output2
	block_size_input = semi_block_size_input1 + semi_block_size_input2

	shape_of_input_data = input_image.shape

	num_cubes_dim1 = np.int(shape_of_input_data[0] // block_size_output)
	num_cubes_dim2 = np.int(shape_of_input_data[1] // block_size_output)
	num_cubes_dim3 = np.int(shape_of_input_data[2] // block_size_output)

	list_input_cubes = []
	list_output_cubes = []
	min_value_image = np.min(input_image)

	diff_semi_block1 = semi_block_size_input1 - semi_block_size_output1
	diff_semi_block2 = semi_block_size_input2 - semi_block_size_output2
	print('size of input image going in padding operation')
	print(input_image.shape)
	print(diff_semi_block1)
	print(diff_semi_block2)

	input_image_padded = np.pad(input_image, ((diff_semi_block1, diff_semi_block2),
		(diff_semi_block1, diff_semi_block2), (diff_semi_block1, diff_semi_block2),
		(0,0)), mode='constant', constant_values = min_value_image)

	for i in range(num_cubes_dim1):
		for j in range(num_cubes_dim2):
			for k in range(num_cubes_dim3):

				### extract segmentation space 3D cube ###
				list_output_cubes.append(np.ones((block_size_output,block_size_output,block_size_output,1))*output_image)
				print(list_output_cubes[-1].shape)

				### extract raw input space 3D cube ###
				volumetric_block = 	input_image_padded[block_size_output*i:(block_size_output*i+block_size_input),
					block_size_output*j:(block_size_output*j+block_size_input),
					block_size_output*k:(block_size_output*k+block_size_input),:]
				gender_3d_block = np.ones((block_size_input,
					block_size_input, block_size_input, 1)) * np.float(gender_image)
				whole_block = np.concatenate((volumetric_block, gender_3d_block),axis=-1)

				list_input_cubes.append(whole_block)
				print(list_input_cubes[-1].shape)

	list_output_cubes =  np.stack(list_output_cubes)
	list_input_cubes = np.stack(list_input_cubes)

	return list_input_cubes, list_output_cubes


def check_input_space(input_image, ROI_end_points, diff_semi_block):

	#########################################################################################################
	### input_image -- shape (121,145,121,2) -- if using both gm and wm from spm12 segmentations ############

	#########################################################################################################
	### diff_semi_block -- difference between semi_size_block_input_space and semi_size_block_output_space ##
	#########################################################################################################

	current_shape = input_image.shape

	##################################################
	### to be used at testing time -- if needed, 
	### it pads the input space so that the bigger input space of the U-NET is covered
	### as we iterate over the smaller output space 3D blocks of the U-NET

	### ROI_end_points -- dictionary 
	### ROi_end_points[0] = [lower_coord, upper_coord]
	### ROi_end_points[1] = [lower_coord, upper_coord]
	### ROi_end_points[2] = [lower_coord, upper_coord]

	padding_dimensions = []
	control = 0				

	for _ in range(3):

		dim_list = []

		################################
		### checks in the lower part ###
		################################

		if ROI_end_points[_][0] - diff_semi_block < 0:			
			
			dim_list.append(np.abs(ROI_end_points[_][0] - diff_semi_block))
			control+=1

		else:

			dim_list.append(0)

		################################
		### checks in the upper part ###
		################################

		if ROI_end_points[_][1] + diff_semi_block > (current_shape[_]-1):
			
			dim_list.append(np.abs(ROI_end_points[_][1] + diff_semi_block - current_shape[_]+1))
			control+=1

		else:

			dim_list.append(0)

		padding_dimensions.append(tuple(dim_list))

	if control > 0:

		padding_dimensions = tuple(padding_dimensions)
		padding_dimensions_extra = list(padding_dimensions)
		padding_dimensions_extra.append(tuple([0,0]))
		padding_dimensions_extra = tuple(padding_dimensions_extra)
		input_image = np.pad(input_image, padding_dimensions_extra, mode='constant', constant_values = 0.0)	
		#central_points = [central_points[_]+padding_dimensions[_][0] for _ in range(3)]

	return input_image


def extract_3D_cubes_input_seg_regression_ROI_bound(input_image, output_scalar, gender_image, semi_block_size_input1, semi_block_size_output1,
	semi_block_size_input2, semi_block_size_output2, dim_output, ROI_end_points, mask):

	### ROI_end_points -- dictionary 
	### ROi_end_points[0] = [lower_coord, upper_coord]
	### ROi_end_points[1] = [lower_coord, upper_coord]
	### ROi_end_points[2] = [lower_coord, upper_coord]

	########################################################################################
	##### input_image -- shape (121,145,121,2) -- usual scenario using both gm and wm ######
	########################################################################################

	block_size_output = semi_block_size_output1 + semi_block_size_output2
	block_size_input = semi_block_size_input1 + semi_block_size_input2
	diff_semi_block = np.abs(semi_block_size_output1 - semi_block_size_input1)

	print('size of difference between semi blocks')
	print(diff_semi_block)

	### crop the whole brain gm and wm images to obtain just the expanded ROI space ###################################
	### Warning -- this might be a bit non-optimal for weird shapes of ROIs, such as corpus callosm or hippocampus ####

	output_image = crop_image(input_image, ROI_end_points)
	output_mask = crop_image(np.expand_dims(mask,axis=-1), ROI_end_points)
	shape_of_data = output_image.shape

	print('shape of output image after bounding it to the ROI')
	print(shape_of_data)


	print('*** ROI end points ****')
	print(ROI_end_points[0])
	print(ROI_end_points[1])
	print(ROI_end_points[2])	


	print('padding needed for dimension 1')
	
	### dimension 1 ###
	diff_dim1 = shape_of_data[0] %  block_size_output
	if diff_dim1!=0:
		diff_dim1 = block_size_output - diff_dim1
	print(diff_dim1)

	print('padding needed for dimension 2')

	### dimension 2 ###
	diff_dim2 = shape_of_data[1] % block_size_output
	if diff_dim2!=0:
		diff_dim2 = block_size_output - diff_dim2
	print(diff_dim2)

	print('padding needed for dimension 3')
	
	### dimension 3 ###
	diff_dim3 = shape_of_data[2] % block_size_output
	if diff_dim3!=0:
		diff_dim3 = block_size_output - diff_dim3					
	print(diff_dim3)

	#####################################################################
	### pad output space so that it is divisible by block_size_output ###
	#####################################################################

	output_image = np.pad(array = output_image, pad_width = ((diff_dim1,0), (diff_dim2,0), (diff_dim3,0), (0,0)), mode='constant')
	output_mask = np.pad(array = output_mask, pad_width = ((diff_dim1,0), (diff_dim2,0), (diff_dim3,0), (0,0)), mode='constant')
	padding_output_space = [diff_dim1, diff_dim2, diff_dim3]

	###################################################################################################################################################################
	#### need to account that semi_block_size is quite big and for example using MNI structural looking at Cerebellum, it might go overboard with the selection #######
	###################################################################################################################################################################

	input_image = crop_as_much_as_possible_then_pad(input_image, ROI_end_points, diff_semi_block, padding_output_space)

	###################################################################################################################################
	###### Remainder -- to get from output_image_coordinates to input_image_coordinates add on all dimensions + diff_semi_block #######
	###################################################################################################################################


	######  Extract non-overlapping 3D cubes in Regression output space ###############################################
	###### also extracts the overlapping bigger 3D cubes in raw input space ###########################################
	###### extracts non-overlapping 3D blocks contrained by an ROI box  ###############################################

	shape_of_input_data = input_image.shape
	shape_of_output_data = output_image.shape

	print('size of input image')
	print(shape_of_input_data)
	
	print('size of output image')
	print(shape_of_output_data)	

	num_cubes_dim1 = np.int(shape_of_output_data[0] // block_size_output)
	num_cubes_dim2 = np.int(shape_of_output_data[1] // block_size_output)
	num_cubes_dim3 = np.int(shape_of_output_data[2] // block_size_output)

	list_input_cubes = []
	list_output_cubes = []
	min_value_image = np.min(input_image) ## obviously 0 for brain-age prediction using spm12 seg


	for i in range(num_cubes_dim1):
		for j in range(num_cubes_dim2):
			for k in range(num_cubes_dim3):

				#print('**************************************')

				###############################################
				### extract regression output space 3D cube ###
				###############################################

				list_output_cubes.append(np.ones((block_size_output,block_size_output,block_size_output,1)) * output_scalar)
				#print(list_output_cubes[-1].shape)

				#######################################
				### extract raw input space 3D cube ###
				#######################################

				volumetric_block = 	input_image[block_size_output*i:(block_size_output*(i+1) + 2 * diff_semi_block),
					block_size_output*j:(block_size_output*(j+1) + 2 * diff_semi_block),
					block_size_output*k:(block_size_output*(k+1) + 2 * diff_semi_block),:]
				#print(volumetric_block.shape)

				gender_3d_block = np.ones((block_size_input,
					block_size_input, block_size_input, 1)) * np.float(gender_image)
				whole_block = np.concatenate((volumetric_block, gender_3d_block),axis=-1)

				list_input_cubes.append(whole_block)
				#print(list_input_cubes[-1].shape)

	list_output_cubes = np.stack(list_output_cubes)
	list_input_cubes = np.stack(list_input_cubes)

	return list_input_cubes, list_output_cubes, shape_of_output_data, output_mask



