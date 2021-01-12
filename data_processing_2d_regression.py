import numpy as np
import tensorflow as tf
from collections import defaultdict
import random
from sklearn import preprocessing
import os

def apply_mask(array, mask):

	##### array --- (dim1, dim2)
	##### mask --- (dim1, dim2)
	##### masks the array so that the non-ROI part is zero ####

	array[mask < 1.0] = 0.0

	return array

def blockshaped(arr, nrows, ncols):

	############################################################    
	#### Used to get non-overlapping patches from an image #####
	############################################################

    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


def unblockshaped(arr, h, w):
	
	"""
	Return an array of shape (h, w) where
	h * w = arr.size

	If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
	then the returned array preserves the "physical" layout of the sublocks.
	"""
	
	n, nrows, ncols = arr.shape
	return (arr.reshape(h//nrows, -1, nrows, ncols)
		.swapaxes(1,2)
		.reshape(h, w))


def crop_2D_block(image, central_points, semi_block_size1, semi_block_size2):

	#### basically crops a small 2D cube from a bigger 2D object ####

	### image -- shape (height, width, channels)
	### central_points -- (c1,c2)
	### semi_block_size -- (l1,l2)

	plm = image[central_points[0]-semi_block_size1:central_points[0]+semi_block_size2,
		central_points[1]-semi_block_size1:central_points[1]+semi_block_size2,:]
	#print(plm.shape)

	return plm




def check_and_add_zero_padding_2d_image(input_image, output_image, central_points, semi_block_size1, semi_block_size2):

	#### checks if extracting a patch need padding or not 
	#### accounts for the case where the central_points are close to the boundary of the image and expands it with the minimum of the image

	### image -- shape (height, width, channels)
	### central_points -- (c1,c2)
	### semi_block_size -- (l1,l2)

	current_shape = input_image.shape 
	min_value_image = np.min(input_image)
	padding_dimensions = []
	control=0				

	for _ in range(2):

		dim_list = []

		if central_points[_]-semi_block_size1 < 0:			
			dim_list.append(np.abs(central_points[_]-semi_block_size1))
			control+=1

		else:

			dim_list.append(0)

		if central_points[_]+semi_block_size2 > current_shape[_]:
			dim_list.append(np.abs(central_points[_]+semi_block_size2 - current_shape[_]))
			control+=1
		else:
			dim_list.append(0)

		padding_dimensions.append(tuple(dim_list))


	if control > 0:

		padding_dimensions = tuple(padding_dimensions)
		padding_dimensions_extra = list(padding_dimensions)
		padding_dimensions_extra.append(tuple([0,0]))
		padding_dimensions_extra = tuple(padding_dimensions_extra)
		#print(padding_dimensions_extra)
		input_image = np.pad(input_image, padding_dimensions_extra, mode='constant', constant_values = min_value_image)
		output_image = np.pad(output_image, padding_dimensions_extra, mode='constant')
		central_points = [central_points[_]+padding_dimensions[_][0] for _ in range(2)]

	return input_image, output_image, central_points





######  Extract non-overlapping 2D patches in segmentation space #############
###### also extracts the overlapping bigger 2D patches in raw input space ####

def extract_2D_cubes_input_seg(input_image, output_image, semi_block_size_input1, semi_block_size_output1,
	semi_block_size_input2, semi_block_size_output2, dim_output):

	#### input_image -- shape (height, width, num_raw_modalities)
	#### output_image -- shape (height, width)

	block_size_output = semi_block_size_output1 + semi_block_size_output2
	block_size_input = semi_block_size_input1 + semi_block_size_input2

	shape_of_input_data = input_image.shape

	num_cubes_dim1 = np.int(shape_of_input_data[0] // block_size_output)
	num_cubes_dim2 = np.int(shape_of_input_data[1] // block_size_output)

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
		(diff_semi_block1, diff_semi_block2), (0,0)), mode='constant', constant_values = min_value_image)

	for i in range(num_cubes_dim1):
		for j in range(num_cubes_dim2):

			### extract segmentation space 3D cube ###
			list_output_cubes.append(output_image[block_size_output*i:block_size_output*(i+1),
				block_size_output*j:block_size_output*(j+1)])
			print(list_output_cubes[-1].shape)

			### extract raw input space 3D cube ###
			list_input_cubes.append(input_image_padded[block_size_output*i:(block_size_output*i+block_size_input),
				block_size_output*j:(block_size_output*j+block_size_input),:])
			print(list_input_cubes[-1].shape)

	list_output_cubes =  np.stack(list_output_cubes)
	list_output_cubes = output_transformation(list_output_cubes)
	shape_of_seg_output = list_output_cubes.shape
	list_output_cubes = list_output_cubes.reshape((-1,1))
	enc = preprocessing.OneHotEncoder()
	enc.fit(list_output_cubes)
	list_output_cubes = enc.transform(list_output_cubes).toarray()
	list_output_cubes =  list_output_cubes.reshape((shape_of_seg_output[0],shape_of_seg_output[1],shape_of_seg_output[2], dim_output))	
	list_input_cubes = np.stack(list_input_cubes)

	return list_input_cubes, list_output_cubes
