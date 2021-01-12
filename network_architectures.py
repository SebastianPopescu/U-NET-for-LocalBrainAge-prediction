import tensorflow as tf
import numpy as np
from propagate_layers import *
DTYPE=tf.float32

def UNET_network(inputul, num_encoding_layers, mode, keep_prob,
	num_layers_same_scale,  dim_output, num_filters, unet_type):

	print('********************')
	print('shape of input to UNET')
	print(inputul.get_shape().as_list())
	print('*********************')
	intermediate_layers = []
	intermediate_layers_shape = []

	############################################
	########## Encoding Part of UNET ###########
	############################################

	for _ in range(num_encoding_layers):

		###################################
		####### Same scale part ###########
		###################################

		for num_layer_same_scale in range(num_layers_same_scale):

			inputul = propagate_dropout(l=_, num_layer_same_scale = num_layer_same_scale, input_prev_layer = inputul, num_stride=1,
				dim_filter=3, num_filters = num_filters[_], 
				padding='valid', unet_type=unet_type, mode=mode,
				keep_prob=keep_prob, convolution_type='encoding', deconvolution_shape=None)

			print('********************')
			print('shape in the encoding part of UNET')
			print(inputul.get_shape().as_list())
			print('*********************')
		
		intermediate_layers.append(inputul)
		intermediate_layers_shape.append(inputul.get_shape().as_list())

		###################################
		##### Down-sampling part ##########
		###################################

		inputul = tf.nn.avg_pool3d(
			input = inputul,
			ksize = (1,2,2,2,1),
			strides = (1,2,2,2,1),
			padding = "VALID",
			data_format='NDHWC',
			name='average_pooling_3d')

		print('********************')
		print('shape after downsampling UNET')
		print(inputul.get_shape().as_list())
		print('*********************')

	######################
	#### Context part ####
	######################
	
	for num_layer_same_scale in range(num_layers_same_scale):

		inputul =propagate_dropout(l=0, num_layer_same_scale = num_layer_same_scale, input_prev_layer = inputul, num_stride=1,
			dim_filter=3, num_filters = num_filters[-1]*2, 
			padding='valid', unet_type=unet_type, mode=mode, keep_prob=keep_prob, convolution_type='context', deconvolution_shape=None)

		print('********************')
		print('shape in the context part of UNET')
		print(inputul.get_shape().as_list())
		print('*********************')

	############################################
	########## Decoding Part of UNET ###########
	############################################

	#### This is the Decoding part of the UNET ####
	for _, __ in zip(reversed(range(num_encoding_layers)), range(num_encoding_layers)):

		############################################
		############ Upsampling part ###############
		######################	######################

		inputul = tf.keras.layers.UpSampling3D(size=(2,2,2))(inputul)

		####################################
		########## Cropping part ###########
		####################################

		current_shape_upsampled = inputul.get_shape().as_list()
		if unet_type=='3D':

			cropping_starting_points = [ (intermediate_layers_shape[_][plm] - current_shape_upsampled[plm])//2 for plm in range(1,4)]

			inputul = tf.concat((inputul, tf.slice(intermediate_layers[_], [0, tf.cast(cropping_starting_points[0],tf.int32), tf.cast(cropping_starting_points[1],tf.int32),
				tf.cast(cropping_starting_points[2],tf.int32),0],[-1, tf.cast(current_shape_upsampled[1],tf.int32), tf.cast(current_shape_upsampled[2],tf.int32), 
				tf.cast(current_shape_upsampled[3],tf.int32),
				-1])),axis=-1)

		else:

			cropping_starting_points = [ (intermediate_layers_shape[_][plm] - current_shape_upsampled[plm])//2 for plm in range(1,3)]
			inputul = tf.concat((inputul, tf.slice(intermediate_layers[_], [0, tf.cast(cropping_starting_points[0],tf.int32), tf.cast(cropping_starting_points[1],tf.int32),
				0],[-1, tf.cast(current_shape_upsampled[1],tf.int32), tf.cast(current_shape_upsampled[2],tf.int32),
				-1])),axis=-1)				

		##########################################
		######## Same scale part #################
		##########################################

		for num_layer_same_scale in range(num_layers_same_scale):

			inputul =propagate_dropout(l=_, num_layer_same_scale = num_layer_same_scale, input_prev_layer = inputul,
				num_stride=1, dim_filter=3, num_filters = num_filters[_], 
				padding='valid', unet_type=unet_type, mode=mode, keep_prob=keep_prob,
				convolution_type='decoding', deconvolution_shape=intermediate_layers_shape[_])

			print('********************')
			print('shape of transformation in decoding part of UNET')
			print(inputul.get_shape().as_list())
			print('*********************')

	##########################################################	
	#### This is the last Convolution to get the output ######
	##########################################################
	inputul = propagate_last(l=0, num_layer_same_scale = 0, input_prev_layer = inputul, num_stride=1, dim_filter=1, num_filters = dim_output, 
		padding='valid', unet_type=unet_type, mode=mode, keep_prob=1.0, convolution_type='classification', deconvolution_shape=None)

	print('********************')
	print('shape of UNET output')
	print(inputul.get_shape().as_list())
	print('*********************')

	return inputul


def UNET_network_context_enhanced(inputul, num_encoding_layers, mode, keep_prob,
	num_layers_same_scale,  dim_output, num_filters, unet_type):

	print('********************')
	print('shape of input to UNET')
	print(inputul.get_shape().as_list())
	print('*********************')
	intermediate_layers = []
	intermediate_layers_shape = []

	############################################
	########## Encoding Part of UNET ###########
	############################################
	list_pred_context = []
	list_pred_context_global = []

	for _ in range(num_encoding_layers):

		###################################
		####### Same scale part ###########
		###################################

		for num_layer_same_scale in range(num_layers_same_scale):

			inputul = propagate_dropout(l=_, num_layer_same_scale = num_layer_same_scale, input_prev_layer = inputul, num_stride=1,
				dim_filter=3, num_filters = num_filters[_], 
				padding='valid', unet_type=unet_type, mode=mode,
				keep_prob=keep_prob, convolution_type='encoding', deconvolution_shape=None)

			print('********************')
			print('shape in the encoding part of UNET')
			print(inputul.get_shape().as_list())
			print('*********************')
		

		###################################################################
		#### Get Prediction at context level in the Downsampling part #####
		###################################################################

		if _!=0:

			with tf.variable_scope('context_level_predition_'+str(_), reuse = tf.AUTO_REUSE):

				list_pred_context.append(tf.layers.conv3d(inputs=inputul,
					filters=1, kernel_size=(1, 1, 1),
					strides=(1, 1, 1), padding='VALID', data_format='channels_last',
					dilation_rate=(1, 1, 1), activation=None, use_bias=False,
					kernel_initializer=tf.initializers.variance_scaling(),
					bias_initializer=tf.zeros_initializer(),
					kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
					bias_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
					activity_regularizer=None, kernel_constraint=None,
					bias_constraint=None, trainable=True,
					name='context_level_'+str(_)+'_conv3d_layer',
					reuse=tf.AUTO_REUSE))

				input_context = tf.reduce_mean(inputul, axis=[1,2,3])
				list_pred_context_global.append(tf.layers.dense(
					inputs = input_context,
					units = 1,
					activation=None,
					use_bias=False,
					kernel_initializer=tf.initializers.variance_scaling(),
					bias_initializer=tf.zeros_initializer(),
					kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
					bias_regularizer=None,
					activity_regularizer=None,
					kernel_constraint=None,
					bias_constraint=None,
					trainable=True,
					name='context_level_dense_layers',
					reuse=tf.AUTO_REUSE))


		intermediate_layers.append(inputul)
		intermediate_layers_shape.append(inputul.get_shape().as_list())

		###################################
		##### Down-sampling part ##########

		inputul = tf.nn.avg_pool3d(
			input = inputul,
			ksize = (1,2,2,2,1),
			strides = (1,2,2,2,1),
			padding = "VALID",
			data_format='NDHWC',
			name='average_pooling_3d')

		print('********************')
		print('shape after downsampling UNET')
		print(inputul.get_shape().as_list())
		print('*********************')

	######################
	#### Context part ####
	######################
	
	for num_layer_same_scale in range(num_layers_same_scale):

		inputul =propagate_dropout(l=0, num_layer_same_scale = num_layer_same_scale, input_prev_layer = inputul, num_stride=1,
			dim_filter=3, num_filters = num_filters[-1]*2, 
			padding='valid', unet_type=unet_type, mode=mode, keep_prob=keep_prob, convolution_type='context', deconvolution_shape=None)

		print('********************')
		print('shape in the context part of UNET')
		print(inputul.get_shape().as_list())
		print('*********************')


	#inputul_context = inputul

	##########################################
	#### Get Prediction at context level #####
	##########################################

	with tf.variable_scope('context_level_predition_3', reuse = tf.AUTO_REUSE):

	
		### This is the old way of doing Deep Supervision ###
		input_context = tf.reduce_mean(inputul, axis=[1,2,3])
		list_pred_context_global.append(tf.layers.dense(
			inputs = input_context,
			units = 1,
			activation=None,
			use_bias=False,
			kernel_initializer=tf.initializers.variance_scaling(),
			bias_initializer=tf.zeros_initializer(),
			kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
			bias_regularizer=None,
			activity_regularizer=None,
			kernel_constraint=None,
			bias_constraint=None,
			trainable=True,
			name='context_level_dense_layers',
			reuse=tf.AUTO_REUSE))
		

		list_pred_context.append(tf.layers.conv3d(inputs=inputul,
			filters=1,
			kernel_size=(1, 1, 1),
			strides=(1, 1, 1),
			padding='VALID',
			data_format='channels_last',
			dilation_rate=(1, 1, 1),
			activation=None,
			use_bias=False,
			kernel_initializer=tf.initializers.variance_scaling(),
			bias_initializer=tf.zeros_initializer(),
			kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
			bias_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
			activity_regularizer=None,
			kernel_constraint=None,
			bias_constraint=None,
			trainable=True,
			name='context3_level_conv3d_layer',
			reuse=tf.AUTO_REUSE
			))

	############################################
	########## Decoding Part of UNET ###########
	############################################

	#### This is the Decoding part of the UNET ####
	for _, __ in zip(reversed(range(num_encoding_layers)), range(num_encoding_layers)):

		############################################
		############ Upsampling part ###############
		######################	######################

		inputul = tf.keras.layers.UpSampling3D(size=(2,2,2))(inputul)
		'''
		if __ !=0:

			####################################
			### add context-level embedding ####
			####################################

			inputul_additional = tf.keras.layers.UpSampling3D(size=(2+(__*2),2+(__*2),2+(__*2)))(inputul_context)
			
			############################################################
			#### crop the upsampled version of the context embedding ###
			############################################################

			current_shape_upsampled = inputul_additional.get_shape().as_list()
			current_shape_upsampled2 = inputul.get_shape().as_list()
			cropping_starting_points = [ np.abs(current_shape_upsampled2[plm] - current_shape_upsampled[plm])//2 for plm in range(1,4)]
			inputul = tf.concat((inputul, tf.slice(inputul_additional, [0, tf.cast(cropping_starting_points[0],tf.int32), tf.cast(cropping_starting_points[1],tf.int32),
				tf.cast(cropping_starting_points[2],tf.int32),0],[-1, tf.cast(current_shape_upsampled2[1],tf.int32), tf.cast(current_shape_upsampled2[2],tf.int32), 
				tf.cast(current_shape_upsampled2[3],tf.int32),
				-1])), axis=-1)
		'''
		####################################
		########## Cropping part ###########
		####################################

		current_shape_upsampled = inputul.get_shape().as_list()
		if unet_type=='3D':

			cropping_starting_points = [ (intermediate_layers_shape[_][plm] - current_shape_upsampled[plm])//2 for plm in range(1,4)]

			inputul = tf.concat((inputul, tf.slice(intermediate_layers[_], [0, tf.cast(cropping_starting_points[0],tf.int32), tf.cast(cropping_starting_points[1],tf.int32),
				tf.cast(cropping_starting_points[2],tf.int32),0],[-1, tf.cast(current_shape_upsampled[1],tf.int32), tf.cast(current_shape_upsampled[2],tf.int32), 
				tf.cast(current_shape_upsampled[3],tf.int32),
				-1])),axis=-1)

		else:

			cropping_starting_points = [ (intermediate_layers_shape[_][plm] - current_shape_upsampled[plm])//2 for plm in range(1,3)]
			inputul = tf.concat((inputul, tf.slice(intermediate_layers[_], [0, tf.cast(cropping_starting_points[0],tf.int32), tf.cast(cropping_starting_points[1],tf.int32),
				0],[-1, tf.cast(current_shape_upsampled[1],tf.int32), tf.cast(current_shape_upsampled[2],tf.int32),
				-1])),axis=-1)				

		##########################################
		######## Same scale part #################
		##########################################

		for num_layer_same_scale in range(num_layers_same_scale):

			inputul =propagate_dropout(l=_, num_layer_same_scale = num_layer_same_scale, input_prev_layer = inputul,
				num_stride=1, dim_filter=3, num_filters = num_filters[_], 
				padding='valid', unet_type=unet_type, mode=mode, keep_prob=keep_prob,
				convolution_type='decoding', deconvolution_shape=intermediate_layers_shape[_])

			print('********************')
			print('shape of transformation in decoding part of UNET')
			print(inputul.get_shape().as_list())
			print('*********************')


		if __ == 0:

			###########################################
			#### Get Prediction at context level4 #####
			###########################################

			with tf.variable_scope('context_level_predition_4', reuse = tf.AUTO_REUSE):
				
				input_context = tf.reduce_mean(inputul, axis=[1,2,3])
				list_pred_context_global.append(tf.layers.dense(
					inputs = input_context,
					units = 1,
					activation=None,
					use_bias=False,
					kernel_initializer=tf.initializers.variance_scaling(),
					bias_initializer=tf.zeros_initializer(),
					kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
					bias_regularizer=None,
					activity_regularizer=None,
					kernel_constraint=None,
					bias_constraint=None,
					trainable=True,
					name='context_level_dense_layers2',
					reuse=tf.AUTO_REUSE))
				

				list_pred_context.append(tf.layers.conv3d(inputs=inputul,
					filters=1,
					kernel_size=(1, 1, 1),
					strides=(1, 1, 1),
					padding='VALID',
					data_format='channels_last',
					dilation_rate=(1, 1, 1),
					activation=None,
					use_bias=False,
					kernel_initializer=tf.initializers.variance_scaling(),
					bias_initializer=tf.zeros_initializer(),
					kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
					bias_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
					activity_regularizer=None,
					kernel_constraint=None,
					bias_constraint=None,
					trainable=True,
					name='context_level4_conv3d_layer',
					reuse=tf.AUTO_REUSE
					))


	##########################################################	
	#### This is the last Convolution to get the output ######
	##########################################################
	inputul = propagate_last(l=0, num_layer_same_scale = 0, input_prev_layer = inputul, num_stride=1, dim_filter=1, num_filters = dim_output, 
		padding='valid', unet_type=unet_type, mode=mode, keep_prob=1.0, convolution_type='classification', deconvolution_shape=None)

	print('********************')
	print('shape of UNET output')
	print(inputul.get_shape().as_list())
	print('*********************')

	return inputul, list_pred_context, list_pred_context_global


def FCN_one_path_network(inputul, num_encoding_layers, mode, keep_prob,
	num_layers_same_scale,  dim_output, num_filters, unet_type):

	print('********************')
	print('shape of input to UNET')
	print(inputul.get_shape().as_list())
	print('*********************')

	############################################
	########## Encoding Part of FCN ############
	############################################

	### num_encoding_layers = 8 ###
	num_filters = [60,60,80,80,80,80,100,100]

	for _ in range(num_encoding_layers):

		###################################
		####### Same scale part ###########
		###################################

		inputul = propagate_dropout(l=_, num_layer_same_scale = 0, input_prev_layer = inputul, num_stride=1,
			dim_filter=3, num_filters = num_filters[_], 
			padding='valid', unet_type=unet_type, mode=mode, keep_prob=keep_prob, convolution_type='encoding', deconvolution_shape=None)

		print('********************')
		print('shape in the encoding part of FCN')
		print(inputul.get_shape().as_list())
		print('*********************')

	##########################################################	
	#### Fully Connected Layers part #########################
	##########################################################

	num_filters = [300, 300]
	for _ in range(2):
	
		inputul = propagate_dropout(l=_, num_layer_same_scale = 0, input_prev_layer = inputul, num_stride=1,
			dim_filter=1, num_filters = num_filters[_], 
			padding='valid', unet_type=unet_type, mode=mode, keep_prob=keep_prob, convolution_type='fully_connected', deconvolution_shape=None)

	print('********************')
	print('shape in the fully_connected_part of FCN')
	print(inputul.get_shape().as_list())
	print('*********************')

	inputul = propagate_last(l=0, num_layer_same_scale = 0, input_prev_layer = inputul, num_stride=1, dim_filter=1,
		num_filters = dim_output, 
		padding='valid', unet_type=unet_type, mode=mode, keep_prob=1.0, convolution_type='classification', deconvolution_shape=None)

	print('********************')
	print('shape in the fully_connected_part of FCN')
	print(inputul.get_shape().as_list())
	print('*********************')

	return inputul

	


