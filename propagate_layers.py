import tensorflow as tf
import numpy as np
from dropblock import DropBlock3D

def condition(X):

	return X + tf.eye(tf.shape(X)[0]) * 1e-3

def propagate_dropblock(l, num_layer_same_scale, input_prev_layer, num_stride, dim_filter, num_filters, 
	padding, unet_type, mode, keep_prob, convolution_type, deconvolution_shape=None):

	#########################################################
	###### this is using DropBlock for regularization #######
	#########################################################

	with tf.variable_scope( str(convolution_type)+'_layer_'+str(l)+'_dim_'+str(num_layer_same_scale), reuse=tf.AUTO_REUSE):

		if unet_type=='3D':

			if convolution_type=='upsampling':

				input_prev_layer = tf.layers.conv3d_transpose(
					inputs = input_prev_layer,
					filters = num_filters,
					kernel_size = (dim_filter,dim_filter, dim_filter),
					strides=(num_stride, num_stride, num_stride),
					padding=padding,
					data_format='channels_last',
					activation=None,
					use_bias=True,
					kernel_initializer=tf.initializers.variance_scaling(),
					bias_initializer=tf.zeros_initializer(),
					kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
					bias_regularizer=None,
					activity_regularizer=None,
					kernel_constraint=None,
					bias_constraint=None,
					trainable=True,
					name='deconv3d_layer',
					reuse=tf.AUTO_REUSE)

			else:

				input_prev_layer = tf.layers.conv3d(inputs=input_prev_layer,
					filters=num_filters,
					kernel_size=(dim_filter, dim_filter, dim_filter),
					strides=(num_stride, num_stride, num_stride),
					padding=padding,
					data_format='channels_last',
					dilation_rate=(1, 1, 1),
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
					name='conv3d_layer',
					reuse=tf.AUTO_REUSE
					)

		else:

			if convolution_type=='upsampling':

				input_prev_layer = tf.layers.conv2d_transpose(
					inputs = input_prev_layer,
					filters = num_filters,
					kernel_size = (dim_filter, dim_filter),
					strides=(num_stride, num_stride),
					padding=padding,
					data_format='channels_last',
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
					name='deconv2d_layer',
					reuse=tf.AUTO_REUSE)

			else:

				input_prev_layer = tf.layers.conv2d(inputs=input_prev_layer,
					filters=num_filters,
					kernel_size=(dim_filter, dim_filter),
					strides=(num_stride, num_stride),
					padding=padding,
					data_format='channels_last',
					dilation_rate=(1, 1),
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
					name='conv2d_layer',
					reuse=tf.AUTO_REUSE
					)
		'''
		input_prev_layer = tf.layers.batch_normalization(
			inputs = input_prev_layer,
			axis=-1,
			momentum=0.99,
			epsilon=0.001,
			center=True,
			scale=True,
			beta_initializer=tf.zeros_initializer(),
			gamma_initializer=tf.ones_initializer(),
			moving_mean_initializer=tf.zeros_initializer(),
			moving_variance_initializer=tf.ones_initializer(),
			beta_regularizer=None,
			gamma_regularizer=None,
			beta_constraint=None,
			gamma_constraint=None,
			training=mode,
			trainable=True,
			name='batch_norm',
			reuse=tf.AUTO_REUSE,
			renorm=False,
			renorm_clipping=None,
			renorm_momentum=0.99,
			fused=None,
			virtual_batch_size=None,
			adjustment=None
		)
		'''
		input_prev_layer = tf.nn.leaky_relu(features = input_prev_layer, alpha = 0.1, name='leaky_relu')
		drop_block = DropBlock3D(keep_prob=keep_prob, block_size=3)
		input_prev_layer = drop_block(input_prev_layer, True)
		#input_prev_layer = DropBlock(inputul = input_prev_layer, keep_prob=keep_prob, block_size=3)
		
	return input_prev_layer

def propagate_last(l, num_layer_same_scale, input_prev_layer, num_stride, dim_filter, num_filters, 
	padding, unet_type, mode, keep_prob, convolution_type, deconvolution_shape=None):

	with tf.variable_scope( str(convolution_type)+'_layer_'+str(l)+'_dim_'+str(num_layer_same_scale),reuse=tf.AUTO_REUSE):

		if unet_type=='3D':

			input_prev_layer = tf.layers.conv3d(inputs=input_prev_layer,
				filters=num_filters,
				kernel_size=(dim_filter, dim_filter, dim_filter),
				strides=(num_stride, num_stride, num_stride),
				padding=padding,
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
				name='conv3d_layer',
				reuse=tf.AUTO_REUSE
				)

		else:

			input_prev_layer = tf.layers.conv2d(inputs=input_prev_layer,
				filters=num_filters,
				kernel_size=(dim_filter, dim_filter),
				strides=(num_stride, num_stride),
				padding=padding,
				data_format='channels_last',
				dilation_rate=(1, 1),
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
				name='conv2d_layer',
				reuse=tf.AUTO_REUSE
				)

	return input_prev_layer


def propagate_dropout(l, num_layer_same_scale, input_prev_layer, num_stride, dim_filter, num_filters, 
	padding, unet_type, mode, keep_prob, convolution_type, deconvolution_shape=None):

	#################################################
	##### this is using channel-wise dropout ########
	#################################################

	with tf.variable_scope( str(convolution_type)+'_layer_'+str(l)+'_dim_'+str(num_layer_same_scale), reuse=tf.AUTO_REUSE):

		if unet_type=='3D':

			if convolution_type=='upsampling':

				input_prev_layer = tf.layers.conv3d_transpose(
					inputs = input_prev_layer,
					filters = num_filters,
					kernel_size = (dim_filter,dim_filter, dim_filter),
					strides=(num_stride, num_stride, num_stride),
					padding=padding,
					data_format='channels_last',
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
					name='deconv3d_layer',
					reuse=tf.AUTO_REUSE)

			else:

				input_prev_layer = tf.layers.conv3d(inputs=input_prev_layer,
					filters=num_filters,
					kernel_size=(dim_filter, dim_filter, dim_filter),
					strides=(num_stride, num_stride, num_stride),
					padding=padding,
					data_format='channels_last',
					dilation_rate=(1, 1, 1),
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
					name='conv3d_layer',
					reuse=tf.AUTO_REUSE
					)

		else:

			if convolution_type=='upsampling':

				input_prev_layer = tf.layers.conv2d_transpose(
					inputs = input_prev_layer,
					filters = num_filters,
					kernel_size = (dim_filter, dim_filter),
					strides=(num_stride, num_stride),
					padding=padding,
					data_format='channels_last',
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
					name='deconv2d_layer',
					reuse=tf.AUTO_REUSE)

			else:

				input_prev_layer = tf.layers.conv2d(inputs=input_prev_layer,
					filters=num_filters,
					kernel_size=(dim_filter, dim_filter),
					strides=(num_stride, num_stride),
					padding=padding,
					data_format='channels_last',
					dilation_rate=(1, 1),
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
					name='conv2d_layer',
					reuse=tf.AUTO_REUSE
					)

		### input_prev_layer -- shape (num_batch, height, width, depth, num_channels)
		
		##############################################
		######### Squeeze and Excitation Block #######
		##############################################

		with tf.variable_scope('squeeze_excite_block', reuse = tf.AUTO_REUSE):

			se = tf.reduce_mean(input_prev_layer, axis = [1,2,3], keepdims = False)
			se = tf.layers.dense(inputs = se, units = num_filters/8,					
				activation = None,
				use_bias = False,
				kernel_initializer = tf.initializers.variance_scaling(),
				bias_initializer = tf.zeros_initializer(),
				kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-4),
				trainable=True,
				name='se_dense_layer_1',
				reuse = tf.AUTO_REUSE)
			se = tf.nn.leaky_relu(features = se, alpha = 0.1, name='leaky_relu')		
			se = tf.layers.dense(inputs = se, units = num_filters,					
				activation = None, use_bias = False,
				kernel_initializer = tf.initializers.variance_scaling(),
				bias_initializer = tf.zeros_initializer(),
				kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-4),
				trainable=True, name='se_dense_layer_2',
				reuse = tf.AUTO_REUSE)
			se = tf.math.sigmoid(se)	
			se = tf.reshape(se, [tf.shape(se)[0], 1, 1, 1, tf.shape(se)[1]])
		input_prev_layer = input_prev_layer * se

		batch_size = tf.shape(input_prev_layer)[0]
		input_prev_layer = tf.nn.dropout(
			x = input_prev_layer,
			keep_prob = keep_prob,
			noise_shape = [batch_size,1,1,1,input_prev_layer.get_shape().as_list()[-1]])

		input_prev_layer = tf.nn.leaky_relu(features = input_prev_layer, alpha = 0.1, name='leaky_relu')
		
	return input_prev_layer



