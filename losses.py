import tensorflow as tf
import numpy as np
from collections import defaultdict
from data_processing import *

def re_error(inputul, outputul, dim_output, unet_type):

	##################################
	####### Segmentation Loss ########
	##################################

	### inputul -- shape (num_batch, height, width, depth, num_classes)
	### outputul -- shape (num_batch, height, width, depth, num_classes)

	shape_of_input_data = inputul.get_shape().as_list()
	shape_of_label_data = outputul.get_shape().as_list()

	inputul = tf.reshape(inputul, [-1, dim_output])
	outputul = tf.reshape(outputul, [-1, dim_output])

	re_error_final = tf.nn.softmax_cross_entropy_with_logits(logits=inputul, labels=outputul)

	return tf.reduce_mean(re_error_final), tf.nn.softmax(inputul)

def mae_error(inputul, outputul, dim_output, unet_type, masks):

	################################
	####### Regression Loss ########
	################################

	### inputul -- shape (num_batch, height, width, depth, 1)
	### outputul -- shape (num_batch, height, width, depth, 1)
	### mask -- shape (num_batch, height, width, depth, 1)

	shape_of_input_data = inputul.get_shape().as_list()
	shape_of_label_data = outputul.get_shape().as_list()

	inputul = tf.reshape(inputul, [-1, ])
	outputul = tf.reshape(outputul, [-1, ])
	masks = tf.reshape(masks, [-1,])
	print('summation of booean mask')
	masked_inputul = tf.boolean_mask(inputul, masks)
	masked_outputul = tf.boolean_mask(outputul, masks)
	print(masked_outputul)
	print(masked_inputul)

	re_error_final = tf.abs(masked_inputul - masked_outputul)

	return tf.reduce_mean(re_error_final), inputul, outputul


def mae_error_context_enhanced(inputul, outputul, dim_output, unet_type, masks, loss_weights):

	####################################################
	####### Regression Loss -- context enhanced ########
	####################################################

	### inputul[0] -- shape (num_batch, height, width, depth, 1)
	### inputul[1...4] -- shape (num_batch, height_context_..., width_context_..., depth_context_..., 1)

	### outputul[0] -- shape (num_batch, height, width, depth, 1)
	### outputul[1...4] -- shape (num_batch, height_context_..., width_context_..., depth_context_..., 1)

	### mask[0] -- shape (num_batch, height, width, depth, 1)
	### mask[1..4] -- shape (num_batch, height_context_..., width_context_..., depth_context_..., 1)

	#### Unpack everything ####

	inputul_context = inputul[1] 
	inputul_context = [inputul_context[_] for _ in range(3)]

	inputul_context_global = inputul[2] 
	inputul_context_global = [inputul_context_global[_] for _ in range(3)]

	inputul = inputul[0]

	outputul_context = [outputul[_] for _ in range(1,4)]
	outputul_global = outputul[-1]
	outputul = outputul[0]

	shape_of_input_data = inputul.get_shape().as_list()
	shape_of_label_data = outputul.get_shape().as_list()

	mask = masks[0]
	mask_context = [masks[_] for _ in range(1,4)]

	inputul = tf.reshape(inputul, [-1, ])
	outputul = tf.reshape(outputul, [-1, ])
	mask = tf.reshape(mask, [-1,])
	masked_inputul = tf.boolean_mask(inputul, mask)
	masked_outputul = tf.boolean_mask(outputul, mask)

	inputul_context = [tf.reshape(inputul_context[_], [-1, ]) for _ in range(3)]
	outputul_context = [tf.reshape(outputul_context[_], [-1, ]) for _ in range(3)]
	mask_context = [tf.reshape(mask_context[_], [-1,]) for _ in range(3)]

	masked_inputul_context = [tf.boolean_mask(inputul_context[_], mask_context[_]) for _ in range(3)]
	masked_outputul_context = [tf.boolean_mask(outputul_context[_], mask_context[_]) for _ in range(3)]

	re_error_final_regression = tf.reduce_mean(tf.abs(masked_inputul - masked_outputul))
	re_error_final_context = [tf.reduce_mean(tf.abs(masked_inputul_context[_] - masked_outputul_context[_])) for _ in range(3)] 
	re_error_final_context_global =  [tf.reduce_mean(tf.abs(inputul_context_global[_] - outputul_global)) for _ in range(3)] 

	re_error_final = loss_weights[0] * re_error_final_regression 
	for _ in range(3):
		re_error_final += loss_weights[_+1] * re_error_final_context[_]
		re_error_final += loss_weights[_+4] * re_error_final_context_global[_]

	return re_error_final, re_error_final_regression, re_error_final_context, re_error_final_context_global


