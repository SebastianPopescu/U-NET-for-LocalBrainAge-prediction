# -*- coding: utf-8 -*-
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
from sklearn.preprocessing import binarize
from sklearn.metrics import accuracy_score,confusion_matrix
#import umap
from sklearn.manifold import TSNE as tsne
DTYPE=tf.float32
import subprocess
from sklearn.feature_extraction.image import extract_patches_2d
import nibabel as nib
from data_processing_3d_regression import *
from loading_data import *
from network_architectures import *
from propagate_layers import *
from losses import *

def extract_3d_blocks_training_regression(inputul, outputul, iteration,
	block_size_input, block_size_output, list_block_size_output_context, mask,
	num_subjects, num_voxels_per_subject, gender, indices_structural_plm):

	### gender(num_batch, 1)
	### size of brain_scan (121, 145, 121)
	### mask -- (121, 145, 121) -- if using both GM and WM
	### inputul -- shape (num_batch, width, height, depth, num_imaging_modalities)
	### outputul -- shape (num_batch, 1)
	### current_shape = mask.shape 

	lista = np.arange(len(inputul.keys()))
	np.random.seed(iteration)
	np.random.shuffle(lista)
	current_index = lista[:num_subjects]
	semi_block_size_input = int(block_size_input//2)
	semi_block_size_input2 = block_size_input - semi_block_size_input
	semi_block_size_output = int(block_size_output//2)
	semi_block_size_output2 = block_size_output - semi_block_size_output

	list_semi_block_size_output_context = [int(list_block_size_output_context[_]//2) for _ in range(3)]
	list_semi_block_size_output2_context = [list_block_size_output_context[_] - list_semi_block_size_output_context[_] for _ in range(3)]

	list_blocks_input = []
	list_blocks_segmentation = []
	list_block_masks = []
	list_block_masks_context = defaultdict()
	list_blocks_segmentation_context = defaultdict()
	list_age = []

	for _ in range(3):
		list_block_masks_context[_] = []
		list_blocks_segmentation_context[_] = []

	for _ in current_index:

		##### iterating over brain scans #####
		### pad current input and output scan to avoid problems ####
		current_input = inputul[_]
		current_output = outputul[_]
		current_gender = gender[_]
		

		#### shape of current scan ####
		current_shape = inputul[_].shape

		#indices_tumor = np.where(mask[...] > 0.0
		indices_tumor_dim1 = indices_structural_plm[0]
		indices_tumor_dim2 = indices_structural_plm[1]
		indices_tumor_dim3 = indices_structural_plm[2]
				
		list_of_random_places = random.sample(range(0, len(indices_tumor_dim1)), num_voxels_per_subject)

		for __ in range(num_voxels_per_subject):

			central_points = [indices_tumor_dim1[list_of_random_places[__]],
				indices_tumor_dim2[list_of_random_places[__]], indices_tumor_dim3[list_of_random_places[__]]]
			print(central_points)
			plm = check_mask(mask, central_points, semi_block_size_output, semi_block_size_output2)
			print(plm.shape)	

			for current_iteration in range(3):
				plm_context = check_mask(mask, central_points, list_semi_block_size_output_context[current_iteration], 
					list_semi_block_size_output2_context[current_iteration])
				print(plm_context.shape)	
				list_block_masks_context[current_iteration].append(plm_context > 0.0)

			current_input_padded, central_points = check_and_add_zero_padding_regression(current_input,central_points,
				semi_block_size_input, semi_block_size_input2)
			list_blocks_segmentation.append(np.ones((block_size_output,block_size_output,block_size_output,1))*current_output)
			for current_iteration in range(3):

				list_blocks_segmentation_context[current_iteration].append(np.ones((list_block_size_output_context[current_iteration],
					list_block_size_output_context[current_iteration],
					list_block_size_output_context[current_iteration],1))*current_output)
		
			cropped_input_block = crop_3D_block(current_input_padded, central_points, semi_block_size_input, semi_block_size_input2)
			print(cropped_input_block.shape)
			gender_3d_block = np.ones((block_size_input,
				block_size_input, block_size_input, 1)) * np.float(current_gender)
			cropped_input_block = np.concatenate((cropped_input_block,gender_3d_block),axis=-1)
			
			list_blocks_input.append(cropped_input_block)
			list_block_masks.append(plm > 0.0)
			list_age.append(current_output)

	list_blocks_input = np.stack(list_blocks_input)
	list_blocks_segmentation = np.stack(list_blocks_segmentation)
	list_block_masks = np.stack(list_block_masks)
	for current_iteration in range(3):
		list_block_masks_context[current_iteration] = np.stack(list_block_masks_context[current_iteration])
		list_blocks_segmentation_context[current_iteration] = np.stack(list_blocks_segmentation_context[current_iteration])
	list_age = np.stack(list_age)
	list_age = np.reshape(list_age, [-1,1])

	return list_blocks_input, list_blocks_segmentation, list_block_masks, list_blocks_segmentation_context, list_block_masks_context, list_age


def timer(start,end):
       hours, rem = divmod(end-start, 3600)
       minutes, seconds = divmod(rem, 60)
       print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

class UNET_Dropout_ROI_Context_Enhanced(object):

	#############################################################
	######### 3D data -- Regression --Context Enhanced ##########
	#############################################################

	def __init__(self, dim_input, dim_output, num_iterations, num_encoding_layers, 
		num_batch, num_filters, dim_filter, num_stride, 
		use_epistemic_uncertainty, size_cube_input, size_cube_output, 
		learning_rate, num_layers_same_scale, import_model, iteration_restored, unet_type, keep_prob, mean_age,
		num_averaged_gradients, num_subjects, num_voxels_per_subject, testing_time):

		self.testing_time = testing_time
		self.num_subjects = num_subjects
		self.num_voxels_per_subject = num_voxels_per_subject
		self.num_averaged_gradients = num_averaged_gradients
		self.mean_age = mean_age
		self.keep_prob = keep_prob #### keepprob for DropBlock #####
		self.unet_type = unet_type #### could be "2D" or "3D" ####  
		self.iteration_restored = iteration_restored #### the iteration/epoch at which we are retriving the saved model ####
		self.import_model = import_model ##### boolean, wheter to use for training or testing
		self.num_layers_same_scale = num_layers_same_scale #### the number of layers at the same scale inside the UNET architecture
		self.learning_rate = learning_rate #### constant learning rate to be used
		self.size_cube_input = size_cube_input #### shape of the input data -- scalar
		self.size_cube_output = size_cube_output ### shaoe of the output data -- scalar --- you have to calculate it
		self.use_epistemic_uncertainty = use_epistemic_uncertainty #### boolean, wheter to compute epistemic uncertaintiy
		self.num_stride = num_stride #### scalar, num stride -- I thinks its useless 
		self.dim_filter = dim_filter #### scalar, usuallY 3
		self.num_batch = num_batch #### size of minibatch
		self.dim_input = dim_input #### number of input channels
		self.dim_output = dim_output #### number of classes for classification
		self.num_iterations = num_iterations #### number of training iterations
		self.num_encoding_layers = num_encoding_layers ### scalar , number of scales for UNET
		self.num_filters = num_filters #### number of filters at each convolution operation
		

	def setup_train(self):

		outputul_list = [self.Y_train]
		outputul_list.extend(self.list_Y_train_context)
		outputul_list.append(self.Y_train_global)

		masks_list = [self.X_train_mask]
		masks_list.extend(self.list_X_train_mask_context)

		self._loss_op, self.mae_training, self.list_mae_training_context, self.list_mae_training_context_global = mae_error_context_enhanced(inputul = UNET_network_context_enhanced(inputul = self.X_train,
			num_encoding_layers = self.num_encoding_layers, unet_type = self.unet_type, mode=True,
			keep_prob = self.keep_prob, 
			num_layers_same_scale = self.num_layers_same_scale, dim_output = self.dim_output, num_filters = self.num_filters),
			outputul = outputul_list, 
			unet_type = self.unet_type, dim_output = self.dim_output, 
			masks = masks_list, loss_weights = self.loss_weights)	

		extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

		if self.import_model:
			self.global_step = tf.Variable(self.iteration_restored, trainable = False)
		else:
			self.global_step = tf.Variable(0, trainable = False)
		starter_learning_rate = self.learning_rate
		learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step, 100000, 0.1, staircase=True)
		# Passing global_step to minimize() will increment it at each step.

		if self.num_averaged_gradients == 1:

			with tf.control_dependencies(extra_update_ops):
				self._train_op = tf.train.AdamOptimizer(learning_rate).minimize(self._loss_op, global_step = self.global_step)

		else:

			# here 'train_op' only applies gradients passed via placeholders stored
			# in 'grads_placeholders. The gradient computation is done with 'grad_op'.
			optimizer = tf.train.AdamOptimizer(learning_rate)
			with tf.control_dependencies(extra_update_ops):

				grads_and_vars = optimizer.compute_gradients(self._loss_op)
			
			avg_grads_and_vars = []
			self._grad_placeholders = []
			for grad, var in grads_and_vars:
				grad_ph = tf.placeholder(grad.dtype, grad.shape)
				self._grad_placeholders.append(grad_ph)
				avg_grads_and_vars.append((grad_ph, var))

			self._grad_op = [x[0] for x in grads_and_vars]
			self._train_op = optimizer.apply_gradients(avg_grads_and_vars)
			self._gradients = [] # list to store gradients

	def train(self, session, X_train_feed, Y_train_feed, X_training_mask_feed, list_Y_train_context_feed,
		list_X_training_mask_feed_context, Y_train_global):

		feed_dict = {
			self.X_train: X_train_feed,
			self.Y_train: Y_train_feed,
			self.X_train_mask : X_training_mask_feed,
			self.Y_train_global : Y_train_global}
		for _ in range(3):
			dictionar = {
				self.list_Y_train_context[_] : list_Y_train_context_feed[_],
				self.list_X_train_mask_context[_] : list_X_training_mask_feed_context[_]}
			feed_dict.update(dictionar)

		if self.num_averaged_gradients == 1:
			loss, _ = session.run([self._loss_op, self._train_op], feed_dict = feed_dict)

		else:
			
			loss, grads = session.run([self._loss_op, self._grad_op], feed_dict = feed_dict)
			self._gradients.append(grads)
			
			if len(self._gradients) == self.num_averaged_gradients:
				for i, placeholder in enumerate(self._grad_placeholders):
		  			feed_dict[placeholder] = np.stack([g[i] for g in self._gradients], axis=0).mean(axis=0)
				session.run(self._train_op, feed_dict=feed_dict)
				self._gradients = []

		return loss

	def session_TF(self, X_training, Y_training, gender_training, X_testing, Y_testing, gender_testing, mask,
		affine, dataset_name, X_testing_names):

	
		#### get the structural atlas ####
		structural_atlas_object = nib.load('./data/combined_atlas.nii.gz')
		structural_atlas_data = structural_atlas_object.get_data()

		indices_structural = np.where(structural_atlas_data == 1.0)
		indices_X = indices_structural[0]
		indices_Y = indices_structural[1]
		indices_Z = indices_structural[2]
		
		ROI_end_points = defaultdict()
		ROI_end_points[0] = [np.min(indices_X), np.max(indices_X)]
		ROI_end_points[1] = [np.min(indices_Y), np.max(indices_Y)]
		ROI_end_points[2] = [np.min(indices_Z), np.max(indices_Z)]

		num_voxels_structural_ROI = len(indices_structural[0])
		
		print('*************************')
		print('number of voxels for ROI :'+str(num_voxels_structural_ROI))
		print('*************************')
		
		gpu_options = tf.GPUOptions(allow_growth=True)
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

		if not self.testing_time:

			self.loss_weights = [0.5, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]

			self.X_train = tf.placeholder(tf.float32, shape=(None, self.size_cube_input, self.size_cube_input,
				self.size_cube_input, self.dim_input), name='X_train')
			self.Y_train = tf.placeholder(tf.float32, shape=(None, self.size_cube_output, self.size_cube_output,
				self.size_cube_output, self.dim_output), name='Y_train')			
			self.X_train_mask = tf.placeholder(tf.bool, shape=(None, self.size_cube_output, self.size_cube_output,
				self.size_cube_output), name='X_train_mask')

			self.X_test = tf.placeholder(tf.float32, shape=(None, self.size_cube_input, self.size_cube_input, self.size_cube_input,
				self.dim_input), name='X_test')
			self.Y_test = tf.placeholder(tf.float32, shape=(None, self.size_cube_output, self.size_cube_output, self.size_cube_output,
				self.dim_output), name='Y_test')
			self.X_test_mask = tf.placeholder(tf.bool, shape=(None, self.size_cube_output, self.size_cube_output, self.size_cube_output),
				name='X_test_mask')

			self.Y_train_global = tf.placeholder(tf.float32, shape=(None,
				self.dim_output), name='Y_test')
			self.Y_test_global = tf.placeholder(tf.float32, shape=(None, 
				self.dim_output), name='Y_test')

			##########################################
			##### Modified Training Procedure ########
			##########################################

			predictions_testing, list_predictions_testing_context, list_predictions_testing_context_global = UNET_network_context_enhanced(inputul = self.X_test, 
				num_encoding_layers = self.num_encoding_layers, unet_type = self.unet_type, mode=True, keep_prob = self.keep_prob, 
				num_layers_same_scale = self.num_layers_same_scale, dim_output = self.dim_output, num_filters = self.num_filters)

			### get the size of outputs at intermediate levels ###
			print(list_predictions_testing_context)
			self.list_size_cube_output_context = [list_predictions_testing_context[_].get_shape().as_list()[1] for _ in range(3)] 

			self.list_Y_train_context = [tf.placeholder(tf.float32, shape=(None, self.list_size_cube_output_context[_], self.list_size_cube_output_context[_],
				self.list_size_cube_output_context[_], self.dim_output),
				name='Y_train_context_'+str(_)) for _ in range(3)]
	
			self.list_Y_test_context = [tf.placeholder(tf.float32, shape=(None, self.list_size_cube_output_context[_], self.list_size_cube_output_context[_],
				self.list_size_cube_output_context[_], self.dim_output),
				name='Y_test_context_'+str(_)) for _ in range(3)]

			self.list_X_train_mask_context = [tf.placeholder(tf.float32, shape=(None, self.list_size_cube_output_context[_], self.list_size_cube_output_context[_],
				self.list_size_cube_output_context[_]),
				name='X_train_mask_context_'+str(_)) for _ in range(3)]

			self.list_X_test_mask_context = [tf.placeholder(tf.float32, shape=(None, self.list_size_cube_output_context[_], self.list_size_cube_output_context[_],
				self.list_size_cube_output_context[_]),
				name='X_test_mask_context_'+str(_)) for _ in range(3)]

			self.setup_train()

			predictions_testing += self.mean_age
			list_predictions_testing_context = [ list_predictions_testing_context[_] + self.mean_age for _ in range(3)]
			list_predictions_testing_context_global = [ list_predictions_testing_context_global[_] + self.mean_age for _ in range(3)]

			if self.import_model:

				v1 = [v for v in tf.global_variables() if "Adam" not in v.name]
				saver_grabber = tf.train.Saver(var_list=v1)
				#saver_grabber = tf.train.Saver()
				saver_grabber.restore(sess,tf.train.latest_checkpoint('./saved_model_3D_UNET_Dropout/iteration_'+str(self.iteration_restored)))

			else:
				pass

			#####################################################################
			#### Need to eliminate background voxels at testing time for MAE ####
			#####################################################################

			flattened_boolean_mask_testing = tf.reshape(self.X_test_mask,[-1,])
			correct_pred = tf.abs(tf.boolean_mask(tf.reshape(predictions_testing,[-1,]),
				flattened_boolean_mask_testing)-tf.boolean_mask(tf.reshape(self.Y_test,[-1,]),flattened_boolean_mask_testing))
			mae = tf.reduce_mean(correct_pred)

			#######################################################################################
			#### Need to eliminate background voxels at testing time for MAE at Context Levels ####
			#######################################################################################
			
			list_mae_context = []
			list_mae_context_global = []
			for _ in range(3):
					
				flattened_boolean_mask_testing_context = tf.reshape(self.list_X_test_mask_context[_], [-1,])
				correct_pred_context = tf.abs(tf.boolean_mask(tf.reshape(list_predictions_testing_context[_], [-1,]),
					flattened_boolean_mask_testing_context) - tf.boolean_mask(tf.reshape(self.list_Y_test_context[_],[-1,]),
					flattened_boolean_mask_testing_context))
				list_mae_context.append(tf.reduce_mean(correct_pred_context))

				list_mae_context_global.append(tf.reduce_mean(tf.abs(self.Y_test_global - list_predictions_testing_context_global[_])))


			tf.summary.scalar('mae_testing', tf.squeeze(mae))
			for _ in range(3):
				tf.summary.scalar('mae_testing_context', tf.squeeze(list_mae_context[_]))
				tf.summary.scalar('mae_testing_context_global', tf.squeeze(list_mae_context_global[_]))					

			tf.summary.scalar('mae_training', tf.squeeze(self.mae_training))
		
			for _ in range(3):

				tf.summary.scalar('mae_training_context', tf.squeeze(self.list_mae_training_context[_]))
				tf.summary.scalar('mae_training_context', tf.squeeze(self.list_mae_training_context_global[_]))				
			
			tf.summary.scalar('re_cost', tf.squeeze(self._loss_op))
			merged = tf.summary.merge_all()
			train_writer = tf.summary.FileWriter('./tensorboard_3D_UNET_Dropout')

			saver = tf.train.Saver()
			if not self.import_model:
				sess.run(tf.global_variables_initializer())
			else:
				### initalize Adam variables ###
				v1 = [v for v in tf.global_variables() if "Adam" in v.name]
				print(v1)
				sess.run(tf.initialize_variables(var_list = v1))

			graph = tf.get_default_graph()

			cmd = 'mkdir -p ./saved_model_3D_UNET_Dropout'
			os.system(cmd)

			cmd = 'mkdir -p ./whole_segmentations_testing_3D_UNET_Dropout'
			os.system(cmd)

			for i in range(self.iteration_restored, self.num_iterations - self.iteration_restored):

				if i<100000:
					self.loss_weights = [0.5, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]										
				elif i >= 100000 and i <150000:
					self.loss_weights = [0.5, 0.15, 0.15, 0.15, 0.0, 0.0, 0.0]					
				elif i >= 150000 and i <200000:
					self.loss_weights = [0.5, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0]
				elif i >= 200000 and i <250000:
					self.loss_weights = [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]			
				elif i>=250000:
					self.loss_weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]	
				else:
					print('error')

				costul_actual_overall = []

				for separate_minibatch in range(self.num_averaged_gradients):

					X_training_feed, Y_training_feed, X_training_feed_mask, list_Y_training_feed_context, list_X_training_feed_mask_context, Y_train_global_feed = extract_3d_blocks_training_regression(inputul = X_training,
						outputul = Y_training, iteration = i, block_size_input = self.size_cube_input,
						block_size_output = self.size_cube_output, 
						list_block_size_output_context = self.list_size_cube_output_context, 
						mask = mask, num_subjects = self.num_subjects,
						num_voxels_per_subject = self.num_voxels_per_subject,
						gender = gender_training, 
						indices_structural_plm = indices_structural)
										
					print('******* things from data processing part *********')
					print(X_training_feed.shape)
					print(Y_training_feed.shape)
					print(X_training_feed_mask.shape)
					for _ in range(3):

						print(list_Y_training_feed_context[_].shape)
						print(list_X_training_feed_mask_context[_].shape)	
						
					
					costul_actual = self.train(session = sess, X_train_feed = X_training_feed, Y_train_feed = Y_training_feed,
						X_training_mask_feed = X_training_feed_mask, 
						list_Y_train_context_feed = list_Y_training_feed_context,
						list_X_training_mask_feed_context = list_X_training_feed_mask_context,
						Y_train_global=Y_train_global_feed)

					costul_actual_overall.append(costul_actual)
				costul_actual = np.mean(costul_actual_overall)
		
				if i % 500 ==0 and i!=0:

					X_training_feed, Y_training_feed, X_training_feed_mask, list_Y_training_feed_context, list_X_training_feed_mask_context, Y_train_global_feed   = extract_3d_blocks_training_regression(inputul = X_training,
						outputul = Y_training, iteration = i, block_size_input = self.size_cube_input,
						block_size_output = self.size_cube_output, mask = mask, 
						list_block_size_output_context = self.list_size_cube_output_context,
						num_subjects = self.num_subjects, num_voxels_per_subject = self.num_voxels_per_subject,
						gender = gender_training, indices_structural_plm = indices_structural)
			
					X_testing_feed, Y_testing_feed, X_testing_feed_mask, list_Y_testing_feed_context, list_X_testing_feed_mask_context, Y_test_global_feed   = extract_3d_blocks_training_regression(inputul = X_testing,
						outputul = Y_testing, iteration = i, block_size_input = self.size_cube_input,
						block_size_output = self.size_cube_output, mask = mask, 
						list_block_size_output_context = self.list_size_cube_output_context ,
						num_subjects = self.num_subjects, num_voxels_per_subject = self.num_voxels_per_subject,
						gender = gender_testing, indices_structural_plm = indices_structural)
			
					print(Y_training_feed.shape)
					print(Y_testing_feed.shape)

					### create dictionary ###


					feed_dict={self.X_train : X_training_feed,
						self.Y_train : Y_training_feed, 
						self.X_test : X_testing_feed,
						self.Y_test : Y_testing_feed, 
						self.X_train_mask : X_training_feed_mask,
						self.X_test_mask : X_testing_feed_mask,
						self.Y_train_global : Y_train_global_feed,
						self.Y_test_global : Y_test_global_feed
						}

					for _ in range(3):
						dictionar = {
							self.list_Y_train_context[_] : list_Y_training_feed_context[_],
							self.list_X_train_mask_context[_] : list_X_training_feed_mask_context[_],
							self.list_Y_test_context[_] : list_Y_testing_feed_context[_],
							self.list_X_test_mask_context[_] : list_X_testing_feed_mask_context[_]}
						feed_dict.update(dictionar)

					summary  = sess.run(merged,
						feed_dict=feed_dict)

					train_writer.add_summary(summary,i)			
										
				if i % 10000 ==0 and i!=0:

					cmd = './saved_model_3D_UNET_Dropout/iteration_'+str(i)
					os.system(cmd)

					saver.save(sess, './saved_model_3D_UNET_Dropout/iteration_'+str(i)+'/saved_UNET', global_step=i)  
					print('Saved checkpoint')

				print('at iteration '+str(i) + ' we have nll : '+str(costul_actual))

		elif self.testing_time:

			X_test = tf.placeholder(tf.float32, shape=(None, self.size_cube_input, self.size_cube_input, self.size_cube_input,
				self.dim_input), name='X_test')
			Y_test = tf.placeholder(tf.float32, shape=(None, self.size_cube_output, self.size_cube_output, self.size_cube_output,
				self.dim_output), name='Y_test')
			Y_test_context = tf.placeholder(tf.float32, shape=(None, self.dim_output),
				name='Y_test_context')		
			X_test_mask = tf.placeholder(tf.bool, shape=(None, self.size_cube_output, self.size_cube_output, self.size_cube_output),
				name='X_test_mask')

			predictions_testing, predictions_testing_context, predictions_testing_context2 = UNET_network_context_enhanced(inputul = X_test, 
				num_encoding_layers = self.num_encoding_layers, unet_type = self.unet_type, mode=True, keep_prob = self.keep_prob, 
				num_layers_same_scale = self.num_layers_same_scale, dim_output = self.dim_output, num_filters = self.num_filters)
			
			predictions_testing += self.mean_age
			#predictions_testing_context += self.mean_age
			#predictions_testing_context2 += self.mean_age
		
			print('attempting to grab ... ./saved_model_3D_UNET_Dropout/iteration_'+str(self.iteration_restored))
			saver_grabber = tf.train.Saver()
			saver_grabber.restore(sess, tf.train.latest_checkpoint('./saved_model_3D_UNET_Dropout/iteration_'+str(self.iteration_restored)))

			###################################################
			### Whole 3D Brain scan Image Regressions #########
			###################################################

			#################################################################################
			### if Image size is not divizikbila by patch_size we need to do some padding ###
			#################################################################################

			cmd = 'mkdir -p ./whole_segmentations_'+str(dataset_name)+'_3D_UNET_Dropout/iteration_'+str(self.iteration_restored)
			os.system(cmd)
			#subproscess.call(["mkdir", "-p","./sanity_checks_testing/iteration_'+str(i)"])

			for _ in range(len(X_testing.keys())):
			
				ROI_end_points = defaultdict()
				ROI_end_points[0] = [np.min(indices_X), np.max(indices_X)]
				ROI_end_points[1] = [np.min(indices_Y), np.max(indices_Y)]
				ROI_end_points[2] = [np.min(indices_Z), np.max(indices_Z)]

				#######################################################################
				####### we are iterating over brain scans in the testing set now ######
				#######################################################################

				print('*******************************')
				print('we are at subjects num '+str(_))
				print('********************************')

				current_image = X_testing[_]
				current_gender = gender_testing[_]
				current_name = X_testing_names[_]
				#### We pad each brain scans so that we can take non-overlapping cubic blocks over it #####
				shape_of_data = X_testing[_].shape

				#current_mask = np.logical_not(np.equal(current_image,np.zeros_like(current_image)))
				current_label = Y_testing[_]
				
				size_cube_input1 = self.size_cube_input//2
				size_cube_output1 = self.size_cube_output//2
				size_cube_input2 = self.size_cube_input - size_cube_input1
				size_cube_output2 = self.size_cube_output - size_cube_output1
				
				print('size of the semi cubes')
				print(size_cube_input1)
				print(size_cube_input2)
				print(size_cube_output1)
				print(size_cube_output2)
				'''
				patches, patches_labels = extract_3D_cubes_input_seg_regression(input_image=current_image, output_image = current_label, gender_image = current_gender,
					semi_block_size_input1 = size_cube_input1, semi_block_size_output1 = size_cube_output1,
					semi_block_size_input2 = size_cube_input2, semi_block_size_output2 = size_cube_output2, dim_output = self.dim_output)
				'''
				
				patches, patches_labels, shape_of_ROI_data, mask_output_space = extract_3D_cubes_input_seg_regression_ROI_bound(input_image = current_image,
					output_scalar = current_label,
					gender_image = current_gender, semi_block_size_input1 = size_cube_input1,
					semi_block_size_output1 = size_cube_output1,
					semi_block_size_input2 = size_cube_input2, semi_block_size_output2 = size_cube_output2,
					dim_output = self.dim_output, ROI_end_points = ROI_end_points, mask = mask)

				
				#### get labels non-overlapping patches #### 
				
				print('size of what we got from custom made non-overlapping 3D cuube extraction')
				print(patches.shape)
				print(patches_labels.shape)
			
				num_iterate_over = patches.shape[0] 
				num_batches = num_iterate_over // self.num_subjects
				lista_batches = [np.arange(kkt*self.num_subjects,(kkt+1)*self.num_subjects) for kkt in range(num_batches-1)]
				lista_batches.append(np.arange((num_batches-1)*self.num_subjects, num_iterate_over))
				predictions_testing_np = []

				##############################################################################################
				#######  Forward Monte Carlo Samples to get a better picture of epistemic uncertainty ########
				##############################################################################################

				num_MC_samples = 25

				t1 = time.time()
				list_samples_predictions = []

				for plm_MC in range(num_MC_samples):

					predictions_testing_now = []
				
					for i_batch in range(num_batches):

						predictions_testing_now_now= sess.run(predictions_testing,
							feed_dict={X_test:patches[lista_batches[i_batch]], Y_test:patches_labels[lista_batches[i_batch]]})

						predictions_testing_now.append(predictions_testing_now_now)
					
					predictions_testing_now = np.concatenate(predictions_testing_now, axis=0)
					list_samples_predictions.append(predictions_testing_now)

				t2 = time.time()
				print('how much time it takes per subject')
				timer(t1,t2)

				list_samples_predictions = np.stack(list_samples_predictions)
				mean_segmentation = np.mean(list_samples_predictions, axis=0)
				epistemic_variance_naive = np.var(list_samples_predictions, axis=0)
				#epistemic_variance_naive = epistemic_variance_naive.reshape(epistemic_variance_naive.shape[:4])

				shape_of_data_after_padding = shape_of_ROI_data[:3]

				mean_segmentation = uncubify(mean_segmentation[...,0], (shape_of_data_after_padding[0],
					shape_of_data_after_padding[1], shape_of_data_after_padding[2]))

				epistemic_variance_naive = uncubify(epistemic_variance_naive[...,0], (shape_of_data_after_padding[0],
					shape_of_data_after_padding[1], shape_of_data_after_padding[2]))
	
				print(mean_segmentation.shape)
				print(epistemic_variance_naive.shape)

				cmd='mkdir -p ./whole_segmentations_'+str(dataset_name)+'_3D_UNET_Dropout/iteration_'+str(self.iteration_restored)+'/'+current_name
				os.system(cmd)					
				
				text_de_scris='chronological age : '+str(Y_testing[_])
				with open('./whole_segmentations_'+str(dataset_name)+'_3D_UNET_Dropout/iteration_'+str(self.iteration_restored)+'/'+current_name+'/details.txt','w') as f:
					f.write(text_de_scris)

				
				###################################
				####### Sampled Segmentations #####
				###################################

				cmd='mkdir -p ./whole_segmentations_'+str(dataset_name)+'_3D_UNET_Dropout/iteration_'+str(self.iteration_restored)+'/'+current_name+'/sampled_seg'
				os.system(cmd)

				mask_output_space = mask_output_space.reshape((-1, ))
				mask_output_space = np.array(mask_output_space, dtype=bool)
				mean_segmentation = mean_segmentation.reshape((-1, ))

				#predictions_testing_np = predictions_testing_np[current_mask]

				whole_brain_segmentation = np.zeros((121*145*121, ))
				whole_brain_segmentation[mask.reshape((-1,)) == 1] = mean_segmentation[mask_output_space]
				whole_brain_segmentation = whole_brain_segmentation.reshape((121, 145, 121))

			
				img = nib.Nifti1Image(whole_brain_segmentation, affine)
				nib.save(img,'./whole_segmentations_'+str(dataset_name)+'_3D_UNET_Dropout/iteration_'+str(self.iteration_restored)+'/'+current_name+'/sampled_seg'+'/segmentation.nii.gz' )

				####################################
				####### Uncertainties ##############
				####################################

				cmd='mkdir -p ./whole_segmentations_'+str(dataset_name)+'_3D_UNET_Dropout/iteration_'+str(self.iteration_restored)+'/'+current_name+'/uncertainty'
				os.system(cmd)

				whole_brain_variance = np.zeros((121*145*121, ))
				epistemic_variance_naive = epistemic_variance_naive.reshape((-1,))
				whole_brain_variance[mask.reshape((-1,)) == 1.0] = epistemic_variance_naive[mask_output_space]
				whole_brain_variance = whole_brain_variance.reshape(121, 145, 121)		
				img = nib.Nifti1Image(whole_brain_variance, affine)
				nib.save(img,'./whole_segmentations_'+str(dataset_name)+'_3D_UNET_Dropout/iteration_'+str(self.iteration_restored)+'/'+current_name+'/uncertainty'+'/epistemic_uncertainty.nii.gz' )

				####################################
				####### BRAIN-PAD ##################
				####################################

				cmd='mkdir -p ./whole_segmentations_'+str(dataset_name)+'_3D_UNET_Dropout/iteration_'+str(self.iteration_restored)+'/'+current_name+'/brain_pad'
				os.system(cmd)
				whole_brain_brain_pad = np.zeros((121*145*121, ))
				mean_segmentation = mean_segmentation.reshape((-1,))
				#predictions_testing_np = predictions_testing_np[current_mask]
				brain_pad = mean_segmentation - Y_testing[_]
				whole_brain_brain_pad[mask.reshape((-1,))==1.0] = brain_pad[mask_output_space]
				whole_brain_brain_pad = whole_brain_brain_pad.reshape(121,145,121)

				img = nib.Nifti1Image(whole_brain_brain_pad, affine)
				nib.save(img,'./whole_segmentations_'+str(dataset_name)+'_3D_UNET_Dropout/iteration_'+str(self.iteration_restored)+'/'+current_name+'/brain_pad'+'/brain_pad.nii.gz' )
				







