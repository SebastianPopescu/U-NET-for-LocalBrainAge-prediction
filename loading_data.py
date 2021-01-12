import numpy as np
from collections import defaultdict
import nibabel as nib
import pandas as pd


def parse_string_list(string_list, index):

	new_list = [string_list[index_now] for index_now in index]

	return new_list


def parse_info(list_of_nifty_files_gm, subject_info, list_extract_subjects):

	subject_info.set_index("Subject", inplace=True)

	#########################################
	### subject_info -- PANDAS data frame ###
	#########################################
	
	lista_outcomes = defaultdict()
	
	
	control=0
	for current_subject in list_extract_subjects:

		print(current_subject)
		#try:
	
		############################################
		###### check if there is a nifty file ######
		############################################
		plm = [s for s in list_of_nifty_files_gm if current_subject in s]

		if len(plm)>0:

			current_row_of_interest = subject_info.loc[current_subject]
			lista_outcomes[control] = current_row_of_interest.Age
			control+=1
			
		else:
			print('Did not have nifty file ')
		#except:
		
		#	print('We could not find the nifti files')

	return lista_outcomes


'''
def data_factory_whole_brain(list_of_nifty_files_gm,
	list_of_nifty_files_wm, subject_info, list_extract_subjects):

	subject_info.set_index("Subject", inplace=True)

	#########################################
	### subject_info -- PANDAS data frame ###
	#########################################

	lista_imagini = defaultdict()
	lista_outcomes = defaultdict()
	lista_gender = defaultdict()
	lista_name = defaultdict()

	### parse the GM and WM nifty lists for the ones presment in list_extract_subjects ###
	list_parsed_gm = []
	list_parsed_wm = []

	control=0
	for current_subject in list_extract_subjects:

		print(current_subject)
		#try:
		############################################
		###### check if there is a nifty file ######
		############################################
		plm = [s for s in list_of_nifty_files_gm if current_subject in s]
		print(plm)
		if len(plm)>0:

			list_parsed_gm.append( [s for s in list_of_nifty_files_gm if current_subject in s][0])
			list_parsed_wm.append( [s for s in list_of_nifty_files_wm if current_subject in s][0])
			current_row_of_interest = subject_info.loc[current_subject]
			lista_outcomes[control] = current_row_of_interest.Age
			print(current_row_of_interest.Age)
			lista_gender[control] =  current_row_of_interest.Gender
			print(current_row_of_interest.Gender)
			lista_name[control] = current_subject
			control+=1
			
		else:
			print('Did not have nifty file ')
		#except:
		
		#	print('We could not find the nifti files')

	control=0
	for sth in list_parsed_gm:	
	
		lista_imagini[control] = []
		control+=1	


	##### load GM data #####
	control = 0
	for nifty_file in list_parsed_gm:

		
		nifti_name = nifty_file.rsplit('/')[-1]
		if 'run-02' in nifti_name:
			nifti_name = nifti_name.rsplit('_run-02')[0]+'_run-01_T1w.nii'
		if 'run-03' in nifti_name:
			nifti_name = nifti_name.rsplit('_run-03')[0]+'_run-01_T1w.nii'
		if 'run-04' in nifti_name:
			nifti_name = nifti_name.rsplit('_run-04')[0]+'_run-01_T1w.nii'
		if 'run-05' in nifti_name:
			nifti_name = nifti_name.rsplit('_run-05')[0]+'_run-01_T1w.nii'
		
		nifti_name = '/data/my_data/OASIS3/gm_data/'+nifti_name
		print('loading ... '+str(nifti_name))

		
		temporar_object = nib.load(nifti_name)
		temporar_data = temporar_object.get_data()
		temporar_object.uncache()
		#print(temporar_data_gm.shape)

		lista_imagini[control].append(np.expand_dims(temporar_data,axis=-1))
		control+=1

	##### load WM data #####
	control = 0
	for nifty_file in list_parsed_wm:

		nifti_name = nifty_file.rsplit('/')[-1]
		if 'run-02' in nifti_name:
			nifti_name = nifti_name.rsplit('_run-02')[0]+'_run-01_T1w.nii'
		if 'run-03' in nifti_name:
			nifti_name = nifti_name.rsplit('_run-03')[0]+'_run-01_T1w.nii'
		if 'run-04' in nifti_name:
			nifti_name = nifti_name.rsplit('_run-04')[0]+'_run-01_T1w.nii'
		if 'run-05' in nifti_name:
			nifti_name = nifti_name.rsplit('_run-05')[0]+'_run-01_T1w.nii'
		nifti_name = '/data/my_data/OASIS3/wm_data/'+nifti_name
		print('loading ... '+str(nifti_name))
		temporar_object = nib.load(nifti_name)
		temporar_data = temporar_object.get_data()
		temporar_object.uncache()
		#print(temporar_data_gm.shape)

		lista_imagini[control].append(np.expand_dims(temporar_data,axis=-1))
		control+=1

	###################################################################
	########### concatenate the dictionary entries ####################
	###################################################################		

	for key in lista_imagini.keys():

		print('concatenating --- '+str(key))
		lista_imagini[key] = np.concatenate(lista_imagini[key], axis=-1)

	return lista_imagini, lista_outcomes, lista_gender, list_parsed_gm, list_parsed_wm, lista_name

'''



def data_factory_whole_brain(list_of_nifty_files_gm,
	list_of_nifty_files_wm, subject_info, list_extract_subjects):

	subject_info.set_index("Subject", inplace=True)

	#########################################
	### subject_info -- PANDAS data frame ###
	#########################################

	lista_imagini = defaultdict()
	lista_outcomes = defaultdict()
	lista_gender = defaultdict()
	lista_name = defaultdict()

	### parse the GM and WM nifty lists for the ones presment in list_extract_subjects ###
	list_parsed_gm = []
	list_parsed_wm = []

	control=0
	for current_subject in list_extract_subjects:

		print(current_subject)
		try:
			############################################
			###### check if there is a nifty file ######
			############################################
			plm = [s for s in list_of_nifty_files_gm if current_subject in s]
			print(plm)
			if len(plm)>0:

				list_parsed_gm.append( [s for s in list_of_nifty_files_gm if current_subject in s][0])
				list_parsed_wm.append( [s for s in list_of_nifty_files_wm if current_subject in s][0])
				current_row_of_interest = subject_info.loc[current_subject]
				lista_outcomes[control] = current_row_of_interest.Age
				print(current_row_of_interest.Age)
				lista_gender[control] =  current_row_of_interest.Gender
				print(current_row_of_interest.Gender)
				lista_name[control] = current_subject
			
				
				lista_imagini[control] = []

				##### load GM data #####
				nifty_file = list_parsed_gm[-1]

				nifti_name = nifty_file.rsplit('/')[-1]
				if 'run-02' in nifti_name:
					nifti_name = nifti_name.rsplit('_run-02')[0]+'_run-01_T1w.nii'
				if 'run-03' in nifti_name:
					nifti_name = nifti_name.rsplit('_run-03')[0]+'_run-01_T1w.nii'
				if 'run-04' in nifti_name:
					nifti_name = nifti_name.rsplit('_run-04')[0]+'_run-01_T1w.nii'
				if 'run-05' in nifti_name:
					nifti_name = nifti_name.rsplit('_run-05')[0]+'_run-01_T1w.nii'
				
				nifti_name = '/data/my_data/OASIS3/gm_data/'+nifti_name
				print('loading ... '+str(nifti_name))
		
				temporar_object = nib.load(nifti_name)
				temporar_data = temporar_object.get_data()
				temporar_object.uncache()
				#print(temporar_data_gm.shape)

				lista_imagini[control].append(np.expand_dims(temporar_data,axis=-1))

				##### load WM data #####

				nifty_file = list_parsed_wm[-1]

				nifti_name = nifty_file.rsplit('/')[-1]
				if 'run-02' in nifti_name:
					nifti_name = nifti_name.rsplit('_run-02')[0]+'_run-01_T1w.nii'
				if 'run-03' in nifti_name:
					nifti_name = nifti_name.rsplit('_run-03')[0]+'_run-01_T1w.nii'
				if 'run-04' in nifti_name:
					nifti_name = nifti_name.rsplit('_run-04')[0]+'_run-01_T1w.nii'
				if 'run-05' in nifti_name:
					nifti_name = nifti_name.rsplit('_run-05')[0]+'_run-01_T1w.nii'
				nifti_name = '/data/my_data/OASIS3/wm_data/'+nifti_name
				print('loading ... '+str(nifti_name))
				temporar_object = nib.load(nifti_name)
				temporar_data = temporar_object.get_data()
				temporar_object.uncache()
				#print(temporar_data_gm.shape)

				lista_imagini[control].append(np.expand_dims(temporar_data,axis=-1))
				
				control+=1
				
			else:
				print('Did not have nifty file ')
		except:
		
			print('We could not find the nifti files')



	###################################################################
	########### concatenate the dictionary entries ####################
	###################################################################		

	for key in lista_imagini.keys():

		print('concatenating --- '+str(key))
		lista_imagini[key] = np.concatenate(lista_imagini[key], axis=-1)

	return lista_imagini, lista_outcomes, lista_gender, list_parsed_gm, list_parsed_wm, lista_name





def data_factory_whole_brain_training(list_of_nifty_files_gm,
	list_of_nifty_files_wm, subject_info, list_extract_subjects):

	subject_info.set_index("Subject", inplace=True)

	#########################################
	### subject_info -- PANDAS data frame ###
	#########################################

	lista_imagini = defaultdict()
	lista_outcomes = defaultdict()
	lista_gender = defaultdict()
	lista_name = defaultdict()

	### parse the GM and WM nifty lists for the ones presment in list_extract_subjects ###
	list_parsed_gm = []
	list_parsed_wm = []

	control=0
	for current_subject in list_extract_subjects:

		print(current_subject)
		#try:
		############################################
		###### check if there is a nifty file ######
		############################################
		plm = [s for s in list_of_nifty_files_gm if current_subject in s]

		if len(plm)>0:

			list_parsed_gm.append( [s for s in list_of_nifty_files_gm if current_subject in s][0])
			list_parsed_wm.append( [s for s in list_of_nifty_files_wm if current_subject in s][0])
			current_row_of_interest = subject_info.loc[current_subject]
			lista_outcomes[control] = current_row_of_interest.Age
			print(current_row_of_interest.Age)
			lista_gender[control] =  current_row_of_interest.Gender
			print(current_row_of_interest.Gender)
			lista_name[control] = current_subject
			control+=1
			
		else:
			print('Did not have nifty file ')
		#except:
		
		#	print('We could not find the nifti files')

	control=0
	for sth in list_parsed_gm:	
	
		lista_imagini[control] = []
		control+=1	

	##### load GM data #####
	control = 0
	for nifty_file in list_parsed_gm:
		print('name of subject')
		print(lista_name[control])
		print('loading ... '+str(nifty_file))

		temporar_object = nib.load(nifty_file)
		temporar_data = temporar_object.get_data()
		temporar_object.uncache()
		#print(temporar_data_gm.shape)

		lista_imagini[control].append(np.expand_dims(temporar_data,axis=-1))
		control+=1

	##### load WM data #####
	control = 0
	for nifty_file in list_parsed_wm:
		print('name of subject')
		print(lista_name[control])
		print('loading ... '+str(nifty_file))

		temporar_object = nib.load(nifty_file)
		temporar_data = temporar_object.get_data()
		temporar_object.uncache()
		#print(temporar_data_gm.shape)

		lista_imagini[control].append(np.expand_dims(temporar_data,axis=-1))
		control+=1

	###################################################################
	########### concatenate the dictionary entries ####################
	###################################################################		

	for key in lista_imagini.keys():

		print('concatenating --- '+str(key))
		lista_imagini[key] = np.concatenate(lista_imagini[key], axis=-1)

	return lista_imagini, lista_outcomes, lista_gender, list_parsed_gm, list_parsed_wm, lista_name

