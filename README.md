# U-NET-for-LocalBrainAge-prediction
Code for upcoming paper "A U-NET model for Local Brain-Age"


How to use:

As mentioned in the paper, the brain scans have to go through the Dartel pipeline in spm12. 

Code for obtaining these segmentations will be added shortly.

Using LocalBrainAge_testing.py you can obtain new 3D heatmaps of local brain-age for new subjects. The file has the following arguments that you need to specify:

--filepath_csv = location of .csv file that contains the dataset metadata, look inside the .py file to find details regarding formating
--dirpath_gm = location of directory where the gray matter SPM12 segmentations are stored
--dirpath_wm = location of directory where the white matter SPM12 segmentations are stored
--dataset_name = name of your dataset, it just impacts the name of the location where all the subsequent .nii files get saved
