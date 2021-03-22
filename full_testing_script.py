import argparse
import os
from dartel_pipeline import batched_spm12_dartel


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_encoding_layers', type = int, default = 2, help = 'keep the default setting')
    parser.add_argument('--num_filters', type = int, default = 64, help = 'keep the default setting')
    parser.add_argument('--num_subjects', type = int, default = 2, help = 'keep the default setting')
    parser.add_argument('--num_voxels_per_subject', type = int , default = 1, help = 'keep the default setting')
    parser.add_argument('--filepath_csv', type = str, help = 'the location of the .csv file containing the meta-data assocated to the dataset in cause')	
    parser.add_argument('--dirpath_raw_data', type = str, help = 'the location of the directory containing the raw T1 nifti files')
    parser.add_argument('--dataset_name', type = str, help = 'the name of the dataset in cause, it will influence where the results are written')
    parser.add_argument('--size_batch_preprocessing', type = int, help = 'how many nifti files to process at the same time')
    args = parser.parse_args()

    ##### spm12 pre-processing #####
    batched_spm12_dartel(img_dir = args.dirpath_raw_data, name_of_dataset = args.dataset_name, size_batch = args.size_batch_preprocessing)


    ##### getting LocalBrainAge predictions #####
    cmd = 'python3 ./LocalBrainAge_testing.py --filepath_csv='+str(args.filepath_csv)+' --dirpath_gm='+str(args.dirpath_raw_data)+'/gm_data --dirpath_wm='+str(args.dirpath_raw_data)+'/wm_data --dataset_name='+str(args.dataset_name)