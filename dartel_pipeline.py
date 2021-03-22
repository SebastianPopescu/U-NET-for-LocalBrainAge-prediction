import numpy as np
from collections import defaultdict
import os
import sys
import subprocess
import argparse
    
def absoluteFilePaths(dirpath):
   
    ### dirpath has to be absolute path ####

    #sys.path.append('/data/my_programs/freesufer/bin')

    list_realpaths = []
    filenames = os.listdir(dirpath)

    for f in filenames:
        list_realpaths.append(os.path.abspath(os.path.join(dirpath, f)))

    return list_realpaths


def batched_spm12_dartel(img_dir, name_of_dataset, size_batch):


    cmd='mkdir -p ./dump_dartel_output_'+str(name_of_dataset)
    os.system(cmd)

    list_t1_scans = absoluteFilePaths(img_dir)
    total_number_of_scans = len(list_t1_scans)

    number_batches = total_number_of_scans // size_batch
    lista_batches = [np.arange(kkt*size_batch,(kkt+1)*size_batch) for kkt in range(number_batches-1)]
    lista_batches.append(np.arange((number_batches-1)*size_batch, total_number_of_scans))

    for current_batch in lista_batches:

        batched_cmd = ''

        for number_scan in current_batch:

            nifti = list_t1_scans[number_scan]
            print(nifti.rsplit('/')[-1][-4:])
            if nifti.rsplit('/')[-1][-4:]!='.nii' or nifti.rsplit('/')[-1][:4]=='smwc' or nifti.rsplit('/')[-1][:2]=='rc' or nifti.rsplit('/')[-1][:2]=='u_' or nifti.rsplit('/')[-1][:1]=='c':
                ### this is the case that we are actually not dealing with a raw T1 scan, but instead some preprocessed spm12 output #####
                pass
            
            else:

                boolean_result =  os.path.exists(img_dir+'/smwc1'+nifti.rsplit('/')[-1])

                if boolean_result:

                    print('dartel output already exists')

                else:

                    print('bash ./wrap_spm_brain_age_preprocess.sh '+nifti+' > ./dump_dartel_output_'+str(name_of_dataset)+'/'+nifti.rsplit('/')[-1][:-4] +' & ')
                    #> ./dump_dartel_output_'+str(name_of_dataset)+'/'+nifti.rsplit('/')[-1][:-4] +'&')
                    cmd='bash ./wrap_spm_brain_age_preprocess.sh '+nifti+' > ./dump_dartel_output_'+str(name_of_dataset)+'/'+nifti.rsplit('/')[-1][:-4] +' & '
                    #> ./dump_dartel_output_'+str(name_of_dataset)+'/'+nifti.rsplit('/')[-1][:-4] +'&'
                    batched_cmd+=cmd
    
        print('*************************************')	
        print(batched_cmd[:-2])
        os.system(batched_cmd[:-2])

        print('*************************************')

    ### after dartel pipeline finishes delete temporary files ####


    print('********************************')
    print('********** cleaning up *********')
    print('********************************')

    cmd='rm '+img_dir+'/*_seg8.mat'
    os.system(cmd)

    cmd='rm '+img_dir+'/*_tissue_volumes.csv'
    os.system(cmd)
    
    cmd='rm '+img_dir+'/c1*'
    os.system(cmd)
    
    cmd='rm '+img_dir+'/c2*'
    os.system(cmd)
    
    cmd='rm '+img_dir+'/c3*'
    os.system(cmd)
    
    cmd='rm '+img_dir+'/rc1*'
    os.system(cmd)
    
    cmd='rm '+img_dir+'/rc2*'
    os.system(cmd)
    
    cmd='rm '+img_dir+'/rc3*'
    os.system(cmd)

    cmd='rm '+img_dir+'/u_rc1*'
    os.system(cmd)


    print('********************************')
    print('********** moving data *********')
    print('********************************')

    cmd = 'mkdir -p '+img_dir+'/gm_data'
    os.system(cmd)

    cmd = 'mv '+img_dir+'/smwc1* '+img_dir+'/gm_data'
    os.system(cmd)

    cmd = 'mkdir -p '+img_dir+'/wm_data'
    os.system(cmd)

    cmd = 'mv '+img_dir+'/smwc2* '+img_dir+'/wm_data'
    os.system(cmd)



if __name__=='__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type = str, help = 'the directory where the raw nifti files are located')
    parser.add_argument('--dataset_name', type = str, help = 'name of the dataset, this will influence where the spm12 output gets written')
    parser.add_argument('--size_batch', type = int, help = 'number of scans to be processed at the same time')
    args = parser.parse_args()

    batched_spm12_dartel(img_dir=args.img_dir,
        name_of_dataset=args.dataset_name, size_batch=args.size_batch)
