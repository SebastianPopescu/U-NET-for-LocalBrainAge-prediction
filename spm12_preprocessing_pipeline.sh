#!/bin/sh
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --partition=long
#SBATCH -o spm12_preprocessing_-%A_%a.out
#SBATCH -e spm12_preprocessing_-%A_%a.err
#SBATCH -J spm12_pipeline

module load matlab
module load anaconda
module load fsl

nifti_files=()
filename="$1"

while read -r line; do
    nifti_files+=("$line")
    echo "nifti file read from text - $line"
done < "$filename"

cd "$2"
echo $PWD

mkdir -p gm_data
mkdir -p wm_data

cp -n ../wrap_spm_brain_age_preprocess.sh .
cp -n ../spm_brain_age_preprocess_b23d.m .

echo "${nifti_files[$SLURM_ARRAY_TASK_ID]}"
bash  wrap_spm_brain_age_preprocess.sh "${nifti_files[$SLURM_ARRAY_TASK_ID]}"

mv smwc1${nifti_files[$SLURM_ARRAY_TASK_ID]} ./gm_data
mv smwc2${nifti_files[$SLURM_ARRAY_TASK_ID]} ./wm_data
