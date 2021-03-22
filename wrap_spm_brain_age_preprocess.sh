#!/bin/sh

usage() {
    cat <<EOF

wrap_spm_brain_age_preprocess.sh 

Usage: wrap_spm_brain_age_preprocess.sh <t1image> 
e.g.   wrap_spm_brain_age_preprocess.sh t1.nii 

<t1image>   T1 structural image

EOF
    exit 1
}

############################################################################

[ "$#" -lt 1 ] && usage

T1Str="$1"

cmd="matlab -nosplash -nodesktop -r \"spm_brain_age_preprocess_b23d('$T1Str'),exit\"";

echo ${cmd}
eval ${cmd}



