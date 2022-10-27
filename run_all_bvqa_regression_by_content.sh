#!/bin/bash

MODELS=(
 'NIQE_LGN1_LGN2_NSS1_NSS2_SLEEQ18_SLEEQ1_fusion'

)

DATASETS=(
  'LIVE_VQC'
  #'KoNVid'
  #'Youtube-UGC'
)


for m in "${MODELS[@]}"
do
for DS in "${DATASETS[@]}"
do
#${DS}_${m}_feats.mat
  feature_file=features/${DS}_${m}_feats.mat
  mos_file=features/${DS}_metadata.csv
  out_file=result/${DS}_${m}_corr.mat
  log_file=logs/${DS}_${m}_.log

#   echo "$m" 
#   echo "${feature_file}"
#   echo "${out_file}"
#   echo "${log_file}"
#VQC_evaluate_bvqa_features_by_content_regression.py
  cmd="python src/evaluate_bvqa_features_completely_blind.py"
  cmd+=" --model_name $m"
  cmd+=" --dataset_name ${DS}"
  cmd+=" --feature_file ${feature_file}"
  cmd+=" --mos_file ${mos_file}"
  cmd+=" --out_file ${out_file}"
  cmd+=" --log_file ${log_file}"
  cmd+=" --use_parallel"
  cmd+=" --log_short"
  cmd+=" --num_iterations 50"

  echo "${cmd}"

  eval ${cmd}
done
done
