PARAMS=(
       --wireup_method                       "nccl-openmpi"
       --run_tag                             "${SLURM_JOBID}"
       --output_dir                          "${OUTPUT_DIR}}"
#       --checkpoint                          "None"
       --data_dir_prefix                     "${TRAIN_DATA_PREFIX}"
#       --max_inter_threads                   "1"
       --max_epochs                          "200"
       --save_frequency                      "400"
       --validation_frequency                "200"
       --max_validation_steps                "50"
       --logging_frequency                   "0"
       --training_visualization_frequency    "200"
       --validation_visualization_frequency  "40"
       --local_batch_size                    "2"
#       --channels                            "{[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}"
       --optimizer                           "LAMB"
       --start_lr                            "1e-3"
#       --adam_eps                            "1e-8"
       --weight_decay                        "1e-2"
#       --loss_weight_pow                     "-0.125"
       --lr_warmup_steps                     "0"
       --lr_warmup_factor                    "1.0"
       --lr_schedule                         "{\"type\":\"multistep\",\"milestones\":\"15000 25000\",\"decay_rate\":\"0.1\"}"
#       --target_iou                          "0.82"
       --model_prefix                        "classifier"
       --amp_opt_level                       "O1"
#       --enable_wandb
#       --resume_logging
#       |& tee
#       -a ${output_dir}/train.out
)