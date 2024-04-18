timestamp=$(date +"%Y%m%d%H%M%S")
jobname="sbrnet_array_training_${timestamp}"

sweep_config="/projectnb/tianlabdl/rjbona/sbrnet/configs_deprecated/sweep_config.yaml"
wandb sweep "$sweep_config"

qsub -N "${jobname}" -o "/projectnb/tianlabdl/rjbona/sbrnet/.log/qsubs/${jobname}.qlog" /projectnb/tianlabdl/rjbona/sbrnet/scripts/array_train.qsub
