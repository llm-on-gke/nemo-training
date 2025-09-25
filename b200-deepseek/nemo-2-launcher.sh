export LD_LIBRARY_PATH=$NCCL_PLUGIN_PATH:$LD_LIBRARY_PATH
ldconfig $LD_LIBRARY_PATH
echo "Added $LD_LIBRARY_PATH to ldconfig:"
ldconfig -p | grep libcuda | sed 's/^/  /'
echo ""
if [[ -n "${EXPLICIT_LOG_DIR}" ]]; then
  explicit_log_dir=${EXPLICIT_LOG_DIR}
else
  explicit_log_dir=workload_logs
fi
echo "Logging to ${explicit_log_dir}"
if [[ -n "${TOKENIZER_PATH}" ]]; then
  echo "Getting tokenizer files"
  cp ${TOKENIZER_PATH}/* .
  echo ""
fi
echo "Launching Torch distributed on the node rank $JOB_COMPLETION_INDEX out of $NNODES nodes"


# Update nemo run so we can export the config.
#pip install git+https://github.com/NVIDIA/NeMo-Run.git@6550ff68204e5095452098eed3765ed765de5d33
export NEMO_HOME=/workspace/NeMo
git clone https://github.com/NVIDIA/NeMo

# 2. Install dependencies
cd NeMo
git checkout r2.4.0
pip install '.[all]'
pip install megatron-core@git+https://github.com/NVIDIA/Megatron-LM.git@core_r0.13.0
#pip install nemo_run@git+https://github.com/NVIDIA/NeMo-Run.git
# Export the nemo2 config to yaml.
python ${NEMO_LAUNCH_SCRIPT} --factory "recipe()" \
trainer.num_nodes="$NNODES" \
log.explicit_log_dir="${explicit_log_dir}" \
trainer.max_steps=10 trainer.devices=8 trainer.strategy.tensor_model_parallel_size=1 trainer.strategy.expert_model_parallel_size=8 data.global_batch_size=2048 \
--to-yaml exported_nemo_config.yaml

# Create the nsys directory.
mkdir -p ${explicit_log_dir}/nsys

cd /home/nemo-training/b200-deepseek

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600 
unset NCCL_NVLS_ENABLE 
OMP_NUM_THREADS=12 NSYS_CONFIG_DIRECTIVES="AgentLaunchTimeoutSec=240;AppLaunchTimeoutSec=240" TORCH_NCCL_ENABLE_MONITORING=0 \
/usr/local/bin/nsys profile -s none -t nvtx,cuda --capture-range=cudaProfilerApi --capture-range-end=stop \
-o ${explicit_log_dir}/nsys/noderank-${JOB_COMPLETION_INDEX} \
--session-new "nemo-rank${JOB_COMPLETION_INDEX}"-$RANDOM \
--wait all \
torchrun \
--nproc-per-node="${GPUS_PER_NODE}" \
--nnodes="${NNODES}" \
--node_rank="${JOB_COMPLETION_INDEX}" \
--rdzv_id="${JOB_IDENTIFIER}" \
--master_addr="${MASTER_ADDR}" \
--master_port="${MASTER_PORT}" \
${NEMO_LAUNCH_SCRIPT} --factory "recipe()" \
trainer.num_nodes="$NNODES" \
log.explicit_log_dir="${explicit_log_dir}" \
trainer.max_steps=20 trainer.num_nodes=${NNODES} trainer.devices=8 

if [[ "$JOB_COMPLETION_INDEX" == "0" ]]; then
  mkdir -p ${ARTIFACT_DIR}
  cp -r ${explicit_log_dir}/* ${ARTIFACT_DIR}/
  cp ${NEMO_LAUNCH_SCRIPT} ${ARTIFACT_DIR}/run-cli.py
  cp exported_nemo_config.yaml ${ARTIFACT_DIR}/nemo-configuration.yaml
  env > ${ARTIFACT_DIR}/environ.txt
  ls ${ARTIFACT_DIR}
fi
echo "Training completed"
echo "Pod on $(hostname --fqdn) is exiting"
