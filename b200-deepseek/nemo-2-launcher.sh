export LD_LIBRARY_PATH=$NCCL_PLUGIN_PATH:$LD_LIBRARY_PATH
ldconfig $LD_LIBRARY_PATH
echo "Added $LD_LIBRARY_PATH to ldconfig:"
ldconfig -p | grep libcuda | sed 's/^/  /'
echo ""
source /usr/local/gib/scripts/set_nccl_env.sh
export NCCL_SOCKET_IFNAME="eth0,eth1"
export NCCL_TUNER_CONFIG_PATH=/usr/local/gib/configs/tuner_config_a4.txtpb

export TRITON_CACHE_DIR="/tmp/triton-cache/"
#export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NVTE_FWD_LAYERNORM_SM_MARGIN=20
export NVTE_BWD_LAYERNORM_SM_MARGIN=20
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false # to avoid HF warnings
export DEEPEP_COMM_TIMEOUT_MS=30000

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
#export NEMO_HOME=/workspace/NeMo
#git clone https://github.com/NVIDIA/NeMo

# 2. Install dependencies
# cd NeMo
#git checkout r2.4.0
#pip install '.[all]'
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
NVSHMEM_DEBUG=INFO OMP_NUM_THREADS=12 NSYS_CONFIG_DIRECTIVES="AgentLaunchTimeoutSec=240;AppLaunchTimeoutSec=240" TORCH_NCCL_ENABLE_MONITORING=0 \
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