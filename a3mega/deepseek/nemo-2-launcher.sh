export NCCL_TELEMETRY_MODE=0
export LD_LIBRARY_PATH="$NCCL_PLUGIN_PATH"
ldconfig $LD_LIBRARY_PATH
echo "Added $LD_LIBRARY_PATH to ldconfig:"
ldconfig -p | grep libcuda | sed 's/^/  /'
echo ""
if [[ -n "${EXPLICIT_LOG_DIR}" ]]; then
  explicit_log_dir=${EXPLICIT_LOG_DIR}
else
  explicit_log_dir=${ARTIFACT_DIR}
fi
echo "Logging to ${explicit_log_dir}"
if [[ -n "${TOKENIZER_PATH}" ]]; then
  echo "Getting tokenizer files"
  cp ${TOKENIZER_PATH}/* .
  echo ""
fi
echo "Launching Torch distributed on the node rank $JOB_COMPLETION_INDEX out of $NNODES nodes"


# Update nemo run so we can export the config.
pip install git+https://github.com/NVIDIA/NeMo-Run.git@6550ff68204e5095452098eed3765ed765de5d33

# Export the nemo2 config to yaml.
python ${NEMO_LAUNCH_SCRIPT} --factory "recipe()" \
trainer.num_nodes="$NNODES" \
log.explicit_log_dir="${explicit_log_dir}" \
trainer.max_steps=10 trainer.devices="${GPUS_PER_NODE}" \
--to-yaml exported_nemo_config.yaml

# Create the nsys directory.
mkdir -p ${explicit_log_dir}/nsys

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
--rdzv_backend static \
--master_addr="${MASTER_ADDR}" \
--master_port="${MASTER_PORT}" \
    ${NEMO_LAUNCH_SCRIPT} --factory "recipe()" \
trainer.num_nodes="$NNODES" \
log.explicit_log_dir="${explicit_log_dir}" \
trainer.max_steps=10  trainer.devices="${GPUS_PER_NODE}"

if [[ "$JOB_COMPLETION_INDEX" == "0" ]]; then
  cp ${NEMO_LAUNCH_SCRIPT} ${explicit_log_dir}/run-cli.py
  cp exported_nemo_config.yaml ${explicit_log_dir}/nemo-configuration.yaml
  ls ${explicit_log_dir}
  env > ${explicit_log_dir}/environ.txt
fi
echo "Training completed"
echo "Pod on $(hostname --fqdn) is exiting"