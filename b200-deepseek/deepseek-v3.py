"""Nemo2 pretraining recipe for Deepseek v3 model."""

from nemo.collections import llm
from nemo.collections.llm.recipes import deepseek_v3
from nemo.lightning.pytorch.callbacks import NsysCallback
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
import nemo_run as run
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed, bf16_with_fp8_current_scaling_mixed
import os
from nemo.lightning.pytorch.callbacks.moe_token_drop import MegatronTokenDropCallback
from nemo.lightning.pytorch.callbacks.megatron_enable_experimental_callback import MegatronEnableExperimentalCallback


def recipe(
    profile_enabled: bool = False,
    profile_start_step: int = 0,
    profile_end_step: int = 0,
    profile_ranks: str = "0",
) -> run.Partial:
  """Returns a Nemo2 training recipe for Deepseek v3 model.

  Args:
      profile_enabled: Whether to enable Nsys profiling.
      profile_start_step: The step to start profiling.
      profile_end_step: The step to end profiling.
      profile_ranks: The ranks to profile, comma separated.

  Returns:
      A Nemo2 training pretrain.
  """
  print("LOCAL_RANK: ", os.environ["LOCAL_RANK"])
  local_rank=os.environ['LOCAL_RANK']
  os.environ['NVSHMEM_ENABLE_NIC_PE_MAPPING'] = '1'
  os.environ['NVSHMEM_HCA_LIST'] = f'mlx5_{local_rank}:1'

  # Start from the Nemo standard pretrain.
  pretrain = deepseek_v3.pretrain_recipe(
      num_nodes=1, num_gpus_per_node=8, performance_mode=True
  )

  pretrain.trainer.limit_val_batches = 0.0
  pretrain.trainer.val_check_interval = 100

  # Add the Nsys profiling callback if enabled.
  if profile_enabled:
    pretrain.trainer.callbacks.append(
        run.Config(
            NsysCallback,
            start_step=profile_start_step,
            end_step=profile_end_step,
            ranks=[int(x) for x in profile_ranks.split(",")],
            gen_shape=False,
        )
    )

  # Add the FLOPs measurement callback.
  pretrain.trainer.callbacks.append(
      run.Config(
          FLOPsMeasurementCallback,
          model_name="deepseekv3",
          model_config=pretrain.model.config,
          data_config=pretrain.data,
      )
  )
  enable_deepep=True
  # Disable checkpointing.
  pretrain.log.ckpt = None
  pretrain.trainer.enable_checkpointing = False
  
  pretrain.model.config.recompute_granularity = None
  pretrain.model.config.recompute_method = None
  pretrain.model.config.recompute_num_layers = None
  pretrain.model.config.recompute_modules = None
  
  if enable_deepep :
    pretrain.model.config.moe_token_dispatcher_type = "flex"
    pretrain.model.config.moe_enable_deepep = True
    pretrain.model.config.moe_shared_expert_overlap = False
   #pretrain.trainer.callbacks.append(run.Config(MegatronTokenDropCallback))
  else: 
    pretrain.model.config.moe_token_dispatcher_type = "alltoall"
    pretrain.model.config.moe_enable_deepep = False
    pretrain.model.config.moe_shared_expert_overlap = true
    pretrain.trainer.callbacks.append(run.Config(MegatronTokenDropCallback))
  pretrain.model.config.moe_permute_fusion = True
  pretrain.model.config.apply_rope_fusion = True
  pretrain.trainer.callbacks.append(run.Config(MegatronEnableExperimentalCallback))

  # Pipeline parallelism configs. We infer PP layout from the provided PP and VP size
  map_pp_vp_to_layout = {
      (1, 1): None,
      (4, 1): [['embedding'] + ['decoder'] * 16, ['decoder'] * 16, ['decoder'] * 16, ['decoder'] * 13 + ['loss']],
      (8, 1): [['embedding'] + ['decoder'] * 8] + [['decoder'] * 8] * 6 + [['decoder'] * 5 + ['loss']],
      (4, 2): [['embedding'] + ['decoder'] * 8] + [['decoder'] * 8] * 6 + [['decoder'] * 5 + ['loss']],
      (16, 1): [['embedding'] + ['decoder'] * 4] + [['decoder'] * 4] * 14 + [['decoder', 'loss']],
      (8, 2): [['embedding'] + ['decoder'] * 4] + [['decoder'] * 4] * 14 + [['decoder', 'loss']],
      (4, 4): [['embedding'] + ['decoder'] * 4] + [['decoder'] * 4] * 14 + [['decoder', 'loss']],
  }
  pp_size = pretrain.trainer.strategy.pipeline_model_parallel_size or 1
  vp_size = pretrain.trainer.strategy.virtual_pipeline_model_parallel_size or 1
  if (pp_size, vp_size) not in map_pp_vp_to_layout:
      raise ValueError(
          f"Invalid PP and VP size: {pp_size} and {vp_size} to infer PP layout "
          f"for DeepSeek V3. Known PP and VP combinations: {map_pp_vp_to_layout.keys()}"
      )
  
  layout = map_pp_vp_to_layout[(pp_size, vp_size)]
 
     
  if layout is not None:
      layout = list([list(x) for x in layout])  # yield all the elements
  if enable_deepep:
      recipe.trainer.strategy.pipeline_model_parallel_layout="Et*3|(tt|)*29m|L" #"Et*2|(tt|)*22t|(tt|)*7mL"
  else:
      recipe.trainer.strategy.pipeline_model_parallel_layout = layout
  

  # The following knobs are not needed if we specify layout
  pretrain.trainer.strategy.account_for_embedding_in_pipeline_split = False
  pretrain.trainer.strategy.account_for_loss_in_pipeline_split = False
  pretrain.trainer.strategy.num_layers_in_first_pipeline_stage = None
  pretrain.trainer.strategy.num_layers_in_last_pipeline_stage = None

  
  # pretrain.trainer.plugins = bf16_with_fp8_mixed()
  pretrain.trainer.plugins = bf16_with_fp8_current_scaling_mixed()
  # disable first/last layer bf16 for benchmarking
  pretrain.trainer.plugins.first_last_layers_bf16 = False
  pretrain.trainer.plugins.grad_reduce_in_fp32 = False

  #pretrain.model.config.enable_cuda_graph = True
  pretrain.trainer.strategy.use_te_rng_tracker = True

  pretrain.trainer.strategy.sequence_parallel = True

  pretrain.data.micro_batch_size=1
  pretrain.data.global_batch_size = 2048
  pretrain.trainer.strategy.tensor_model_parallel_size = 1
  pretrain.trainer.strategy.pipeline_model_parallel_size = 8
  pretrain.trainer.strategy.expert_model_parallel_size = 32
  pretrain.trainer.strategy.virtual_pipeline_model_parallel_size = None


  # Log every step.
  pretrain.trainer.log_every_n_steps = 1

  return pretrain


if __name__ == "__main__":
  run.cli.main(llm.pretrain, default_factory=recipe)