"""Nemo2 pretraining recipe for Deepseek v3 model."""

from nemo.collections import llm
from nemo.collections.llm.recipes import deepseek_v3
from nemo.lightning.pytorch.callbacks import NsysCallback
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed

from pretrain_deepseek_v3 import override_recipe_configs
import nemo_run as run


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
      A Nemo2 training recipe.
  """
  # Start from the Nemo standard recipe.
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

  # Disable checkpointing.
  pretrain.log.ckpt = None
  pretrain.trainer.enable_checkpointing = False

  #pretrain.trainer.plugins = bf16_with_fp8_mixed()
  #pretrain.trainer.strategy.pipeline_model_parallel_size = 16
  #pretrain.trainer.strategy.tensor_model_parallel_size = 1
  #pretrain.trainer.strategy.expert_model_parallel_size = 8
  #pretrain.trainer.strategy.virtual_pipeline_model_parallel_size = None
  #pretrain.trainer.strategy.expert_tensor_parallel_size = 1

  # Log every step.
  pretrain.trainer.log_every_n_steps = 1

  pretrain = override_recipe_configs(
        pretrain,
        32, #num_nodes
        1, #mbs
        1024, #gbs
        20, #max_steps
        2, #TP
        16, #PP
        1, #CP
        1, #VP
        8, #EP
        1, #ETP
        False, #enable_cuda_graphs
        False, #use_mcore_fsdp
        0, #recompute_layers
        0, #activation_offload_layers
        None, #recompute_layers
        False, #use_user_buffer_registration
        False, #use_sharp
        True, #enable deepep
        "fp8", #compute_type
        "cs", #fp8_recipe
    )
  pretrain.model.config.moe_expert_capacity_factor=None #rick
  return pretrain


if __name__ == "__main__":
  run.cli.main(llm.pretrain, default_factory=recipe)
