"""Nemo2 pretraining recipe for Deepseek v3 model."""

from nemo.collections import llm
from nemo.collections.llm.recipes import deepseek_v3
from nemo.lightning.pytorch.callbacks import NsysCallback
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback

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
  pretrain = deepseek_v3.pretrain_recipe(num_nodes=1, num_gpus_per_node=8, performance_mode=True)

  # Set the number of steps to 50 for a quicker benchmark.
  #pretrain.trainer.max_steps = 50
  # Disable validation batches.
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
  
  #Recipe 2 layer:
  #pretrain.model.config.num_layers = 2
  #pretrain.model.config.moe_layer_freq = [0, 1]
  pretrain.trainer.strategy.pipeline_model_parallel_size = 1
  pretrain.trainer.strategy.tensor_model_parallel_size=8
  pretrain.trainer.strategy.expert_model_parallel_size = 8 
  #retrain.trainer.strategy.virtual_pipeline_model_parallel_size = None
  pretrain.trainer.strategy.expert_tensor_parallel_size = 1
  
  #DATA parallism
  pretrain.trainer.strategy.data_parallel_shard_degree=-1
  
  # Disable checkpointing.
  pretrain.log.ckpt = None
  pretrain.trainer.enable_checkpointing = False

  # Log every step.
  pretrain.trainer.log_every_n_steps = 1

  return pretrain


if __name__ == "__main__":
  run.cli.main(llm.pretrain, default_factory=recipe)
