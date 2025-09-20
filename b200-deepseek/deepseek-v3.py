"""Nemo2 pretraining recipe for Deepseek v3 model."""

from nemo.collections import llm
from nemo.collections.llm.recipes import deepseek_v3
from nemo.lightning.pytorch.callbacks import NsysCallback
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed

from .pretrain_deepseek_v3 import override_recipe_configs
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

  pretrain.trainer.plugins = bf16_with_fp8_mixed()
  pretrain.trainer.strategy.pipeline_model_parallel_size = 16
  pretrain.trainer.strategy.tensor_model_parallel_size = 1
  pretrain.trainer.strategy.expert_model_parallel_size = 8
  pretrain.trainer.strategy.virtual_pipeline_model_parallel_size = None
  pretrain.trainer.strategy.expert_tensor_parallel_size = 1

  # Log every step.
  pretrain.trainer.log_every_n_steps = 1

  #[NeMo I 2025-09-19 17:33:25 nemo_logging:393] num_nodes=32
  #[NeMo I 2025-09-19 17:33:25 nemo_logging:393] num_gpus_per_node=8
  #[NeMo I 2025-09-19 17:33:25 nemo_logging:393] mbs=1
  #[NeMo I 2025-09-19 17:33:25 nemo_logging:393] gbs=2048
  #[NeMo I 2025-09-19 17:33:25 nemo_logging:393] tp_size=2
  #[NeMo I 2025-09-19 17:33:25 nemo_logging:393] pp_size=16
  #[NeMo I 2025-09-19 17:33:25 nemo_logging:393] cp_size=1
  #[NeMo I 2025-09-19 17:33:25 nemo_logging:393] vp_size=1
  #[NeMo I 2025-09-19 17:33:25 nemo_logging:393] ep_size=8
  #[NeMo I 2025-09-19 17:33:25 nemo_logging:393] etp_size=1
  #[NeMo I 2025-09-19 17:33:25 nemo_logging:393] enable_cuda_graphs=True
  #[NeMo I 2025-09-19 17:33:25 nemo_logging:393] use_mcore_fsdp=False
  #[NeMo I 2025-09-19 17:33:25 nemo_logging:393] recompute_layers=0
  #[NeMo I 2025-09-19 17:33:25 nemo_logging:393] activation_offload_layers=0
  #[NeMo I 2025-09-19 17:33:25 nemo_logging:393] recompute_modules=None
  #[NeMo I 2025-09-19 17:03:27 nemo_logging:393] keep_fsdp_fp8_transpose_cache=False
  #[NeMo I 2025-09-19 17:03:27 nemo_logging:393] use_user_buffer_registration=False
  #[NeMo I 2025-09-19 17:03:27 nemo_logging:393] use_sharp=False

  pretrain = override_recipe_configs(
        pretrain,
        $NNODES,
        1, #mbs
        2048, #gbs
        2, #TP
        16,
        1,
        1,
        8,
        1,
        True,
        False,
        0,
        0,
        None,
        False,
        False,
    )

  return pretrain


if __name__ == "__main__":
  run.cli.main(llm.pretrain, default_factory=recipe)
