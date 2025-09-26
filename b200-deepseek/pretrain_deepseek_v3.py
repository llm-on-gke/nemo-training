# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import basename, splitext
from typing import List, Optional

import nemo_run as run

from nemo.collections.llm.recipes.deepseek_v3 import pretrain_recipe
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.pytorch.callbacks.megatron_enable_experimental_callback import MegatronEnableExperimentalCallback
from nemo.lightning.pytorch.callbacks.moe_token_drop import MegatronTokenDropCallback
from nemo.lightning.run.plugins import MemoryProfilePlugin, NsysPlugin
from argument_parser import parse_additional_slurm_params, parse_cli_args
#from ..executors import slurm_executor
from helpers import (
    args_sanity_check,
    build_perf_env_plugin,
    get_user_configs,
    set_exp_logging_configs,
    set_primary_perf_configs,
)
from utils import dump_config_diff_from_base_recipe, hf_tokenizer

HF_MODEL_URI = "deepseek-ai/DeepSeek-V3-Base"



def override_recipe_configs(
    recipe,
    num_nodes: int,
    mbs: int,
    gbs: int,
    max_steps: int,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    vp_size: int,
    ep_size: int,
    etp_size: int,
    enable_cuda_graphs: bool,
    use_mcore_fsdp: bool,
    recompute_layers: int,
    activation_offload_layers: int,
    recompute_modules: Optional[List[str]] = None,
    use_user_buffer_registration: Optional[bool] = None,
    use_sharp: Optional[bool] = None,
    enable_deepep: Optional[bool] = None,
    compute_dtype: Optional[str]='bf16',
    fp8_recipe: Optional[str]=None,
):
    """
    DeepSeek V3 pre-train recipe aimed at achieving best possible performance.
    """
    #recipe = pretrain_recipe(performance_mode=True)

    # reset recompute args in the default recipe
    if recompute_modules is None:
        recipe.model.config.recompute_granularity = None
        recipe.model.config.recompute_method = None
        recipe.model.config.recompute_num_layers = None
        recipe.model.config.recompute_modules = None

    #if not hasattr(recipe.trainer, "callbacks") or recipe.trainer.callbacks is None:
    #    recipe.trainer.callbacks = []

    # Token dispatcher configs. For H100 we use deepEP and for Blackwell,
    # because deepEP is not supported yet, we use all-to-all dispatcher with
    # token drop. After deepEP is supported, we can use deepEP dispatcher.
    # For DeepEP
    #if args.gpu.lower() in ['h100']:
    if enable_deepep:
      recipe.model.config.moe_token_dispatcher_type = "flex"
      recipe.model.config.moe_enable_deepep = True
      recipe.model.config.moe_shared_expert_overlap = False  # not supported for deepEP
      #recipe.model.config.moe_expert_capacity_factor=None #rick
      #use force load balance for reducing variance in benchmarking
      recipe.model.config.moe_router_force_load_balancing = True
      USE_TOKEN_DROP = False
      #recipe.trainer.callbacks.append(run.Config(DeepEPCallback))

    else:
      recipe.model.config.moe_token_dispatcher_type = "alltoall"
      recipe.model.config.moe_enable_deepep = False
      recipe.model.config.moe_shared_expert_overlap = True
      recipe.model.config.moe_expert_capacity_factor=1.0
      USE_TOKEN_DROP = True  # Use token drop callback
   
    if USE_TOKEN_DROP:
        recipe.trainer.callbacks.append(run.Config(MegatronTokenDropCallback))

    # Performance optimization knobs
    recipe.model.config.moe_permute_fusion = True
    recipe.model.config.apply_rope_fusion = True
    recipe.trainer.callbacks.append(run.Config(MegatronEnableExperimentalCallback))

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
    pp_size = pp_size or 1
    vp_size = vp_size or 1
    if (pp_size, vp_size) not in map_pp_vp_to_layout:
        raise ValueError(
            f"Invalid PP and VP size: {pp_size} and {vp_size} to infer PP layout "
            f"for DeepSeek V3. Known PP and VP combinations: {map_pp_vp_to_layout.keys()}"
        )
    layout = map_pp_vp_to_layout[(pp_size, vp_size)]

    if layout is not None:
        layout = list([list(x) for x in layout])  # yield all the elements
    
    if enable_deepep:
      recipe.trainer.strategy.pipeline_model_parallel_layout="Et*2|(tt|)*22t|(tt|)*7mL"
    else:
      recipe.trainer.strategy.pipeline_model_parallel_layout = layout
    # The following knobs are not needed if we specify layout
    recipe.trainer.strategy.account_for_embedding_in_pipeline_split = False
    recipe.trainer.strategy.account_for_loss_in_pipeline_split = False
    recipe.trainer.strategy.num_layers_in_first_pipeline_stage = None
    recipe.trainer.strategy.num_layers_in_last_pipeline_stage = None

    recipe = set_primary_perf_configs(
        recipe,
        "pre_train",
        num_nodes,
        8,
        mbs,
        gbs,
        10, #max_steps
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
        etp_size,
        enable_cuda_graphs=enable_cuda_graphs,
        use_mcore_fsdp=use_mcore_fsdp,
        use_fsdp_double_buffer=False,
        use_user_buffer_registration=use_user_buffer_registration,
        use_sharp=use_sharp,
        recompute_layers=recompute_layers,
        activation_offload_layers=activation_offload_layers,
        compute_dtype=compute_dtype,
        fp8_recipe=fp8_recipe,
        recompute_modules=recompute_modules,
        use_te_act_func=None,
        act_func_fp8_input_store=None,
    )
    #recipe = set_exp_logging_configs(
    #    recipe,
    #    "pre_train",
    #   "llm",
    #  "deepseekv3",
    #    args.tensorboard,
    #   args.wandb,
    #    args.wandb_prj_name,
    #    args.wandb_job_name,
    #)

    # data module configs
    #if args.use_hf_tokenizer:
    recipe.data.tokenizer = hf_tokenizer(HF_MODEL_URI)
    #else:
    #recipe.data.tokenizer = run.Config(
    #        get_nmt_tokenizer, library="null", model_name="NullTokenizer", vocab_size=129280
    #)
    recipe.model.tokenizer = recipe.data.tokenizer

    return recipe

