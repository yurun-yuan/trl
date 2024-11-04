# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import os
from dataclasses import dataclass
from typing import Iterable, Literal, Optional

from transformers import TrainingArguments

@dataclass
class SCOREConfig(TrainingArguments):
    r"""
    Configuration class for the [`SCORETrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        exp_name (`str`, *optional*, defaults to `os.path.basename(__file__)[: -len(".py")]`):
            Name of this experiment.
        reward_model_path (`str`, *optional*, defaults to `"EleutherAI/pythia-160m"`):
            Path to the reward model.
        num_ppo_epochs (`int`, *optional*, defaults to `4`):
            Number of epochs to train.
        whiten_rewards (`bool`, *optional*, defaults to `False`):
            Whether to whiten the rewards.
        kl_coef (`float`, *optional*, defaults to `0.05`):
            KL coefficient.
        cliprange (`float`, *optional*, defaults to `0.2`):
            Clip range.
        rloo_k (`int`, *optional*, defaults to `2`):
            REINFORCE Leave-One-Out (RLOO) number of online samples per prompt.
    """

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    num_ppo_epochs: int = 4
    whiten_rewards: bool = False
    cliprange: float = 0.2
    rloo_k: int = 1
    prompt_templates: Iterable[str] = None
    num_turns: int = 2
    beta1: float = 0.1
    beta2: float = 1
    bonus_coef: float = 10



    run_name: Optional[str] = None
    dataset_num_proc: Optional[int] = None
    num_mini_batches: int = 1
    total_episodes: Optional[int] = None
    local_rollout_forward_batch_size: int = 64
    num_sample_generations: int = 10
    response_length: int = 53
    stop_token: Optional[Literal["eos"]] = None
    stop_token_id: Optional[int] = None
    temperature: float = 0.7
    missing_eos_penalty: Optional[float] = None
    sft_model_path: str = "EleutherAI/pythia-160m"
    world_size: Optional[int] = None
    num_total_batches: Optional[int] = None
    micro_batch_size: Optional[int] = None
    local_batch_size: Optional[int] = None
    batch_size: Optional[int] = None
    local_mini_batch_size: Optional[int] = None
    mini_batch_size: Optional[int] = None
    push_to_hub: bool = False


# @dataclass
# class SCOREConfig(OnPolicyConfig):
#     r"""
#     Configuration class for the [`SCORETrainer`].

#     Using [`~transformers.HfArgumentParser`] we can turn this class into
#     [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
#     command line.

#     Parameters:
#         exp_name (`str`, *optional*, defaults to `os.path.basename(__file__)[: -len(".py")]`):
#             Name of this experiment.
#         reward_model_path (`str`, *optional*, defaults to `"EleutherAI/pythia-160m"`):
#             Path to the reward model.
#         num_ppo_epochs (`int`, *optional*, defaults to `4`):
#             Number of epochs to train.
#         whiten_rewards (`bool`, *optional*, defaults to `False`):
#             Whether to whiten the rewards.
#         kl_coef (`float`, *optional*, defaults to `0.05`):
#             KL coefficient.
#         cliprange (`float`, *optional*, defaults to `0.2`):
#             Clip range.
#         rloo_k (`int`, *optional*, defaults to `2`):
#             REINFORCE Leave-One-Out (RLOO) number of online samples per prompt.
#     """

#     exp_name: str = os.path.basename(__file__)[: -len(".py")]
#     num_ppo_epochs: int = 4
#     whiten_rewards: bool = False
#     cliprange: float = 0.2
#     rloo_k: int = 1
#     prompt_templates: Iterable[str] = None
#     num_turns: int = 2
#     beta1: float = 0.1
#     beta2: float = 1
#     bonus_coef: float = 10
