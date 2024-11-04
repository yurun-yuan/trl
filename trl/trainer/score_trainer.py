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

import gc
import math
import os
import textwrap
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, PrinterCallback

from ..models.utils import unwrap_model_for_generation
from ..trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    prepare_deepspeed,
    print_rich_table,
    truncate_response,
    count_Leading_trailing_values,
)
from .score_config import SCOREConfig
from .utils import generate_model_card


if is_wandb_available():
    import wandb

INVALID_LOGPROB = 1.0


class SCORETrainer(Trainer):
    _tag_names = ["trl", "score"]

    def __init__(
        self,
        config: SCOREConfig,
        tokenizer: PreTrainedTokenizer,
        apply_ids_chat_template: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], # apply_ids_chat_template(history_chat_ids, new_prompt_ids) -> Tensor
        policy: nn.Module,
        ref_policy: nn.Module,
        get_reward: Callable[[str, int], float], # get_reward(response, row_idx) -> scalar reward
        train_dataset: Dataset,
        stage: int,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> None:
        if ref_policy is policy:
            raise ValueError(
                "`policy` and `ref_policy` cannot be the same object. If you want `ref_policy` to be the "
                "same as `policy`, you must mass a copy of it, or `None` if you use peft."
            )

        self.args = config
        args = config
        self.tokenizer = tokenizer
        self.policy = policy

        self.policy.generation_config.eos_token_id = (
            None  # disable `pad_token_id` and `eos_token_id` because we just want to
        )
        self.policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

        self.ref_policy = ref_policy
        self.get_reward = get_reward
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator if data_collator is not None else DataCollatorWithPadding(tokenizer)
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.apply_ids_chat_template = apply_ids_chat_template

        assert stage in [0, 1], "`stage` must be either 0 or 1"
        self.stage = stage
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.bonus_coef = args.bonus_coef

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)

        assert args.rloo_k > 0, "`rloo_k` must be a positive integer"
        self.local_dataloader_batch_size = exact_div(
            args.local_batch_size, args.rloo_k, "`local_batch_size` must be a multiple of rloo_k"
        )  # RLOO logic: needed because RLOO repeats the same prompt args.rloo_k times

        #########
        # setup model, optimizer, and others
        #########
        for module in [policy, ref_policy]:
            disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = tokenizer.eos_token_id
        self.model = policy
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            self.ref_policy = prepare_deepspeed(
                self.ref_policy, args.per_device_train_batch_size, args.fp16, args.bf16
            )
            self.deepspeed = self.model
        else:
            self.ref_policy = self.ref_policy.to(self.accelerator.device)

        self.num_turns = args.num_turns
        self.prompt_templates = [torch.tensor(self.tokenizer.encode(prompt, add_special_tokens=False)) for prompt in args.prompt_templates]
        assert len(self.prompt_templates) == self.num_turns, "Number of prompt templates must match `num_turns`"

        if self.num_turns != 2:
            raise NotImplementedError("Only 2-turn conversations are supported at the moment")

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        self.model_wrapped = self.model
        ref_policy = self.ref_policy
        tokenizer = self.tokenizer
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        # entropy_stats = torch.zeros(stats_shape, device=device)
        # ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches * args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = [data["input_ids"].to(device).repeat(args.rloo_k, 1)] + [None] * (self.num_turns - 1)
                query_indices = data["idx"].to(device).repeat(args.rloo_k, 1)
                context_length = [queries[0].shape[1]] + [None] * (self.num_turns - 1)
                query_responses = [None] * self.num_turns
                logitss = [None] * self.num_turns
                responses = [[] for _ in range(self.num_turns)]
                postprocessed_responses = [[] for _ in range(self.num_turns)]
                logprobs = [[] for _ in range(self.num_turns)]
                ref_logprobs = [[] for _ in range(self.num_turns)]
                scores = [[] for _ in range(self.num_turns)]
                sequence_lengths = [[] for _ in range(self.num_turns)]
                padding_masks = [None] * self.num_turns
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    query_responses_t0, logitss_t0 = batch_generation(
                        unwrapped_model,
                        queries[0],
                        args.local_rollout_forward_batch_size,
                        tokenizer.pad_token_id,
                        generation_config,
                    )
                    query_responses[0] = query_responses_t0
                    logitss[0] = logitss_t0


                queries_t1 = []
                for i in range(query_responses_t0.shape[0]):
                    num_leading_pad, num_trailing_pad = count_Leading_trailing_values(query_responses_t0[i], tokenizer.pad_token_id)
                    end = query_responses_t0[i].size(0) - num_trailing_pad
                    query_resp_t0 = query_responses_t0[i][num_leading_pad : end]
                    query_resp_t0 = self.apply_ids_chat_template(query_resp_t0, self.prompt_templates[1].to(query_resp_t0.device))
                    queries_t1.append(query_resp_t0)
                max_length = max(tensor.size(0) for tensor in queries_t1)
                queries_t1 = [F.pad(tensor, (max_length - tensor.size(0), 0), value=tokenizer.pad_token_id) for tensor in queries_t1]
                queries_t1 = torch.stack(queries_t1, dim=0)
                queries[1] = queries_t1

                context_length[1] = queries_t1.shape[1]

                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    query_responses_t1, logitss_t1 = batch_generation(
                        unwrapped_model,
                        queries[1],
                        args.local_rollout_forward_batch_size,
                        tokenizer.pad_token_id,
                        generation_config,
                    )
                    query_responses[1] = query_responses_t1
                    logitss[1] = logitss_t1

                for turn in range(self.num_turns):
                    for i in range(0, queries[turn].shape[0], args.local_rollout_forward_batch_size):
                        query = queries[turn][i : i + args.local_rollout_forward_batch_size]
                        query_response = query_responses[turn][i : i + args.local_rollout_forward_batch_size]
                        response = query_response[:, context_length[turn]:]
                        logits = logitss[turn][i : i + args.local_rollout_forward_batch_size]
                        all_logprob = F.log_softmax(logits, dim=-1)
                        logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                        del logits, all_logprob
                        torch.cuda.empty_cache()

                        ref_output = forward(ref_policy, query_response, tokenizer.pad_token_id)
                        ref_logits = ref_output.logits[:, context_length[turn] - 1 : -1] / (args.temperature + 1e-7)
                        ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                        ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                        del ref_output, ref_logits, ref_all_logprob
                        torch.cuda.empty_cache()

                        # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                        postprocessed_response = response
                        if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                            postprocessed_response = truncate_response(
                                args.stop_token_id, tokenizer.pad_token_id, response
                            )

                        # Response Processing 2. run reward model on the truncated responses
                        sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1

                        query_idx = query_indices[i : i + args.local_rollout_forward_batch_size]
                        score = torch.tensor(
                            [self.get_reward(tokenizer.decode(postprocessed_response[j]), query_idx[j].item()) for j in range(len(sequence_length))],
                            device=device,
                        )

                        responses[turn].append(response)
                        postprocessed_responses[turn].append(postprocessed_response)
                        logprobs[turn].append(logprob)
                        ref_logprobs[turn].append(ref_logprob)
                        sequence_lengths[turn].append(sequence_length)
                        scores[turn].append(score)
                    responses[turn] = torch.cat(responses[turn], 0)
                    postprocessed_responses[turn] = torch.cat(postprocessed_responses[turn], 0)
                    logprobs[turn] = torch.cat(logprobs[turn], 0)
                    ref_logprobs[turn] = torch.cat(ref_logprobs[turn], 0)
                    sequence_lengths[turn] = torch.cat(sequence_lengths[turn], 0)
                    scores[turn] = torch.cat(scores[turn], 0)
                    del (logprob, ref_logprob, score)
                    torch.cuda.empty_cache()
                    gc.collect()

                    # Response Processing 3. filter response. Ensure that the sample contains stop_token_id
                    # responses not passing that filter will receive a low (fixed) score
                    # only query humans on responses that pass that filter
                    contain_eos_token = torch.any(postprocessed_responses[turn] == tokenizer.eos_token_id, dim=-1)
                    if args.missing_eos_penalty is not None:
                        scores[turn][~contain_eos_token] -= self.args.missing_eos_penalty
                    # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                    # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                    response_idxs = torch.arange(responses[turn].shape[1], device=responses[turn].device).repeat(responses[turn].shape[0], 1)
                    padding_masks[turn] = response_idxs > sequence_lengths[turn].unsqueeze(1)
                    logprobs[turn] = torch.masked_fill(logprobs[turn], padding_masks[turn], INVALID_LOGPROB)
                    ref_logprobs[turn] = torch.masked_fill(ref_logprobs[turn], padding_masks[turn], INVALID_LOGPROB)

                # 4. compute rewards
                kl = [logprobs[turn] - ref_logprobs[turn] for turn in range(self.num_turns)]

                if self.stage == 0:
                    non_score_reward = (-args.beta2 * kl[0] - args.beta1 * kl[1]).sum(1)
                    rlhf_reward = scores[1] + non_score_reward
                else:
                    non_score_reward = (-args.beta1 * kl[0] - args.beta1 * kl[1]).sum(1)
                    rlhf_reward = scores[0] + scores[1] + non_score_reward + args.bonus_coef * (score[1] - score[0])

                # vectorized RLOO advantages implementation
                rlhf_reward = rlhf_reward.reshape(args.rloo_k, -1)
                if args.rloo_k > 1:
                    baseline = (rlhf_reward.sum(0) - rlhf_reward) / (args.rloo_k - 1)
                    advantages = rlhf_reward - baseline
                else:
                    advantages = rlhf_reward
                advantages = advantages.flatten()
                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_advantage = advantages[micro_batch_inds]

                            mb_logprobs, new_logprobs  = [], []
                            for turn in range(self.num_turns):
                                mb_responses = responses[turn][micro_batch_inds]
                                mb_query_responses = query_responses[turn][micro_batch_inds]
                                mb_logprobs.append(logprobs[turn][micro_batch_inds])

                                output = forward(model, mb_query_responses, tokenizer.pad_token_id)
                                logits = output.logits[:, context_length[turn] - 1 : -1]/(args.temperature + 1e-7)
                                new_all_logprob = F.log_softmax(logits, dim=-1)
                                new_logprob = torch.gather(new_all_logprob, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                                new_logprobs.append(torch.masked_fill(
                                    new_logprob, padding_masks[turn][micro_batch_inds], INVALID_LOGPROB
                                ))

                            # For stat only
                            # new_ratio = (new_logprobs - mb_logprobs).exp()

                            new_logprobs = new_logprobs[0].sum(1) + new_logprobs[1].sum(1)
                            mb_logprobs = mb_logprobs[0].sum(1) + mb_logprobs[1].sum(1)
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = pg_loss_max.mean()
                            loss = pg_loss
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                                # prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                # entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                # entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                # ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = new_ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    self.state.global_step += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        # output, 
                        logits, new_all_logprobs, new_logprobs,
                        logprobs_diff, ratio, pg_losses, pg_losses2,
                        pg_loss, loss, pg_clipfrac, 
                        # prob_dist, 
                        # entropy, 
                        approxkl,
                        mb_advantage, 
                        # mb_responses, 
                        # mb_query_responses, 
                        mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
            with torch.no_grad():
                # mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl_turn1"] = self.accelerator.gather(kl[0].sum(1).mean()).mean().item()
                metrics["objective/kl_turn2"] = self.accelerator.gather(kl[1].sum(1).mean()).mean().item()
                # metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()
                metrics["objective/rlhf_reward"] = self.accelerator.gather(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather(vf_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather(vf_clipfrac_stats).mean().item()
                # metrics["policy/entropy_avg"] = self.accelerator.gather(entropy_stats).mean().item()
                # metrics["val/ratio"] = self.accelerator.gather(ratio_stats).mean().item()
                # metrics["val/ratio_var"] = self.accelerator.gather(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == tokenizer.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)
            del kl, mean_kl, mean_entropy, scores

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None, metrics=metrics)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            torch.cuda.empty_cache()
            gc.collect()

            # if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
            #     self.generate_completions(sampling=True)

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        raise NotImplementedError("Sampling is not supported in SCORETrainer")

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, List[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str`, *optional*, defaults to `None`):
                The name of the model.
            dataset_name (`str`, *optional*, defaults to `None`):
                The name of the dataset used for training.
            tags (`str`, `List[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        citation = textwrap.dedent("""\
        @inproceedings{ahmadian2024back,
            title        = {{Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs}},
            author       = {Arash Ahmadian and Chris Cremer and Matthias Gall{\'{e}} and Marzieh Fadaee and Julia Kreutzer and Olivier Pietquin and Ahmet {\"{U}}st{\"{u}}n and Sara Hooker},
            year         = 2024,
            booktitle    = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), {ACL} 2024, Bangkok, Thailand, August 11-16, 2024},
            publisher    = {Association for Computational Linguistics},
            pages        = {12248--12267},
            editor       = {Lun{-}Wei Ku and Andre Martins and Vivek Srikumar},
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            trainer_name="SCORE",
            trainer_citation=citation,
            paper_title="Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs",
            paper_id="2402.14740",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
