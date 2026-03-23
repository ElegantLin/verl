# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import ray
from omegaconf import OmegaConf
from tqdm import tqdm

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local
from verl.utils.reward_score.hybrid_reward_eif import select_primary_response
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.one_step_eif import compute_one_step_scores, summarize_one_step_estimator


@ray.remote
def process_item(config, data_source, response_item, reward_data, aux_tau_values=None):
    reward_fn = get_custom_reward_fn(config) or default_compute_score
    ground_truth = reward_data["ground_truth"]
    one_step_cfg = config.get("one_step_eif", {})
    one_step_enabled = one_step_cfg.get("enable", False)
    primary_response_index = int(one_step_cfg.get("primary_response_index", 0))

    if one_step_enabled:
        response_lst = [select_primary_response(response_item, primary_response_index=primary_response_index)]
    elif isinstance(response_item, list):
        response_lst = response_item
    else:
        response_lst = [response_item]

    score_lst = []
    for response in response_lst:
        score = reward_fn(data_source, response, ground_truth)
        if isinstance(score, dict):
            score = score["score"]
        score_lst.append(float(score))
    result = {
        "data_source": data_source,
        "naive_score": float(np.mean(score_lst)),
    }

    if one_step_enabled:
        if aux_tau_values is None:
            raise ValueError("one_step_eif is enabled but auxiliary tau values are missing for a row.")
        primary_score = score_lst[0]
        eif_score = compute_one_step_scores([primary_score], [aux_tau_values])[0]
        result.update(
            {
                "primary_score": float(primary_score),
                "one_step_eif_score": float(eif_score),
                "aux_tau_values": list(aux_tau_values),
            }
        )
    return result


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]
    one_step_eif_enable = config.get("one_step_eif", {}).get("enable", False)
    aux_tau_key = config.data.get("aux_tau_key", None)
    aux_tau_values = [None] * len(dataset)
    if one_step_eif_enable:
        if aux_tau_key is None:
            raise ValueError("one_step_eif is enabled, but data.aux_tau_key is not configured.")
        if aux_tau_key not in dataset.columns:
            raise ValueError(
                f"one_step_eif is enabled, but aux_tau_key `{aux_tau_key}` is missing from dataset columns."
            )
        aux_tau_values = dataset[aux_tau_key]

    total = len(dataset)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(**OmegaConf.to_container(config.ray_kwargs.get("ray_init", {})))

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)
    data_source_eif = defaultdict(list)
    data_source_primary = defaultdict(list)
    data_source_aux_tau = defaultdict(list)
    # Create remote tasks
    remote_tasks = [
        process_item.remote(config, data_sources[i], responses[i], reward_model_data[i], aux_tau_values[i]) for i in range(total)
    ]

    # Process results as they come in
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                item_result = ray.get(result_id)
                data_source = item_result["data_source"]
                data_source_reward[data_source].append(item_result["naive_score"])
                if one_step_eif_enable:
                    data_source_eif[data_source].append(item_result["one_step_eif_score"])
                    data_source_primary[data_source].append(item_result["primary_score"])
                    data_source_aux_tau[data_source].append(item_result["aux_tau_values"])
                pbar.update(1)

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f"test_score/{data_source}/naive_mean"] = float(np.mean(rewards))

    if one_step_eif_enable:
        for data_source, eif_scores in data_source_eif.items():
            metric_dict[f"test_score/{data_source}/one_step_eif_mean"] = float(np.mean(eif_scores))
            summary = summarize_one_step_estimator(
                data_source_primary[data_source],
                data_source_aux_tau[data_source],
            )
            for key, value in summary.items():
                metric_dict[f"test_score/{data_source}/{key}"] = float(value)

    print(metric_dict)


if __name__ == "__main__":
    main()
