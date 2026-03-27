# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Custom reward functions used by the HERO paper baselines."""


def compute_score_math_verify(data_source: str, solution_str: str, ground_truth: str, extra_info: dict, **kwargs):
    from verl.utils.reward_score.math_verify import compute_score

    return float(compute_score(model_output=solution_str, ground_truth=ground_truth))


def compute_score_math_reward(data_source: str, solution_str: str, ground_truth: str, extra_info: dict, **kwargs):
    from verl.utils.reward_score.math_reward import compute_score

    return float(compute_score(solution_str=solution_str, ground_truth=ground_truth))
