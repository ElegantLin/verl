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

import asyncio
import logging
import os
from typing import Any

import aiohttp
import numpy as np
import ray
import torch
from omegaconf import DictConfig, open_dict
from PIL import Image
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_tokenizer
from verl.utils.experimental.reward_utils import pil_image_to_base64, prepare_query_for_multi_modal
from verl.utils.fs import copy_to_local
from verl.utils.ray_utils import get_event_loop
from verl.utils.reward_score.hero import apply_hero_shaping, apply_naive_reward_combine
from verl.utils.reward_score.hybrid_reward_eif import (
    AceMathRewardScorer,
    MarginalLLMScorer,
    TauLLMScorer,
    compute_response_algorithm1_eif,
    extract_question_text,
)
from verl.utils.reward_score.one_step_eif import compute_algorithm1_scores

from .reward_model import RewardModelManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))



def _normalize_hybrid_eif_scalar(value, *, key: str) -> float:
    if isinstance(value, torch.Tensor):
        scalar = value.detach().cpu().numpy()
    else:
        if hasattr(value, 'tolist') and not isinstance(value, str):
            value = value.tolist()
        scalar = np.asarray(value, dtype=np.float64)

    if scalar.ndim == 0:
        return float(scalar)
    if scalar.ndim == 1 and scalar.shape[0] == 1:
        return float(scalar[0])
    raise ValueError(f'`{key}` must contain scalar values, but got shape={scalar.shape}.')


def _load_hybrid_eif_scalar_values(data: DataProto, *, key: str) -> np.ndarray:
    if data.batch is not None and key in data.batch.keys():
        values = data.batch[key]
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            return values.astype(np.float64, copy=False)
        if values.ndim == 2 and values.shape[1] == 1:
            return values[:, 0].astype(np.float64, copy=False)
        raise ValueError(f'`{key}` in batch must be 1D scalar-aligned values, got shape={values.shape}.')

    if data.non_tensor_batch is not None and key in data.non_tensor_batch:
        return np.asarray([_normalize_hybrid_eif_scalar(value, key=key) for value in data.non_tensor_batch[key]])

    raise KeyError(f'`reward.reward_manager.name=hybrid_eif` requires `{key}` in batch or non_tensor_batch.')


def migrate_legacy_reward_impl(config):
    """
    Migrate the legacy reward model implementation to the new one.
    """
    # 1. reward workers migration
    # config.reward_model.num_workers -> config.reward.num_workers
    if config.reward_model.num_workers is not None:
        config.reward.num_workers = config.reward_model.num_workers

    # 2. reward manager migration
    # config.reward_model.reward_manager -> config.reward.reward_manager
    if config.reward_model.reward_manager is not None:
        config.reward.reward_manager.name = config.reward_model.reward_manager
    if config.reward_model.reward_loop_source is not None:
        config.reward.reward_manager.source = config.reward_model.reward_loop_source
        config.reward.reward_manager.module.path = config.reward_model.reward_loop_module_path
        config.reward.reward_manager.module.name = config.reward_model.reward_loop_class_name

    # 3. custom reward function migration
    # config.custom_reward_function -> config.reward.custom_reward_function
    if not all(v is None for v in config.custom_reward_function.values()):
        config.reward.custom_reward_function = config.custom_reward_function

    # 4. reward model migration
    # config.reward_model -> config.reward.reward_model
    for key in ["enable", "enable_resource_pool", "n_gpus_per_node", "nnodes"]:
        if config.reward_model.get(key) is not None:
            config.reward.reward_model[key] = config.reward_model[key]
    if config.reward_model.model.path is not None:
        config.reward.reward_model.model_path = config.reward_model.model.path
    # config.reward_model.reward_kwargs -> config.reward.reward_kwargs (for dapo algo)
    if config.reward_model.get("reward_kwargs") is not None:
        with open_dict(config.reward):
            config.reward["reward_kwargs"] = config.reward_model["reward_kwargs"]
    # config.reward_model.rollout -> config.reward.reward_model.rollout
    legacy_rollout = config.reward_model.rollout
    for key in legacy_rollout.keys():
        if legacy_rollout[key] is not None:
            config.reward.reward_model.rollout[key] = legacy_rollout[key]

    # 5. sandbox_fusion migration
    # config.sandbox_fusion -> reward.sandbox_fusion
    if not all(v is None for v in config.sandbox_fusion.values()):
        config.reward.sandbox_fusion = config.sandbox_fusion

    # 6. delete legacy config from configs
    with open_dict(config):
        del config.reward_model
        del config.custom_reward_function
        del config.sandbox_fusion

    return config


class RewardLoopWorker:
    """
    RewardLoopWork can tackle reward computation:
    (1) rule-based reward computation
    (2) reward model-based reward computation (both disrm and genrm)
    (3) high-flexible user-customized reward function (can access rm by posting requests to reward_model_router)

    Reward Computation Logic:
    - if user-customized reward function is provided:
        -> directly use user-customized reward function
    - if user-customized reward function is not provided:
        -> rm is not enabled: use default rule-based reward function
        -> rm is disrm: compute reward score using disrm
        -> rm is genrm: raise error (user-costomized reward func must be provided)
    """

    def __init__(self, config: DictConfig, reward_router_address: str = None):
        """
        Args:
            config: DictConfig, the config for reward loop worker.
            reward_router_address: str, the address of reward router.
        """
        self.config = config
        self.reward_router_address = reward_router_address
        self._init_reward_fn()
        self.loop = get_event_loop()

    def _init_reward_fn(self):
        input_tokenizer_path = self.config.actor_rollout_ref.model.tokenizer_path
        if input_tokenizer_path is None:
            input_tokenizer_path = self.config.actor_rollout_ref.model.path
        input_tokenizer_local_path = copy_to_local(input_tokenizer_path)
        self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=True)
        self.reward_model_tokenizer = None
        if self.config.reward.reward_model.enable:
            reward_model_tokenizer_local_path = copy_to_local(self.config.reward.reward_model.model_path)
            self.reward_model_tokenizer = hf_tokenizer(reward_model_tokenizer_local_path, trust_remote_code=True)

        self.reward_manager = load_reward_manager(
            self.config,
            self.input_tokenizer,
            reward_router_address=self.reward_router_address,
            reward_model_tokenizer=self.reward_model_tokenizer,
        )

    async def compute_score_batch(self, data: DataProto) -> list[dict]:
        tasks = []
        for i in range(len(data)):
            tasks.append(asyncio.create_task(self.compute_score(data[i : i + 1])))
        outputs = await asyncio.gather(*tasks)
        return outputs

    async def compute_score(self, data: DataProto) -> dict:
        assert len(data) == 1, "RewardLoopWorker only support single data item"
        if self.config.reward.reward_manager.name in {"hero", "naive_combine"}:
            return await self.compute_score_hero(data)
        if self.config.reward.reward_manager.name == "hybrid_eif":
            # hybrid_eif uses precomputed tau samples from the batch, so the worker only computes phi.
            return await self.reward_manager.run_single(data)
        if self.config.reward.reward_manager.name == "hybrid_eif_online":
            # hybrid_eif_online computes verifier-side phi in workers and applies online tau aggregation in the manager.
            return await self.reward_manager.run_single(data)

        if self.config.reward.custom_reward_function.path is not None:
            # directly use user-customized reward function
            return await self.reward_manager.run_single(data)
        else:
            if self.config.reward.reward_model.enable:
                # we assume the rm is disrm
                # genrm must set custom_reward_function
                return await self.compute_score_disrm(data)
            else:
                return await self.reward_manager.run_single(data)

    async def compute_score_hero(self, data: DataProto) -> dict:
        """Compute HERO inputs: verifier-side score + dense RM score."""
        hero_output = await self.reward_manager.run_single(data)
        reward_extra_info = dict(hero_output.get("reward_extra_info", {}))
        rule_score = float(hero_output.get("reward_score", 0.0))

        # fallback to rule score when RM is not enabled
        rm_score = rule_score
        if self.config.reward.reward_model.enable:
            rm_result = await self.compute_score_disrm(data)
            rm_score = float(rm_result["reward_score"])

        reward_extra_info["hero_rule_score"] = rule_score
        reward_extra_info["hero_rm_score"] = rm_score
        # Keep acc for pass@k/val metric aggregation.
        reward_extra_info.setdefault("acc", rule_score)
        return {"reward_score": rule_score, "reward_extra_info": reward_extra_info}

    async def _post_request(self, payload: dict, endpoint: str, max_retries: int = 16):
        url = f"http://{self.reward_router_address}/{endpoint}"
        last_exception = None
        for attempt in range(max_retries):
            try:
                # It's safer to have a timeout instead of None, which can hang indefinitely.
                timeout = aiohttp.ClientTimeout(total=None)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except aiohttp.ClientResponseError as e:
                # Do not retry on 4xx client errors, but retry on 5xx server errors.
                if 400 <= e.status < 500:
                    logger.error(f"Request to {url} failed with client error HTTP {e.status}: {e}. Not retrying.")
                    raise
                last_exception = e
                logger.warning(
                    f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed with HTTP {e.status}: {e}. "
                    "Retrying..."
                )
            except (asyncio.TimeoutError, aiohttp.ClientConnectorError) as e:
                last_exception = e
                logger.warning(f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed: {e}. Retrying...")
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed with unexpected error: {e}. "
                    "Retrying..."
                )

            if attempt < max_retries - 1:
                # Using exponential backoff is generally better than a fixed sleep.
                backoff_seconds = 2**attempt
                await asyncio.sleep(min(backoff_seconds, 30))

        logger.error(f"Max retries ({max_retries}) reached for request to {url}.")
        if last_exception:
            raise last_exception

    async def _preprocess_reward_inputs(self, data: DataProto) -> str:
        assert len(data) == 1, "RewardLoopWorker only support single data item"
        data_item = data[0]
        assert "raw_prompt" in data_item.non_tensor_batch

        # extract raw prompt
        chat: list = list(data_item.non_tensor_batch["raw_prompt"])

        # extract response
        response = data_item.batch["responses"]
        if response.ndim == 3:
            # handling multi-modal response
            response_image = response
            if isinstance(response_image, torch.Tensor):
                response_image = response_image.float().permute(1, 2, 0).cpu().numpy()
            assert response_image.shape[-1] == 3, "must be in HWC format"
            response_image = (response_image * 255).round().clip(0, 255).astype(np.uint8)
            response_image = Image.fromarray(response_image)

            image_base64 = await self.loop.run_in_executor(None, pil_image_to_base64, response_image)
            query = prepare_query_for_multi_modal(image_base64)

            chat.append({"role": "assistant", "content": query})
        else:
            response_ids = response
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            rollout_response = self.input_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            rollout_response = rollout_response.replace(self.input_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": rollout_response})

        rm_prompt = self.reward_model_tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=False,
            tokenize=False,
        )

        # llama tokenizer will add bos token by default
        # will be removed in vllm >= 0.11.2, where we can add "add_special_tokens" = False
        if self.reward_model_tokenizer.bos_token is not None and rm_prompt.startswith(
            self.reward_model_tokenizer.bos_token
        ):
            rm_prompt = rm_prompt[len(self.reward_model_tokenizer.bos_token) :]

        return rm_prompt

    async def compute_score_disrm(self, data: DataProto) -> dict:
        disrm_prompt = await self._preprocess_reward_inputs(data)
        engine_name = self.config.reward.reward_model.rollout.name
        model_name = self.config.reward.reward_model.model_path
        if engine_name == "vllm":
            payloads = {
                "model": model_name,
                "input": disrm_prompt,
                "activation": False,
            }
            output = await self._post_request(payloads, "classify")
            rm_score = output["data"][-1]["probs"][-1]
        elif engine_name == "sglang":
            payloads = {
                "model": model_name,
                "input": disrm_prompt,
            }
            output = await self._post_request(payloads, "v1/embeddings")
            rm_score = output["data"][-1]["embedding"][-1]
        elif engine_name == "trtllm":
            # TODO: remove this once TRT-LLM switches to TorchSampler
            raise ValueError("TensorRT-LLM backend does not support reward models currently.")

            payloads = {
                "model": model_name,
                "prompt": disrm_prompt,
                "return_context_logits": True,
            }
            output = await self._post_request(payloads, "v1/completions")
            rm_score = output["choices"][0]["context_logits"]
            assert isinstance(rm_score, list) and len(rm_score) > 0, (
                "TensorRT-LLM OpenAI server response for reward score is not in the expected format."
            )

            rm_score = float(rm_score[0][0])
            logger.debug(f"rm score: {rm_score}")
        else:
            raise NotImplementedError(f"RewardLoopManager does not support {engine_name}")

        return {"reward_score": rm_score}


class RewardLoopManager:
    """
    RewardLoopManager run in single controller.
    This class will create reward loop workers and manage them.
    """

    def __init__(self, config: DictConfig, rm_resource_pool: RayResourcePool = None):
        self.config = config
        self._hero_sigma_bar: float | None = None
        self._hybrid_eif_online_input_tokenizer = None
        self._hybrid_eif_online_aux_scorer: AceMathRewardScorer | None = None
        self._hybrid_eif_online_tau_scorer: TauLLMScorer | None = None
        self._hybrid_eif_online_m_scorer: MarginalLLMScorer | None = None
        if self.config.reward.reward_model.enable:
            self.reward_model_manager = RewardModelManager(config.reward.reward_model, rm_resource_pool)
            self.reward_router_address = self.reward_model_manager.get_router_address()
        else:
            self.reward_model_manager = None
            self.reward_router_address = None

        self.reward_loop_workers_class = ray.remote(RewardLoopWorker)
        self._init_reward_loop_workers()

    def _init_reward_loop_workers(self):
        self.reward_loop_workers = []
        num_workers = self.config.reward.num_workers
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]

        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]

            self.reward_loop_workers.append(
                self.reward_loop_workers_class.options(
                    name=f"reward_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=True,
                    ),
                ).remote(self.config, self.reward_router_address)
            )

    def _init_hybrid_eif_online_components(self):
        if self._hybrid_eif_online_input_tokenizer is None:
            input_tokenizer_local_path = copy_to_local(self.config.actor_rollout_ref.model.path)
            self._hybrid_eif_online_input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=True)

        cfg = self.config.reward.get("reward_kwargs", {}).get("hybrid_eif_online", {})
        if self.reward_model_manager is None or self.reward_router_address is None:
            raise ValueError(
                'hybrid_eif_online requires reward.reward_model.enable=True so AceMath-7B-RM can provide r_i.'
            )

        if self._hybrid_eif_online_aux_scorer is None:
            self._hybrid_eif_online_aux_scorer = AceMathRewardScorer(
                model=str(self.config.reward.reward_model.model_path),
                base_url=str(self.reward_router_address),
                engine_name=str(self.config.reward.reward_model.rollout.name),
                reward_model_tokenizer=self.reward_model_manager.tokenizer,
                timeout=float(cfg.get("aux_timeout", 300.0)),
            )

        if self._hybrid_eif_online_tau_scorer is None:
            self._hybrid_eif_online_tau_scorer = TauLLMScorer(
                model=str(cfg.get("tau_model", "Qwen/Qwen3.5-397B-A17B-FP8")),
                base_url=str(cfg.get("tau_base_url", "http://localhost:8000/v1")),
                api_key_env=str(cfg.get("tau_api_key_env", "OPENAI_API_KEY")),
                temperature=float(cfg.get("tau_temperature", 0.0)),
                max_tokens=int(cfg.get("tau_max_tokens", 16)),
                timeout=float(cfg.get("tau_timeout", 300.0)),
            )

        if self._hybrid_eif_online_m_scorer is None:
            self._hybrid_eif_online_m_scorer = MarginalLLMScorer(
                model=str(cfg.get("m_model", cfg.get("tau_model", "Qwen/Qwen3.5-397B-A17B-FP8"))),
                base_url=str(cfg.get("m_base_url", cfg.get("tau_base_url", "http://localhost:8000/v1"))),
                api_key_env=str(cfg.get("m_api_key_env", cfg.get("tau_api_key_env", "OPENAI_API_KEY"))),
                temperature=float(cfg.get("m_temperature", 0.0)),
                max_tokens=int(cfg.get("m_max_tokens", 16)),
                timeout=float(cfg.get("m_timeout", 300.0)),
            )

    def _decode_response_texts(self, data: DataProto) -> list[str]:
        self._init_hybrid_eif_online_components()
        tokenizer = self._hybrid_eif_online_input_tokenizer
        assert tokenizer is not None

        prompt_length = data.batch["prompts"].size(1)
        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=1)

        decoded = []
        for i in range(len(data)):
            length = int(valid_response_length[i].item())
            decoded.append(tokenizer.decode(response_ids[i, :length], skip_special_tokens=True))
        return decoded

    def _apply_hybrid_eif_online(self, data: DataProto, reward_extra_infos: list[dict], scores: list[float]) -> list[float]:
        self._init_hybrid_eif_online_components()
        cfg = self.config.reward.get("reward_kwargs", {}).get("hybrid_eif_online", {})
        fail_open_to_phi = bool(cfg.get("fail_open_to_phi", True))
        aux_concurrency = int(cfg.get("aux_concurrency", 32))
        tau_concurrency = int(cfg.get("tau_concurrency", 128))
        m_concurrency = int(cfg.get("m_concurrency", tau_concurrency))

        decoded_responses = self._decode_response_texts(data)
        raw_prompts = data.non_tensor_batch.get("raw_prompt", np.array([None] * len(data), dtype=object))
        extra_infos = data.non_tensor_batch.get("extra_info", np.array([None] * len(data), dtype=object))

        aux_semaphore = asyncio.Semaphore(aux_concurrency) if aux_concurrency > 0 else None
        tau_semaphore = asyncio.Semaphore(tau_concurrency) if tau_concurrency > 0 else None
        m_semaphore = asyncio.Semaphore(m_concurrency) if m_concurrency > 0 else None

        aux_scorer = self._hybrid_eif_online_aux_scorer
        tau_scorer = self._hybrid_eif_online_tau_scorer
        m_scorer = self._hybrid_eif_online_m_scorer
        assert aux_scorer is not None and tau_scorer is not None and m_scorer is not None

        async def _score_row(sample_idx: int, *, aux_session) -> dict[str, Any]:
            try:
                question = extract_question_text(raw_prompts[sample_idx], extra_info=extra_infos[sample_idx])
                result = await compute_response_algorithm1_eif(
                    question=question,
                    response=decoded_responses[sample_idx],
                    phi_score=float(scores[sample_idx]),
                    aux_scorer=aux_scorer,
                    tau_scorer=tau_scorer,
                    m_scorer=m_scorer,
                    raw_prompt=raw_prompts[sample_idx],
                    aux_session=aux_session,
                    aux_semaphore=aux_semaphore,
                    tau_semaphore=tau_semaphore,
                    m_semaphore=m_semaphore,
                )
                return {"ok": True, "result": result}
            except Exception as exc:  # pragma: no cover - exercised by integration paths.
                return {"ok": False, "error": exc}

        async def _run_row_scoring() -> list[dict[str, Any]]:
            aux_timeout = aiohttp.ClientTimeout(total=aux_scorer.timeout)
            aux_connector = aiohttp.TCPConnector(limit=0)
            async with aiohttp.ClientSession(timeout=aux_timeout, connector=aux_connector) as aux_session:
                tasks = [
                    _score_row(sample_idx, aux_session=aux_session)
                    for sample_idx in range(len(data))
                ]
                return list(await asyncio.gather(*tasks))

        row_results = asyncio.run(_run_row_scoring())

        final_scores = np.asarray(scores, dtype=np.float64)
        for sample_idx, result in enumerate(row_results):
            try:
                if not result["ok"]:
                    raise result["error"]

                row_result = result["result"]
                final_scores[sample_idx] = row_result["score"]
                diagnostics = row_result["diagnostics"]

                info = reward_extra_infos[sample_idx]
                phi_score = float(diagnostics["phi"])
                info["hybrid_eif_online_phi"] = phi_score
                info["hybrid_eif_online_aux_reward"] = float(row_result["aux_reward"])
                info["hybrid_eif_online_tau"] = float(diagnostics["tau"])
                info["hybrid_eif_online_m"] = float(diagnostics["m"])
                info["hybrid_eif_online_final_score"] = float(final_scores[sample_idx])
                info["hybrid_eif_online_fallback"] = 0
                info["hybrid_eif_online_error"] = None
                info.setdefault("acc", phi_score)
            except Exception as exc:
                if not fail_open_to_phi:
                    raise
                logger.warning("hybrid_eif_online failed for sample_idx=%s: %s. Falling back to phi.", sample_idx, exc)
                phi_score = float(scores[sample_idx])
                final_scores[sample_idx] = phi_score
                info = reward_extra_infos[sample_idx]
                info["hybrid_eif_online_phi"] = phi_score
                info["hybrid_eif_online_final_score"] = phi_score
                info["hybrid_eif_online_fallback"] = 1
                info["hybrid_eif_online_error"] = str(exc)
                info.setdefault("acc", phi_score)

        return final_scores.tolist()

    def compute_rm_score(self, data: DataProto) -> DataProto:
        if self.reward_model_manager is not None:
            self.reward_model_manager.wake_up()

        chunks = data.chunk(len(self.reward_loop_workers))
        outputs = ray.get(
            [
                worker.compute_score_batch.remote(chunk)
                for worker, chunk in zip(self.reward_loop_workers, chunks, strict=True)
            ]
        )
        outputs_flat = [item for sublist in outputs for item in sublist]

        reward_extra_infos = [output.get("reward_extra_info", {}) for output in outputs_flat]
        # compute rm score
        scores = [float(item["reward_score"]) for item in outputs_flat]

        if self.config.reward.reward_manager.name == "hero":
            hero_cfg = self.config.reward.get("reward_kwargs", {}).get("hero", {})
            rule_scores = np.asarray(
                [info.get("hero_rule_score", score) for info, score in zip(reward_extra_infos, scores, strict=True)],
                dtype=np.float32,
            )
            dense_scores = np.asarray(
                [info.get("hero_rm_score", score) for info, score in zip(reward_extra_infos, scores, strict=True)],
                dtype=np.float32,
            )
            if "uid" in data.non_tensor_batch:
                uids = np.asarray(data.non_tensor_batch["uid"], dtype=object)
            else:
                # Fallback: each sample forms its own group when uid is unavailable.
                uids = np.asarray([f"sample_{i}" for i in range(len(scores))], dtype=object)
            hero_scores, hero_diag, self._hero_sigma_bar = apply_hero_shaping(
                rule_scores=rule_scores,
                rm_scores=dense_scores,
                uids=uids,
                alpha=float(hero_cfg.get("alpha", 0.1)),
                beta=float(hero_cfg.get("beta", 0.1)),
                eps=float(hero_cfg.get("eps", 1e-6)),
                w_min=float(hero_cfg.get("w_min", 0.4)),
                w_max=float(hero_cfg.get("w_max", 3.0)),
                k=float(hero_cfg.get("k", 6.0)),
                sigma_bar=self._hero_sigma_bar,
                sigma_ema=float(hero_cfg.get("sigma_ema", 0.9)),
            )
            scores = hero_scores.tolist()
            for i, info in enumerate(reward_extra_infos):
                info["hero_rule_score"] = float(rule_scores[i])
                info["hero_rm_score"] = float(dense_scores[i])
                info["hero_stratified_score"] = float(hero_diag["stratified_reward"][i])
                info["hero_difficulty_weight"] = float(hero_diag["difficulty_weight"][i])
                info["hero_group_sigma"] = float(hero_diag["group_sigma"][i])
                info["hero_sigma_bar"] = float(self._hero_sigma_bar)
                info["hero_final_score"] = float(scores[i])
                info.setdefault("acc", float(rule_scores[i]))
        elif self.config.reward.reward_manager.name == "naive_combine":
            naive_cfg = self.config.reward.get("reward_kwargs", {}).get("naive_combine", {})
            alpha = float(naive_cfg.get("alpha", 0.5))
            rule_scores = np.asarray(
                [info.get("hero_rule_score", score) for info, score in zip(reward_extra_infos, scores, strict=True)],
                dtype=np.float32,
            )
            dense_scores = np.asarray(
                [info.get("hero_rm_score", score) for info, score in zip(reward_extra_infos, scores, strict=True)],
                dtype=np.float32,
            )
            scores = apply_naive_reward_combine(
                rule_scores=rule_scores,
                rm_scores=dense_scores,
                alpha=alpha,
            ).tolist()
            for i, info in enumerate(reward_extra_infos):
                info["naive_combine_rule_score"] = float(rule_scores[i])
                info["naive_combine_rm_score"] = float(dense_scores[i])
                info["naive_combine_alpha"] = alpha
                info["naive_combine_final_score"] = float(scores[i])
                info.setdefault("acc", float(rule_scores[i]))
        elif self.config.reward.reward_manager.name == "hybrid_eif":
            hybrid_eif_cfg = self.config.reward.get("reward_kwargs", {}).get("hybrid_eif", {})
            phi_scores = np.asarray(scores, dtype=np.float64)
            tau_key = str(hybrid_eif_cfg.get('tau_key', 'tau_llm_value'))
            m_key = str(hybrid_eif_cfg.get('m_key', 'm_llm_value'))

            tau_scores = _load_hybrid_eif_scalar_values(data, key=tau_key)
            m_scores = _load_hybrid_eif_scalar_values(data, key=m_key)
            eif_scores = compute_algorithm1_scores(phi_scores, tau_scores, m_scores)
            scores = eif_scores.tolist()

            for i, info in enumerate(reward_extra_infos):
                info['hybrid_eif_phi'] = float(phi_scores[i])
                info['hybrid_eif_tau'] = float(tau_scores[i])
                info['hybrid_eif_m'] = float(m_scores[i])
                info['hybrid_eif_final_score'] = float(scores[i])
                info.setdefault('acc', float(phi_scores[i]))
        elif self.config.reward.reward_manager.name == "hybrid_eif_online":
            scores = self._apply_hybrid_eif_online(data, reward_extra_infos, scores)

        if self.config.reward.reward_manager.name == "visual":
            # visual reward only has one score for the whole response
            rm_scores = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)
        else:
            prompt_length = data.batch["prompts"].size(1)
            valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=1)
            rm_scores = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            rm_scores[torch.arange(rm_scores.size(0)), valid_response_length - 1] = torch.tensor(
                scores, dtype=torch.float32
            )
        batch = TensorDict({"rm_scores": rm_scores}, batch_size=len(data))

        reward_extra_keys = sorted(set().union(*(info.keys() for info in reward_extra_infos)))
        non_tensor_batch = {}
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info.get(key, None) for info in reward_extra_infos], dtype=object)

        if self.reward_model_manager is not None:
            self.reward_model_manager.sleep()

        return DataProto(
            batch=batch, non_tensor_batch=non_tensor_batch, meta_info={"reward_extra_keys": reward_extra_keys}
        )

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            return await asyncio.gather(*tasks)

        return asyncio.run(run_all())
