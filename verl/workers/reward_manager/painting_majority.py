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

from collections import defaultdict, Counter
import ipdb
st = ipdb.set_trace
import torch
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

logger = logging.getLogger(__name__)


@register("painting_majority")
class PaintingMajorityRewardManager:
    """The reward manager for painting caption/qa tasks with hierarchical evaluation."""

    def __init__(
        self, 
        tokenizer, 
        num_examine, 
        compute_score=None, 
        reward_fn_key="data_source",
        judge_api_url: str = "http://localhost:8000/v1/chat/completions",
        judge_model_name: str = "judge-model",
        quality_threshold: float = 0.7,
        similarity_threshold: float = 0.8,
        min_majority_ratio: float = 0.3,
        max_majority_ratio: float = 0.8
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.judge_api_url = judge_api_url
        self.judge_model_name = judge_model_name
        self.quality_threshold = quality_threshold
        self.similarity_threshold = similarity_threshold
        self.min_majority_ratio = min_majority_ratio  # 对应原来的 1 < majority_num
        self.max_majority_ratio = max_majority_ratio  # 对应原来的 majority_num < n

    def __call__(self, data: DataProto, return_dict=False):
        """Hierarchical evaluation for painting caption/qa tasks"""

        # If there is rm score, we directly return rm score
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # Group responses by uid (same question/image)
        id2responses = defaultdict(list)
        
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            uid = data_item.non_tensor_batch["uid"]
            data_source = data_item.non_tensor_batch["data_source"]
            
            # 提取meta信息
            meta_info = data_item.non_tensor_batch.get("meta_info", {})
            task_type = "caption" if "caption" in data_source else "qa"
            
            id2responses[uid].append({
                "index": i,
                "response": response_str,
                "data_source": data_source,
                "meta_info": meta_info,
                "task_type": task_type
            })

        # Process each group
        id2rewards = {}
        
        for uid, responses in id2responses.items():
            if len(responses) == 0:
                continue
                
            # Step 1: Factual correctness filtering using judge model
            factual_correct_responses = []
            all_scores = []
            
            for resp_info in responses:
                try:
                    # 构建extra_info for scoring
                    extra_info = {
                        "meta_info": resp_info["meta_info"],
                        "task_type": resp_info["task_type"],
                        "api_url": self.judge_api_url,
                        "model_name": self.judge_model_name
                    }
                    
                    score_result = self.compute_score(
                        data_source=resp_info["data_source"],
                        solution_str=resp_info["response"],
                        ground_truth="",  # 开放性任务通常没有标准答案
                        extra_info=extra_info
                    )
                    
                    if isinstance(score_result, dict):
                        factual_score = score_result.get("factual_score", 0.5)
                        has_errors = score_result.get("has_errors", True)
                        overall_score = score_result.get("score", 0.5)
                    else:
                        factual_score = float(score_result)
                        has_errors = factual_score < 0.5
                        overall_score = factual_score
                    
                    resp_info["factual_score"] = factual_score
                    resp_info["has_errors"] = has_errors
                    resp_info["overall_score"] = overall_score
                    all_scores.append(overall_score)
                    
                    # 只有事实正确的回答才进入候选
                    if not has_errors and factual_score >= self.quality_threshold:
                        factual_correct_responses.append(resp_info)
                        
                except Exception as e:
                    logger.warning(f"Error computing score for response: {e}")
                    resp_info["factual_score"] = 0.0
                    resp_info["has_errors"] = True
                    resp_info["overall_score"] = 0.0
                    all_scores.append(0.0)

            # Step 2: Apply majority logic similar to original framework
            n_total = len(responses)
            n_factual_correct = len(factual_correct_responses)
            
            # 计算majority ratio (类似原来的 majority_num / n)
            majority_ratio = n_factual_correct / n_total if n_total > 0 else 0
            
            # 判断是否符合"有争议但倾向正确"的条件
            should_give_reward = (
                self.min_majority_ratio < majority_ratio < self.max_majority_ratio and
                n_factual_correct > 1 and  # 至少有2个正确答案
                n_total > n_factual_correct  # 存在一些不够好的答案（有争议）
            )
            
            if should_give_reward:
                # Step 3: 在事实正确的回答中选择质量最高的作为beat_one
                if factual_correct_responses:
                    best_response = max(factual_correct_responses, key=lambda x: x["overall_score"])
                    beat_one_score = best_response["overall_score"]
                    
                    # 给与beat_one相近质量的回答奖励
                    quality_threshold_dynamic = beat_one_score * 0.9  # 90%的质量阈值
                    
                    rewards = []
                    for resp_info in responses:
                        if (resp_info in factual_correct_responses and 
                            resp_info["overall_score"] >= quality_threshold_dynamic):
                            rewards.append(1.0)
                        else:
                            rewards.append(0.0)
                else:
                    # 没有事实正确的回答，都不给奖励
                    rewards = [0.0] * n_total
            else:
                # 不符合majority条件，都不给奖励
                rewards = [0.0] * n_total
            
            # 存储每个回答的奖励
            id2rewards[uid] = {}
            for i, resp_info in enumerate(responses):
                id2rewards[uid][resp_info["index"]] = rewards[i]

        # Assign rewards to tensor
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            
            uid = data_item.non_tensor_batch["uid"]
            if uid in id2rewards and i in id2rewards[uid]:
                reward = id2rewards[uid][i]
                reward_tensor[i, valid_response_length - 1] = reward

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor 