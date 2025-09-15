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

from collections import defaultdict
import ipdb
st = ipdb.set_trace
import torch
from collections import Counter
import json
import re
import requests
import logging
import base64
import os

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.workers.reward_manager import register

logger = logging.getLogger(__name__)

@register("majority")
class MajorityRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        
        # ch_painting相关配置
        self.judge_api_url = "http://10.160.199.227:8003/v1/chat/completions"
        self.judge_model_name = "holo-model"

    def _image_to_base64(self, image_path: str) -> str:
        """将本地图片路径转换为base64编码"""
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image file not found: {image_path}")
                return ""
            
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_string = base64.b64encode(image_data).decode('utf-8')
                
                # 根据文件扩展名确定MIME类型
                ext = os.path.splitext(image_path)[1].lower()
                if ext in ['.jpg', '.jpeg']:
                    mime_type = 'image/jpeg'
                elif ext == '.png':
                    mime_type = 'image/png'
                elif ext == '.gif':
                    mime_type = 'image/gif'
                elif ext == '.webp':
                    mime_type = 'image/webp'
                else:
                    mime_type = 'image/jpeg'  # 默认
                
                return f"data:{mime_type};base64,{base64_string}"
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            return ""

    def _call_judge_model_for_pairwise_comparison(self, response_a: str, response_b: str, meta_info: dict, original_question: str = "", image_url: str = ""):
        """调用judge模型进行两两比较评判"""
        
        # 构建pairwise comparison的prompt模板
        prompt_template = """你是一个艺术作品专家。请在三个维度上比较两个回答的质量：

画作信息：
- 作者：{author}
- 作品名：{artwork}
- 时期：{period}

原始问题：
{original_question}

回答A：
{response_a}

回答B：
{response_b}

请在以下三个维度分别比较哪个回答更好：

1. 视觉感知准确性：哪个回答的视觉元素描述（颜色、构图、风格等）更准确？
2. 事实关联准确性：哪个回答的作者、时期、历史背景等事实信息更正确？
3. 回答相关性：哪个回答更完整、准确地回应了原始问题？

请以JSON格式回复：
{{
    "visual_winner": "<A/B/tie>",
    "factual_winner": "<A/B/tie>",
    "relevance_winner": "<A/B/tie>",
    "reasoning": "详细说明每个维度的判断理由，包括为什么选择该winner"
}}"""

        # 格式化prompt
        formatted_prompt = prompt_template.format(
            author=meta_info.get("author", "未知"),
            artwork=meta_info.get("artwork", "未知"),
            period=meta_info.get("period", "未知"),
            original_question=original_question,
            response_a=response_a,
            response_b=response_b
        )
        
        # 构建API请求，支持图片
        messages = [
            {
                "role": "user", 
                "content": []
            }
        ]
        
        # 如果有图片URL，添加图片内容
        if image_url:
            # 如果是本地路径，转换为base64
            if os.path.exists(image_url):
                base64_image = self._image_to_base64(image_url)
                if base64_image:
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image
                        }
                    })
            else:
                # 如果已经是URL或base64格式，直接使用
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                })
        
        messages[0]["content"].append({
            "type": "text",
            "text": formatted_prompt
        })
        
        payload = {
            "model": self.judge_model_name,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 800
        }
        
        try:
            response = requests.post(self.judge_api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # 解析JSON回复
            try:
                parsed_result = json.loads(content)
                return {
                    "visual_winner": str(parsed_result.get("visual_winner", "tie")).lower(),
                    "factual_winner": str(parsed_result.get("factual_winner", "tie")).lower(),
                    "relevance_winner": str(parsed_result.get("relevance_winner", "tie")).lower(),
                    "reasoning": str(parsed_result.get("reasoning", ""))
                }
            except json.JSONDecodeError:
                # 如果无法解析JSON，尝试从文本中提取信息
                logger.warning(f"Failed to parse JSON response: {content}")
                return self._extract_comparison_from_text(content)
                
        except Exception as e:
            logger.error(f"Error calling judge model API: {e}")
            return {
                "visual_winner": "tie",
                "factual_winner": "tie", 
                "relevance_winner": "tie",
                "reasoning": f"API调用失败: {str(e)}"
            }

    def _extract_comparison_from_text(self, text: str):
        """从文本中提取两两比较的评判结果（当JSON解析失败时的fallback）"""
        visual_winner = "tie"
        factual_winner = "tie"
        relevance_winner = "tie"
        reasoning = "无法解析评判结果"
        
        # 尝试提取winner
        visual_match = re.search(r'visual_winner["\s]*:\s*["\s]*([^"]+)', text)
        if visual_match:
            visual_winner = visual_match.group(1).strip().lower()
            
        factual_match = re.search(r'factual_winner["\s]*:\s*["\s]*([^"]+)', text)
        if factual_match:
            factual_winner = factual_match.group(1).strip().lower()
            
        relevance_match = re.search(r'relevance_winner["\s]*:\s*["\s]*([^"]+)', text)
        if relevance_match:
            relevance_winner = relevance_match.group(1).strip().lower()
        
        # 尝试提取reasoning
        reasoning_match = re.search(r'reasoning["\s]*:\s*["\s]*([^"]+)', text)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        return {
            "visual_winner": visual_winner,
            "factual_winner": factual_winner,
            "relevance_winner": relevance_winner,
            "reasoning": reasoning
        }

    def _extract_answer(self, response_str, data_source: str):
        """提取数学问题的答案"""
        try:
            if "gsm8k" in data_source:
                from verl.utils.reward_score import gsm8k
                return gsm8k.extract_solution(response_str)
            elif "math" in data_source.lower():
                from verl.utils.reward_score import math
                string_in_last_boxed = math.last_boxed_only_string(response_str)
                if string_in_last_boxed is not None:
                    return math.remove_boxed(string_in_last_boxed)
                else:
                    return None
            elif "multiply" in data_source.lower():
                from verl.utils.reward_score import multiply
                return multiply.extract_solution(response_str)
            else:
                # 默认尝试提取数字答案
                pattern = r"(?:答案|answer|result)(?:\s*(?:is|为|=)\s*)([-+]?\d*\.?\d+)"
                match = re.search(pattern, response_str, re.IGNORECASE)
                if match:
                    return match.group(1)
                return None
        except Exception as e:
            print(f"Error in _extract_answer: {e}")
            return None

    def _perform_pairwise_comparisons(self, responses_info):
        """对同一uid的所有回答进行两两比较"""
        comparisons = {}
        
        for uid, info in responses_info.items():
            responses = info['responses']
            if len(responses) < 2:
                # 少于2个回答，无法比较
                comparisons[uid] = []
                continue
                
            uid_comparisons = []
            # 进行所有可能的两两比较
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    response_a = responses[i]
                    response_b = responses[j]
                    
                    # 调用judge模型进行比较
                    comparison_result = self._call_judge_model_for_pairwise_comparison(
                        response_a['text'], 
                        response_b['text'],
                        response_a['meta_info'],
                        response_a['original_question'],
                        response_a['image_url']
                    )
                    
                    uid_comparisons.append({
                        'index_a': response_a['index'],
                        'index_b': response_b['index'], 
                        'visual_winner': comparison_result['visual_winner'],
                        'factual_winner': comparison_result['factual_winner'],
                        'relevance_winner': comparison_result['relevance_winner'],
                        'reasoning': comparison_result['reasoning']
                    })
            
            comparisons[uid] = uid_comparisons
            
        return comparisons
    
    def _compute_win_statistics(self, comparisons, responses_info):
        """计算每个回答在各个维度的胜负统计"""
        win_stats = {}
        for uid, info in responses_info.items():
            responses = info['responses']
            n_responses = len(responses)
            
            # 初始化胜负统计
            win_stats[uid] = {}
            for response in responses:
                index = response['index']
                win_stats[uid][index] = {
                    'visual_wins': 0,
                    'factual_wins': 0,
                    'relevance_wins': 0,
                    'total_comparisons': 0
                }
            
            # 统计胜负
            for comparison in comparisons[uid]:
                index_a = comparison['index_a']
                index_b = comparison['index_b']
                
                # 更新比较次数
                win_stats[uid][index_a]['total_comparisons'] += 1
                win_stats[uid][index_b]['total_comparisons'] += 1
                
                # 统计各维度胜负
                for dimension in ['visual', 'factual', 'relevance']:
                    winner = comparison[f'{dimension}_winner']
                    if winner == 'a':
                        win_stats[uid][index_a][f'{dimension}_wins'] += 1
                    elif winner == 'b':
                        win_stats[uid][index_b][f'{dimension}_wins'] += 1
                    # tie的情况不给任何一方加分
            
            # 计算胜率
            for index in win_stats[uid]:
                stats = win_stats[uid][index]
                total = stats['total_comparisons']
                if total > 0:
                    stats['visual_win_rate'] = stats['visual_wins'] / total
                    stats['factual_win_rate'] = stats['factual_wins'] / total
                    stats['relevance_win_rate'] = stats['relevance_wins'] / total
                    stats['overall_win_rate'] = (stats['visual_wins'] + stats['factual_wins'] + stats['relevance_wins']) / (3 * total)
                else:
                    stats['visual_win_rate'] = 0
                    stats['factual_win_rate'] = 0
                    stats['relevance_win_rate'] = 0
                    stats['overall_win_rate'] = 0
            
        return win_stats
    
    def _detect_controversy_and_assign_rewards(self, win_stats, responses_info):
        """检测争议性并分配奖励"""
        solver_rewards = {}
        proposer_rewards = {}
        
        for uid, stats_dict in win_stats.items():
            if not stats_dict:
                continue
                
            # 收集所有回答的整体胜率
            overall_win_rates = [stats['overall_win_rate'] for stats in stats_dict.values()]
            
            if len(overall_win_rates) < 2:
                # 少于2个回答，不给奖励
                for index in stats_dict:
                    solver_rewards[index] = 0.0
                    proposer_rewards[uid] = 0.0 # proposer reward for this uid
                continue
                
            max_win_rate = max(overall_win_rates)
            min_win_rate = min(overall_win_rates)
            win_rate_variance = max_win_rate - min_win_rate
            
            # 争议性检测：胜率分布要有适当的分散度
            is_controversial = 0.2 < win_rate_variance < 0.8
            has_sufficient_samples = len(overall_win_rates) >= 3
            has_clear_winners = max_win_rate > 0.5
            
            if is_controversial and has_sufficient_samples and has_clear_winners:
                # 分配奖励：胜率高的答案获得更多奖励
                for index, stats in stats_dict.items():
                    win_rate = stats['overall_win_rate']
                    if win_rate >= 0.7:
                        solver_rewards[index] = 1.0  # 高质量答案
                    elif win_rate >= 0.4:
                        solver_rewards[index] = 0.5  # 中等质量答案
                    else:
                        solver_rewards[index] = 0.0  # 低质量答案
            else:
                # 不符合争议性条件，不给奖励
                for index in stats_dict:
                    solver_rewards[index] = 0.0
                
            # 计算proposer奖励
            proposer_rewards[uid] = self._compute_proposer_reward(uid, stats_dict, self._perform_pairwise_comparisons({uid: responses_info[uid]})[uid])
            
        return solver_rewards, proposer_rewards

    def _compute_proposer_reward(self, uid, win_stats, comparisons):
        """计算proposer（问题生成者）的奖励"""
        if not win_stats or len(win_stats) < 2:
            return 0.0
            
        # 收集所有回答的整体胜率
        overall_win_rates = [stats['overall_win_rate'] for stats in win_stats.values()]
        
        # 计算分歧度 (variance in win rates)
        max_win_rate = max(overall_win_rates)
        min_win_rate = min(overall_win_rates)
        win_rate_spread = max_win_rate - min_win_rate
        
        # 计算多样性 (基于比较结果的分布)
        total_comparisons = len(comparisons)
        tie_count = sum(1 for comp in comparisons 
                       if comp['visual_winner'] == 'tie' and 
                          comp['factual_winner'] == 'tie' and 
                          comp['relevance_winner'] == 'tie')
        diversity_score = 1.0 - (tie_count / total_comparisons if total_comparisons > 0 else 1.0)
        
        # 计算维度间争议 (不同维度winner不一致)
        cross_dimensional_disagreements = 0
        for comp in comparisons:
            winners = [comp['visual_winner'], comp['factual_winner'], comp['relevance_winner']]
            if len(set(winners)) > 1:  # 如果三个维度的winner不全相同
                cross_dimensional_disagreements += 1
        
        cross_dim_controversy = cross_dimensional_disagreements / total_comparisons if total_comparisons > 0 else 0
        
        # 好问题的特征权重
        # 1. 适中的胜率分散度 (0.2-0.8)
        spread_score = 1.0 if 0.2 <= win_rate_spread <= 0.8 else 0.0
        
        # 2. 足够的多样性 (避免所有回答都一样)
        diversity_weight = min(diversity_score, 1.0)
        
        # 3. 维度间争议 (说明问题有复杂性)
        controversy_weight = min(cross_dim_controversy, 1.0)
        
        # 4. 有明显的好坏区分 (最高胜率不能太低)
        has_quality_distinction = 1.0 if max_win_rate >= 0.4 else 0.0
        
        # 综合评分
        proposer_reward = (spread_score * 0.4 + 
                          diversity_weight * 0.3 + 
                          controversy_weight * 0.2 + 
                          has_quality_distinction * 0.1)
        
        return proposer_reward

    def __call__(self, data: DataProto, return_dict=False):
        """使用Pairwise Comparison + Multi-dimensional评估的reward manager"""
        # Initialize the reward tensor
        reward_tensor = torch.zeros(len(data), data.batch["attention_mask"].shape[-1])

        # Initialize the reward extra info
        reward_extra_info = defaultdict(list)
        
        # 第一轮：收集同一uid的所有响应
        responses_info = {}
        for i in range(len(data)):
            data_item = data[i]
            uid = data_item.non_tensor_batch.get("uid", [f"default_uid_{i}"])[0]
            response_str = self.tokenizer.decode(data_item.batch["attention_mask"], skip_special_tokens=True)
            
            if uid not in responses_info:
                responses_info[uid] = {
                    'responses': [],
                    'indices': [],
                    'original_question': data_item.non_tensor_batch.get("original_question", [""])[0],
                    'image_url': data_item.non_tensor_batch.get("image_url", [""])[0]
                }
            
            responses_info[uid]['responses'].append(response_str)
            responses_info[uid]['indices'].append(i)
        
        # 第二轮：进行两两比较并计算奖励
        comparisons = self._perform_pairwise_comparisons(responses_info)
        win_statistics = self._compute_win_statistics(comparisons, responses_info)
        solver_rewards, proposer_rewards = self._detect_controversy_and_assign_rewards(win_statistics, responses_info)
        
        # 存储统计信息到reward_extra_info
        for uid, proposer_reward in proposer_rewards.items():
            print(f"UID {uid}: proposer_reward={proposer_reward:.3f}")
            # 为每个样本添加proposer reward
            for idx in responses_info[uid]['indices']:
                reward_extra_info[f"proposer_reward"].append(proposer_reward)
                reward_extra_info[f"uid"].append(uid)
                
        for uid, solver_reward_dict in solver_rewards.items():
            for idx, solver_reward in zip(responses_info[uid]['indices'], solver_reward_dict.values()):
                print(f"Sample {idx}: uid={uid}, "
                      f"reward={solver_reward:.1f}")
            
            # 存储proposer reward到extra_info中
            reward_extra_info[f"proposer_reward_{uid}"] = proposer_reward
        
        # 第三轮：将奖励分配到reward_tensor
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            
            # 获取该index的reward
            reward = solver_rewards.get(i, 0.0)
            reward_tensor[i, valid_response_length - 1] = reward
            
            print(f"Index {i}: reward = {reward}")
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),  # 转换为普通dict
            }
        else:
            return reward_tensor


