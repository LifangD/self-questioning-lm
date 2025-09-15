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

import json
import re
from typing import Dict, Any, Optional
import requests
import logging

logger = logging.getLogger(__name__)


def call_judge_model_api(
    response_text: str,
    meta_info: Dict[str, Any],
    task_type: str = "caption",
    api_url: str = "http://localhost:8000/v1/chat/completions",
    model_name: str = "judge-model"
) -> Dict[str, Any]:
    """
    调用judge模型API进行事实正确性评判
    
    Args:
        response_text: 待评判的回答文本
        meta_info: 包含作者和作品信息的元数据 {"author": "xxx", "artwork": "xxx", "period": "xxx"}
        task_type: 任务类型，"caption" 或 "qa"
        api_url: judge模型的API地址
        model_name: judge模型名称
    
    Returns:
        Dict包含: {"factual_score": float, "reasoning": str, "has_errors": bool}
    """
    
    # 构建prompt模板
    if task_type == "caption":
        prompt_template = """你是一个艺术作品专家。请根据给定的画作信息，评判以下描述是否存在明显的事实错误。

画作信息：
- 作者：{author}
- 作品名：{artwork}
- 时期：{period}

待评判的描述：
{response_text}

请评判这个描述是否有明显的事实错误（如作者错误、时期错误、风格错误等）。
请以JSON格式回复：
{{
    "factual_score": <0.0-1.0的分数，1.0表示完全正确，0.0表示有严重错误>,
    "has_errors": <true/false，是否有明显事实错误>,
    "reasoning": "<简短说明评判理由>"
}}"""
    else:  # qa
        prompt_template = """你是一个艺术作品专家。请根据给定的画作信息，评判以下问答是否存在明显的事实错误。

画作信息：
- 作者：{author}
- 作品名：{artwork}  
- 时期：{period}

待评判的回答：
{response_text}

请评判这个回答是否有明显的事实错误（如作者错误、时期错误、历史事实错误等）。
请以JSON格式回复：
{{
    "factual_score": <0.0-1.0的分数，1.0表示完全正确，0.0表示有严重错误>,
    "has_errors": <true/false，是否有明显事实错误>,
    "reasoning": "<简短说明评判理由>"
}}"""

    # 格式化prompt
    formatted_prompt = prompt_template.format(
        author=meta_info.get("author", "未知"),
        artwork=meta_info.get("artwork", "未知"),
        period=meta_info.get("period", "未知"),
        response_text=response_text
    )
    
    # 构建API请求
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user", 
                "content": formatted_prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # 解析JSON回复
        try:
            parsed_result = json.loads(content)
            return {
                "factual_score": float(parsed_result.get("factual_score", 0.0)),
                "has_errors": bool(parsed_result.get("has_errors", True)), 
                "reasoning": str(parsed_result.get("reasoning", ""))
            }
        except json.JSONDecodeError:
            # 如果无法解析JSON，尝试从文本中提取信息
            logger.warning(f"Failed to parse JSON response: {content}")
            return extract_score_from_text(content)
            
    except Exception as e:
        logger.error(f"Error calling judge model API: {e}")
        return {
            "factual_score": 0.5,  # 默认中性分数
            "has_errors": False,
            "reasoning": f"API调用失败: {str(e)}"
        }


def extract_score_from_text(text: str) -> Dict[str, Any]:
    """从文本中提取评分信息（当JSON解析失败时的fallback）"""
    factual_score = 0.5
    has_errors = False
    reasoning = "无法解析评判结果"
    
    # 尝试提取分数
    score_match = re.search(r'factual_score["\s]*:\s*([0-9.]+)', text)
    if score_match:
        factual_score = float(score_match.group(1))
    
    # 尝试提取错误标志
    error_match = re.search(r'has_errors["\s]*:\s*(true|false)', text, re.IGNORECASE)
    if error_match:
        has_errors = error_match.group(1).lower() == 'true'
    
    # 尝试提取reasoning
    reasoning_match = re.search(r'reasoning["\s]*:\s*["\s]*([^"]+)', text)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    return {
        "factual_score": factual_score,
        "has_errors": has_errors,
        "reasoning": reasoning
    }


def extract_answer_content(response_str: str, task_type: str = "caption") -> str:
    """
    从模型回答中提取核心内容
    
    Args:
        response_str: 模型的完整回答
        task_type: 任务类型
    
    Returns:
        提取的核心内容
    """
    # 简单清理，移除多余的空白和特殊标记
    cleaned = response_str.strip()
    
    # 如果是QA任务，可能需要提取答案部分
    if task_type == "qa":
        # 尝试提取"答案："后的内容
        answer_pattern = r'(?:答案|答|Answer|answer)[：:]\s*(.+?)(?:\n|$)'
        match = re.search(answer_pattern, cleaned, re.IGNORECASE | re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
    
    return cleaned


def compute_score(
    solution_str: str, 
    ground_truth: str, 
    extra_info: Optional[Dict[str, Any]] = None,
    api_url: str = "http://localhost:8000/v1/chat/completions",
    model_name: str = "judge-model"
) -> Dict[str, Any]:
    """
    计算painting caption/qa的评分
    
    Args:
        solution_str: 模型生成的回答
        ground_truth: 标准答案（可能为空，因为是开放性任务）
        extra_info: 包含meta信息和任务类型的字典
            {
                "meta_info": {"author": "xxx", "artwork": "xxx", "period": "xxx"},
                "task_type": "caption" or "qa",
                "api_url": "...",
                "model_name": "..."
            }
    
    Returns:
        评分结果字典
    """
    if extra_info is None:
        extra_info = {}
    
    meta_info = extra_info.get("meta_info", {})
    task_type = extra_info.get("task_type", "caption")
    api_url = extra_info.get("api_url", api_url)
    model_name = extra_info.get("model_name", model_name)
    
    # 提取回答内容
    answer_content = extract_answer_content(solution_str, task_type)
    
    # 调用judge模型进行事实正确性评判
    judge_result = call_judge_model_api(
        response_text=answer_content,
        meta_info=meta_info,
        task_type=task_type,
        api_url=api_url,
        model_name=model_name
    )
    
    # 计算最终分数
    factual_score = judge_result["factual_score"]
    has_errors = judge_result["has_errors"]
    
    # 分层评判逻辑：
    # 1. 如果有明显事实错误，分数较低
    # 2. 如果事实正确，给予基础分数，后续可以用其他维度进一步评判
    if has_errors:
        final_score = factual_score * 0.3  # 有错误时，最多给30%分数
    else:
        final_score = 0.7 + factual_score * 0.3  # 无错误时，基础分70%，事实准确性再加30%
    
    return {
        "score": final_score,
        "factual_score": factual_score,
        "has_errors": has_errors,
        "reasoning": judge_result["reasoning"],
        "task_type": task_type,
        "extracted_content": answer_content
    } 