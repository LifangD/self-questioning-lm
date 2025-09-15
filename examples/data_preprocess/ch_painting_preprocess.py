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
将国画JSONL数据预处理为parquet格式
"""
import sys
sys.path.append("/data/dlf/code/self-questioning-lm")
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import datasets
import pandas as pd

from verl.utils.hdfs_io import copy, makedirs
from PIL import Image

def parse_grouding_data(raw_output=""):
    import re
    #print(raw_output)
    #raw_output = raw_output.split("[对原答案进行润色和引用标记添加后的完整答案]")[-1]
    question_objects, answer = raw_output.split("**答案：**")
    question, objects = question_objects.split("**物象/技法列表：**")
    question = question.split("**原问题：**")[-1].strip()
    question = "先检测物象/技法，再回答问题:" + question 

    # 保留ref标志
    object_lines = re.findall(r'(ref\d+):\s*([^\<]+)<([\d,]+)>', objects)
    formatted_objects = []
    for ref, name, coords in object_lines:
        name = name.strip()
        if '"' in name:
            name = name.replace('"','')
        coords = [int(x) for x in coords.split(',')]
        box = f'({coords[0]},{coords[1]}),({coords[2]},{coords[3]})'
        obj_str = f'<|object_ref_start|>{name}<|object_ref_end|>'
        box_str = f'<|box_start|>{box}<|box_end|>'
        formatted_objects.append(f'{ref}:{obj_str}{box_str}')
    objects_str = '\n'.join(formatted_objects)

    answer = "<objects>"+  objects_str + "</objects>" + '\n' + answer.strip()

    return question, answer

def load_jsonl_data(data_path: str) -> List[Dict]:
    """加载JSONL格式的国画数据"""
    processed_data = []
    
 
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            
            item = json.loads(line)
            
            # 处理图像路径
            image_path = item.get("img", "")
            if not image_path or not Path(image_path).exists():
                print(f"⚠️  [DATA] Image not found: {image_path} (line {line_num + 1})")
                continue
            
            # 提取作品信息
            work = item["meta"].get("work", {})
            author = item["meta"].get("author", {})
            author_name = author.get("name", {})
            res= parse_grouding_data(item.get("raw_model_output",""))
            
            question,answer=res 
            # 构建元数据
            # grounding = item["current_prompt"].split("**画作尺寸")[0].split("</object>")[0].split("<object>")[1].strip()
            # if "/n" in grounding:
            #     grounding = grounding.replace("/n", "")
            
            # 构建问题和答案 - 使用与data_loader.py相同的格式
            # 构建上下文信息
            
            context_parts = []
            for key, value in {
                "title": work.get("title", ""),
                "artist": author_name.get("cn", "未知"),
                "dynasty": work.get("dynasty", ""),
                "media_type": work.get("media_type", ""),
                "material_type": work.get("material_type", ""),
                "style": ", ".join([tag.replace("类型/", "") for tag in work.get("tags", []) if "类型" in tag]),
                "subjects": ", ".join([tag.replace("主题/", "") for tag in work.get("tags", []) if "主题" in tag]),
                #"grounding": grounding
            }.items():
                if key == "description":
                    continue
                context_parts.append(f"{key}: {value}")
            
            context = " | ".join(context_parts) if context_parts else "未知作品"
            
            
#             # Proposer阶段：生成关于国画的专业问题
#             proposer_prompt = f"""
# <image>
# 你是一名“用户模拟器”，任务是根据下面这幅国画及其作品信息，模拟真实用户可能提出的各类赏析问题。  
# 作品信息：{context}

# 模拟要点  
# 1. 用户身份随机：可能是刚入门，也可能像行家。  
# 2. 问题形式自然：  
#    • 可在末尾加“/think”表示想要推理过程；不加则默认不思考。  
#    • 可直接说“先帮我看看画里都有什么”或“指出所有印章”表示需要检测；若无此类措辞则默认不检测。  
#    • 可能指定格式（如“用 JSON 告诉我”），也可能一句话都不提格式。  
# 3. 语气口语化，不暴露内部规则。

# 请仅输出一条用户口吻的问题。"""

            proposer_prompt = f"""<image>提出一个关于国画的问题，作品信息：{context}
            """

            
            processed_item = {
                "image_path": str(image_path),
                "question": question,
                "answer": answer,
                "proposer_prompt": proposer_prompt,
                "metadata": {
                    "title": work.get("title", ""),
                    "artist": author_name.get("cn", "未知"),
                    "dynasty": work.get("dynasty", ""),
                    "media_type": work.get("media_type", ""),
                    "material_type": work.get("material_type", ""),
                    "description": item.get("description", ""),
                    "style": ", ".join([tag.replace("类型/", "") for tag in work.get("tags", []) if "类型" in tag]),
                    "subjects": ", ".join([tag.replace("主题/", "") for tag in work.get("tags", []) if "主题" in tag]),
                    #"grounding": grounding
                }
            }
            
            processed_data.append(processed_item)
                

    
    print(f"✅ [DATA] Loaded {len(processed_data)} valid samples from {data_path}")
    return processed_data
    



def process_painting_data(raw_data: List[Dict], split: str, data_source: str = "ch_painting") -> datasets.Dataset:
    """将原始数据处理为VERL格式，适配Proposer-Solver架构"""
    
    def process_fn(example, idx,split):
        proposer_prompt = example["proposer_prompt"]  # Proposer的输入prompt
        image_path = example["image_path"]
        metadata = example["metadata"]
        question = example["question"]
        answer = example["answer"]

        image = image_path
        # if image.mode != "RGBA":
        #     image = image.convert("RGBA")
        if split == "train":
            proposer_prompt = proposer_prompt 
        else:
            proposer_prompt = "<image>" + question
        
        print(proposer_prompt)
        
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user", 
                    "content": proposer_prompt,
                }
            ],
            "images": [{"image_url": image,"min_pixels": 3136,"max_pixels": 1003520}],  # 图像路径列表
            "ability": "proposer_solver_painting",  # 表明这是Proposer-Solver任务
            "reward_model": {"style": "model"},  # 使用LLM-as-Judge评估最终答案
            "extra_info": {
                "split": split,
                "index": idx,
                "task_type": "proposer",  # 标记这是Proposer阶段
                "proposer_prompt": proposer_prompt,
                "metadata": metadata,
                "image_path": image_path,
                "qustion": question,
                "answer": answer,
                # 预期的工作流程：
                # 1. Proposer根据prompt生成问题
                # 2. Solver回答生成的问题  
                # 3. LLM-as-Judge评估Solver的回答质量
            },
        }
        
        return data
    
    # 创建数据集
    df = pd.DataFrame(raw_data)
    dataset = datasets.Dataset.from_pandas(df)
    dataset = dataset.map(function=process_fn, with_indices=True, num_proc=16,fn_kwargs={"split": split})
    
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", default="/data/dlf/code/VLM-ChinesePainting-Alignment/Data/SpecificQA/sub_8k_data/sample8k_qa_ground.jsonl")
    parser.add_argument("--output_dir", default="/data/dlf/code/self-questioning-lm/selfplay_data/ch_painting")
    parser.add_argument("--hdfs_dir", default=None, help="HDFS输出目录（可选）")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--data_source", default="ch_painting", help="数据源名称")
    parser.add_argument("--include_reference", action="store_true", help="是否包含参考答案（用于调试，不用于reward计算）")

    args = parser.parse_args()

    # 加载原始数据
    print(f"🔄 [PREPROCESS] Loading data from {args.input_jsonl}")
    raw_data = load_jsonl_data(args.input_jsonl)
    
    if not raw_data:
        print("❌ [PREPROCESS] No valid data found, exiting...")
        exit(1)
    
    # 分割训练集和测试集
    total_samples = len(raw_data)
    train_size = int(total_samples * args.split_ratio)
    
    train_data = raw_data[:32]
    test_data = raw_data[32:40]
    
    print(f"📊 [PREPROCESS] Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    # 处理数据
    print("🔄 [PREPROCESS] Processing training data...")
    train_dataset = process_painting_data(train_data, "train", args.data_source)
    
    print("🔄 [PREPROCESS] Processing test data...")
    test_dataset = process_painting_data(test_data, "test", args.data_source)
    
    # 保存为parquet格式
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_parquet_path = os.path.join(args.output_dir, "train.parquet")
    test_parquet_path = os.path.join(args.output_dir, "test.parquet")
    
    print(f"💾 [PREPROCESS] Saving train dataset to {train_parquet_path}")
    train_dataset.to_parquet(train_parquet_path)
    
    print(f"💾 [PREPROCESS] Saving test dataset to {test_parquet_path}")
    test_dataset.to_parquet(test_parquet_path)
    
    # 上传到HDFS（如果指定）
    if args.hdfs_dir is not None:
        print(f"☁️  [PREPROCESS] Uploading to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=args.output_dir, dst=args.hdfs_dir)
    
    print("✅ [PREPROCESS] Data preprocessing completed!")
    print(f"📁 Train dataset: {train_parquet_path} ({len(train_dataset)} samples)")
    print(f"📁 Test dataset: {test_parquet_path} ({len(test_dataset)} samples)") 