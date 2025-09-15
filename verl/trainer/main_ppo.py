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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import hydra
import ray
import debugpy


import os
import sys

import subprocess

def get_available_gpus(num_gpus=4, memory_threshold=0.1, utilization_threshold=10):
    """
    使用nvidia-smi命令获取可用的GPU设备
    
    Args:
        num_gpus: 需要的GPU数量
        memory_threshold: 内存使用率阈值 (0.0-1.0)
        utilization_threshold: GPU使用率阈值 (0-100)
    
    Returns:
        List[int]: 可用的GPU ID列表
    """
    try:
        # 使用nvidia-smi获取GPU信息
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            raise Exception("nvidia-smi命令执行失败")
        
        lines = result.stdout.strip().split('\n')
        available_gpus = []
        gpu_info = []
        
        print("GPU状态信息:")
        print("-" * 60)
        print(f"{'ID':<3} {'内存使用':<15} {'GPU使用率':<10} {'状态':<8}")
        print("-" * 60)
        
        for line in lines:
            parts = line.split(', ')
            if len(parts) >= 4:
                gpu_id = int(parts[0])
                memory_used = int(parts[1])
                memory_total = int(parts[2])
                gpu_util = int(parts[3])
                
                memory_ratio = memory_used / memory_total if memory_total > 0 else 1.0
                memory_str = f"{memory_used}MB/{memory_total}MB"
                util_str = f"{gpu_util}%"
                
                gpu_info.append({
                    'id': gpu_id,
                    'memory_ratio': memory_ratio,
                    'gpu_util': gpu_util
                })
                
                # 判断是否可用
                is_available = memory_ratio < memory_threshold and gpu_util < utilization_threshold
                status = "可用" if is_available else "占用"
                
                print(f"{gpu_id:<3} {memory_str:<15} {util_str:<10} {status:<8}")
                
                if is_available:
                    available_gpus.append(gpu_id)
        
        print("-" * 60)
        
        if len(available_gpus) >= num_gpus:
            selected = available_gpus[:num_gpus]
            print(f"选择到足够的可用GPU: {selected}")
        else:
            print(f"可用GPU不足({len(available_gpus)}/{num_gpus})，选择占用率最低的GPU")
            # 按内存使用率和GPU使用率排序，选择最空闲的GPU
            sorted_gpus = sorted(gpu_info, key=lambda x: (x['memory_ratio'], x['gpu_util']))
            selected = [gpu['id'] for gpu in sorted_gpus[:num_gpus]]
            print(f"选择GPU: {selected}")
        
        return selected
        
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
        print(f"使用默认GPU: [0,1,2,3]")
        return list(range(num_gpus))

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    try:
        selected_gpus = get_available_gpus(num_gpus=4)
        gpu_str = ','.join(map(str, selected_gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        print(f"设置 CUDA_VISIBLE_DEVICES = {gpu_str}")
    except Exception as e:
        print(f"自动选择GPU失败: {e}")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        print("使用默认设置: CUDA_VISIBLE_DEVICES = 0,1,2,3")

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config) -> None:
    # Check if Ray is not initialized
    if not ray.is_initialized():

    
        # 设置环境变量，用于子进程调试
        debug_env = {
            "TOKENIZERS_PARALLELISM": "true", 
            "NCCL_DEBUG": "WARN", 
            "VLLM_LOGGING_LEVEL": "WARN", 
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"
        }

        
        ray.init(
            runtime_env={"env_vars": debug_env},
            num_cpus=config.ray_init.num_cpus,
        )


    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    runner = TaskRunner.remote()
    
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)




@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
    
        
      
          
        
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True,max_pixels=28*28*1280)

        # Version validation for vllm.
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")
        


        # Define worker classes based on the actor strategy.
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker classes.
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Add a reference policy worker if KL loss or KL reward is used.
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}), is_train=True)
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        print("finish dataset  creation")
        # Initialize the PPO trainer.
        import debugpy
        # debug_port = int(os.environ.get("DEBUGPY_WORKER_PORT", 5678))
        # debugpy.listen(("0.0.0.0", debug_port))
        # print(f"等待调试器连接到端口 {debug_port}...")
        # debugpy.wait_for_client()
        #debugpy.breakpoint()
        #breakpoint()
        print("调试器已连接！")

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        # If a data generation strategy is specified, use the DynamicGenDataset class
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        dataset_cls = DynamicGenDataset
        print("Using DynamicGenDataset for data generation.")

    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
