#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional, Tuple, Dict, Any, List, Callable


def setup_distributed(backend: str = 'nccl',
                      init_method: Optional[str] = None,
                      world_size: Optional[int] = None,
                      rank: Optional[int] = None) -> Tuple[int, int]:
    """
    设置分布式环境

    参数:
        backend: 分布式后端 ('nccl', 'gloo')
        init_method: 初始化方法 (默认为环境变量)
        world_size: 总进程数 (默认为环境变量)
        rank: 当前进程的排名 (默认为环境变量)

    返回:
        (local_rank, world_size)
    """
    # 如果尚未初始化
    if not dist.is_initialized():
        # 从环境变量获取参数
        if world_size is None:
            if "WORLD_SIZE" in os.environ:
                world_size = int(os.environ["WORLD_SIZE"])
            else:
                world_size = 1

        if rank is None:
            if "RANK" in os.environ:
                rank = int(os.environ["RANK"])
            else:
                rank = 0

        # 获取本地进程排名
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
        else:
            local_rank = rank % torch.cuda.device_count()

        # 设置设备
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        # 初始化方法
        if init_method is None:
            if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
                init_method = f"env://"
            else:
                # 默认使用文件共享
                init_method = f"file:///tmp/torch_distributed_init"

        # 初始化进程组
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank
        )

        # 同步所有进程
        dist.barrier()

        return local_rank, world_size
    else:
        # 如果已经初始化，返回当前状态
        return torch.cuda.current_device(), dist.get_world_size()


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def spawn_processes(fn: Callable,
                    world_size: int,
                    backend: str = 'nccl',
                    **kwargs):
    """
    生成多进程

    参数:
        fn: 要运行的函数
        world_size: 总进程数
        backend: 分布式后端
        **kwargs: 传递给函数的参数
    """
    # 在多GPU环境中启动多个进程
    mp.spawn(
        fn,
        args=(world_size, backend, kwargs),
        nprocs=world_size,
        join=True
    )


def is_main_process() -> bool:
    """检查是否为主进程"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def reduce_dict(input_dict: Dict[str, Any],
                average: bool = True) -> Dict[str, Any]:
    """
    在所有进程间减少字典值

    参数:
        input_dict: 输入字典
        average: 是否取平均值

    返回:
        减少后的字典
    """
    if not dist.is_initialized():
        return input_dict

    world_size = dist.get_world_size()
    if world_size == 1:
        return input_dict

    # 创建新字典以避免修改输入
    output_dict = {}

    for k, v in input_dict.items():
        # 跳过非张量值
        if not isinstance(v, torch.Tensor):
            output_dict[k] = v
            continue

        # 确保张量在CUDA上
        if v.device.type != "cuda":
            v = v.cuda()

        # 求和
        dist.all_reduce(v, op=dist.ReduceOp.SUM)

        # 如果需要，取平均值
        if average:
            v = v / world_size

        output_dict[k] = v

    return output_dict


def all_gather(data):
    """
    收集所有进程的数据

    参数:
        data: 要收集的数据

    返回:
        所有进程的数据列表
    """
    if not dist.is_initialized():
        return [data]

    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]

    # 序列化数据
    buffer = torch.ByteTensor(torch.ByteStorage.from_buffer(pickle.dumps(data)))

    # 获取每个进程的buffer大小
    local_size = torch.tensor([buffer.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # 填充buffer到最大大小
    if local_size != max_size:
        padding = torch.zeros(max_size - local_size, dtype=torch.uint8, device="cuda")
        buffer = torch.cat((buffer, padding), dim=0)

    # 收集所有buffer
    tensor_list = [torch.zeros(max_size, dtype=torch.uint8, device="cuda") for _ in range(world_size)]
    dist.all_gather(tensor_list, buffer)

    # 反序列化
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        data_list.append(pickle.loads(tensor.cpu().numpy().tobytes()[:size]))

    return data_list


def broadcast_tensors(tensors: List[torch.Tensor],
                      src: int = 0) -> List[torch.Tensor]:
    """
    从源进程广播张量到所有进程

    参数:
        tensors: 张量列表
        src: 源进程的排名

    返回:
        广播后的张量列表
    """
    if not dist.is_initialized():
        return tensors

    world_size = dist.get_world_size()
    if world_size == 1:
        return tensors

    # 广播每个张量
    for tensor in tensors:
        dist.broadcast(tensor, src=src)

    return tensors


def get_data_parallel_model(model: torch.nn.Module,
                            use_ddp: bool = True,
                            device_ids: Optional[List[int]] = None,
                            **kwargs) -> torch.nn.Module:
    """
    获取数据并行模型

    参数:
        model: 原始模型
        use_ddp: 是否使用DistributedDataParallel
        device_ids: 设备ID列表
        **kwargs: 其他参数

    返回:
        并行化的模型
    """
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        return model

    # 如果未指定设备ID，使用所有可用的GPU
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    # 如果只有一个GPU，直接返回模型
    if len(device_ids) == 1:
        return model.to(f"cuda:{device_ids[0]}")

    # 使用DistributedDataParallel
    if use_ddp and dist.is_initialized():
        # 确保模型在正确的设备上
        model = model.to(torch.cuda.current_device())

        # 如果模型包含BatchNorm层，启用同步BN
        if kwargs.get('sync_bn', False):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # 封装为DDP模型
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),
            find_unused_parameters=kwargs.get('find_unused_parameters', False),
            gradient_as_bucket_view=kwargs.get('gradient_as_bucket_view', False)
        )
    else:
        # 使用DataParallel
        model = torch.nn.DataParallel(
            model,
            device_ids=device_ids,
            output_device=device_ids[0]
        )
        model = model.to(f"cuda:{device_ids[0]}")

    return model


def get_fsdp_model(model: torch.nn.Module,
                   sharding_strategy: str = 'full',
                   cpu_offload: bool = False,
                   auto_wrap_policy: Optional[Callable] = None,
                   **kwargs) -> torch.nn.Module:
    """
    获取完全分片数据并行(FSDP)模型 (需要PyTorch 1.12+)

    参数:
        model: 原始模型
        sharding_strategy: 分片策略 ('full', 'shard_grad_op', 'no_shard')
        cpu_offload: 是否将参数卸载到CPU
        auto_wrap_policy: 自动包装策略
        **kwargs: 其他参数

    返回:
        FSDP封装的模型
    """
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        return model

    # 检查是否支持FSDP
    if not hasattr(torch.distributed, 'FullyShardedDataParallel'):
        print("警告: 当前PyTorch版本不支持FSDP，回退到DDP")
        return get_data_parallel_model(model, True)

    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            BackwardPrefetch,
            ShardingStrategy,
            CPUOffload
        )
        from torch.distributed.fsdp.wrap import (
            transformer_auto_wrap_policy,
            size_based_auto_wrap_policy,
            enable_wrap,
            wrap
        )

        # 确保模型在正确的设备上
        model = model.to(torch.cuda.current_device())

        # 转换同步BN
        if kwargs.get('sync_bn', False):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # 配置分片策略
        if sharding_strategy == 'full':
            sharding_strategy = ShardingStrategy.FULL_SHARD
        elif sharding_strategy == 'shard_grad_op':
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        elif sharding_strategy == 'no_shard':
            sharding_strategy = ShardingStrategy.NO_SHARD
        else:
            raise ValueError(f"不支持的分片策略: {sharding_strategy}")

        # CPU卸载配置
        cpu_offload_config = CPUOffload(offload_params=cpu_offload)

        # 混合精度配置
        if kwargs.get('mixed_precision', False):
            if kwargs.get('dtype', 'float16') == 'float16':
                mp_dtype = torch.float16
            elif kwargs.get('dtype', 'float16') == 'bfloat16':
                mp_dtype = torch.bfloat16
            else:
                mp_dtype = torch.float32

            mixed_precision_config = MixedPrecision(
                param_dtype=mp_dtype,
                reduce_dtype=mp_dtype,
                buffer_dtype=mp_dtype
            )
        else:
            mixed_precision_config = None

        # 自动包装策略
        if auto_wrap_policy is None:
            if 'transformer_layer_cls' in kwargs:
                # 针对Transformer的包装策略
                transformer_cls = kwargs['transformer_layer_cls']
                auto_wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=transformer_cls
                )
            else:
                # 基于大小的包装策略
                auto_wrap_policy = functools.partial(
                    size_based_auto_wrap_policy,
                    min_num_params=kwargs.get('min_num_params', 100_000)
                )

        # 创建FSDP模型
        model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload_config,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_config,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=torch.cuda.current_device()
        )

        return model

    except ImportError as e:
        print(f"FSDP导入错误: {e}，回退到DDP")
        return get_data_parallel_model(model, True)
    except Exception as e:
        print(f"FSDP初始化错误: {e}，回退到DDP")
        return get_data_parallel_model(model, True)


def get_distributed_sampler(dataset,
                            shuffle: bool = True,
                            drop_last: bool = False,
                            seed: int = 42) -> torch.utils.data.Sampler:
    """
    获取分布式采样器

    参数:
        dataset: 数据集
        shuffle: 是否打乱
        drop_last: 是否丢弃最后不完整的批次
        seed: 随机种子

    返回:
        分布式采样器
    """
    if not dist.is_initialized():
        if shuffle:
            return torch.utils.data.RandomSampler(dataset)
        else:
            return torch.utils.data.SequentialSampler(dataset)

    return torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed
    )


def save_on_main(state: Dict[str, Any], filename: str):
    """
    仅在主进程上保存模型

    参数:
        state: 状态字典
        filename: 文件名
    """
    if is_main_process():
        torch.save(state, filename)

    # 确保所有进程等待主进程完成保存
    if dist.is_initialized():
        dist.barrier()


def train_distributed(rank: int,
                      world_size: int,
                      backend: str,
                      train_fn: Callable,
                      args: Dict[str, Any]):
    """
    分布式训练函数

    参数:
        rank: 当前进程的排名
        world_size: 总进程数
        backend: 分布式后端
        train_fn: 训练函数
        args: 传递给训练函数的参数
    """
    # 设置进程组
    setup_distributed(backend=backend, rank=rank, world_size=world_size)

    # 设置随机种子
    seed = args.get('seed', 42)
    torch.manual_seed(seed + rank)

    try:
        # 运行训练函数
        train_fn(rank=rank, **args)
    finally:
        # 清理分布式环境
        cleanup_distributed()


# 辅助函数
def create_distributed_optimizer(optimizer: torch.optim.Optimizer,
                                 model: torch.nn.Module,
                                 use_fp16: bool = False,
                                 fp16_scale_window: int = 1000) -> torch.optim.Optimizer:
    """
    创建分布式优化器包装

    参数:
        optimizer: 原始优化器
        model: 模型
        use_fp16: 是否使用混合精度
        fp16_scale_window: FP16缩放窗口

    返回:
        包装后的优化器
    """
    if not use_fp16:
        return optimizer

    # 检查是否支持Apex
    try:
        from apex import amp

        # 混合精度初始化
        model, optimizer = amp.initialize(
            model,
            optimizer,
            opt_level="O1",
            loss_scale="dynamic"
        )

        return optimizer
    except ImportError:
        # 使用PyTorch原生混合精度
        print("Apex不可用，使用PyTorch原生混合精度")
        return optimizer


def create_zero_optimizer(optimizer: torch.optim.Optimizer,
                          model: torch.nn.Module,
                          stage: int = 2,
                          **kwargs) -> torch.optim.Optimizer:
    """
    创建ZeRO优化器

    参数:
        optimizer: 原始优化器
        model: 模型
        stage: ZeRO阶段
        **kwargs: 其他参数

    返回:
        ZeRO优化器
    """
    # 检查是否支持DeepSpeed
    try:
        import deepspeed
        from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

        # 配置DeepSpeed优化器
        if stage == 1:
            # ZeRO Stage 1
            ds_config = {
                "train_batch_size": kwargs.get('batch_size', 32),
                "zero_optimization": {
                    "stage": 1
                },
                "fp16": {
                    "enabled": kwargs.get('fp16', False)
                }
            }
        elif stage == 2:
            # ZeRO Stage 2
            ds_config = {
                "train_batch_size": kwargs.get('batch_size', 32),
                "zero_optimization": {
                    "stage": 2,
                    "contiguous_gradients": True,
                    "overlap_comm": True
                },
                "fp16": {
                    "enabled": kwargs.get('fp16', False)
                }
            }
        elif stage == 3:
            # ZeRO Stage 3
            ds_config = {
                "train_batch_size": kwargs.get('batch_size', 32),
                "zero_optimization": {
                    "stage": 3,
                    "contiguous_gradients": True,
                    "overlap_comm": True,
                    "stage3_prefetch_bucket_size": kwargs.get('prefetch_bucket_size', 0.9 * 1024 * 1024 * 1024),
                    "stage3_param_persistence_threshold": kwargs.get('param_persistence_threshold', 10 * 1024),
                },
                "fp16": {
                    "enabled": kwargs.get('fp16', False)
                }
            }
        else:
            raise ValueError(f"不支持的ZeRO阶段: {stage}")

        # 创建DeepSpeed引擎
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=ds_config
        )

        return optimizer
    except ImportError:
        print("DeepSpeed不可用，使用原始优化器")
        return optimizer


if __name__ == "__main__":
    # 测试分布式功能

    def example_train_fn(rank, world_size, epochs=2):
        """示例训练函数"""
        # 设置进程
        local_rank, _ = setup_distributed(rank=rank, world_size=world_size)

        # 创建模型
        model = torch.nn.Linear(10, 1)
        model = model.to(local_rank)

        # 封装为DDP模型
        model = get_data_parallel_model(model, use_ddp=True)

        # 打印信息
        if is_main_process():
            print(f"训练开始: 世界大小 = {world_size}, 共 {epochs} 轮")

        # 模拟训练
        for epoch in range(epochs):
            # 仅在主进程上打印
            if is_main_process():
                print(f"Epoch {epoch + 1}/{epochs}")

            # 同步所有进程
            if dist.is_initialized():
                dist.barrier()

        # 清理
        cleanup_distributed()


    # 如果有多个GPU并且是主程序，启动多进程
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and __name__ == "__main__":
        world_size = torch.cuda.device_count()
        print(f"启动 {world_size} 个进程进行分布式训练")
        spawn_processes(
            example_train_fn,
            world_size,
            backend='nccl',
            epochs=2
        )
    else:
        print("分布式测试需要多个GPU")
        if torch.cuda.is_available():
            print(f"可用GPU数量: {torch.cuda.device_count()}")
        else:
            print("CUDA不可用")