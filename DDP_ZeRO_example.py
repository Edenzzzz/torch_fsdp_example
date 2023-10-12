from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import FullyShardedDataParallel
from functools import partial 
import torch.distributed as dist 
import torch.multiprocessing as mp
import os 
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5' 
from torch.cuda import reset_peak_memory_stats
from torch.cuda import max_memory_allocated
import torch
from torch import nn
from torch.optim import SGD

from torch import autocast
from torch.nn.parallel import DistributedDataParallel

num_gpus = torch.cuda.device_count()

class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            *(nn.Linear(6000, 6000) for _ in range(10))
        )

    def forward(self, x):
        return self.linear(x)
    
def count_params(model):
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(p.numel() * p.element_size() for p in model.buffers()) # no grad params like mean and var in BN
    return (params + buffers) / 1024 ** 3

def test_fp32():
    reset_peak_memory_stats()

    model = Layer().cuda()
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    data = torch.ones(6000).cuda()

    for i in range(10):
        optimizer.zero_grad()
        output = model(data)
        loss = output.sum()
        loss.backward()
        optimizer.step()
    memory = max_memory_allocated()
    print(f'fp32 peak memory allocated: {memory / 1e9:.3f}G')


def test_fp16():
    reset_peak_memory_stats()

    model = Layer().cuda()
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    data = torch.ones(6000).cuda()

    for _ in range(10):
         # cache_enabled retains a fp16 copy of params within the same autocast context for potential reuse 
         # (like calling forward twice). The copies are recreated only when entering a new context.
         # see good blog: https://discuss.pytorch.org/t/autocast-and-torch-no-grad-unexpected-behaviour/93475/2
        with autocast('cuda', cache_enabled=False):
            optimizer.zero_grad()
            output = model(data)
            loss = output.sum()
            loss.backward()
            optimizer.step()
    memory = max_memory_allocated()
    print(f'fp16 peak memory allocated: {memory / 1e9:.3f}G')
    reset_peak_memory_stats()


def test_fsdp_fp16(rank, world_size):
    dist.init_process_group('nccl', init_method="env://", rank=rank, world_size=world_size)
    reset_peak_memory_stats()
    torch.cuda.set_device(rank)
    
    #NOTE: you must specify a suitable module wrap policy
    # otherwise FSDP will just wrap the top level of the model.
    
    fsdp_model = FullyShardedDataParallel(
        module=Layer(), device_id=rank,
        auto_wrap_policy=partial(
            size_based_auto_wrap_policy,
            min_num_params=1e4,))
    optimizer = SGD(fsdp_model.parameters(), lr=0.1, momentum=0.9)
    data = torch.ones(6000).cuda()

    for _ in range(10):
        optimizer.zero_grad()
        output = fsdp_model(data)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        memory = max_memory_allocated()
    if rank == 0:
        print(f'ZeRO fp16 peak memory allocated using {num_gpus} GPUs: {memory / 1e9:.3f}G')


def test_ddp_fp16(rank, world_size):
    
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    reset_peak_memory_stats()

    model = DistributedDataParallel(Layer().cuda())
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    data = torch.ones(6000).cuda()

    for _ in range(10):
        with autocast('cuda', cache_enabled=False):
            optimizer.zero_grad()
            output = model(data)
            loss = output.sum()
            loss.backward()
            optimizer.step()
    memory = max_memory_allocated()
    if rank == 0:
        print(f'vanilla DDP fp16 peak memory allocated: {memory / 1e9:.3f}G')
        
        
# ensures that the code doesn't get executed when each new process is started.
if __name__ == '__main__':
    print(f"Model params: {count_params(Layer()):.1f}G")
    world_size = num_gpus
    test_fp32()
    test_fp16()
    mp.spawn(test_ddp_fp16, args=(world_size, ), nprocs=world_size, join=True)
    mp.spawn(test_fsdp_fp16, args=(world_size, ), nprocs=world_size, join=True)