**A minial usage of torch FSDP(Fully shared data parrallel, which implements ZeRO3 https://arxiv.org/pdf/1910.02054.pdf)**
Requires 4 GPUs but you can tune that to run the tests one by one. Shows linear memory reduction regarding the number of GPUs compared to default torch DDP. Also shows why you should pass in cache_enabled=False in torch.autocastwhen there's only one forward pass.
