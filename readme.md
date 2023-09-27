# Training with DistributedDataParallel

This is an end-to-end example of training a simple Logistic Regression Pytorch model with [DistributedDataParallel](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) (DDP; single-node, multi-GPU data parallel training) on a fake dataset. The dataset gets distributed to multiple GPUs by `DistributedSampler`. This builds off of [this tutorial](https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51) and the [Pytorch DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).

Let's say you have 8 GPUs and want to run it on GPUs 5, 6, and 7, since GPUs 0-4 are in use by others. Then, it can be run with: `CUDA_VISIBLE_DEVICES=5,6,7 python3 main.py`

Additional resources
- TODO: Implement validation in DistributedDataParallel [forum link here](https://discuss.pytorch.org/t/how-to-validate-in-distributeddataparallel-correctly/94267)
- [DDP video tutorials](https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series)
- [Distributed Data Parallel Model Training in PyTorch](https://www.youtube.com/watch?v=SivkGd6LQoU)