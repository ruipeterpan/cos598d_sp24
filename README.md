# COS 598D Project: Distributed Training of Language Models

## Overview

This assignment is designed to familiarize you with the basics of training language models with a toy example on a small scale. You will first get hands-on experience with fine-tuning pre-trained language models like BERT (implemented in PyTorch and Huggingface Transformers) on a single machine. Then, you will implement data parallelism, the most common paradigm of distributed training, using primitives from PyTorch Distributed and perform distributed training on multiple machines. You will understand the tradeoffs of different approaches of performing distributed training.


## Learning Outcomes

- Load a pretrained language model from Huggingface and fine-tune it on GLUE datasets
- Write distributed training applications with PyTorch distributed
- Understand the trade-off between different methods of performing distributed training
- Describe how ML frameworks (PyTorch) interact with lower-level communication frameworks (e.g. OpenMPI, Gloo, and NCCL)


## Environment Setup

You may choose to either use GPUs or CPUs for this project. You are required to use up to 4 nodes in this assignment, and the nodes should be able to communicate with each other through a network interface. The nodes should have at least ~12 GB of RAM (in case of CPU training) or GPU memory (in case of GPU training) and at least ~10G of disk space.

The following setup process caters to CPU-only environments. Note that this assignment is best done if you have bare-metal access to your compute nodes (i.e. you can access the nodes using a terminal not through a Slurm scheduler). You can get access to CPU nodes on CloudLab -- for instructions on accessing CloudLab, see [cloudlab.md](cloudlab.md).

- Create a fork of this repository and clone your own fork on all machines
- Install software dependencies via apt: `sudo apt-get update ; sudo apt-get install htop dstat python3-pip`
- Download the RTE dataset, one of the datasets within GLUE. You are only required to fine-tune on RTE in this assignment. This command downloads all datasets within GLUE: `cd $HOME ; mkdir glue_data ; cd cos598d ; python3 download_glue_data.py --data_dir $HOME/glue_data`
- Optional: create a virtual environment (conda/venv) to install the following dependencies
- Install CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`. If you are using GPUs, install the appropriate PyTorch [here](https://pytorch.org/get-started/locally/).
- Install dependencies: `pip install numpy scipy scikit-learn tqdm pytorch_transformers`


## Part 1: Fine-tuning BERT on GLUE Datasets on a Single Node

We have provided a base script ([`run_glue_skeleton.py`](run_glue_skeleton.py)) for you to get started, which provides some model/dataset setup and training setup to fine-tune BERT-base on the RTE dataset. The base script bakes in a handful of optimizations such as mixed-precision training and gradient accumulation -- we will not be needing them, and you can ignore the related code. We also provide `utils_glue.py` which includes some helper functions.

Here is a command you can use to run `run_glue.py`:

```shell
export GLUE_DIR=$HOME/glue_data
export TASK_NAME=RTE

python3 run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir
```

**Task 1:** Load the pretrained BERT-base model from Huggingface and fill in the standard training loop of a minibatch – forward pass (already implemented), backward pass, loss computation (already implemented), and parameter updates (via optimizer step). Record the loss values of the first five minibatches by printing the loss value after every iteration. Afterward, run training for 3 epochs (an epoch is a complete pass over a dataset -- when doing fine-tuning, we typically only need a small number of epochs) with batch size 64 and the default hyperparameters. All required code changes are marked with `TODO(cos598d)` comments. 

There are several examples for training that describe these four steps. Some good resources include the [PyTorch examples repository](https://github.com/pytorch/examples) and the [Pytorch tutorials](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html). This script is also a starting point for later parts of the assignment. Familiarize yourself with the script and run training on a single machine.


## Part 2: Distributed Data Parallel Training

Next, you will modify the script used in Part 1 to enable distributed data parallel training. There are primarily two ways distributed training is performed: i) Data Parallel, ii) Model Parallel. In the case of Data parallel, each of the participating workers trains the same network, but on different data points from the dataset. After each iteration (forward and backward pass), the workers average their local gradients to come up with a single update. In model parallel training, the model is partitioned among a number of workers. Each worker performs training on part of the model and sends its output to the worker which has the next partition during the forward pass and vice-versa in the backward pass. Model parallel is usually used when the size of the network is very large and doesn’t fit on a single worker. In this assignment, we solely focus on Data Parallel Training. For more information about parallelism in ML, see the FAQs. For data parallel training, you will need to partition the data among other nodes. Look at the FAQs to find details on how to partition the data.

To understand more about the communication primitives in ML, please refer to these two articles: [PyTorch documentation](https://pytorch.org/tutorials/intermediate/dist_tuto.html#collective-communication), [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html). We will be using `gather`, `scatter`, and `all_reduce` in this assignment.

### Part 2(a): Gradient Synchronization with gather and scatter

PyTorch comes with the `gloo` communication backend built-in. On CPU nodes, we will use this to implement gradient aggregation using the `gather` and `scatter` calls. On GPU nodes, use the `nccl` backend.

We will do the following:

1. Set up PyTorch in distributed mode using [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html). Initialize the distributed environment using [init_process_group()](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group), which only needs to be invoked once. You are encouraged to write a small test script to make sure you can initialize a process group on four nodes before proceeding. For details of the initialization process, see the FAQs.
2. Next, to perform gradient aggregation, you will need to read the gradients after the backward pass. PyTorch performs gradient computation using auto grad when you call `.backward` on a computation graph. The gradient is stored in the `.grad` attribute of the parameters. The parameters can be accessed using APIs like `model.parameters()` or `model.named_parameters()`.
3. Finally, to perform gradient aggregation, you will need to use `gather` and `scatter` communication collectives. Specifically, worker 0 (with rank 0) in the group will gather the gradients from all the participating workers, perform element-wise averaging, and then scatter the mean vector to all workers. The workers update the `grad` variable with the received vector and then continue training. 

**Task 2(a):** Implement gradient synchronization using `gather` and `scatter` in `gloo` or `nccl`. Verify that you are using the same total batch size, where total batch size = batch size on one worker * number of workers. With the same total batch size and the same seed, you should get similar loss values as in the single-node training case. Remember that you trained with a total batch size of 64 in Task 1.


### Part 2(b): Gradient Synchronization with all_reduce

Ring AllReduce is an extremely scalable technique for performing gradient synchronization. Read [here](https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/) for more intuition and visualization of Ring Allreduce (and how it has been applied to distributed ML). Instead of using the `gather` and `scatter` collectives separately, you will next use the built-in `all_reduce` collective to sync gradients among different nodes. Again, read the gradients after the backward pass and perform `all_reduce` on the gradients. Note that the PyTorch `all_reduce` call does not have an "average" mode, so you will need to use the `sum` operation and then get the average on each node by dividing the sum by the number of workers. After averaging, update the gradients of the model as in the previous part.

**Task 2(b):** Implement gradient synchronization using the `all_reduce` collective in `gloo` or `nccl`. In this case, if you have set the same random seed while using the same total batch size, you should see the same loss values as in Task 2(a).


## Part 3: Distributed Data Parallel Training with PyTorch DistributedDataParallel

Now, instead of writing your own gradient synchronization, use the distributed functionality provided by PyTorch. 

**Task 3:** Register your model with [distributed data parallel](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) and perform distributed training. Unlike in Part 2, you will not need to read the gradients for each layer as DistributedDataParallel performs these steps automatically. For more details, read [here](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).

## [Optional] Part 4: Benchmarking Data Parallel Training

In this optional section, your task is to benchmark and profile the communication aspect of data parallel training and observe how application-level hyperparameters (e.g. batch size and precision level) affect system performance. In your code for task 2(b), before you perform an `all_reduce`, record the shape/dimension and the data type (e.g. `float32`) of the tensor being communicated between nodes. Given this information, you should be able to calculate the size of the tensor in bytes (`size_in_bytes = num_elements * element_size`). Next, record the total amount of network traffic over an epoch. The expected amount of network traffic should equal `num_iterations * traffic_per_iteration` due to the iterative nature of ML training. You can do this by using Linux CLI tools like [dstat](https://linux.die.net/man/1/dstat) to generate network traffic statistics and export them into a csv file. Other similar tools include [tshark](https://www.wireshark.org/docs/man-pages/tshark.html), [tcpdump](https://www.tcpdump.org/manpages/tcpdump.1.html), etc.

**Task 4:** Adjust the batch size (e.g. reduce it from 64 to 32 and 16) and observe how the network traffic (aggregated over an epoch) and tensor size (for every `all_reduce`) changes. Compare your observed traffic with the expected amount of traffic. Reason about the difference (or the lack of difference) of training runs with different batch sizes.

## Deliverables

A *brief* report in PDF format (filename: `$NetID$_$firstname$_$lastname$.pdf`) and the code for each task. Please compress your source code into a .tar.gz file (using [tar](https://www.howtogeek.com/248780/how-to-compress-and-extract-files-using-the-tar-command-on-linux/)) and upload it to Gradescope.

In the report, include the following content:
- Run task 1 and report the evaluation metric after every epoch.
- Run each task for 40 iterations (40 minibatches of data). Discard the timings of the first iteration and report the average time per iteration for the remaining iterations for each task (2(a), 2(b), and 3). To time your code, see [this](https://realpython.com/python-timer/).
- Reason about the difference (or the lack of difference) among different setups. Feel free to refer to the [PyTorch Distributed [VLDB '18]](https://arxiv.org/pdf/2006.15704.pdf) paper for more context.
- Comment on the scalability of distributed ML based on your results.
- [Optional] If you chose to do task 4, include a comparison of the network traffic in different setups and your reasoning of the differences.
- Provide any implementation details.
- Code for each task should be in different directories: `task2a`, `task2b`, and `task3`.
- All your code that requires PyTorch distributed should be runnable using the following command:

```shell
python run_glue.py [other input args] --master_ip $ip_address$ --master_port $port$ --world_size 4 --local_rank $rank$
```


## Common FAQs and Resources

- **What is BERT?** BERT, which stands for Bidirectional Encoder Representations from Transformers, is an NLP model introduced by Google in 2018. It is an encoder-only model, and it is capable of various NLP tasks, such as question answering, sentiment analysis, and language translation. For more information, check this out: [A Visual Guide to Using BERT for the First Time
](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/). If you want more insights into which parameters we are doing gradient synchronization on, you can use a model profiler (e.g. [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html), [DeepSpeed Profiler](https://www.deepspeed.ai/tutorials/flops-profiler/)) to inspect the model. Some profilers may require a GPU to run.

- **What is GLUE?** The General Language Understanding Evaluation benchmark (GLUE) is a collection of datasets used for training, evaluating, and analyzing NLP models relative to one another. The collection consists of nine difficult and diverse task datasets designed to test a model’s language understanding. For more information, check this out: [GLUE Explained: Understanding BERT Through Benchmarks](https://mccormickml.com/2019/11/05/GLUE/)

- **What is fine-tuning?** Fine-tuning in the context of NLP refers to the process of taking a pre-trained model (trained on a huge text corpus) and further training it on a specific dataset to tailor it for a particular task.

- **Testing programs:** We suggest you write small programs to test the functionality of communication collectives at first. For example, create a tensor and send it to another node. Next, try to perform all reduce on it. This will help you get comfortable with communication collectives.

- **Setting up distributed init:** In this setup we will use the TCP init method “tcp://10.10.1.1:6585”. Instead of a shared file system, we want to use TCP to communicate. In this example, we use `10.10.1.1` as the IP address for rank 0. You can use `ifconfig` to find the IP address of your nodes. On CloudLab, you are encouraged to use the experimental network, which listens on `10.10.1.*` interfaces, but you can use any network interface as long as the nodes can communicate with each other (e.g. through a `ping`). The port has to be a non-privileged port, i.e. greater than 1023.

- **Data Partitioning:** In the case of data parallel training, the workers train on non-overlapping data partitions. You will use the distributed sampler to distribute the data among workers. For more details, look at [torch.utils.data.distributed_sampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)

- **Parallelism in ML:** HuggingFace has a nice [writeup](https://huggingface.co/docs/transformers/v4.15.0/parallelism) that summarizes the common parallelism paradigms in (large-scale) ML training. 

## Acknowledgements

- The skeleton code is modified from [run_glue.py](https://github.com/huggingface/transformers/blob/7e7fc53da5f230db379ece739457c81b2f50f13e/examples/run_glue.py) in an older version of the [`transformers/examples`](https://github.com/huggingface/transformers/tree/main/examples) repository. The source code supports DDP, but it will not work out of the box for Task 3.
- This assignment took a lot of inspiration from [this assignment](https://pages.cs.wisc.edu/~shivaram/cs744-fa22/assignment2.html) in CS 744 designed by [Prof. Shivaram Venkataraman](https://shivaram.org/). 
