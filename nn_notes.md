Doing research on neural networks usually require expensive meta-parameter grid searches. With limited hardware resources (CPU, GPU, memory, transfers, disks IOs) shared with different users, it’s good to follow some best-practices.

The following guidelines are general and might be irrelevant or incomplete depending on the specific experiments.

# General experimentation process

In the following guidelines, I make the assumption that we train deep neural networks with gradient descent on batch of images stored in a dataset too large to fit in memory. The data typically follows the following flow:

 1. Pre-processing of raw data from the dataset
 2. Data-augmentation
 3. Forward pass through the neural network
 4. Backward pass through the neural network (only at **training**)
 5. Metrics computation (only at **evaluation**)
 6. Output generation (only at **inference**)

With this, the data inevitably undergoes the following transfers:

1. Read data from disk to CPU
2. Transfer data from the CPU to the GPU
3. Transfer results form GPU to CPU
4. Write results to disk

# Guidelines
(Most of the guidelines fall in the two first)

### Use resources sparingly
The first guideline is probably to avoid doing something that is not **necessary**. For examples evaluate your model at each epoch, computing tensors twice, transfer model’s outputs to the CPU at training or computing something at each epoch when it can be pre-processed only once.

### Use resources efficiently
The second guideline is probably to avoid using a device that is not **efficient** for what you want to achieve. For example reading your dataset from spinning disks or through the network, implementing loops when you can use a library to parallelise the computation or transfer your data to the GPU in float32 instead of uint8.

### Dataset localisation
Raw datasets can contain more information than necessary. Additionally some pre-processing can be done once instead of every epoch. For this reason, raw datasets should be stored on a **long-term-storage** device, while pre-processed datasets would be placed on a local **scratch** where read speed is the fastest.

### Prefetching
In a simple workflow, CPU is idle when the GPU is processing the batch and vice-versa. Fetching the data in a different **thread** allows to work asynchronously. Having more than 1 additional thread means that the generation of one batch takes more time than the forward and backward passes through the neural network.

### Data format
Some processing don’t require to work with `float32` tensors (typically crop, scale, flip, …). As the bandwidth between the CPU and the GPU is limited, it’s good to implement the processing that requires to work with `float32` on the GPU (color, noise, …)
And actually, did you try `float16`? On most GPUs, the speed gain is huge, while the performance loss is un-noticable.

# Bottlenecks investigation
For bottleneck investigation, I recommend using the following tools:

- `htop`: For CPU time, memory usage and disk I/Os
- `gpustat`: For general GPU time and memory usage
- `iostat /dev/sda --human -x 2`: For more in-depth disk IOs
- Profiling libraries for detailed GPU time and CPU to GPU transfers

When the GPU is not the bottleneck (more than ~90% of the GPU used at training):


Symptom | Possible explanation |Possible solution
--------|----------------------|------------------
One single CPU core runs at ~100%. | Your process uses costly for loops. | Use array operations with numpy
Disk IOs are slow | Un-necessary write or read to disk | Avoid writing un-necessary outputs, …
GPU is saturating and CPUs unused | Too many operations are implemented on the GPU | Move part of the computation on the CPU
Workstation is slow | Processes are being swapped | Memory leak (or un-necessary retention into memory)

 
