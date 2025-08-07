Asynchronous Saving with Distributed Checkpoint (DCP)
=====================================================

Created On: Jul 22, 2024 | Last Updated: Jul 10, 2025 | Last Verified: Nov 05, 2024

**Author:** [Lucas Pasqualin](https://github.com/lucasllc), [Iris Zhang](https://github.com/wz337), [Rodrigo Kumpera](https://github.com/kumpera), [Chien-Chin Huang](https://github.com/fegin)

Checkpointing is often a bottle-neck in the critical path for distributed training workloads, incurring larger and larger costs as both model and world sizes grow. One excellent strategy for offsetting this cost is to checkpoint in parallel, asynchronously. Below, we expand the save example from the [Getting Started with Distributed Checkpoint Tutorial](https://github.com/pytorch/tutorials/blob/main/recipes_source/distributed_checkpoint_recipe.rst) to show how this can be integrated quite easily with `torch.distributed.checkpoint.async_save`.

 What you will learn

*   How to use DCP to generate checkpoints in parallel
    
*   Effective strategies to optimize performance
    

 Prerequisites

*   PyTorch v2.4.0 or later
    
*   [Getting Started with Distributed Checkpoint Tutorial](https://github.com/pytorch/tutorials/blob/main/recipes_source/distributed_checkpoint_recipe.rst)
    

Asynchronous Checkpointing Overview
-----------------------------------

Before getting started with Asynchronous Checkpointing, it’s important to understand it’s differences and limitations as compared to synchronous checkpointing. Specifically:

*   Memory requirements - Asynchronous checkpointing works by first copying models into internal CPU-buffers.
    
    This is helpful since it ensures model and optimizer weights are not changing while the model is still checkpointing, but does raise CPU memory by a factor of `checkpoint_size_per_rank X number_of_ranks`. Additionally, users should take care to understand the memory constraints of their systems. Specifically, pinned memory implies the usage of `page-lock` memory, which can be scarce as compared to `pageable` memory.
    
*   Checkpoint Management - Since checkpointing is asynchronous, it is up to the user to manage concurrently run checkpoints. In general, users can
    
    employ their own management strategies by handling the future object returned form `async_save`. For most users, we recommend limiting checkpoints to one asynchronous request at a time, avoiding additional memory pressure per request.
    

import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn

from torch.distributed.fsdp import fully\_shard
from torch.distributed.checkpoint.state\_dict import get\_state\_dict, set\_state\_dict
from torch.distributed.checkpoint.stateful import Stateful

CHECKPOINT\_DIR \= "checkpoint"

class AppState(Stateful):
 """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
 with the Stateful protocol, DCP will automatically call state\_dict/load\_stat\_dict as needed in the
 dcp.save/load APIs.

 Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
 and optimizer.
 """

    def \_\_init\_\_(self, model, optimizer\=None):
        self.model \= model
        self.optimizer \= optimizer

    def state\_dict(self):
        \# this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED\_STATE\_DICT
        model\_state\_dict, optimizer\_state\_dict \= get\_state\_dict(self.model, self.optimizer)
        return {
            "model": model\_state\_dict,
            "optim": optimizer\_state\_dict
        }

    def load\_state\_dict(self, state\_dict):
        \# sets our state dicts on the model and optimizer, now that we've loaded
        set\_state\_dict(
            self.model,
            self.optimizer,
            model\_state\_dict\=state\_dict\["model"\],
            optim\_state\_dict\=state\_dict\["optim"\]
        )

class ToyModel(nn.Module):
    def \_\_init\_\_(self):
        super(ToyModel, self).\_\_init\_\_()
        self.net1 \= nn.Linear(16, 16)
        self.relu \= nn.ReLU()
        self.net2 \= nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def setup(rank, world\_size):
    os.environ\["MASTER\_ADDR"\] \= "localhost"
    os.environ\["MASTER\_PORT"\] \= "12355 "

    \# initialize the process group
    dist.init\_process\_group("gloo", rank\=rank, world\_size\=world\_size)
    torch.cuda.set\_device(rank)

def cleanup():
    dist.destroy\_process\_group()

def run\_fsdp\_checkpoint\_save\_example(rank, world\_size):
    print(f"Running basic FSDP checkpoint saving example on rank {rank}.")
    setup(rank, world\_size)

    \# create a model and move it to GPU with id rank
    model \= ToyModel().to(rank)
    model \= fully\_shard(model)

    loss\_fn \= nn.MSELoss()
    optimizer \= torch.optim.Adam(model.parameters(), lr\=0.1)

    checkpoint\_future \= None
    for step in range(10):
        optimizer.zero\_grad()
        model(torch.rand(8, 16, device\="cuda")).sum().backward()
        optimizer.step()

        \# waits for checkpointing to finish if one exists, avoiding queuing more then one checkpoint request at a time
        if checkpoint\_future is not None:
            checkpoint\_future.result()

        state\_dict \= { "app": AppState(model, optimizer) }
        checkpoint\_future \= dcp.async\_save(state\_dict, checkpoint\_id\=f"{CHECKPOINT\_DIR}\_step{step}")

    cleanup()

if \_\_name\_\_ \== "\_\_main\_\_":
    world\_size \= torch.cuda.device\_count()
    print(f"Running async checkpoint example on {world\_size} devices.")
    mp.spawn(
        run\_fsdp\_checkpoint\_save\_example,
        args\=(world\_size,),
        nprocs\=world\_size,
        join\=True,
    )

Even more performance with Pinned Memory
----------------------------------------

If the above optimization is still not performant enough, you can take advantage of an additional optimization for GPU models which utilizes a pinned memory buffer for checkpoint staging. Specifically, this optimization attacks the main overhead of asynchronous checkpointing, which is the in-memory copying to checkpointing buffers. By maintaining a pinned memory buffer between checkpoint requests users can take advantage of direct memory access to speed up this copy.

Note

The main drawback of this optimization is the persistence of the buffer in between checkpointing steps. Without the pinned memory optimization (as demonstrated above), any checkpointing buffers are released as soon as checkpointing is finished. With the pinned memory implementation, this buffer is maintained between steps, leading to the same peak memory pressure being sustained through the application life.

import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn

from torch.distributed.fsdp import fully\_shard
from torch.distributed.checkpoint.state\_dict import get\_state\_dict, set\_state\_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint import FileSystemWriter as StorageWriter

CHECKPOINT\_DIR \= "checkpoint"

class AppState(Stateful):
 """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
 with the Stateful protocol, DCP will automatically call state\_dict/load\_stat\_dict as needed in the
 dcp.save/load APIs.

 Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
 and optimizer.
 """

    def \_\_init\_\_(self, model, optimizer\=None):
        self.model \= model
        self.optimizer \= optimizer

    def state\_dict(self):
        \# this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED\_STATE\_DICT
        model\_state\_dict, optimizer\_state\_dict \= get\_state\_dict(self.model, self.optimizer)
        return {
            "model": model\_state\_dict,
            "optim": optimizer\_state\_dict
        }

    def load\_state\_dict(self, state\_dict):
        \# sets our state dicts on the model and optimizer, now that we've loaded
        set\_state\_dict(
            self.model,
            self.optimizer,
            model\_state\_dict\=state\_dict\["model"\],
            optim\_state\_dict\=state\_dict\["optim"\]
        )

class ToyModel(nn.Module):
    def \_\_init\_\_(self):
        super(ToyModel, self).\_\_init\_\_()
        self.net1 \= nn.Linear(16, 16)
        self.relu \= nn.ReLU()
        self.net2 \= nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def setup(rank, world\_size):
    os.environ\["MASTER\_ADDR"\] \= "localhost"
    os.environ\["MASTER\_PORT"\] \= "12355 "

    \# initialize the process group
    dist.init\_process\_group("gloo", rank\=rank, world\_size\=world\_size)
    torch.cuda.set\_device(rank)

def cleanup():
    dist.destroy\_process\_group()

def run\_fsdp\_checkpoint\_save\_example(rank, world\_size):
    print(f"Running basic FSDP checkpoint saving example on rank {rank}.")
    setup(rank, world\_size)

    \# create a model and move it to GPU with id rank
    model \= ToyModel().to(rank)
    model \= fully\_shard(model)

    loss\_fn \= nn.MSELoss()
    optimizer \= torch.optim.Adam(model.parameters(), lr\=0.1)

    \# The storage writer defines our 'staging' strategy, where staging is considered the process of copying
    \# checkpoints to in-memory buffers. By setting \`cached\_state\_dict=True\`, we enable efficient memory copying
    \# into a persistent buffer with pinned memory enabled.
    \# Note: It's important that the writer persists in between checkpointing requests, since it maintains the
    \# pinned memory buffer.
    writer \= StorageWriter(cache\_staged\_state\_dict\=True, path\=CHECKPOINT\_DIR)
    checkpoint\_future \= None
    for step in range(10):
        optimizer.zero\_grad()
        model(torch.rand(8, 16, device\="cuda")).sum().backward()
        optimizer.step()

        state\_dict \= { "app": AppState(model, optimizer) }
        if checkpoint\_future is not None:
            \# waits for checkpointing to finish, avoiding queuing more then one checkpoint request at a time
            checkpoint\_future.result()
        dcp.async\_save(state\_dict, storage\_writer\=writer, checkpoint\_id\=f"{CHECKPOINT\_DIR}\_step{step}")

    cleanup()

if \_\_name\_\_ \== "\_\_main\_\_":
    world\_size \= torch.cuda.device\_count()
    print(f"Running fsdp checkpoint example on {world\_size} devices.")
    mp.spawn(
        run\_fsdp\_checkpoint\_save\_example,
        args\=(world\_size,),
        nprocs\=world\_size,
        join\=True,
    )

Conclusion
----------

In conclusion, we have learned how to use DCP’s `async_save()` API to generate checkpoints off the critical training path. We’ve also learned about the additional memory and concurrency overhead introduced by using this API, as well as additional optimizations which utilize pinned memory to speed things up even further.

*   [Saving and loading models tutorial](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
    
*   [Getting started with FullyShardedDataParallel tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)