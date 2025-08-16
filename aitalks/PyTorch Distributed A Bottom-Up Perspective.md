PyTorch Distributed Overview
============================

`torch.distributed` is a native PyTorch submodule providing a flexible set of Python APIs for distributed model training. Many of the state-of-the-art Large Language Model (LLM) training libraries such as DeepSpeed and Megatron-LM are built on top of it.

In this note, we will examine components from `torch.distributed` from a bottom-up angle:

*   Basic Cross-Node Communication (c10d)
*   Distributed Data-Parallel Training (DDP)
*   Remote Procedure Call (RPC)

Basic Cross-Node Communication
==============================

`torch.distributed` provides basic Python APIs to send tensors across processes/nodes. In particular, it provides both Point-to-Point (P2P) APIs, e.g., `torch.distributed.send` and `torch.distributed.recv`, and collective communication APIs, e.g., `torch.distributed.all_reduce`.

Roughly speaking, collective communication generalized P2P communication by allowing communication to involve multiple parties with possibly multiple sources and/or destinations. For example, `all_reduce` is the collective communication to compute the sum/average of tensors from each individual party and dispatch the result to each party. For a `all_reduce` with $n$ parties, each of which holds a tensor of size $B$ bytes, if one only uses P2P communication, $B(n-1)^2$ payload is needed as each party needs to pull $n-1$ tensors of size B. Even if we could add an aggregator, e.g., a parameter server, to assist the computation, we still need $n$uplink passes to the aggregator and $n$ downlink passes from the aggregator such that the total payload is $2Bn$. In practice, the aggregator often becomes the communication bottleneck as all traffic is through it. In contrast, collective communication with ring `all_reduce` chunks tensors into small pieces of size $B/n$ and flows these pieces across nodes along a ring order with aggregation on the fly such that the total payload is only $2B(n-1)/n$ for each node and $2B(n-1)$ overall. Thisis $2/(n-1)$ fraction of that in native P2P. Compared with P2P with an additional aggregator, the traffic reduction fraction $(n-1)/n$ is not significant but traffic in the ring `all_reduce` is evenly split between nodes and hence does not create communication bottlenecks.

Efficient collective communication has been a key technique in the long existing High Performance Computation (HPC) community. `all reduce` was first introduced to distributed deep learning by [Baidu](https://github.com/baidu-research/baidu-allreduce) to replace the once dominating parameter server architecture. Later, Uber created a popular library [Horovod](https://github.com/horovod/horovod) to allow people to apply collective communication in their Tensorflow or PyTorch model training codes. As the primary usage of collective communication is data parallel distributed training, both PyTorch and Tensorflow later introduced (distributed) data parallel training APIs into the library which leverages all-reduce underneath. PyTorch did an even better job job to create generic Python APIs to expose almost all collective communication primitives.

As the efficiency of collective communication is highly coupled with the hardware and low-level (communication) divers/protocols, which should be implemented via C/CPP, PyTorch enables its collective communication to be hardware and software flexible by allowing users to configure the backends of `torch.distributed` to invoke collective communication libraries such as NCCL (created by Nvidia for GPUs), MPI (developed in HPC community) and GLOO (created by Meta). In fact, `pytroch.distributed` even allows a user/company to implement and compile its own collective communication library by C/CPP and invoke it as a new backend. This design pattern can make the `pytorch.distributed` codes completely independent of collective communication backends.

Following the convention in MPI, `pytorch.distributed` requires initializing a group of processes/nodes, where each is assigned a unique rank and can explicitly communicate with one another using this rank. That is, each process/node needs to invoke the following function before any collective or P2P communication:

```python
torch.distributed.init_process_group(  
    backend=None,  
    init_method=None,  
    timeout=datetime.timedelta(seconds=1800),  
    world_size=-1,  
    rank=-1,  
    store=None,  
    group_name='',  
    pg_options=None)
```
where `backend` is used to specify a backend from nccl/gloo/mpi; `init_method` (a URL string) indicates where and how to discover peers, e.g., tcp or shared file-system; `world_size` is the total # of nodes/processes; and `rank` indicates the 0-based index for the node invoking this API.

Once the group is initialized, we could have P2P or collective communication between nodes. The following is an example where rank 0 sends a tensor to rank 1.  
On rank 0, we run codes

```python
import torch  
import torch.distributed as dist  

dist.init_process_group(backend="gloo", init_method="tcp://localhost:29500", rank=0, world_size=2)  
t = torch.tensor([1.0]*10)  
dist.send(tensor=t, dst=1)
```
On rank 1, we run codes
```python
import torch  
import torch.distributed as dist  

dist.init_process_group(backend="gloo", init_method="tcp://localhost:29500", rank=1, world_size=2)  
tensor = torch.tensor([0.0]*10)  
dist.recv(tensor=tensor, src=0)  
print(f'rank 1 received tensor {tensor} from rank 0')
```

Note that `torch.distributed.recv` needs to accept a tenor of the same size as the one to be received from `src` node.

`torch.distributed.send` and `torch.distributed.recv` are synchronous P2P APIs in the sense that codes after it is blocked until P2P communication is completed. In the case, that the sent/received tensor is not immediately needed, we can use its asynchronous version `.isend` and `.irecv` to improve the efficiency.

P2P communication codes are asymmetric as the sender and the receiver need to trigger `.send` and `.recv` respectively. The codes of collective communication like "all reduce" or "all gather" are often simpler as all nodes are almost symmetric. For example, once `pytorch.distributed` is initialized with `.init_process_group`, all nodes can use codes like the below for `all_reduce`

```python
t = torch.tensor([XXXXXX]) # define the tensor at this rank  
torch.distributed.all_reduce(t, op= torch.distributed.ReduceOp.SUM)  
print(f"after all reduce, t={t}")
```

After `torch.distributed.all_reduce`, tensor `t` will be updated as the sum of individual tensors in place. Note that we don't need to specify what other nodes are involved in `all_reduce` because we have initialized the group and involved all nodes for this operation. It is also possible to conduct `all_reduce` in a subgroup by defining such a subgroup and passing it to `all_reduce` API.

Distributed Data-Parallel Training (DDP)
========================================

Data-parallel training has become the most popular (and probably necessary) technique for large model training with large datasets since the famous paper [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677).

In DDP, multiple GPUs in parallel sample their mini-batches, compute gradients, all-reduce gradients, and update individual models with the all-reduced gradients. Suppose we have N GPUs, DDP can in theory boost the throughput by N times. (In practice, it is less than N, as all-reduce takes extra time.)

In fact, with the `pytorch.distributed.all_reduce` API, one can directly implement DDP with only a few lines of code. The codes are roughly sketched as follows:

```python
import torch  
import torch.distributed as dist  
# initialize the communication group  
# define dataloader, loss_fun, and optimizer before starting the training for loop (of an epoch)  
for inp, label in dataloader:  
    optimizer.zero_grad()  
    out = model(inp)  
    loss = loss_fun(out, lable)  
    loss.backward()  
    # --start--  
    for param in model.parameters():  
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)   
        param.grad.data /= dist.get_world_size()  
    # --end--  
    optimizer.step()
```

Compared with standard single node training, only 3 lines of codes between `# --start--` and `# --end--` are needed to enable DDP and these 3 lines of codes just simply trigger `all_reduce` for each gradient. (Note that `module.parameters()` in PyTorch returns of (flattened) list of model weight tensors.)

At this point, we should appreciate that PyTorch provides us with the full set of collective communication APIs for tensors. This provides machine learning scientists/engineers tremendous opportunities to develop distributed training.

However, PyTorch still provides us with dedicated APIs for DDP and does not want users to use my above-sketched simple codes for DDP. This is because PyTorch did extra cool optimization/speedup with low-level C/CPP implementations. Let me explain what is the cool optimization.

Consider a deep neural network with n layers. A complete SGD iteration in DDP is composed of the following:

(1) Each node compute: layer 1 forward -> layer 2 forward -> … -> layer n forward -> layer n backward -> layer n-1 backward -> … -> layer 1 backward
(2) all-reduce gradients of all layers  
(3) update the weight of each layer

Note that the forward and backward in step (1) must be conducted sequentially. However, we shall note that we don’t necessarily need to wait until all operations in step (1) are completed to start step (2). In fact, once layer n backward is completed, we can immediately start to all-reduce layer n gradient and then even update the weight using this all-reduced gradient (as layer n weight is not needed in the backward for layer n-1 to layer 1.) Furthermore, we should step (2) are communication operations and backward in step (1) are computation operations. So overlapping them should not slow down either.

What DDP in `pytorch.distributed` does is an efficient low-level implementation of the above overlapping/pipelining idea. (Instead of overlapping by layer, it actually divides n layers into buckets of multiple layers to optimize the tradeoff between frequent all-reduce of small tensors and poor communication/computation overlapping.) To invoke DDP in `pytorch.distributed` is simple, after initializing the process group and defining the model, we only need the following one-line code:
```python
from torch.nn.parallel import DistributedDataParallel as DDP  
ddp_model = DDP(model, device_ids=[rank])
```
where `model` is the PyTorch model defined in the traditional way; `device_ids=[rank]` consumes the `rank` of the process/node executing the current code. In the remaining codes, we just use `ddp_model` to replace `model` in our single-node training.

Remote Procedure Call (RPC)
===========================

DDP with `pytorch.distributed` is simple and convenient. However, it is restricted to the homogeneous scenario where all nodes follow the same operations. Other than data parallelism, model parallelism is another popular distributed model training pattern where multiple nodes jointly train a single model and different tensors are passed between these nodes. Technically, one can implement model parallelism directly with P2P communication APIs `.send` or `.recv` from `pytroch.distributed` to exchange intermediate layer values and partial gradients. Obviously, such an implementation can be convoluted and poorly structured. It is also hard to use P2P communication APIs to do conditionally tensor passing or to let one node remotely trigger logic on another node. For example, to conditionally pass tensor $y$ from node A to node B based on the value of tensor $x$ on node A will require one to first pass tensor $x$ from node A to node B and then trigger `.send` on node A and `.recv` on node B by checking the value of tensor `x` on both nodes. Ideally, we prefer to get rid of passing tensor `x` and only let node A hold the logic of passing tensor `y` if needed.

RPC provided by `pytorch.distributed` is to address the above pain points and hence handy in model parallelism or other training scenarios. Literarily, PyTorch's PRCs extend standard [remote procedure calls](https://en.wikipedia.org/wiki/Remote_procedure_call) for tensor computations by providing the following functionalities:

*   Python APIs of remote procedure call: These are APIs like standard RPC that can run a function on the specified node with the passed arguments and receive the return values.
*   Remote Reference (RRef): This distributed shared pointer allows us to manage data objects across nodes. By passing or receiving the RRef to an object, a remote node can manage or call its method of it.
*   Distributed Autograd and Optimizer: The computation graph of forward passes across nodes with RPC APIs will be tracked so that the gradient in the backward passes can be computed automatically with the newly distributed autograd. Further, distributed optimizers that could update parameters (pointed by RRef) placed on different nodes are also provided.

Initialization
==============

To use RPC, each node should be initialized from an API different from `torch.distributed.init_process_group` used in the previous sections. (Note that a node can be in a RPC group and a collection communication/DDP group simultaneously.)

```python
torch.distributed.rpc.init_rpc(name,   
    backend=None,  
    rank=-1,  
    world_size=None,  
    rpc_backend_options=None)
```

where `name` is the global unique name of this node, which is later used to specify where an RPC is executed; `backend` is to specify the backend of RPC; (Though RPC is very close to P2P communication and gloo as a collective communication backend supports P2P. `pytorch.distributed` created a new backend `TensorPipe` that is specifically optimized for tensor computation. ) `rank` and `world_size` are used in the same way as in `torch.distributed.init_process_group`; `rpc_backend_options` expects a `torch.distributed.rpc.RpcBackendOptions` object which has properties like `rpc_timeout` and `init_method` with the same purpose as `init_method` in `torch.distributed.init_process_group`. (If `rpc_backend_options=None`, then we use `os.environ['MASTER_ADDR']` as the address and `os.environ['MASTER_PORT']` as the port to establish a TPC-based transport for RPC communication.

RPC calls
=========

Once the RPC group is initialized, one can use APIs `rpc.rpc_sync` (for a synchronous/blocking PRC call), `rpc.rpc_async` (for an asynchronous/non-blocking PRC call), `rpc.remote` (for an asynchronous call that remotes an `RRef`). These APIs take the same arguments as in the following `rpc_sync` signature:

```python
torch.distributed.rpc.rpc_sync(to, func, args=None, kwargs=None, timeout=-1.0)
```

where `to` takes the str name or int rank to specify where the func is executed, `func` is the function to be executed, and `args`/`kwargs` specify the tuple/dictionary of the arguments for `func`.

RRef (Remote Reference)
=======================

`pytorch.distributed` also introduces an important concept `RRef` (Remote Reference) which is a reference to an object on a remote worker. Recall that `rpc.remote` returns an `RRef` object so that we can access and manage the returned remote object. For an `RRef` object, we can use `.to_here()` to get a copy of its value at the local node, where `.to_here()` is executed. With an `RRef` object, we can conveniently access/manage a remote object, e.g., trigger a member function on a remote object pointed from an `RRef`. Alternatively, we can also define a `RRef` to point to a local object and pass it to a remote node via the above-mentioned `rpc` API so that a remote node can access/manage this object. In summary, an `RRef` provides extra convenience on top of `rpc_sync`/`rpc_async`/`remote` to manage remote function calls as it is not restricted by functions but works for objects. (It is also needed by `pytorch.distributed` to manage object lifecycle across nodes because it is hard for a node to know whether its local object is needed by any remote nodes and hence if the local object can be erased from memory. With `RRef`, this is doable by tracking how many remote references are still there.)

Distributed Autograd and Optimizer
==================================

Auto differentiation is the major functionality advantage that PyTorch/Tensorflow can provide when compared with Numpy. In PyTorch, one only implements the forward logic for a `nn.module` using PyTorch computation APIs such that backward auto differentiation process is implemented automatically by PyTorch. You may worry if things will fail when one use `pytorch.distributed` RPC to implement cross-nodes forward. PyTorch eases your concerns by providing **distributed autograd** which tracks the PRC cross-nodes computation graph for you and conducts the autograd computation for you when needed. The codes achieving distributed autograd are as follows:

```python
import torch.distributed.autograd as dist_autograd  
with dist_autograd.context() as context_id:  
    pred = model.forward()  
    loss = loss_func(pred, loss)  
    dist_autograd.backward(context_id, loss)
```

The codes are very similar to single node backward and have two key steps:

*   we need to wrap the forward and backward codes under a `with dist_autograd.context() as context_id` statement to define a context id
*   Instead of triggering backward with `tensor.backward()` in the conventional cases, we use `dist_autograd.backward()` which takes in the context id and the tensor that is to compute `backward`.

After providing distributed autograd, `Pytorch.distributed` further provides a distributed optimizer that can allow an existing optimizer to work with distributed nodes. Its usage is very straightforward as described below:

```python
torch.distributed.optim.DistributedOptimizer(optimizer_class, params_rref, args, kwargs)
```

where `optimizer_class` is to take a standard optimizer class, e.g., `torch.optim.SGD`; `params_rref` is a list of `RRef` that points to local or remote model parameters involved in the optimizer (recall that standard optimizer also takes in such a parameter and the only difference is that distributed optimizer takes in `RRef`s rather than model parameters directly); `args/kwars` are just arguments to specify the optimizer (as in the standard optimizer)

An Example using RPC
====================

In the following, we consider a simple example where node A remotes create a model on node B, trigger SGD training with its own local data, and then pull the trained node B model weight to itself. This example covers everything mentioned in this PRC section.

```python
import os  
import argparse  
import torch  
from torch import nn, optim  
from torch.distributed import rpc, autograd  
from torch.distributed.optim import DistributedOptimizer  
import torch.multiprocessing as mp  

# define the model to be created on node B  
class Net(nn.Module):  
    def __init__(self, input_dim=2, output_dim=1):  
        super(Net, self).__init__()  
        self.fc = nn.Linear(input_dim, output_dim, bias=False)  
    def forward(self, x):  
        out = self.fc(x)  
        out = out.to("cpu")  
        return out  
    def get_param_rrefs(self):  
        return [rpc.RRef(param) for param in  self.fc.parameters()]  
    def get_model(self):  
        return  self.fc  

def run_nodeB(rank, world_size):  
    rpc.init_rpc(name="node_B", rank=rank, world_size=world_size)  
    rpc.shutdown()  

def run_nodeA(rank, world_size)  
    rpc.init_rpc(name="node_A", rank=rank, world_size=world_size)  
    # this will create the Net remotely on node B and obtain the RRef  
    rref = rpc.remote("node_B", Net)  
    # obtain the rref params  
    param_rrefs = rref.rpc_sync().get_params_rrefs()  
    # define the distributed optimizer  
    opt = DistributedOptimizer(optim.SGD, params_rrefs, lr=0.1)  
    # define a synthetic dataset and dataloader  
    # prepare dataset and dataloader for the server node  
    x = np.array([[1, 2] for i in range(8)], dtype=np.float32)  
    y = np.array([1]*4+[0]*4, dtype=np.float32).reshape(-1,1)  
    ds = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))  
    dataloader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)  
    # train the model on remote node A by consuming node B's data  
    for (x, y_true) in dataloader:  
        with autograd.context() as cid:  
            y_pred = rref.rpc_sync().forward(x)  
            loss = torch.nn.functional.mse_loss(y_pred, y_true)  
            autograd.backward(cid, [loss])  
            opt.step(cid)  
    # pull the model from node A and print it out  
    node_A_nn = rref.rpc_sync().get_model()  
    print(f"Node A NET weight: {node_A_nn.fc.weight.detach().numpy()}")  
    rpc.shutdown()  

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--world_size", type=int, default=2)  
    parser.add_argument("--rank", type=int, default=None)  
    args = parser.parse_args()  
    os.environ['MASTER_ADDR'] = "localhost"  
    os.environ["MASTER_PORT"] = 29500  
    processes = []  
    if args.rank == 0:  
        p = mp.Process(target=run_server, args=(0, args.world_size))  
        p.start()  
    else:  
        p = mp.Process(target=run_client, args=(args.rank, args.world_size))  
        p.start()  
    p.join()
```

The above codes to emulate model training across 2 nodes can be run locally as the `os.environ['MASTER_ADDR']` is hardcoded to "localhost". We need two terminals and run `python XX.py --rank=0 --world_size=2` and `python XX.py --rank=1 --world_size=2`, respectively.

