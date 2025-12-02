
# Understanding PyTorch Buffers

In essence, PyTorch buffers are tensor attributes associated with a PyTorch module or model similar to parameters, but unlike parameters, buffers are not updated during training.

Buffers in PyTorch are particularly useful when dealing with GPU computations, as they need to be transferred between devices (like from CPU to GPU) alongside the model's parameters. Unlike parameters, buffers do not require gradient computation, but they still need to be on the correct device to ensure that all computations are performed correctly.

In chapter 3, we use PyTorch buffers via `self.register_buffer`, which is only briefly explained in the book. Since the concept and purpose are not immediately clear, this code notebook offers a longer explanation with a hands-on example.

## An example without buffers

Suppose we have the following code, which is based on code from chapter 3. This version has been modified to exclude buffers. It implements the causal self-attention mechanism used in LLMs:


```python
import torch
import torch.nn as nn

class CausalAttentionWithoutBuffers(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec
```

We can initialize and run the module as follows on some example data:


```python
torch.manual_seed(123)

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

batch = torch.stack((inputs, inputs), dim=0)
context_length = batch.shape[1]
d_in = inputs.shape[1]
d_out = 2

ca_without_buffer = CausalAttentionWithoutBuffers(d_in, d_out, context_length, 0.0)

with torch.no_grad():
    context_vecs = ca_without_buffer(batch)

print(context_vecs)
```

    tensor([[[-0.4519,  0.2216],
             [-0.5874,  0.0058],
             [-0.6300, -0.0632],
             [-0.5675, -0.0843],
             [-0.5526, -0.0981],
             [-0.5299, -0.1081]],
    
            [[-0.4519,  0.2216],
             [-0.5874,  0.0058],
             [-0.6300, -0.0632],
             [-0.5675, -0.0843],
             [-0.5526, -0.0981],
             [-0.5299, -0.1081]]])


So far, everything has worked fine so far.

However, when training LLMs, we typically use GPUs to accelerate the process. Therefore, let's transfer the `CausalAttentionWithoutBuffers` module onto a GPU device.

Please note that this operation requires the code to be run in an environment equipped with GPUs.


```python
has_cuda = torch.cuda.is_available()
has_mps = torch.backends.mps.is_available()

print("Machine has GPU:", has_cuda or has_mps)

if has_mps:
    device = torch.device("mps")   # Apple Silicon GPU (Metal)
elif has_cuda:
    device = torch.device("cuda")  # NVIDIA GPU
else:
    device = torch.device("cpu")   # CPU fallback

print(f"Using device: {device}")

batch = batch.to(device)
ca_without_buffer = ca_without_buffer.to(device)
```

    Machine has GPU: True
    Using device: cuda


Now, let's run the code again:


```python
with torch.no_grad():
    context_vecs = ca_without_buffer(batch)

print(context_vecs)
```


    ---------------------------------------------------------------------------
    
    RuntimeError                              Traceback (most recent call last)
    
    <ipython-input-4-1e0d2e6638f6> in <cell line: 1>()
          1 with torch.no_grad():
    ----> 2     context_vecs = ca_without_buffer(batch)
          3 
          4 print(context_vecs)


    /usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
       1530             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1531         else:
    -> 1532             return self._call_impl(*args, **kwargs)
       1533 
       1534     def _call_impl(self, *args, **kwargs):


    /usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
       1539                 or _global_backward_pre_hooks or _global_backward_hooks
       1540                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1541             return forward_call(*args, **kwargs)
       1542 
       1543         try:


    <ipython-input-1-cf1dad0dd611> in forward(self, x)
         21 
         22         attn_scores = queries @ keys.transpose(1, 2)
    ---> 23         attn_scores.masked_fill_(
         24             self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
         25         attn_weights = torch.softmax(


    RuntimeError: expected self and mask to be on the same device, but got mask on cpu and self on cuda:0


Running the code resulted in an error. What happened? It seems like we attempted a matrix multiplication between a tensor on a GPU and a tensor on a CPU. But we moved the module to the GPU!?


Let's double-check the device locations of some of the tensors:


```python
print("W_query.device:", ca_without_buffer.W_query.weight.device)
print("mask.device:", ca_without_buffer.mask.device)
```

    W_query.device: cuda:0
    mask.device: cpu



```python
type(ca_without_buffer.mask)
```




    torch.Tensor



As we can see, the `mask` was not moved onto the GPU. That's because it's not a PyTorch parameter like the weights (e.g., `W_query.weight`).

This means we  have to manually move it to the GPU via `.to("cuda")`:


```python
ca_without_buffer.mask = ca_without_buffer.mask.to(device)
print("mask.device:", ca_without_buffer.mask.device)
```

    mask.device: cuda:0


Let's try our code again:


```python
with torch.no_grad():
    context_vecs = ca_without_buffer(batch)

print(context_vecs)
```

    tensor([[[-0.4519,  0.2216],
             [-0.5874,  0.0058],
             [-0.6300, -0.0632],
             [-0.5675, -0.0843],
             [-0.5526, -0.0981],
             [-0.5299, -0.1081]],
    
            [[-0.4519,  0.2216],
             [-0.5874,  0.0058],
             [-0.6300, -0.0632],
             [-0.5675, -0.0843],
             [-0.5526, -0.0981],
             [-0.5299, -0.1081]]], device='cuda:0')


This time, it worked!

However, remembering to move individual tensors to the GPU can be tedious. As we will see in the next section, it's easier to use `register_buffer` to register the `mask` as a buffer.

## An example with buffers

Let's now modify the causal attention class to register the causal `mask` as a buffer:


```python
import torch
import torch.nn as nn

class CausalAttentionWithBuffer(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # Old:
        # self.mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)

        # New:
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec
```

Now, conveniently, if we move the module to the GPU, the mask will be located on the GPU as well:


```python
ca_with_buffer = CausalAttentionWithBuffer(d_in, d_out, context_length, 0.0)
ca_with_buffer.to(device)

print("W_query.device:", ca_with_buffer.W_query.weight.device)
print("mask.device:", ca_with_buffer.mask.device)
```

    W_query.device: cuda:0
    mask.device: cuda:0



```python
with torch.no_grad():
    context_vecs = ca_with_buffer(batch)

print(context_vecs)
```

    tensor([[[0.4772, 0.1063],
             [0.5891, 0.3257],
             [0.6202, 0.3860],
             [0.5478, 0.3589],
             [0.5321, 0.3428],
             [0.5077, 0.3493]],
    
            [[0.4772, 0.1063],
             [0.5891, 0.3257],
             [0.6202, 0.3860],
             [0.5478, 0.3589],
             [0.5321, 0.3428],
             [0.5077, 0.3493]]], device='cuda:0')


As we can see above, registering a tensor as a buffer can make our lives a lot easier: We don't have to remember to move tensors to a target device like a GPU manually.

## Buffers and `state_dict`

- Another advantage of PyTorch buffers, over regular tensors, is that they get included in a model's `state_dict`
- For example, consider the `state_dict` of the causal attention object without buffers


```python
ca_without_buffer.state_dict()
```




    OrderedDict([('W_query.weight',
                  tensor([[-0.2354,  0.0191, -0.2867],
                          [ 0.2177, -0.4919,  0.4232]], device='cuda:0')),
                 ('W_key.weight',
                  tensor([[-0.4196, -0.4590, -0.3648],
                          [ 0.2615, -0.2133,  0.2161]], device='cuda:0')),
                 ('W_value.weight',
                  tensor([[-0.4900, -0.3503, -0.2120],
                          [-0.1135, -0.4404,  0.3780]], device='cuda:0'))])



- The mask is not included in the `state_dict` above
- However, the mask *is* included in the `state_dict` below, thanks to registering it as a buffer


```python
ca_with_buffer.state_dict()
```




    OrderedDict([('mask',
                  tensor([[0., 1., 1., 1., 1., 1.],
                          [0., 0., 1., 1., 1., 1.],
                          [0., 0., 0., 1., 1., 1.],
                          [0., 0., 0., 0., 1., 1.],
                          [0., 0., 0., 0., 0., 1.],
                          [0., 0., 0., 0., 0., 0.]], device='cuda:0')),
                 ('W_query.weight',
                  tensor([[-0.1362,  0.1853,  0.4083],
                          [ 0.1076,  0.1579,  0.5573]], device='cuda:0')),
                 ('W_key.weight',
                  tensor([[-0.2604,  0.1829, -0.2569],
                          [ 0.4126,  0.4611, -0.5323]], device='cuda:0')),
                 ('W_value.weight',
                  tensor([[ 0.4929,  0.2757,  0.2516],
                          [ 0.2377,  0.4800, -0.0762]], device='cuda:0'))])



- A `state_dict` is useful when saving and loading trained PyTorch models, for example
- In this particular case, saving and loading the `mask` is maybe not super useful, because it remains unchanged during training; so, for demonstration purposes, let's assume it was modified where all `1`'s were changed to `2`'s:


```python
ca_with_buffer.mask[ca_with_buffer.mask == 1.] = 2.
ca_with_buffer.mask
```




    tensor([[0., 2., 2., 2., 2., 2.],
            [0., 0., 2., 2., 2., 2.],
            [0., 0., 0., 2., 2., 2.],
            [0., 0., 0., 0., 2., 2.],
            [0., 0., 0., 0., 0., 2.],
            [0., 0., 0., 0., 0., 0.]], device='cuda:0')



- Then, if we save and load the model, we can see that the mask is restored with the modified value


```python
torch.save(ca_with_buffer.state_dict(), "model.pth")

new_ca_with_buffer = CausalAttentionWithBuffer(d_in, d_out, context_length, 0.0)
new_ca_with_buffer.load_state_dict(torch.load("model.pth"))

new_ca_with_buffer.mask
```




    tensor([[0., 2., 2., 2., 2., 2.],
            [0., 0., 2., 2., 2., 2.],
            [0., 0., 0., 2., 2., 2.],
            [0., 0., 0., 0., 2., 2.],
            [0., 0., 0., 0., 0., 2.],
            [0., 0., 0., 0., 0., 0.]])



- This is not true if we don't use buffers:


```python
ca_without_buffer.mask[ca_without_buffer.mask == 1.] = 2.

torch.save(ca_without_buffer.state_dict(), "model.pth")

new_ca_without_buffer = CausalAttentionWithoutBuffers(d_in, d_out, context_length, 0.0)
new_ca_without_buffer.load_state_dict(torch.load("model.pth"))

new_ca_without_buffer.mask
```




    tensor([[0., 1., 1., 1., 1., 1.],
            [0., 0., 1., 1., 1., 1.],
            [0., 0., 0., 1., 1., 1.],
            [0., 0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0.]])

