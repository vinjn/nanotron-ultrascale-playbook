ParallelKittens: Simple and Fast Multi-GPU AI Kernels
=====================================================

[Stuart Sul](https://stuartsul.com/), [Simran Arora](https://arorasimran.com/), [Benjamin Spector](https://benjaminfspector.com/), [Chris Ré](https://cs.stanford.edu/people/chrismre/)

_Note: This is an introduction post for our two longer blog posts, new ThunderKittens updates, new multi-GPU kernels, and our paper release. For more details:_

*   [Part 1: How do we perform multi-GPU communication using NVLink/NVSwitch in our kernels?](https://hazyresearch.stanford.edu/blog/2025-09-22-pgl)
*   [Part 2: What are the general principles for building efficient multi-GPU kernels on modern hardware?](https://hazyresearch.stanford.edu/blog/2025-11-17-fluffy-kittens)
*   [Code](https://github.com/HazyResearch/ThunderKittens)
*   [Paper](https://hazyresearch.stanford.edu/static/posts/2025-11-17-pk/ParallelKittens.pdf)

![Sequence-parallel kittens](https://hazyresearch.stanford.edu/static/posts/2025-11-17-pk/sp-kittens.png)

_Figure 1: "Sequence-parallel" kittens_

In the last few years, we have really put a lot of effort into making AI more efficient. We have worked on making AI use less compute (e.g., BASED), be more hardware-aware (e.g., FlashAttention), be more easy to map to hardware (ThunderKittens), overlap its execution sequence (Megakernels), and run efficiently on multiple vendors (ThunderMittens for Apple Silicon, HipKittens for AMD).

At this point, we believe _GPU networking_ offers many new and exciting opportunities for AI efficiency. This is especially true due to recent advancements in GPU networking hardware. For instance, with the introduction of NVSwitch 4th generation, we now have compute inside the networking fabric (_in-network compute_), we can perform asynchronous network transfers from the device-side with the Tensor Memory Accelerator (TMA) [\[1\]](https://hazyresearch.stanford.edu/blog/2025-11-17-pk#footnote-1), and we are shifting from scale-out architectures to scale-up architectures (e.g., Nvidia planning a single system with 576 GPUs by 2027). That will open up new opportunities and challenges in both AI (e.g., how do we build models that are native to such systems?) and systems (e.g., fault tolerance). It will be more than just faster data or model parallelism!

To lay early foundations for exploring these technologies, we did three things. First, [we extended ThunderKittens to support multi-GPU kernels](https://hazyresearch.stanford.edu/blog/2025-09-22-pgl). Second, [we explored hardware-driven principles that can be reused to write a variety of efficient multi-GPU kernels](https://hazyresearch.stanford.edu/blog/2025-11-17-fluffy-kittens). Third, [we built a handful of new kernels to demonstrate the approach](https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/parallel). So far, our observations are simple:

*   _Transfer mechanisms_. There are multiple ways to initiate GPU networking, each with different costs. The right choice depends on the workload and scheduling strategy.
*   _Scheduling strategies_. We can overlap communication and compute at different levels: host-device, inter-SM, or intra-SM. Modern GPUs offer abundant execution resources and many ways to orchestrate them. The goal, as always, is to keep the tensor cores busy!
*   _Design overheads_. It turns out that off-the-shelf communication libraries (e.g., NCCL, NVSHMEM) are quite slow to adapt to new hardware features. Writing basic communication kernels from scratch (with fewer than 10 lines of device code in ThunderKittens) can easily outperform them.
*   _Tiles_. Tiles are still the way to go. Tile-granularity network communication not only saturates network bandwidth but also lets us keep the familiar ThunderKittens tile abstractions that make our kernel design life simple.

While we continue exploring new directions, we are excited to share that we can already match or surpass state-of-the-art implementations across various parallel strategies with the updated ThunderKittens.

![BF16 all-reduce sum performance on 8xH100s and 8xB200s.](https://hazyresearch.stanford.edu/static/posts/2025-11-17-pk/all-reduce.png)

_Figure 2: BF16 all-reduce sum performance on 8xH100s and 8xB200s._

![BF16 all-gather + GEMM performance on 8xH100s.](https://hazyresearch.stanford.edu/static/posts/2025-11-17-pk/ag-gemm.png)

_Figure 3: BF16 all-gather + GEMM performance on 8xH100s._

![Ring Attention performance on 8xH100s (B = 16, H = 16, D = 128).](https://hazyresearch.stanford.edu/static/posts/2025-11-17-pk/ring-attn.png)

_Figure 4: Ring Attention performance on 8xH100s (B = 16, H = 16, D = 128)._

We plan to keep adding new features, such as inter-node communication (as well as cleaning up the ThunderKittens repo and finally bringing proper documentation). We also have ideas for even more exciting applications (e.g., load-balancing MoEs). That said, the current APIs and kernels are stable, and the existing functionality isn’t expected to change. We are excited to release it to the public. Feedback is always welcome! Feel free to reach out to Stuart at [ssul@cs.stanford.edu](mailto:ssul@cs.stanford.edu).

**Read more: [Part 1](https://hazyresearch.stanford.edu/blog/2025-09-22-pgl) | [Part 2](https://hazyresearch.stanford.edu/blog/2025-11-17-fluffy-kittens) | [Code](https://github.com/HazyResearch/ThunderKittens) | [Paper](https://hazyresearch.stanford.edu/static/posts/2025-11-17-pk/ParallelKittens.pdf)**  

***

\[1\] It looks like AMD is also adding TMA-like hardware features ("TDM") as well! ([link](https://github.com/triton-lang/triton/pull/8333/files))