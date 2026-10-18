# mini-sglang 问答：CUDA graph

测 PR [#160](https://github.com/sgl-project/mini-sglang/pull/160) 时问过的问题和答案，用中文写，方便自己复习。代码都对应 upstream [`9a91cfa`](https://github.com/sgl-project/mini-sglang/tree/9a91cfafe754aa85daee49998176275667eb58f2)。2026-10-18 开始，读到第 6 段（`engine/graph.py`）时接着补。CUDA graph 是什么、decode 为什么要用它，见 nano-vllm 的 [04](../nano-vllm/04-model-runner-and-cuda-graph.md)；两个项目录哪些大小、怎么补齐的对比，见 [QA-vs-nano-vllm](QA-vs-nano-vllm.md) 第 6 部分。

**目录**

1. 启动时录哪些 batch size
2. 一个 decode batch 用哪个 graph
3. PR #160 改了什么，实测
4. 为什么少录一个 bs=4，就少用 20 MiB
5. 第一次在新文件夹里跑，录 graph 为什么慢了 10 倍
6. 自测（还没答）
7. 下一步要想的问题

---

## 启动时录哪些 batch size？

[`_determine_cuda_graph_bs`](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/engine/graph.py#L49-L67)，只有 19 行：

```python
if cuda_graph_bs is not None:      # 直接给了列表，就照用
    return cuda_graph_bs

free_memory_gb = free_memory / (1 << 30)
if cuda_graph_max_bs is None:      # 没给上限：空闲显存大于 80 GiB 用 256，否则 160
    if free_memory_gb > 80:  # H200
        cuda_graph_max_bs = 256
    else:
        cuda_graph_max_bs = 160

if cuda_graph_max_bs < 1:          # 0：不用 CUDA graph
    return []

return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
```

最后一行：1、2、4 不管上限是多少都加上，之后每 8 个一张，一直到上限。

| `--cuda-graph-max-bs` | 录哪些 |
|---|---|
| 不设（我的 8 GB 卡） | 1, 2, 4, 8, 16, …, 160，一共 23 个 |
| 1（shell 模式自动设成 1，[args.py L231-L234](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/server/args.py#L231-L234)） | **1, 2, 4** |
| 3 | **1, 2, 4** |
| 8 | 1, 2, 4, 8 |
| 12 | 1, 2, 4, 8 |
| 0 | 不录 |

加粗的两行就是 PR #160 说的问题：上限设得比 4 小，还是会录到 4。shell 模式同时只跑一个请求（`max_running_req=1`），2 和 4 根本用不上。

---

## 一个 decode batch 用哪个 graph？

[graph.py L149-L166](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/engine/graph.py#L149-L166) 里两个函数：

- `can_use_cuda_graph`：是 decode，而且请求数不超过录过的最大那个（`max_graph_bs`），才用 graph。
- `pad_batch`：用第一个"不小于请求数"的录过的大小，空出来的位置补 dummy 请求。不能用 graph 的就不补，直接走 eager。

| 录了哪些 | 3 个请求 | 10 个请求 |
|---|---|---|
| [1, 2, 4]（原版，上限 3） | 补成 4 个，用 bs=4 的 graph | eager |
| [1, 2]（PR，上限 3） | eager | eager |
| [1, 2, 4, 8]（上限 12） | 补成 4 个 | eager |

所以 PR 合进去以后，上限设 3 实际只用到 2。这不是 PR 带进来的：原来的规则就是这样，上限设 12 时，9 到 12 个请求的 batch 本来就走 eager。PR 只是把"不超过上限"也用到了 1、2、4 上，前后一致。

---

## PR #160 改了什么，实测

改动只有两行：1、2、4 这三个也只保留不超过上限的。

实测（RTX 4060 Laptop，Qwen3-0.6B，`LLM(..., cuda_graph_max_bs=3)`，2026-10-18）：

| | 原版 `9a91cfa` | PR `bd37be3` |
|---|---|---|
| 录了哪些 | [1, 2, 4] | [1, 2] |
| 日志：录 graph 前后的空闲显存 | 0.53 → 0.46 GiB | 0.53 → 0.48 GiB |
| graph 私有池里的块（下一节） | small 2 MiB + **large 20 MiB** | small 2 MiB |
| PR 自带的测试 `test_graph_bs.py` | 4 个里 1 个失败：上限 1 时得到 [1, 2, 4]，应该是 [1] | 4 个全过 |

失败的正好是 shell 模式那种情况，也就是 PR 想修的场景。

---

## 为什么少录一个 bs=4，就少用 20 MiB？

不是 bs=4 的 graph 本身按比例更大，而是 **PyTorch 的显存分配器按整块向 GPU 要显存**，不是用多少要多少。

**分配器的规则**：

| 一次申请多大 | 向驱动要多大的块 |
|---|---|
| ≤ 1 MiB（"小"） | 2 MiB 的块，很多小 tensor 挤在里面 |
| 1 MiB 到 10 MiB（"大"） | 20 MiB 的块 |
| > 10 MiB | 按 2 MiB 取整 |

日志里的空闲显存是驱动那边看到的，所以它按整块减少，不按 tensor 的实际大小减少。

**录 graph 时最大的 tensor 是 logits**。`model.forward()` 返回 bf16 的 `[bs, 151936]`（[qwen3.py L77-L80](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/models/qwen3.py#L77-L80)），decode 时别的中间结果都只有几 KB 到几十 KB：

| bs | logits（bf16） | 归哪类 |
|---|---|---|
| 1 | 0.29 MiB | 小 |
| 2 | 0.58 MiB | 小 |
| 3 | 0.87 MiB | 小 |
| **4** | **1.16 MiB** | **大，要一个 20 MiB 的块** |

**两行日志之间做了三件事**（[graph.py L105-L147](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/engine/graph.py#L105-L147)）：

1. **分配 `GraphCaptureBuffer`**：里面是 fp32 的 logits `[max_bs, 151936]`，原版 2.32 MiB，PR 1.16 MiB。两个都大于 1 MiB，各要一个 20 MiB 的块，**两边一样**。
2. **每个 bs 录之前，先正常跑一遍 forward**：用的是普通的池。bs=4 的 1.16 MiB 塞进上面那个 20 MiB 块剩下的地方，不用再要新块。
3. **录 graph**：用的是 graph 自己的私有池。几个 graph 共用一个池（`pool = graph.pool()` 那句），从大到小录，所以池有多大由**最大的那个 bs** 决定。原版先录 bs=4，logits 1.16 MiB，要一个 20 MiB 的块；PR 最大是 bs=2，logits 0.58 MiB，只用 2 MiB 的小块。

**验证**：录完以后用 `torch.cuda.memory_snapshot()`，把私有池（`segment_pool_id` 不是 `(0, 0)`）里的块列出来，结果就是上一节表里的第 3 行：原版多一个 20 MiB 的块，里面只放了一个 1.16 MiB 的 tensor。日志只保留两位小数（0.01 GiB 约等于 10 MiB），0.07 和 0.05 只能看个大概；块的列表才是准的。

**列出来的块为什么都是"实际用了 0.00 MiB"？** 录 graph 时产生的中间结果，录完以后 Python 里没人引用了，分配器就把它们标成空闲。但这些块不还给驱动，也不借给别人，只属于这个池：replay 时 kernel 会往录的时候用的那些地址写，所以这块地方要一直留着。同一时刻只会 replay 一个 graph，所以几个 graph 可以轮流用同一块地方。先录最大的把池撑够，后面小的都在里面切，不用再要新块。

两边都有的另外约 50 MiB 没拆。graph 池只占其中 2 MiB，其余是第 1 步那个 20 MiB 的块，以及不经过 PyTorch 分配器的东西。我估计有 cuBLAS 的 workspace、graph 对象本身、第一次用到时才加载的 kernel，都没核实。

---

## 第一次在新文件夹里跑，录 graph 为什么慢了 10 倍？

PR 版第一次跑，录 2 个 graph 用了 15 秒（7.80 s/batch），原版录 3 个只要 1 秒多。PR 版再跑一次，只用了 1.4 秒。

mini-sglang 自己的几个 kernel（`index`、`store`）是第一次用到时现场编译的，编译结果放在 `~/.cache/tvm-ffi/`，目录名后面带一个 hash。那 15 秒里正好新出现了两个目录：12:30:59 的 `minisgl__index_…` 和 12:31:07 的 `minisgl__store_…`。同一个 kernel 在缓存里有好几份不同的 hash，数量和我用过的代码目录（upstream、learning、dev、测 PR 用的几个）对得上，看起来每个代码目录都要编译一次（hash 是怎么算的没查）。

所以比较两个版本的速度时，每个版本都要先热身跑一次。第一次起 server 时的 55 秒也是这个原因（[QA-processes](QA-processes.md)，启动日志那一节）。

---

## 自测（还没答）

1. 上限设 3 时，PR 合进去以后，3 个请求的 decode 比原版快还是慢？为什么？
2. 为什么从大到小录？反过来先录 bs=1，私有池里的块会是什么样？
3. 如果录的是 [1, 2, 3]，graph 私有池还要不要 20 MiB 的块？Qwen3-0.6B 的 bs 到多少时，logits 会超过 1 MiB？
4. shell 模式（上限 1）下，原版和 PR 的 graph 私有池里各有哪些块？

---

## 下一步要想的问题

- 两边都有的那约 50 MiB 各是什么：先把默认池里录 graph 前后多出来的块列出来，再看剩下多少不经过 PyTorch。
- 读第 6 段：replay 时输入怎么拷进 `GraphCaptureBuffer`，attention 的元数据怎么更新（`prepare_for_replay`）。
- 每个 bs 录之前，为什么要先正常跑一遍 forward？
