# mini-sglang 问答：overlap scheduling

学 overlap scheduling 时问过的问题和答案，用中文写，方便自己复习。代码都对应 upstream [`9a91cfa`](https://github.com/sgl-project/mini-sglang/tree/9a91cfafe754aa85daee49998176275667eb58f2)。2026-10-08 整理，之后陆续补充。radix cache 本身在 [QA-kv-cache.md](QA-kv-cache.md)，进程结构在 [QA-processes.md](QA-processes.md)。

**目录**（由浅入深）

1. overlap scheduling 要解决什么问题
2. 怎么做到的：先发下一步，再处理上一步
3. 上一步生成的 token 还不知道，下一步怎么出发
4. 代价：调度器手里的信息晚一步
5. radix cache 加上 overlap：为什么会命中不了
6. 命中不了之后：PR #142 / #154 修的那个 bug
7. 这个 bug 是怎么来的：mini-sglang 的来历，以及 SGLang 是怎么写的
8. 读 `overlap_loop` 的代码：差在哪、CPU 在等谁、多张卡怎么等（2026-10-15）
9. 复现 A1：从账本错到算错（#168，2026-10-17）
10. 自测：做过的题和纠正
11. 下一步要想的问题

---

## overlap scheduling 要解决什么问题

每一步都有两部分：

- **CPU 部分**：选这一步跑谁、准备 attention 的元数据；GPU 算完以后把 token 拷回来、接到每个请求后面、判断结束、插进 radix 树、发消息给 detokenizer。这些都是 Python。
- **GPU 部分**：跑模型。

不重叠的话两部分轮流来，CPU 干活时 GPU 在等：

```
不重叠（normal_loop）：
CPU: [准备1][发出]          [处理1][准备2][发出]          [处理2][准备3]…
GPU:             [算 第1步]                     [算 第2步]
                           ↑ GPU 空着                     ↑ 又空着

重叠（overlap_loop）：
CPU: [准备1][发出][准备2][发出][处理1][准备3][发出][处理2]…
GPU:             [算 第1步   ][算 第2步      ][算 第3步     ]
```

在 RTX 4060 Laptop 上，Qwen3-0.6B 一个请求 decode 一步只要约 8 ms。Python 的 CPU 部分哪怕只有几毫秒，比例也不小。模型越小、batch 越大（CPU 要处理的请求越多），重叠越划算。

---

## 怎么做到的：先发下一步，再处理上一步

[scheduler.py L83-L106](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L83-L106)，删减后：

```python
for msg in self.receive_msg(...): self._process_one_msg(msg)   # 收新请求
forward_input = self._schedule_next_batch()                     # 选第 k 步跑谁：第 k−1 步的结果还没处理
with self.engine_stream_ctx:
    self.engine.stream.wait_stream(self.stream)                 # 等第 k 步的输入在 GPU 上准备好
    ongoing_data = (forward_input, self._forward(forward_input))# 把第 k 步交给 GPU，马上返回
self._process_last_data(last_data)                              # 这时才处理第 k−1 步的结果
return ongoing_data
```

不重叠的 `normal_loop`（[L108-L118](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L108-L118)）是"选第 k 步 → 跑第 k 步 → 处理第 k 步"。

GPU 的活是异步的：`_forward` 只把 kernel 排进队列就返回，所以 CPU 处理第 k−1 步结果时，GPU 正在算第 k 步。

两个 CUDA stream：调度器自己的 stream（[L53](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L53)）负责把下一步的输入拷到 GPU；engine 的 stream 负责跑模型；`wait_stream`（[L102](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L102)）保证模型开跑前输入已经拷好。

---

## 上一步生成的 token 还不知道，下一步怎么出发？

**这些 token 根本不经过 CPU。**

| 做什么 | 在哪 | 代码 |
|---|---|---|
| 第 k−1 步采样出的 token 直接写进 GPU 上的 `token_pool` | GPU | [scheduler.py L231](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L231) |
| 第 k 步的输入直接从 `token_pool` 里取 | GPU | [L229](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L229) |
| token 同时以非阻塞方式拷回 CPU，拷完记一个事件 | GPU → CPU | [engine.py L203-L205](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/engine/engine.py#L203-L205) |
| CPU 处理结果前，只等这个拷贝完成 | CPU | [scheduler.py L143](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L143) |

GPU 上的队列按顺序执行：先写 `token_pool`，再读 `token_pool`，所以第 k 步一定拿得到第 k−1 步的 token。CPU 需要 token 的值，只是为了事后记账（接到请求后面、判断 EOS、发给 detokenizer），这些都放到 `_process_last_data` 里晚一步做。

---

## 代价：调度器手里的信息晚一步

1. **结束要晚一步才知道。** 第 k−1 步生成了 EOS，可第 k 步在 CPU 知道之前就发出去了，这个请求还在里面。`_process_last_data` 里的 `finished_reqs`（[L159](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L159)）就是防止同一个请求被释放两次的。
2. **插进 radix 树也晚一步。** 下一节专门讲。
3. **正确性更难保证。** 资源可能在 GPU 还在用时被 CPU 释放或改写。`ForwardInput` 把正在跑的那一步的张量留着（[L34](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L34) 的注释 "to avoid IMA"，IMA 是 illegal memory access）。[PR #103](https://github.com/sgl-project/mini-sglang/pull/103) 修的就是这类时间差：FlashInfer 的 plan 复用了一块 pinned 内存，上一次的异步拷贝还没完成就被改写了。

---

## radix cache 加上 overlap：为什么会命中不了

两条规则：

1. 新请求进场时**查树**（`_schedule_next_batch` → `match_req`），只查得到**已经写进树**的东西。
2. 一个请求的 prompt 写进树，发生在它那一步**记账**的时候（`_process_last_data` 里的 `cache_req(finished=False)`，[L164](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L164)），不是一算出来就写。

关键是每一轮里"查树"和"记账"谁先谁后。X、Y 开头都是同一段 300 token 的 system prompt；X 在第 1 步进场，Y 在第 2 步进场：

```
overlap 关：先记账，再查树

第1轮  ① 选第1步：X 进场（树是空的）
       ② GPU 算第1步：算 X 的 350 个 token
       ③ 记第1步的账：X 写进树          ← 树里有 X 了 ✅
第2轮  ① 选第2步：Y 进场，查树 → 查到 X 的 300 个 ✅ 命中
       ② GPU 算第2步：只算 Y 剩下的 50 个
       ③ 记第2步的账：Y 写进树
```

```
overlap 开：先查树，再记账

第1轮  ① 选第1步：X 进场（树是空的）
       ② 发出第1步（GPU 开始算 X）
       ③ 记第0步的账：没有
第2轮  ① 选第2步：Y 进场，查树 → 树还是空的 ❌ 没命中，Y 只能自己算全部 350 个
       ② 发出第2步
       ③ 记第1步的账：X 这时才写进树    ← 晚了，Y 已经查过了
第3轮  ① ②  …
       ③ 记第2步的账：Y 写进树 → 发现前 300 个树里已经有了（X 的）
```

这就是"overlap 把写进树的时机往后推了一步"：X 算完以后，紧挨着的那一步里进场的请求看不到 X 的前缀。看不到别人前缀的"盲区"，overlap 关时是**同一步**进场的请求（和 nano-vllm [#219](https://github.com/GeeeekExplorer/nano-vllm/issues/219) 一样），overlap 开时再加上**下一步**。

调度器模拟的真实输出（overlap 开）：

```
step 1 发出  | free 650, … protected   0 | prefill X[0->350]
step 2 发出  | free 300, … protected   0 | prefill Y[0->350]      ← Y 没命中，350 个全自己算
step 1 处理完| free 300, … protected 350 |                       ← X 这时才写进树
step 3 发出  | free 298, …               | decode X Y
step 2 处理完| free 598, … protected 401 |                       ← free 一下多了 300
```

---

## 命中不了之后：PR #142 / #154 修的那个 bug

Y 没命中，把那 300 个 token 自己又算了一遍。格子是依次分配的，所以同一段 system prompt 在 KV cache 里有了**两份**：

```
格子：  0 ────── 299 │ 300 ─ 349 │ 350 ────── 649 │ 650 ─ 699 │ 700 … 999
        X 的 system  │ X 自己的   │ Y 的 system    │ Y 自己的   │ 空闲
        prompt       │ 50 个      │ prompt（又一份）│ 50 个      │
```

第 2 步记账时，`cache_req` 把 Y 的 prompt 插进树，发现前 300 个已经有了（[cache.py L67-L79](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/cache.py#L67-L79)，代入 Y 的值）：

```python
page_indices = self.page_table[Y 那一行, :]           # 350, 351, …, 649, 650, …
old_handle   = Y 进场时的匹配结果                       # 没命中：old_handle.cached_len = 0
cached_len, new_handle = insert_prefix(Y 的 prompt, page_indices)   # 树里已有前 300 个：cached_len = 300
self.unlock(old_handle)
self._free(page_indices[0 : 300])                     # 释放 350–649（上面 free 多出来的 300 个）
...
req.cache_handle = new_handle                         # handle 换成树里的（指向 X 的 0–299）
```

Y 的 page_table 那一行**没有被改**：

```
树里的 system prompt：      格子 0 – 299（X 的）
Y 的 page_table（现在）：   格子 350 – 649   ← 已经还回空闲池，Y 还指着它们
Y 的 page_table（修复后）：  格子 0 – 299     ← #142 / #154 加的：先改指，再释放
```

Y 还在 decode，每一步的 attention 都要读这 300 个格子：

- 格子刚还回去时，里面还是 Y 自己算的那份 KV，结果是对的，**看不出问题**；
- 一旦被别的请求拿去写，Y 读到的就是别人的 K、V，输出**悄悄变错，不报错**（格子号还在合法范围里，不是 illegal memory access）；
- 所以只有显存紧张、格子很快被重用时才会出错，很难发现。#142 的作者是在 Qwen3-32B 上反复跑 AIME 长推理时遇到的。

以上是读代码和调度器模拟得出的。2026-10-17 在真实引擎上复现了"输出变错"，开成了 [#168](https://github.com/sgl-project/mini-sglang/issues/168)，见第 9 节。

---

## 这个 bug 是怎么来的

**mini-sglang 是什么。** LMSYS 的[官方博客](https://www.lmsys.org/blog/2025-12-17-minisgl/)（2025-12-17，作者 Ziyi Xu，即维护者 DarkSharpness）：SGLang 已经将近 30 万行 Python，mini-sglang 是从 SGLang 派生的 5 千行精简版，目的是**教学**和**快速做研究原型**；它最初就是团队验证新想法用的原型，2025 年夏天还在上海交大当过实验课的教材。

| 时间 | 发生了什么 |
|---|---|
| 2025-09-08 | 第一次提交 |
| 2025-12 | 公开发布，这个月 82 次提交 |
| 2026-02 | 三周里加了 MoE、page_size > 1、FA4 和 TRTLLM 后端、请求取消、**缓存还没结束的请求** |
| 2026-03-12 | #103 修了 overlap 的竞态 |
| 2026-05-17 | 最后一次合并 |

**引入这个 bug 的提交。** 在 [`c7f800d`](https://github.com/sgl-project/mini-sglang/commit/c7f800d)（[PR #86](https://github.com/sgl-project/mini-sglang/pull/86)，2026-02-26）之前，只有**已经结束**的请求才会被插进树（`free_and_cache_finished_req`），"树里已经有了就释放自己那份"对结束的请求是安全的，没人会再读那些格子。PR #86 的标题是 "[Minor] Style cleanup"，说明写的是 "Several renaming. Part of #82"，但其中一个提交是 "[feature] support cache unfinished request"：prefill 一算完就把**还在跑**的请求插进树，沿用了同一句"释放重复的格子"，没有先改 page_table。

**SGLang 是怎么写的。** SGLang [v0.3.0](https://github.com/sgl-project/sglang/blob/v0.3.0/python/sglang/srt/mem_cache/radix_cache.py#L121-L147)（2024-09-19）的 `cache_unfinished_req`，释放之后紧接着把树里那份格子号写回请求自己的那一行：

```python
new_prefix_len = self.insert(token_ids, kv_indices.clone())
self.token_to_kv_pool.free(kv_indices[len(req.prefix_indices) : new_prefix_len])   # ① 释放重复的格子

# The prefix indices could be updated, reuse it
new_indices, new_last_node = self.match_prefix(token_ids)                          # ② 重新查树
self.req_to_token_pool.req_to_token[
    req.req_pool_idx, len(req.prefix_indices) : len(new_indices)
] = new_indices[len(req.prefix_indices) :]                                         # ③ 写回这个请求的那一行
```

| SGLang | mini-sglang |
|---|---|
| `req_to_token_pool` | page_table |
| `req.prefix_indices` | `old_handle` |
| `cache_unfinished_req`（现在的 main 叫 `checkpoint()`，写回那一步还在） | `cache_req(finished=False)` |

mini-sglang 有 ①，没有 ②③。#142 和 #154 补的就是 ③。

**为什么这类 bug 不容易被发现。**

| 情况 | 依据 |
|---|---|
| 主要是一个人写的 | 主分支 171 次提交里 122 次来自维护者 |
| 功能加得快 | 2026 年 2 月三周里加了上面那一批 |
| 没有跑测试的 CI | 仓库唯一的工作流是 Copilot 自动给 PR 写 review；共 7 个测试文件、11 个测试函数，没有一个检查"开着 radix cache 时输出对不对" |
| 引入的改动没写在 PR 说明里 | PR #86 的标题和说明只提到改名，合并时没有 review 记录 |
| 主要在高端卡上测 | 维护者在 [#89](https://github.com/sgl-project/mini-sglang/pull/89) 里说没在 Hopper 之前的卡上测过；#58、#67、#89 都是用户在 Ampere / Ada 卡上发现的 |
| bug 本身不报错 | 只有格子被重用后输出才变错；博客里和 SGLang 的在线性能对比，两边都关掉了 radix cache（`--cache naive`、`--disable-radix`） |

对照着 SGLang 读 mini-sglang 很有用：mini-sglang 简化掉的地方，往往正是 SGLang 处理边界情况的地方。

---

## 读 `overlap_loop` 的代码：差在哪、CPU 在等谁、多张卡怎么等

2026-10-15 开始按阅读路线逐段读代码，读到 `overlap_loop` 时问的几个问题。

### 8.1 `overlap_loop` 和 `normal_loop` 看起来差不多，为什么前者就是 overlap？

两段代码几乎一样，差别只在最后处理的是**哪一批**（[scheduler.py L83-L118](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L83-L118)）：

```python
# normal_loop
ongoing_data = (forward_input, self._forward(forward_input))
self._process_last_data(ongoing_data)   # 处理刚发出去的这一批

# overlap_loop
ongoing_data = (forward_input, self._forward(forward_input))
self._process_last_data(last_data)      # 处理上一圈留下来的那一批
return ongoing_data                     # 这一批留到下一圈
```

再加上两件事，overlap 就成立了：

- `_forward` 只是把 kernel 排进 GPU 的队列，不等它算完；
- CPU 只在一个地方等 GPU：`_process_last_data` 开头的 `copy_done.synchronize()`（[L143](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L143)）。

`normal_loop` 发出第 k 步就马上等它算完，算完 GPU 的队列是空的；`overlap_loop` 发出第 k 步以后去等第 k−1 步，第 k−1 步一算完，第 k 步已经排在后面，GPU 直接接着算。

```
S = 调度  L = 发出  P = 处理结果  - = CPU 在等  # = GPU 在算  . = GPU 空着

normal_loop
CPU  S1 L1 -- -- -- -- P1 P1 S2 L2 -- -- -- -- P2 P2
GPU  .. .. #1 #1 #1 #1 .. .. .. .. #2 #2 #2 #2 .. ..

overlap_loop
CPU  S1 L1 S2 L2 -- -- P1 P1 S3 L3 P2 P2 S4 L4 P3 P3
GPU  .. .. #1 #1 #1 #1 #2 #2 #2 #2 #3 #3 #3 #3 #4 #4
```

### 8.2 按 GPU 的一步来看更直观

按代码的一圈（选、发出、处理上一步）来看很绕。我换了一种切法：把"GPU 算第 k 步"当成一段，这段时间里 CPU 依次做三件事：

1. 处理第 k−1 步的结果：接上新 token、插进 radix 树、判断谁结束了；
2. 收消息，选第 k+1 步、分格子、发出去，排在第 k 步后面；
3. 等第 k 步算完。算完的那一刻，GPU 接着算已经排好的第 k+1 步。

最开始 GPU 还没有活，CPU 先准备第 1 步，相当于 ramp up。没有哪段代码负责"衔接"：第 k 步早在 CPU 去等第 k−1 步之前就排进队列了。代码里的一圈，等于这个看法里第 k 段的后半加上第 k+1 段的前半。第 9 节都按这个切法写。

### 8.3 还没处理第 k 步的结果，凭什么能准备第 k+1 步？

因为 CPU 准备的是**地址**，不是**内容**：

| 第 k+1 步要的 | 从哪来 | 要等第 k 步吗 |
|---|---|---|
| 每个请求算哪个位置、要几个新格子 | 发出第 k 步时，`complete_one` 已经在 CPU 上把长度加好（[engine.py L199-L200](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/engine/engine.py#L199-L200)） | 不用 |
| 谁到了 `max_tokens` | 发出时 `filter_reqs` 已经把它拿掉 | 不用 |
| 新 K/V 写进哪些格子、attention 读哪些格子 | 从 page table 抄一份编号 | 只要编号，不要数据 |
| 输入的 token | 第 k 步在 GPU 上把采样结果写进 `token_pool`，第 k+1 步在 GPU 上从那里读（[L229-L231](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L229-L231)）；同一条队列，先写后读 | 要，但这一读排在 GPU 的队列里 |

CPU 唯一不知道的是 token 的**值**，所以只有要看值、看结果的事会晚一步：遇到 eos，和插进 radix 树。

另外有两条 stream：`_prepare_batch` 在调度器自己的 stream 上把编号拷到 GPU，模型在 engine 的 stream 上算；`self.engine.stream.wait_stream(self.stream)`（[L102](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L102)）让 engine 的队列先等这些拷贝做完。这一步只在 GPU 上排队，不会让 CPU 停下来。

### 8.4 `copy_done.synchronize()` 怎么知道等的是哪一步？

每一步发出时都新建一个 event，**记在"拷回 CPU"后面**（[engine.py L203-L205](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/engine/engine.py#L203-L205)）：

```python
next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)   # 拷回 CPU，也只是排进队列
copy_done_event = torch.cuda.Event()
copy_done_event.record(self.stream)                              # 在 engine stream 的队列里放一个标记
```

这个 event 放在 `ForwardOutput` 里，经过 `ongoing_data` 一路传到下一圈的 `last_data`。它标记的是队列里的一个**位置**：

```
engine stream：… 第 k-1 步 │ 拷回 │ E(k-1) │ 第 k 步 │ 拷回 │ E(k)
                                  ↑ E(k-1).synchronize() 只等到这里
```

所以它只等第 k−1 步（和更早的），不等已经排在它后面的第 k 步。换成别的写法就不一样了：`torch.cuda.synchronize()` 会等这张卡上所有 stream 的所有任务，overlap 就没了；nano-vllm 的 `.tolist()` 等的是当前 stream 上到目前为止的全部任务。

### 8.5 有很多张 GPU，要等所有卡都做完吗？

不需要有谁去等所有卡。等待分三层：

| 谁等谁 | 怎么等 |
|---|---|
| CPU 等 GPU | 每张卡一个 scheduler 进程，各自只等自己那张卡上的 `copy_done` |
| GPU 等 GPU | 张量并行时，每层要把各张卡的部分结果加起来：o 投影和 MLP 的 down 投影之后各一次 all_reduce（[linear.py](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/layers/linear.py#L91-L127)），embedding 再一次，LM head 一次 all_gather。Qwen3-0.6B 一步有 58 次。通信和 kernel 一样排在 GPU 的队列里：一张卡执行到这里，要等其他卡也执行到，才能往下走 |
| CPU 等 CPU | 每一圈收消息时，rank 0 先用 gloo 广播"这一圈有几条消息"，再把原始消息转发出去；其他 rank 在这个广播上等（[io.py L88-L122](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/io.py#L88-L122)） |

rank 0 的 E(k) 排在这一步最后一次通信后面，它完成的时候，所有卡都已经过了那次通信，所以等于间接等到了所有卡。代价是每一步的速度取决于最慢的那张卡。

每张卡各自调度，靠的是"输入一样、规则一样，选出来的 batch 就一样"。有一个地方原来不一样：`running_reqs` 是一个 set，不同进程里遍历的顺序不同，decode batch 里请求的顺序就对不上。all_reduce 是按位置相加的，会把不同请求的部分结果加在一起，**不报错，但算错**。PR #113 的修法是取 batch 之前先按 uid 排序（[decode.py L35](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/decode.py#L35)）。采样时 logits all_gather 到每张卡，每张卡用同一个随机种子自己采样，所以采样出的 token 不用在卡之间传。

---

## 复现 A1：从账本错到算错（#168）

2026-10-16 到 10-17。英文整理在 [issues/168](issues/168-radix-dedup-stale-page-table.md)，issue 是 [#168](https://github.com/sgl-project/mini-sglang/issues/168)。这里记下过程中想明白的几件事。

### 9.1 先用打印看清每一步

在调度器里加了几行打印：进场时匹配到几个、分到哪些格子；插树时树里已经有几个、还了哪些格子；每一步处理完，再检查一遍"还在跑的请求，有没有指着空闲列表里的格子"。

场景：两个请求 uid0、uid1，分别是 `[7]×300 + [1]×50` 和 `[7]×300 + [2]×50`，每步 prefill 预算 350。overlap 开的时候：

| GPU 在算 | CPU 处理上一步 | CPU 准备下一步 |
|---|---|---|
| （还没开始） | | 第 1 步：uid0 进场，命中 0，拿格子 0–349 |
| 第 1 步：uid0 prefill | | 第 2 步：uid1 进场，**树还是空的**，命中 0，拿格子 350–699 |
| 第 2 步：uid1 prefill | uid0 进树 | 第 3 步：decode，拿 700、701 |
| 第 3 步：decode | uid1 进树：前 300 个已经有了 → **还回 350–649，uid1 那一行没改** ⚠️ | 第 4 步：decode |

overlap 关的时候，uid0 进树排在 uid1 进场之前，uid1 命中 300，只拿 50 个新格子，什么都不还。

### 9.2 GPU 到底读什么：为什么"改树、还格子"不影响正在算的那一步

GPU 算一步只读三样东西：

1. 格子里的 K/V 数据；
2. 这一步自己的编号清单，是发出前从 page table **抄出来的拷贝**：新 K/V 写到哪（`out_loc`，[scheduler.py L210](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L210)），attention 读哪些格子（`indices`，[fi.py L215](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/attention/fi.py#L215)）；
3. `token_pool` 里的输入 token。

radix 树和空闲列表都不读。"还格子"只是把编号放回空闲列表，格子里的数据一个字节都没动。所以第 3 步是安全的。等到**准备 uid1 的下一步**时，又从那一行抄编号，bug 才被带进去。

### 9.3 账本错了，不等于算错了

上面那次运行，uid1 从头到尾都指着 350–649，但输出完全正确。要真的算错，三件事得同时成立：

1. 格子已经还回去，请求还指着它们：✅ 已经发生；
2. 这些格子被分给别的请求，写进了别的 K/V：❌ 还回去的格子接在空闲列表**末尾**，前面还有四万多个，轮不到；
3. 那之后这个请求还要读这些位置：❌ uid1 只生成了 5 个 token。

### 9.4 为什么只靠两个请求不行：预留把重复的那份也算进去了

我最开始的想法是把 KV cache 开小，让 uid0、uid1 自己 decode 时拿到 350–649。这走不通，原因在进场时的预留（[prefill.py L50](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/prefill.py#L50)）：

- uid1 要在第 2 步进场，KV cache 至少要有 699 + 2 × `max_tokens` 个格子；
- uid1 是按"350 个全都要算"来预留的，多算的那一份早就算在账上了。所以两个请求之后 decode 要的格子，空闲列表开头那一段一定够，轮不到末尾。
- 格子再少，uid1 第 2 步进不来；等它进场时 uid0 早就在树里了，它会命中 300，什么都不还。

所以必须有**第三个请求**来拿这些格子。巧的是，预留正好帮了忙。KV 开 740 个格子，uid0、uid1 各生成 12 个，再加一个 200 token、`max_tokens` 为 5 的请求 W：

| 条件 | 算式 | 结果 |
|---|---|---|
| uid1 第 2 步进得来 | 350 + 12 + 11 = 373 ≤ 390 | ✅ |
| W 第 3 步进不来（这时格子还没还） | 205 + 22 = 227 > 40 | ✅ 被挡住 |
| W 第 4 步进得来（刚还回 300 个） | 205 + 20 = 225 ≤ 338 | ✅ |
| W 拿得到末尾的格子 | 要 200 个，开头只剩 38 个 | W 拿到 702–739 和 **350–511** |

预留一直挡着 W，uid1 一还格子就放它进来，它拿到的正好就是那些格子。

### 9.5 第一次：覆盖发生了，输出没变

用 9.1 那组 `[7]/[1]/[2]` 的 prompt 跑。打印显示 W 确实拿到了 350–511，也和 uid1 "共用"了这些格子，但四次运行的输出完全一样：overlap 开、overlap 关、默认格子数、uid1 单独跑，uid1 都是一串 `2`。

原因：uid1 最后那 50 个 `2` 没被覆盖，模型看着它们，就接着输出 `2`；前面 300 个位置里是什么，不影响答案。**要让覆盖在输出上看得出来，答案必须依赖被覆盖的那些位置。**

### 9.6 第二次：一张密码表

共用的 300 个 token 换成一张表：`The secret code of Alice is 4817.`、`The secret code of Bob is 2093.`……uid1 问 Bob 的密码，Bob 那一行在 token 位置 24。长度没变，所以调度和上面一模一样：

| 运行 | uid1，应该答 2093 |
|---|---|
| overlap 开，740 个格子 | `' 23043555555'` ❌ |
| overlap 关；默认格子数；uid1 单独跑 | `' 2093.\nAnswer: 209'` ✅ |
| PR #142、PR #154 | `' 2093.\nAnswer: 209'` ✅ |

uid1 的前两个 token（`' 2'`）是在 W 覆盖之前算的，所以是对的；从第三个开始，它读到的是 W 那段讲天气的话的 K/V，答不出来了。uid0 和 W 每次都答对，因为它们读的格子没人写过。

### 9.7 两个 PR 怎么修，为什么一定要先 `clone`

两个 PR 都在还格子之前，先把这一行的位置 0–299 改成指向树里那一份（格子 0–299）。改完以后，uid1 和 uid0 共用树里那一份，而且有 `self.lock(new_handle)` 锁着，不会被驱逐。#154 的写法：

```python
dup_slice = slice(old_handle.cached_len, cached_len)        # 重复的位置：0–299
dup_indices = page_indices[dup_slice].clone()               # 先把旧编号拷一份：350–649
if not finished:
    page_indices[dup_slice].copy_(new_handle.get_matched_indices()[dup_slice])   # 改指到 0–299
self._free(dup_indices)                                     # 还的是拷贝：350–649
```

为什么一定要先 `clone`：

- `page_indices` 是 page table 的**视图**，改它就是在改 page table；
- `cache_req` 跑在 `lazy_free_region` 里面（[cache.py L93-L104](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/cache.py#L93-L104)），`_free` 只是把视图记进一个列表，等这一批全部处理完才读出里面的数、还回空闲列表；
- 不 `clone` 的话，改指以后这个视图看到的已经是 0–299，最后还回去的就成了**树里正在用的那一份**，而 350–649 永远没人还。

缩小成 3 个格子跑一遍就看清了：

```
这一行 = [350, 351, 352, 650]，树里那一份 = [0, 1, 2]，空闲列表 = [700, 701]
先 clone：这一批处理完，空闲列表 = [700, 701, 350, 351, 352]   ✅
不 clone：这一批处理完，空闲列表 = [700, 701, 0, 1, 2]         ❌
```

#142 只在要改指的时候才 `clone`。已经结束的请求，那一行在这一批里不会被改，用视图也读得对，所以两个 PR 的效果一样。

---

## 自测：做过的题和纠正

2026-10-08：

| 题 | 我的回答 | 结果 | 纠正 |
|---|---|---|---|
| "radix cache 依赖插进树的时机，overlap 把这个时机往后推了一步"是什么意思？ | 没看懂，要求画图 | | 画成两条时间线后看懂了：overlap 开时，"查树"排在"记上一步的账"之前 |
| Y 自己那 300 个格子，代码怎么处理？ | 会被释放，然后接到前面那 300 个 token 上 | ⚠️ 一半 | 释放 ✅；"接到前面"现在的代码**没做**，这正是 #142 / #154 和 SGLang 的第 ③ 步 |
| Y 还在 decode，这些格子被别人写了，Y 读到什么？ | 别人的内容，会出问题 | ✅ | 读到的是别人的 K、V；不报错，输出悄悄变错；格子被重用之前看不出来 |

2026-10-15 到 10-17：

| 题 | 我的回答 | 结果 | 纠正 |
|---|---|---|---|
| `overlap_loop` 和 `normal_loop` 看起来差不多，为什么前者就是 overlap？ | 看不出差别 | ⚠️ | 差在最后处理的是哪一批；第 k 步在 CPU 去等第 k−1 步之前就排进队列了（8.1） |
| 用什么切法看 overlap 最直观？ | 按 GPU 的一步：GPU 算第 k 步时，CPU 处理第 k−1 步、准备并发出第 k+1 步，再等第 k 步 | ✅ | 比按代码的一圈直观得多，第 9 节就是按这个切法写的 |
| 多张 GPU 时，CPU 要等所有卡都做完吗？ | 不知道怎么实现 | | 每张卡一个进程，只等自己那张卡；卡和卡之间在通信那一步互相等；进程之间每一圈广播一次消息条数（8.5） |
| CPU 还没把 uid1 插进树，GPU 这一步会不会没有数据？改树会不会让正在算的那一步出错？ | 担心会 | ❌ | GPU 不读树，也不读空闲列表，只读格子里的数据和这一步自己那份编号拷贝（9.2） |
| 开了 overlap，输出也没错，是不是就没问题？ | 看起来没问题 | ❌ | 账本已经错了，只是格子还没被别人写（9.3） |
| 把 KV 开小，让 uid0、uid1 自己 decode 时覆盖那些格子，行不行？ | 行 | ❌ | 预留把重复的那份也算进去了，两个请求自己永远轮不到末尾那些格子，要第三个请求（9.4） |

---

## 下一步要想的问题

- overlap 到底省了多少？用离线 benchmark（请求数改小）开和关各跑一次，比较吞吐。先猜：0.6B 这样的小模型，提升会大还是小？
- 照着 `normal_loop` 和 `overlap_loop`，各画 3 步的 CPU 和 GPU 时间线。
- 结束晚一步：EOS 之后那一步多算出来的 token 去哪了？同一个请求会不会收到两次"结束"消息？detokenizer 和前端怎么处理？
- 同一步 prefill（overlap 关）：两个前缀相同的请求在**同一步**里都没命中，后插树的那个会不会也还掉自己那份？成立的话，A1 就不只是 overlap 的问题。改复现脚本，让 uid0、uid1 在同一步进场，试一试。
- [#149](https://github.com/sgl-project/mini-sglang/issues/149) 的修复 [PR #165](https://github.com/sgl-project/mini-sglang/pull/165) 检查的是 radix 树和空闲列表。A1 错的是 page table，它能抓到吗？
- 练"读代码看出 bug"：PR #111 的标题是"overlap 下，请求在结束的那一步被 abort，会被释放两次"。先不看 diff，自己读 `_process_last_data` 和处理 abort 的代码，推出 bug 在哪，再看 diff 对答案。
