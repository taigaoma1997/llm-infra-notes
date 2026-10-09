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
8. 自测：做过的题和纠正
9. 下一步要想的问题

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

以上是读代码和调度器模拟得出的；真实引擎上"输出变错"的最小复现还没做。

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

## 自测：做过的题和纠正

2026-10-08：

| 题 | 我的回答 | 结果 | 纠正 |
|---|---|---|---|
| "radix cache 依赖插进树的时机，overlap 把这个时机往后推了一步"是什么意思？ | 没看懂，要求画图 | | 画成两条时间线后看懂了：overlap 开时，"查树"排在"记上一步的账"之前 |
| Y 自己那 300 个格子，代码怎么处理？ | 会被释放，然后接到前面那 300 个 token 上 | ⚠️ 一半 | 释放 ✅；"接到前面"现在的代码**没做**，这正是 #142 / #154 和 SGLang 的第 ③ 步 |
| Y 还在 decode，这些格子被别人写了，Y 读到什么？ | 别人的内容，会出问题 | ✅ | 读到的是别人的 K、V；不报错，输出悄悄变错；格子被重用之前看不出来 |

---

## 下一步要想的问题

- overlap 到底省了多少？用离线 benchmark（请求数改小）开和关各跑一次，比较吞吐。先猜：0.6B 这样的小模型，提升会大还是小？
- 照着 `normal_loop` 和 `overlap_loop`，各画 3 步的 CPU 和 GPU 时间线。
- 结束晚一步：EOS 之后那一步多算出来的 token 去哪了？同一个请求会不会收到两次"结束"消息？detokenizer 和前端怎么处理？
- A1 的最小复现：在调度器模拟里检查"还在跑的请求的 page_table 有没有指向空闲格子"；在真实引擎上用 greedy 解码证明输出会变（单独跑 vs 和同前缀的请求一起跑，并且让被释放的格子被别人重用）；然后开 issue，比较 #142 和 #154。
