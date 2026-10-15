# GPU 问答：有哪些出名的线上课程

看完 [QA-gpu](QA-gpu.md) 和 [QA-triton](QA-triton.md)，还是想听人从头系统地讲一遍，就找了这些公开课，用中文记下来。2026-10-14 整理。

**我当时的问题**：有哪些很出名的线上课程，能把 GPU 硬件、kernel 和 Triton 讲清楚？

**建议的顺序**：CS336 第 5 讲 → CS336 第 6 讲 → GPU MODE 第 4 讲 → GPU MODE 第 14 讲 → 开始做 Triton-Puzzles → GPU MODE 第 34 讲（回头对照 PR #281 的 kernel）。看完这些如果还想深挖硬件，再看 CS149 或 PMPP。

---

## 先看这两套

### 1. Stanford CS336《Language Modeling from Scratch》（2025 春）

- 推荐理由：这门课本身讲大模型，硬件和 kernel 都是从"怎么让模型跑快"讲起，和我的出发点最接近。
- 主讲人：Tatsu Hashimoto 和 Percy Liang。
- **Lecture 5：GPUs**，讲 SM、warp、存储层级，以及为什么要分块和融合。
- **Lecture 6：Kernels, Triton**，演示怎么写 kernel、怎么 benchmark、怎么 profile。
- 后面还可以看 Lecture 7、8（并行）和 Lecture 10（推理）。
- 视频在 Stanford Online 的 YouTube 频道：[播放列表](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_)，课程主页 [cs336.stanford.edu/spring2025](http://cs336.stanford.edu/spring2025/)。
- 2026 春也有一版，第 5 讲变成了 "GPUs, TPUs"。这是从[第三方页面](https://summify.io/discover/xnDHaNUvHBg)看到的，没有核实。

### 2. GPU MODE 讲座系列

YouTube [@GPUMODE](https://www.youtube.com/@GPUMODE)，资料在 [gpu-mode/lectures](https://github.com/gpu-mode/lectures)。

- 推荐理由：这是做 GPU kernel 的人最常推荐的社区讲座，已经有一百多讲，挑着看就行。

| 讲 | 题目 | 有什么用 |
|---|---|---|
| 3 | Getting Started With CUDA（Jeremy Howard） | 从 Python 程序员的角度入门，很友好 |
| 4 | Intro to Compute and Memory Architecture | 正好补 SM、warp、存储层级 |
| 8 | CUDA Performance Checklist | 写 kernel 的原则，和 [QA-gpu](QA-gpu.md) 第 5 节的 ①–⑥ 是同一类东西 |
| 14 | Practitioner's Guide to Triton | Triton 入门 |
| 23 | Tensor Cores | Tensor Core 怎么用 |
| 34 | Low Bit Triton Kernels | 用 Triton 写低比特 kernel，和 PR #281 的 W8A16 直接相关 |
| 29 | Triton Internals | Triton 怎么一步步编译成机器码 |
| 35 | SGLang Performance Optimization | 后面学 SGLang 时再看 |

- 中文笔记：BBuf 的 [how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda) 仓库里有 `cuda-mode/` 目录，是这个系列的笔记。

## 想系统学

### 3. PMPP（Programming Massively Parallel Processors）+ Izzat El Hajj 的课

- PMPP 是 GPU 编程最标准的教材。El Hajj 是作者之一，他在 AUB 开的 GPU Computing 课按这本书讲，有完整录像：
  - [2020/2021 春](https://www.youtube.com/playlist?list=PLZrjSW9GrEZE07f8gLECc0tfJwNDm75RW)
  - [2026 春](https://www.youtube.com/playlist?list=PLZrjSW9GrEZE_uL0qSjc7dzFwnI2AjPm1)
- GPU MODE 第 2 讲是这本书前 3 章的串讲，可以先看这一讲，判断合不合口味。

### 4. Stanford CS149《Parallel Computing》（Kayvon Fatahalian）

- 推荐理由：它最擅长讲"硬件为什么要这样设计"，比如多线程怎么掩盖等显存的时间。
- 公开的录像是 [2023 版](https://www.youtube.com/playlist?list=PLoROMvodv4rMp7MTFr4hQsDEcX7Bx6Odp)（[课程主页](https://gfxcourses.stanford.edu/cs149/fall25)）。重点看前面讲现代多核处理器的那一讲，和 Lecture 7《GPU Architecture & CUDA Programming》。

### 5. MIT 6.5940《TinyML and Efficient Deep Learning Computing》（韩松）

- 量化讲得最好的一门课，适合在 PR #281 之后补量化的原理。
- YouTube 上有，B 站有带中文字幕的版本（见 [csdiy 的介绍](https://csdiy.wiki/en/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%B3%BB%E7%BB%9F/EML/)）。
- [2026 秋这一期](https://hanlab.mit.edu/courses/2026-fall-65940)正在上，课表上 10 月 27 日讲 "LLM Quantization and Deployment"。

## 配合着动手

- **[Triton-Puzzles](https://github.com/srush/Triton-Puzzles)**（Sasha Rush）：从最简单的题一路做到 Flash Attention 和量化矩阵乘。它用 Triton 的解释器运行，不需要 GPU。看完 CS336 第 6 讲或 GPU MODE 第 14 讲就可以开始做。
- **[GPU-Puzzles](https://github.com/srush/GPU-Puzzles)**：同一个作者，用 Numba 写，更偏 CUDA 的思维方式。
- **[Modal GPU Glossary](https://modal.com/gpu-glossary/readme)**：不是课，是一本可以互相跳转的 GPU 术语手册，适合遇到名词时查。也有[中文翻译版](https://gpu-glossary-zh.readthedocs.io/)。

## 来源

- [gpu-mode/lectures](https://github.com/gpu-mode/lectures)
- [gpu-mode/resource-stream](https://github.com/gpu-mode/resource-stream)
- [CS336 Spring 2025](http://cs336.stanford.edu/spring2025/)
- [summify.io: CS336 Spring 2026 Lecture 6](https://summify.io/discover/xnDHaNUvHBg)
- [Izzat El Hajj courses](https://ielhajj.github.io/courses.html)
- [Stanford CS149 Fall 2025](https://gfxcourses.stanford.edu/cs149/fall25)
- [Triton-Puzzles](https://github.com/srush/Triton-Puzzles)
- [GPU Glossary](https://modal.com/gpu-glossary/readme)
- [gpu-glossary-zh](https://gpu-glossary-zh.readthedocs.io/)
- [MIT 6.5940 (csdiy)](https://csdiy.wiki/en/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%B3%BB%E7%BB%9F/EML/)
- [MIT 6.5940 Fall 2026](https://hanlab.mit.edu/courses/2026-fall-65940)
- [BBuf/how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda)
