# 数据来源与摘要

## 表示学习语料：LR-Sum 简体中文训练集

- 数据集卡：<https://huggingface.co/datasets/bltlab/lr-sum>
- 配置：`cmn_s`；split：`train`；固定 revision：`deb2e73ce53e42ceda31297a991afa138ee665a8`
- 固定文件：<https://huggingface.co/datasets/bltlab/lr-sum/resolve/deb2e73ce53e42ceda31297a991afa138ee665a8/cmn_s/train-00000-of-00001.parquet>
- 本地文件：`data/external/lr_sum_cmn_s_train.parquet`，1,528,153 字节，SHA-256 `687db5968b23e492bfb904dc438e6ecff95f711ab93c8ef3d4d36e086d5e8630`
- 许可：数据集卡标注 CC BY 4.0。数据归属 LR-Sum / BLT Lab；再分发及改编应按 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 署名、附许可链接并说明修改。
- 论文：Chester Palen-Michel and Constantine Lignos. 2023. “LR-Sum: Summarization for Less-Resourced Languages.” *Findings of ACL 2023*, pp. 6829–6844. <https://aclanthology.org/2023.findings-acl.427/>. DOI: <https://doi.org/10.18653/v1/2023.findings-acl.427>.

原始 Parquet 含 `id`、`url`、`title`、`summary`、`text` 列。词向量只读取 `id`（来源追踪）和正文 `text`；不使用标题、摘要、URL、标签或验证/测试 split。正文经 Unicode NFKC 规范化、控制字符移除、空白规整、规范化指纹去重及 `jieba==0.42.1` 分词，纯标点 token 被移除。483 行中 1 篇与此前正文重复，实际语料为 482 篇、359,302 个 token。上述操作构成对原数据的规范化、去重和分词处理。

## 分类语料：1998 年 1—6 月教师新闻文本

分类语料来自用户提供的 `1998-上半年-带音(1).rar`，原文件大小 12,093,296 字节、SHA-256：`30c4435bcce7d082c2175720a4549114f80656761df95d3c8b20f0ee48e5d7ab`。仓库不依赖原 RAR；实际输入归档为 `data/source/teacher_1998_h1_gb18030.tar.xz`，10,208,876 字节、SHA-256：`63ac4a05079b391662bde04240cf8995fa80075406ff10a748d7c8aeb146eb5c`。该归档保存从原压缩包提取的六份原始文本，便于在不安装 RAR 解压器的环境复现。来源摘要及成员清单也记录在 `teacher_source_lock.json`。

| 月份 | 成员 | 字节 | SHA-256 |
|---|---|---:|---|
| 1998-01 | `1998-01-2003版-带音.txt` | 5,320,005 | `37b85555df2f72642ff9815ac01f51d4f50024d80ca0e2e8a6be6f93b1038639` |
| 1998-02 | `1998-02-2003版-带音.txt` | 5,441,098 | `a050351664f459aed2247bc45cd46c25da2ce893e11c44a9dea06a54529f19d8` |
| 1998-03 | `1998-03-2003版-带音.txt` | 5,951,910 | `2f9d8ecfd92eec1b8a7009bd146da5faad218a34ae17fe460944bd772b2e97db` |
| 1998-04 | `1998-04-2003版-带音.txt` | 6,107,764 | `43462d53f2f7603168745af3200a9e6fe883ddcdf18618280358499e0ffc3fca` |
| 1998-05 | `1998-05-2003版-带音.txt` | 5,920,187 | `04ae37614a55c2418bcb5c7697c4980afa75a74734e930f06bee6e83022d0aea` |
| 1998-06 | `1998-06-2003版-带音.txt` | 5,946,378 | `f18f8f3d6e886644baf8306e1f08a31e3a8d390709782f913ae1ea12f3e97c5f` |

文本使用记录 ID `YYYYMMDD-page-article-segment`，按 GB18030 对每条记录严格解码；不替换无法解码字节。若任一记录解码失败，则剔除该文章并把源文件成员、行号、字节偏移和记录 ID 写入 `teacher_decode_errors.csv`。有 5 条错误记录，涉及 `19980323-09-005`、`19980430-03-005`、`19980507-04-007` 三篇，故这三篇整篇排除。同一记录 ID、正文相同的重复记录去重；若同 ID 正文冲突则剔除整篇。冲突记录 `19980127-09-001-032` 的两个版本及哈希见 `teacher_conflicting_records.csv`。正文完成规范化后，完全相同的重复文章在标签筛选和拆分前去重。

此教师语料由用户提供，压缩包内未附单独的公开许可或再分发声明；项目保留来源文件摘要以供核验。仓库随附归档供本课程实验复现。
