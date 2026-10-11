# 数据来源、许可与固定摘要

## 独立表示学习语料：LR-Sum 简体中文训练集

- 数据集卡：<https://huggingface.co/datasets/bltlab/lr-sum>
- 配置与分片目录：<https://huggingface.co/datasets/bltlab/lr-sum/tree/main/cmn_s>
- 固定 revision：`deb2e73ce53e42ceda31297a991afa138ee665a8`
- 文件：`cmn_s/train-00000-of-00001.parquet`
- 固定文件 URL：<https://huggingface.co/datasets/bltlab/lr-sum/resolve/deb2e73ce53e42ceda31297a991afa138ee665a8/cmn_s/train-00000-of-00001.parquet>
- 本地归档：`data/external/lr_sum_cmn_s_train.parquet`，1,528,153 字节，SHA-256：`687db5968b23e492bfb904dc438e6ecff95f711ab93c8ef3d4d36e086d5e8630`
- 许可：数据集卡标注 **CC BY 4.0**。再分发和改编时应按 CC BY 4.0 提供适当署名、许可链接并说明修改；此处归属数据集 `LR-Sum` / BLT Lab，链接回数据集卡。
- 许可文本：<https://creativecommons.org/licenses/by/4.0/>
- 论文：Chester Palen-Michel and Constantine Lignos. 2023. “LR-Sum: Summarization for Less-Resourced Languages.” *Findings of the Association for Computational Linguistics: ACL 2023*, pp. 6829–6844. <https://aclanthology.org/2023.findings-acl.427/>. DOI: <https://doi.org/10.18653/v1/2023.findings-acl.427>.

原始 Parquet 有 `id`、`url`、`title`、`summary`、`text` 五列。表示学习代码只把 `id` 用于来源追踪，并以正文 `text` 作为模型输入；标题、摘要、URL 和来源标签不进入预训练输入。项目不使用验证或测试 split。正文经过 Unicode NFKC 规范化、移除控制字符、空白规整、完全重复正文去重和 `jieba==0.42.1` 分词，标点单独 token 被去掉。规范化指纹对大小写、空白与 Unicode 标点不敏感。本项目对原始正文做了规范化、去重和 jieba 分词，属于对数据的处理/修改。固定 train Parquet 有 483 篇记录，其中 1 篇与此前正文完全重复，最终保留 482 篇、359,302 个 token。独立审计重建了清洗结果；与 PeopleDaily 一月语料的正文及前 200 字符精确/前缀重复均为 0，验证/测试正文完全或前缀重复也均为 0。此处只核验规范化后的精确或前 200 字符相同，不代表排除了语义近重复。详细纳入/排除清单在 `unsupervised_corpus_manifest.csv`，清洗正文在 `lr_sum_cmn_s_cleaned_train.jsonl`。

## 监督分类语料：人民日报 1998 年 1 月

使用本地人民日报 1998 分词标注文件，本项目未重复保存原始压缩包。公开语料说明：<https://github.com/chenhui-bupt/PeopleDaily1998>。此前冻结规则时使用本机文件 `作业1.3_ngram编程实验/data/人民日报1998_PFR.zip`，只读 ZIP 成员 `199801/199801.txt`，采用 UTF-8-SIG 和源文件中的词/POS 切分。原始 ZIP 大小为 20,305,578 字节、SHA-256 为 `17474bbf2b360921f2eae0089fd83e22601e5a0649943c3f59403650997b4ed2`；一月成员大小为 10,666,468 字节、SHA-256 为 `41760f40739b0f301ab29e99ab14565e7cd4b06128eb3a6ac18bce277a6075a8`。一月共 3,148 个文章 ID、19,484 条非空记录，日期为 1998-01-01 至 1998-01-31。

分类监督数据只保留现有冻结文件：`cleaned_january_articles.jsonl`、`weak_label_candidates.csv`、`label_filter_audit.csv`、`selection_rules.json` 和 `frozen_split.csv`。规则及 300 篇样本不因更换表示学习语料而重新生成。执行语料切换前记录的 SHA-256 如下；`frozen_input_sha256.json` 是供脚本与审计读取的机器版本：

| 文件 | SHA-256 |
|---|---|
| `frozen_split.csv` | `2345c1b8a8cc1818243bf8378311fcf6e46bfe532866753e565baa74a759c16b` |
| `weak_label_candidates.csv` | `d84d452efabace0d59e632b640869bd9507162112b235848a642f313487592cc` |
| `label_filter_audit.csv` | `e50e301fd02d489571addff353b5dd5b487332353979f1358ff80ff38748c72f` |
| `selection_rules.json` | `1960a825f6ee0fdda2ba5f0ed9e0fe62924e6706986636be68c56169be3ddeaf` |

`corpus_stats.json` 记录 PeopleDaily 原文清理统计；它不再是表示学习词语料来源。表示学习使用的语料由独立 LR-Sum train split 产生。原文中文章 `19980127-09-001` 的 `032/m` 记录在源文件第 20,964–20,965 行重复出现但正文不同，清洗时按源顺序保留两条。
