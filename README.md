# Gensim 表示学习与 fastText 新闻分类

本项目使用彼此隔离的语料完成词向量训练和弱标签分类：Gensim Word2Vec/FastText 只用 LR-Sum 简体中文 `train` 正文训练；分类样本来自用户提供的 1998 年 1—6 月教师新闻语料。实验在同一组嵌套拆分上比较 TF-IDF、冻结 Gensim 词向量均值，以及官方 fastText 的随机和预训练初始化；另用三个分层抽样种子比较固定六类测试集上的 TF-IDF 与 Gensim FastText 均值学习曲线。

## 数据与实验设计

LR-Sum 固定在 `bltlab/lr-sum` 的 revision `deb2e73ce53e42ceda31297a991afa138ee665a8`，只使用正文 `text`，不使用标题、摘要、URL、标签或验证/测试分片。483 篇原始记录去重后保留 482 篇、359,302 个 jieba token。来源、许可、处理说明和 SHA-256 见 [data/SOURCE.md](data/SOURCE.md)。

分类语料归档于 `data/source/teacher_1998_h1_gb18030.tar.xz`，覆盖 1998 年 1—6 月。解析时逐记录严格按 GB18030 解码，剔除含解码错误的整篇文章；同 ID 同文记录去重，冲突文章整体剔除；正文规范化后先去重再选样。标题按 `data/selection_rules.json` 的精确分词关键词生成单类弱标签，正文至少 80 个词。每类候选均不少于 200 篇，按固定 SHA-256 顺序各选 200 篇并拆分为 140/30/30。2、4、6 类任务使用这六类冻结样本的嵌套子集。正文分类输入最多取前 500 个源语料词，不输入标题。标签是标题规则生成的弱标签，不是人工审核的金标准。

## 模型

- unigram TF-IDF + Logistic Regression；
- Gensim Word2Vec 或 FastText 冻结词向量均值 + 训练集 StandardScaler + Logistic Regression；
- 官方 `fasttext.train_supervised` 随机初始化或由 Gensim FastText `.vec` 初始化。

Word2Vec/FastText 使用 Skip-gram、100 维、窗口 5、`min_count=3`、10 轮、单 worker、seed 42。Gensim FastText 使用 2—4 字符 n-gram、100,000 桶。官方 fastText 使用 softmax、100 维、学习率 0.5、词 bigram、单线程，字符 n-gram 关闭。每个任务的官方模型在验证集上比较 25、100、250 epoch，以 Macro-F1 选轮数，平局取较小值；测试集只用于最后评估。

六类监督训练语料的 Gensim 显式词表覆盖率为 25.19%（词类型）与 78.54%（词频）。预训练 `.vec` 只包括六类全局训练词表与 Gensim FastText 显式词表的交集，不导出验证/测试词或合成 OOV 向量。该 `.vec` 供 2/4/6 类任务共用，较小任务的共享文件可能包含来自六类全局训练集、但不在该任务训练子集中的词项。不同语料的分词边界、年代和领域差异会影响该对比。

标题命中弱标签关键词也出现在分类正文前 500 词中的比例为训练集 798/840（95.0%）、验证集 172/180（95.6%）、测试集 173/180（96.1%）。标题不输入模型，所以这不是拆分泄漏；但较高分数可能利用这些直接词面线索。可逐篇复核的统计和关键词见 `results/weak_label_cue_overlap.json`。

## 运行与核验

建议 Python 3.12；依赖版本见 `requirements.txt`。在仓库根目录运行：

```powershell
python -m pip install -r requirements.txt
python run_experiment.py
python run_learning_curve.py
python audit.py
```

`run_experiment.py` 从随附的 LR-Sum Parquet 重建规范化语料，并从教师语料归档重建样本；只有来源和语料摘要不匹配或检查点缺失时才训练 Gensim 表示模型。当前检查点与 LR-Sum 来源及参数匹配，因此普通复跑会载入模型。分类器、18 个官方 fastText epoch 候选、预测、指标、错误表和图表会重算。`run_learning_curve.py` 只读取现有冻结拆分、LR-Sum 来源核验元数据和 Gensim FastText 检查点，重算 18 组小样本/全量分类流水线，并输出逐篇预测、训练样本清单、逐组耗时、汇总表与图表；它不会重新解析语料或训练词向量。`audit.py` 会用实验所用的教师语料解析器重建并比对样本，再核验语料隔离、检查点、向量文件和 12,000 条预测，按预测重算 45 组指标，并核对弱标签关键词与实际分类输入正文的重合统计。它是结果一致性与重算审计，不是使用独立解析实现对教师语料解析器的交叉验证。

## 主要产物

- `data/source/teacher_1998_h1_gb18030.tar.xz`：实际分类使用的六个月来源文本归档。
- `data/frozen_split.csv`：1,200 篇冻结样本，含弱标签、拆分、来源 ID 和分类正文。
- `data/teacher_decode_errors.csv`、`data/teacher_conflicting_records.csv`：被剔除的乱码记录与冲突记录审计。
- `data/unsupervised_corpus.txt`、`data/unsupervised_corpus_manifest.csv`：LR-Sum 清洗、分词语料及纳入记录。
- `models/`：Gensim 检查点、训练词表交集 `.vec` 和六类官方 fastText 预训练模型。
- `results/metrics.csv`、`predictions.csv`、`fasttext_validation_curve.csv`、`audit_report.json`：结果与核验记录。
- `results/weak_label_cue_overlap.json`：标题弱标签关键词与实际分类正文前 500 词的逐篇重合统计。
- `results/data_split_counts.png`、`classification_comparison.png`、`confusion_6class_pretrained.png`：原主实验的三幅结果图。
- `run_learning_curve.py`、`results/learning_curve_samples.csv`、`results/learning_curve_predictions.csv`、`results/learning_curve_metrics.csv`、`results/learning_curve_summary.csv`、`results/learning_curve_metadata.json`、`results/learning_curve_tfidf_vs_gensim_fasttext.png`：新增标注规模实验的脚本、训练文章清单、逐篇预测、逐组时间、汇总与图表。
- [实验报告.md](实验报告.md)：完整方法、实测结果和局限。
