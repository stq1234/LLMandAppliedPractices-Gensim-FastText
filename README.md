# Gensim 表示学习与官方 fastText 文本分类

本项目把表示学习语料和监督分类语料分开：Gensim Word2Vec/FastText 只使用独立的 LR-Sum 简体中文训练集正文；分类任务使用冻结的人民日报 1998 年 1 月六类弱标签样本。官方 fastText 的随机初始化和 Gensim FastText 词向量初始化均在同一套分类划分上比较。

## 数据与实验设计

表示学习使用 Hugging Face `bltlab/lr-sum` 的 `cmn_s` 配置 `train` split，固定 revision 为 `deb2e73ce53e42ceda31297a991afa138ee665a8`。随项目保存的原始 Parquet 含 483 行。预处理会先读取 Parquet 整表，再选择 `id` 与 `text` 构造清洗记录；模型句子只由正文 `text` 生成，标题、摘要、URL 和数据集标签不进入训练输入。正文做 Unicode NFKC 规范化，移除控制字符、合并空白，并按先做 NFKC 与大小写折叠、再移除空白和 Unicode 标点后的正文指纹去重，最后用 `jieba==0.42.1` 分词。当前冻结语料包含 482 篇正文、359,302 个 token；1 篇与前文完全重复而排除。与 PeopleDaily 一月正文及其前 200 字符的完全/前缀重复均为 0，验证/测试正文没有进入预训练语料。原始来源、许可、revision 和 SHA-256 记录见 [data/SOURCE.md](data/SOURCE.md)。

监督样本来自 `作业1.3_ngram编程实验/data/人民日报1998_PFR.zip` 中的 `199801/199801.txt`，使用原始词/POS 切分。`data/frozen_split.csv`、弱标签候选、筛选审计和规则在更换表示学习语料前冻结；其 SHA-256 保存在 `data/frozen_input_sha256.json` 并由 `audit.py` 校验。标题按精确 token 命中体育、经济、文化、国际、政治、社会六类词表；允许同类多词，但跨类命中即排除。国际词表限于外国国名、联合国和外交相关词；政治词表保留核心政治机构/制度词并去除泛词。社会类覆盖广义民生，与其他主题仍可能有语义交叉，因此结果反映弱标签任务而非人工金标准。

每类固定 50 篇，按 SHA-256 确定选样及分层拆分为 30/10/10。2 类和 4 类任务直接取六类样本的嵌套子集。分类只输入正文前 500 个已有分词，不输入标题。验证集用于官方 fastText 25、100、250 epoch 候选轮数选择，按 Macro-F1 最大值选择，平局取较小轮数；测试集仅评估选中的轮数。

## 模型

Gensim Word2Vec 和 Gensim FastText 均为 Skip-gram、100 维、窗口 5、`min_count=3`、10 轮、单 worker、随机种子 42。FastText 字符 n-gram 范围为 2–4、bucket 100,000。监督分类包含 unigram TF-IDF + Logistic Regression、冻结 Word2Vec 均值 + 标准化 + Logistic Regression、冻结 Gensim FastText 均值 + 标准化 + Logistic Regression，以及官方 `fasttext.train_supervised` 的随机初始化和预训练初始化两种方式。官方 fastText 使用 softmax、100 维、学习率 0.5、`wordNgrams=2`、`minn=maxn=0`、单线程、bucket 100,000。预训练 `.vec` 只导出 `分类监督训练集去重词表 ∩ Gensim FastText 的 key_to_index`，不导出训练词表 OOV；官方监督分类器关闭字符子词合成，未覆盖词按其默认行为处理。

两个语料的分词体系不同：PeopleDaily 保留其源文件的词/POS 切分并去掉标点、POS 标记；LR-Sum 则由 jieba 分词。结果文件会报告六类监督训练正文与 Gensim 词表之间的类型和词频加权覆盖率。低覆盖会减少均值向量特征中可用的词，因此跨语料对比须结合覆盖率和语料领域、年代差异解释。

近邻与 OOV 查询只检查向量是否可读及 Gensim FastText 能否由字符 n-gram 合成向量，不作为效果指标。性能结论应以留出的测试集结果为准；每类测试样本仅 10 篇，适合作为小规模编程实验比较。

## 环境与运行

建议使用 Python 3.12。依赖版本列在 `requirements.txt`。项目根目录中的冻结 Parquet、PeopleDaily 分类样本和生成的清洗语料均可直接用于复跑：

```powershell
python -m pip install -r requirements.txt
python run_experiment.py
```

脚本会重建 LR-Sum 的规范化、去重和 jieba 分词结果。只有检查点文件缺失，或来源 revision/摘要与当前语料不匹配时，才会重新训练两种 Gensim 表示；命中检查点后，若模型无法载入或缓存分支中的参数断言失败，程序会报错退出，不会自动回退重训。缓存分支检查 SG、维度、窗口、`min_count`、轮数、worker 和 seed；FastText 字符 n-gram 参数由运行末尾的独立审计检查，审计失败同样会报错。每次都重跑全部 15 个监督模型组合并输出诊断、指标、逐篇预测、错误表和图表，随后调用独立审计。若需强制重训表示模型，可删除 `models/word2vec_sg.model`、`models/gensim_fasttext_sg.model` 与 `results/embedding_training.json` 后运行同一命令。`fasttext-wheel==0.9.2` 提供官方 fastText Python 绑定；pip 会按当前平台选择兼容的发行包，不需要把 Gensim 分类器冒称为官方 fastText。单独重算并核查冻结样本、外部语料、epoch 选择、逐篇指标和混淆矩阵：

```powershell
python audit.py
```

## 主要文件

- `data/external/lr_sum_cmn_s_train.parquet`：固定 revision 的 LR-Sum 原始训练 split。
- `data/lr_sum_cmn_s_cleaned_train.jsonl`、`data/unsupervised_corpus.txt`、`data/unsupervised_corpus_manifest.csv`：规范化后的外部正文、jieba token 语料和逐篇纳入/排除清单。
- `data/frozen_split.csv`：固定监督分类样本；`data/frozen_input_sha256.json` 锁定原始弱标签候选、筛选审计、规则和划分。
- `models/`：Word2Vec、Gensim FastText、监督训练词表 `.vec` 和六类预训练官方 fastText 模型。
- `results/metrics.csv`、`predictions.csv`、`fasttext_validation_curve.csv`、六类错误表及 `audit_report.json`：结果和可复算证据。
- `results/classification_comparison.png`、`data_split_counts.png`、`confusion_6class_pretrained.png`：分类性能、语料/拆分规模和六类预训练混淆矩阵；额外的轮数验证曲线保存在 `fasttext_validation_curves.png`。

完整方法和实际结果见 [实验报告.md](实验报告.md)。
