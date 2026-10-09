# Gensim Word2Vec / FastText 与监督文本分类

项目使用人民日报 1998 年 1 月已有的词性分词训练 Gensim Skip-gram Word2Vec 和 FastText，并比较 TF-IDF、冻结词向量均值和官方 fastText 监督分类。分类输入只包含正文前 500 个词，不含标题。

## 环境

- Python 3.12（本次运行环境为 Anaconda Python 3.12.7）
- 安装依赖：`python -m pip install -r requirements.txt`
- 官方 fastText 绑定由 `fasttext-wheel==0.9.2` 提供。实验使用 `fasttext.train_supervised`，不是用 Gensim 分类器替代 fastText。
- `jieba` 不需要：数据已按词/POS 切分。解析时移除 POS 标记、方括号实体标记和 `/w` 标点。

## 运行

在本目录运行下面单一入口。若省略 `--source`，从随项目保存的冻结清洗数据、分类样本和未标注词语料复跑模型；首次构建或重建数据时传入本机原始 ZIP：

```powershell
python run_experiment.py
python run_experiment.py --source "..\作业1.3_ngram编程实验\data\人民日报1998_PFR.zip"
```

第二条命令会按固定摘要核验 ZIP，重建 `data/` 中的冻结清洗数据、弱标签候选和拆分，然后训练模型并运行独立审计。脚本会将自身重新启动到 `PYTHONHASHSEED=0`，因此从命令行即可复现哈希排序。也可单独重跑数据审计：

```powershell
python audit.py
```

## 冻结规则

标题取文章 `001` 段，正文取其余段；正文至少 80 个去标点词。标签根据 `data/selection_rules.json` 的精确标题 token 词表产生：允许同一类别命中多个词，但标题只要命中多个类别就剔除。类别顺序固定为体育、经济、文化、国际、政治、社会。国际词表只使用外国国名、联合国及外交相关词，排除“国际”“亚洲”“欧洲”等泛词；政治词表排除“国家”“会议”“领导”“总统”“中央”“政策”等泛词。社会类别覆盖广义社会与民生议题，可能和政治、经济、文化交叉；跨类标题命中会被排除，余下弱标签仍不能当作人工金标准。

原文含一个文章段号重复（第 20,964–20,965 行），两条不同正文都按源顺序保留。

六类规则下的可用候选数为体育 50、经济 192、文化 70、国际 124、政治 71、社会 113。每类按 `SHA256("20261009:select:" + article_id)` 选 50 篇，再按独立哈希排序拆成训练 30、验证 10、测试 10。2 类和 4 类任务直接使用该六类样本的嵌套子集。测试集不用于选择 epoch。

词向量只在未标注语料训练：六类验证/测试文章、正文完全相同的文章和与验证/测试文章前 200 个字符相同的文章均排除。官方 fastText 使用 `softmax`、100 维、学习率 0.5、wordNgrams=2、minn=maxn=0、thread=1、bucket=100000；在验证集上比较 25/100/250 epoch，Macro-F1 平局选较小 epoch。

Gensim FastText 的 OOV 查询可由训练得到的字符 n-gram 合成向量；为了避免把验证/测试词带入官方监督模型的初始化文件，`.vec` 只导出监督训练样本词表中也在 Gensim 词表内的词。官方 fastText 按要求设置 `minn=maxn=0`，因此该分类器对未初始化的词不执行字符子词合成。

## 主要产物

- `data/cleaned_january_articles.jsonl`：一月全部清洗后的文章及标题/正文 token。
- `data/frozen_split.csv`：固定的 300 篇监督样本及拆分。
- `data/unsupervised_corpus.txt` 与 `data/unsupervised_corpus_manifest.csv`：词向量训练语料和逐篇纳入/排除原因。
- `models/`：Gensim 模型、监督训练词表的 FastText `.vec`，以及六类预训练 fastText 模型（若模型文件低于 100 MiB）。
- `results/metrics.csv`、`predictions.csv`、`fasttext_validation_curve.csv`、六类错误清单和四张诊断图。
- `results/audit_report.json`：样本拆分、隔离、预测指标和混淆计数的独立核对结果。

近邻和 OOV 子词向量只用于检查模型输出是否可查询，不作为模型质量指标。正式性能结论应以留出的测试集指标为准；该测试集每类 10 篇，结果仅适合作为小规模编程实验比较。

## 实验报告

完整方法、参数、统计图表和结果分析见 [实验报告.md](实验报告.md)。
