# 使用整合后的 construct_training_groups.py

## 功能概述

整合后的 `construct_training_groups.py` 脚本现在包含以下功能：

1. **数据标注**: 使用 LLM (GPT-4) 对候选元组进行三维度评估
2. **训练数据生成**: 基于标注结果自动生成 reranker 的训练数据

## 使用方法

### 基本用法（仅标注数据）

```bash
python construct_training_groups.py \
    --collection /path/to/collection.tsv \
    --qrels /path/to/qrels.tsv \
    --folds /path/to/folds.json \
    --queries /path/to/queries.tsv \
    --retrieved_results /path/to/retrieved_results.tsv \
    --api_key YOUR_OPENAI_API_KEY \
    --api_base https://api.openai.com/v1 \
    --output_path ./annotation_results.jsonl
```

### 完整流程（标注 + 生成训练数据）

```bash
python construct_training_groups.py \
    --collection /path/to/collection.tsv \
    --qrels /path/to/qrels.tsv \
    --folds /path/to/folds.json \
    --queries /path/to/queries.tsv \
    --retrieved_results /path/to/retrieved_results.tsv \
    --api_key YOUR_OPENAI_API_KEY \
    --api_base https://api.openai.com/v1 \
    --output_path ./annotation_results.jsonl \
    --generate_training_data \
    --training_output_path ./training_data.jsonl \
    --negative_samples_per_group 15
```

## 主要参数说明

### 必需参数
- `--collection`: 候选元组集合文件路径
- `--qrels`: 查询-元组关系文件路径
- `--folds`: 数据分割文件路径
- `--queries`: 查询文件路径
- `--retrieved_results`: 检索结果文件路径
- `--api_key`: OpenAI API 密钥
- `--api_base`: OpenAI API 基础URL
- `--output_path`: 标注结果输出路径

### 标注控制参数
- `--max_annotations_per_query`: 每个查询的最大标注数量（默认：30）
- `--min_negative_samples`: 最小负样本数量（默认：15）
- `--min_positive_samples`: 最小正样本数量（默认：1）
- `--model`: 使用的 OpenAI 模型（默认：gpt-4o-mini）

### 训练数据生成参数
- `--generate_training_data`: 启用训练数据生成
- `--training_output_path`: 训练数据输出路径（默认：标注结果文件名后加_training.jsonl）
- `--negative_samples_per_group`: 每个训练组的负样本数量（默认：15）

## 工作流程

1. **初始化**: 加载数据和配置，检查已有标注结果
2. **数据标注**: 对每个查询的候选元组进行标注
   - 使用 GPT 评估三个维度：Existence、Relevance、Logical Consistency
   - 自动计算分数并分类为正负样本
   - 当达到足够样本数量或最大标注数时停止
3. **训练数据生成**（如果启用）: 
   - 分析所有标注结果
   - 为每个正样本构建训练组（1个正样本 + 15个负样本）
   - 输出适用于 reranker 训练的格式

## 输出格式

### 标注结果格式 (annotation_results.jsonl)
```json
{
  "query_id": 123,
  "candidate_id": "tuple_456",
  "evaluation": {
    "full_response": "GPT完整回复...",
    "dimension_scores": {
      "existence": "Yes",
      "relevance": "Highly Relevant", 
      "consistency": "Fully Consistent"
    }
  },
  "score": 4
}
```

### 训练数据格式 (training_data.jsonl)
```json
{
  "qry": {
    "qid": "123",
    "query": "查询元组内容..."
  },
  "pos": [{
    "pid": "positive_tuple_id",
    "passage": "正样本元组内容..."
  }],
  "neg": [
    {
      "pid": "negative_tuple_id_1",
      "passage": "负样本元组内容1..."
    },
    // ... 更多负样本（共15个）
  ]
}
```

## 注意事项

1. **API 成本**: 使用 GPT 标注会产生 API 费用，请根据需要调整 `max_annotations_per_query`
2. **断点续传**: 脚本支持断点续传，重新运行时会跳过已标注的样本
3. **数据质量**: 确保输入数据格式正确，特别是 TSV 文件的分隔符
4. **负样本不足**: 如果某些查询的负样本不足，会在输出中显示相关统计信息


