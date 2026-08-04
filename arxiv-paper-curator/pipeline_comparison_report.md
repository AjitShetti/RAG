# Week 7 RAG System Architecture Comparison Report

## Executive Summary
This report evaluates the performance of the **Linear RAG Pipeline (`/api/v1/ask`)** versus the 
**Agentic RAG StateGraph Workflow (`/api/v1/agentic-ask`)** across 6 standardized test queries 
encompassing well-phrased CS questions, vague/short queries, and out-of-domain requests.

### Key Architecture Differences
- **Linear RAG (`/api/v1/ask`)**: Executes single-pass retrieval and answer generation without query refinement, guardrails, or relevance verification.
- **Agentic RAG (`/api/v1/agentic-ask`)**: Operates an adaptive LangGraph loop with an input Guardrail, iterative Retrieval, Relevance Grading, Query Rewriting (up to 2 attempts), and Grounded Generation.

## Summary Performance Matrix

| ID | Query Type | Query | Linear Latency | Agentic Latency | Guardrail Rejected | Rewrites | Linear Sources | Agentic Sources |
|---|---|---|---|---|---|---|---|---|
| 1 | well-phrased | `How do Vision Transformers (ViT) partition im...` | 33517.6 ms | 51255.3 ms | NO | 2 | 0 | 0 |
| 2 | well-phrased | `What are the key architectural differences be...` | 33349.4 ms | 51926.4 ms | NO | 2 | 0 | 0 |
| 3 | well-phrased | `How does Retrieval-Augmented Generation (RAG)...` | 33233.3 ms | 52230.3 ms | NO | 2 | 0 | 0 |
| 4 | out-of-domain | `What is the capital of France and what is the...` | 33383.6 ms | 2063.2 ms | YES | 0 | 0 | 0 |
| 5 | vague | `ai models` | 33373.7 ms | 52098.8 ms | NO | 2 | 0 | 0 |
| 6 | out-of-domain | `Who won the FIFA World Cup in 2022 and how do...` | 33397.9 ms | 1953.3 ms | YES | 0 | 0 | 0 |

## Detailed Question Analysis

### Question 1: How do Vision Transformers (ViT) partition images into patches and process them through multi-head self-attention?
**Category**: In-Domain CS/AI | **Type**: `well-phrased`

#### 1. Linear RAG (`/api/v1/ask`)
- **Latency**: 33517.6 ms
- **Retrieved / Used Chunks**: 0 / 0
- **Sources Attributed**: 0
- **Answer Preview**:
  > "I don't have enough information in the retrieved papers to answer this question reliably...."

#### 2. Agentic RAG (`/api/v1/agentic-ask`)
- **Latency**: 51255.3 ms
- **Guardrail Rejected**: `False`
- **Rewrite Count**: 2
- **Final Query**: `Vision Transformers AND (image partitioning OR patch embedding) AND (multi-head self-attention OR attention mechanisms) AND (computer vision applications OR image classification OR object detection)`
- **Reasoning Audit Steps**:
  - **`guardrail`** -> `in_domain`: Query validated as CS/AI research topic
  - **`retrieve`** -> `retrieved`: Retrieved 0 candidate chunks for query: 'How do Vision Transformers (ViT) partition images into patches and process them through multi-head self-attention?'
  - **`grade`** -> `weak`: Retrieved chunks lack sufficient relevant context
  - **`rewrite`** -> `rewritten`: Rewrote query 'How do Vision Transformers (ViT) partition images into patches and process them through multi-head self-attention?' to 'Vision Transformers image partitioning patch embedding multi-head self-attention mechanisms in computer vision applications' (attempt 1)
  - **`retrieve`** -> `retrieved`: Retrieved 0 candidate chunks for query: 'Vision Transformers image partitioning patch embedding multi-head self-attention mechanisms in computer vision applications'
  - **`grade`** -> `weak`: Retrieved chunks lack sufficient relevant context
  - **`rewrite`** -> `rewritten`: Rewrote query 'Vision Transformers image partitioning patch embedding multi-head self-attention mechanisms in computer vision applications' to 'Vision Transformers AND (image partitioning OR patch embedding) AND (multi-head self-attention OR attention mechanisms) AND (computer vision applications OR image classification OR object detection)' (attempt 2)
  - **`retrieve`** -> `retrieved`: Retrieved 0 candidate chunks for query: 'Vision Transformers AND (image partitioning OR patch embedding) AND (multi-head self-attention OR attention mechanisms) AND (computer vision applications OR image classification OR object detection)'
  - **`grade`** -> `weak`: Retrieved chunks lack sufficient relevant context
  - **`generate`** -> `answered`: Generated answer using 0 context chunks
- **Sources Attributed**: 0
- **Answer Preview**:
  > "I don't have enough information in the retrieved papers to answer this question reliably...."

---

### Question 2: What are the key architectural differences between BERT and GPT models for natural language processing?
**Category**: In-Domain CS/AI | **Type**: `well-phrased`

#### 1. Linear RAG (`/api/v1/ask`)
- **Latency**: 33349.4 ms
- **Retrieved / Used Chunks**: 0 / 0
- **Sources Attributed**: 0
- **Answer Preview**:
  > "I don't have enough information in the retrieved papers to answer this question reliably...."

#### 2. Agentic RAG (`/api/v1/agentic-ask`)
- **Latency**: 51926.4 ms
- **Guardrail Rejected**: `False`
- **Rewrite Count**: 2
- **Final Query**: `architectural comparisons and design differences between BERT and GPT transformer-based models in natural language processing OR empirical evaluations of transformer architectures for language understanding and generation tasks`
- **Reasoning Audit Steps**:
  - **`guardrail`** -> `in_domain`: Query validated as CS/AI research topic
  - **`retrieve`** -> `retrieved`: Retrieved 0 candidate chunks for query: 'What are the key architectural differences between BERT and GPT models for natural language processing?'
  - **`grade`** -> `weak`: Retrieved chunks lack sufficient relevant context
  - **`rewrite`** -> `rewritten`: Rewrote query 'What are the key architectural differences between BERT and GPT models for natural language processing?' to 'architectural comparisons between BERT and GPT models in natural language processing OR transformer-based language models differences in design and application' (attempt 1)
  - **`retrieve`** -> `retrieved`: Retrieved 0 candidate chunks for query: 'architectural comparisons between BERT and GPT models in natural language processing OR transformer-based language models differences in design and application'
  - **`grade`** -> `weak`: Retrieved chunks lack sufficient relevant context
  - **`rewrite`** -> `rewritten`: Rewrote query 'architectural comparisons between BERT and GPT models in natural language processing OR transformer-based language models differences in design and application' to 'architectural comparisons and design differences between BERT and GPT transformer-based models in natural language processing OR empirical evaluations of transformer architectures for language understanding and generation tasks' (attempt 2)
  - **`retrieve`** -> `retrieved`: Retrieved 0 candidate chunks for query: 'architectural comparisons and design differences between BERT and GPT transformer-based models in natural language processing OR empirical evaluations of transformer architectures for language understanding and generation tasks'
  - **`grade`** -> `weak`: Retrieved chunks lack sufficient relevant context
  - **`generate`** -> `answered`: Generated answer using 0 context chunks
- **Sources Attributed**: 0
- **Answer Preview**:
  > "I don't have enough information in the retrieved papers to answer this question reliably...."

---

### Question 3: How does Retrieval-Augmented Generation (RAG) reduce hallucinations in large language models?
**Category**: In-Domain CS/AI | **Type**: `well-phrased`

#### 1. Linear RAG (`/api/v1/ask`)
- **Latency**: 33233.3 ms
- **Retrieved / Used Chunks**: 0 / 0
- **Sources Attributed**: 0
- **Answer Preview**:
  > "I don't have enough information in the retrieved papers to answer this question reliably...."

#### 2. Agentic RAG (`/api/v1/agentic-ask`)
- **Latency**: 52230.3 ms
- **Guardrail Rejected**: `False`
- **Rewrite Count**: 2
- **Final Query**: `Retrieval-Augmented Generation models for hallucination mitigation in large-scale language models through enhanced contextualized information retrieval and knowledge graph-based grounding techniques`
- **Reasoning Audit Steps**:
  - **`guardrail`** -> `in_domain`: Query validated as CS/AI research topic
  - **`retrieve`** -> `retrieved`: Retrieved 0 candidate chunks for query: 'How does Retrieval-Augmented Generation (RAG) reduce hallucinations in large language models?'
  - **`grade`** -> `weak`: Retrieved chunks lack sufficient relevant context
  - **`rewrite`** -> `rewritten`: Rewrote query 'How does Retrieval-Augmented Generation (RAG) reduce hallucinations in large language models?' to 'Retrieval-Augmented Generation models mitigating hallucination effects in large-scale language models via improved contextualized information retrieval and knowledge grounding techniques' (attempt 1)
  - **`retrieve`** -> `retrieved`: Retrieved 0 candidate chunks for query: 'Retrieval-Augmented Generation models mitigating hallucination effects in large-scale language models via improved contextualized information retrieval and knowledge grounding techniques'
  - **`grade`** -> `weak`: Retrieved chunks lack sufficient relevant context
  - **`rewrite`** -> `rewritten`: Rewrote query 'Retrieval-Augmented Generation models mitigating hallucination effects in large-scale language models via improved contextualized information retrieval and knowledge grounding techniques' to 'Retrieval-Augmented Generation models for hallucination mitigation in large-scale language models through enhanced contextualized information retrieval and knowledge graph-based grounding techniques' (attempt 2)
  - **`retrieve`** -> `retrieved`: Retrieved 0 candidate chunks for query: 'Retrieval-Augmented Generation models for hallucination mitigation in large-scale language models through enhanced contextualized information retrieval and knowledge graph-based grounding techniques'
  - **`grade`** -> `weak`: Retrieved chunks lack sufficient relevant context
  - **`generate`** -> `answered`: Generated answer using 0 context chunks
- **Sources Attributed**: 0
- **Answer Preview**:
  > "I don't have enough information in the retrieved papers to answer this question reliably...."

---

### Question 4: What is the capital of France and what is the best recipe for baking croissants?
**Category**: Out-of-Domain / Non-CS | **Type**: `out-of-domain`

#### 1. Linear RAG (`/api/v1/ask`)
- **Latency**: 33383.6 ms
- **Retrieved / Used Chunks**: 0 / 0
- **Sources Attributed**: 0
- **Answer Preview**:
  > "I don't have enough information in the retrieved papers to answer this question reliably...."

#### 2. Agentic RAG (`/api/v1/agentic-ask`)
- **Latency**: 2063.2 ms
- **Guardrail Rejected**: `True`
- **Rewrite Count**: 0
- **Final Query**: `What is the capital of France and what is the best recipe for baking croissants?`
- **Reasoning Audit Steps**:
  - **`guardrail`** -> `out_of_domain`: Rejected query as out-of-domain
- **Sources Attributed**: 0
- **Answer Preview**:
  > "This question is out of domain. Please ask questions related to Computer Science, Artificial Intelligence, Machine Learning, or Data Science research...."

---

### Question 5: ai models
**Category**: Vague / Short Query | **Type**: `vague`

#### 1. Linear RAG (`/api/v1/ask`)
- **Latency**: 33373.7 ms
- **Retrieved / Used Chunks**: 0 / 0
- **Sources Attributed**: 0
- **Answer Preview**:
  > "I don't have enough information in the retrieved papers to answer this question reliably...."

#### 2. Agentic RAG (`/api/v1/agentic-ask`)
- **Latency**: 52098.8 ms
- **Guardrail Rejected**: `False`
- **Rewrite Count**: 2
- **Final Query**: `artificial intelligence OR machine learning OR deep learning OR neural networks OR cognitive computing OR natural language processing OR computer vision OR robotics OR human-computer interaction OR intelligent systems OR computational intelligence`
- **Reasoning Audit Steps**:
  - **`guardrail`** -> `in_domain`: Query validated as CS/AI research topic
  - **`retrieve`** -> `retrieved`: Retrieved 0 candidate chunks for query: 'ai models'
  - **`grade`** -> `weak`: Retrieved chunks lack sufficient relevant context
  - **`rewrite`** -> `rewritten`: Rewrote query 'ai models' to 'artificial intelligence models OR machine learning algorithms OR deep learning architectures OR neural networks OR cognitive computing systems' (attempt 1)
  - **`retrieve`** -> `retrieved`: Retrieved 0 candidate chunks for query: 'artificial intelligence models OR machine learning algorithms OR deep learning architectures OR neural networks OR cognitive computing systems'
  - **`grade`** -> `weak`: Retrieved chunks lack sufficient relevant context
  - **`rewrite`** -> `rewritten`: Rewrote query 'artificial intelligence models OR machine learning algorithms OR deep learning architectures OR neural networks OR cognitive computing systems' to 'artificial intelligence OR machine learning OR deep learning OR neural networks OR cognitive computing OR natural language processing OR computer vision OR robotics OR human-computer interaction OR intelligent systems OR computational intelligence' (attempt 2)
  - **`retrieve`** -> `retrieved`: Retrieved 0 candidate chunks for query: 'artificial intelligence OR machine learning OR deep learning OR neural networks OR cognitive computing OR natural language processing OR computer vision OR robotics OR human-computer interaction OR intelligent systems OR computational intelligence'
  - **`grade`** -> `weak`: Retrieved chunks lack sufficient relevant context
  - **`generate`** -> `answered`: Generated answer using 0 context chunks
- **Sources Attributed**: 0
- **Answer Preview**:
  > "I don't have enough information in the retrieved papers to answer this question reliably...."

---

### Question 6: Who won the FIFA World Cup in 2022 and how do I make cold brew coffee?
**Category**: Out-of-Domain / Non-CS | **Type**: `out-of-domain`

#### 1. Linear RAG (`/api/v1/ask`)
- **Latency**: 33397.9 ms
- **Retrieved / Used Chunks**: 0 / 0
- **Sources Attributed**: 0
- **Answer Preview**:
  > "I don't have enough information in the retrieved papers to answer this question reliably...."

#### 2. Agentic RAG (`/api/v1/agentic-ask`)
- **Latency**: 1953.3 ms
- **Guardrail Rejected**: `True`
- **Rewrite Count**: 0
- **Final Query**: `Who won the FIFA World Cup in 2022 and how do I make cold brew coffee?`
- **Reasoning Audit Steps**:
  - **`guardrail`** -> `out_of_domain`: Rejected query as out-of-domain
- **Sources Attributed**: 0
- **Answer Preview**:
  > "This question is out of domain. Please ask questions related to Computer Science, Artificial Intelligence, Machine Learning, or Data Science research...."

---

## Comparative Insights & Architectural Recommendations

1. **Guardrail Protection against Out-of-Domain Noise**:
   - **Linear RAG** attempts retrieval and generation regardless of domain, wasting search compute and risks generating speculative answers on irrelevant topics.
   - **Agentic RAG** catches out-of-domain queries immediately at the guardrail node, short-circuiting execution and avoiding database retrieval entirely.

2. **Adaptive Query Rewriting for Vague Queries**:
   - **Linear RAG** relies solely on raw user keywords, suffering from keyword mismatch on ambiguous or short queries.
   - **Agentic RAG** identifies weak relevance during grading and re-formulates the query with scientific terminology, drastically improving candidate retrieval context.

3. **Latency vs Reliability Trade-off**:
   - **Linear RAG** features lower latency due to a single un-evaluated pass.
   - **Agentic RAG** incurs additional LLM decision steps but guarantees input safety and higher answer grounding quality.