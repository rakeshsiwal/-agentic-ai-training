# Task-04 Evaluation Report

**Generated:** 2026-06-27 14:32:07  
**Questions evaluated:** 14  
**Passed:** 13 / 14  
**Pass rate:** 92.9%  
**Threshold:** 80%  
**Outcome:** ✅ PASS

---

## Per-Question Breakdown

### q01 — ✅ PASS

**Question:** What does RLHF stand for and what are its three stages?  
**Source doc:** `sample_ml.txt`  

**Expected facts:**

- Reinforcement Learning from Human Feedback
- supervised fine-tuning
- reward model training
- PPO

**Scores:**

| Dimension | Score (1–5) |
|---|---|
| Faithfulness | 5 |
| Relevance | 5 |
| Fact coverage | 5 |

**Judge explanation:** The agent correctly identified all four expected facts, citing the source document accurately with no hallucinations.

**Agent answer:**

RLHF stands for Reinforcement Learning from Human Feedback. According to the retrieved documents, it has three
main stages: (1) Supervised Fine-Tuning (SFT), where the base model is fine-tuned on high-quality human
demonstrations; (2) Reward Model Training, where human annotators rank outputs by quality and a reward model
is trained to predict preferences; and (3) PPO (Proximal Policy Optimization), where the SFT model is
further refined using RL with the reward model as the reward signal.

---

### q02 — ✅ PASS

**Question:** What is the difference between supervised learning and unsupervised learning?  
**Source doc:** `sample_ml.txt`  

**Expected facts:**

- supervised learning uses labeled data
- unsupervised learning finds hidden patterns without labels
- input-output pairs

**Scores:**

| Dimension | Score (1–5) |
|---|---|
| Faithfulness | 5 |
| Relevance | 5 |
| Fact coverage | 5 |

**Judge explanation:** All three expected facts clearly addressed. Answer is faithful to retrieved passages.

**Agent answer:**

Supervised learning trains on labeled data — input-output pairs where the desired output is known — and the
algorithm learns to map inputs to outputs. Examples include linear regression, logistic regression, and
neural networks. Unsupervised learning, by contrast, finds hidden patterns in data without any labels;
the system learns structure on its own using techniques like K-Means clustering, PCA, or autoencoders.

---

### q03 — ✅ PASS

**Question:** What are the key components of a Transformer architecture?  
**Source doc:** `sample_ai.txt`  

**Expected facts:**

- multi-head self-attention
- position-wise feed-forward networks
- layer normalization
- positional encoding

**Scores:**

| Dimension | Score (1–5) |
|---|---|
| Faithfulness | 5 |
| Relevance | 5 |
| Fact coverage | 5 |

**Judge explanation:** All four Transformer components mentioned explicitly.

**Agent answer:**

The Transformer architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017), has four key
components: (1) Multi-head self-attention, which weighs the relevance of different words in a sentence; (2)
Position-wise feed-forward networks; (3) Layer normalization; and (4) Positional encoding, which gives the
model information about the order of tokens.

---

### q04 — ✅ PASS

**Question:** What is RAG and what are its main phases?  
**Source doc:** `vector_databases.md`  

**Expected facts:**

- Retrieval-Augmented Generation
- indexing phase
- retrieval phase
- generation phase
- vector database

**Scores:**

| Dimension | Score (1–5) |
|---|---|
| Faithfulness | 5 |
| Relevance | 5 |
| Fact coverage | 5 |

**Judge explanation:** Correctly named RAG and all three phases; mentioned vector database as the storage layer.

**Agent answer:**

RAG stands for Retrieval-Augmented Generation. It combines a vector database with an LLM and operates in
three phases: (1) Indexing — documents are chunked, embedded, and stored in a vector database; (2) Retrieval
— the user query is embedded and the most similar chunks are fetched; (3) Generation — the retrieved chunks
are combined with the query and sent to an LLM to produce an answer. RAG reduces hallucinations, supports
private knowledge bases, and is more cost-effective than fine-tuning for knowledge injection.

---

### q05 — ✅ PASS

**Question:** What is HNSW and which vector databases use it?  
**Source doc:** `vector_databases.md`  

**Expected facts:**

- Hierarchical Navigable Small World
- graph-based index
- Chroma
- Weaviate

**Scores:**

| Dimension | Score (1–5) |
|---|---|
| Faithfulness | 5 |
| Relevance | 5 |
| Fact coverage | 5 |

**Judge explanation:** All four facts present and accurately attributed.

**Agent answer:**

HNSW stands for Hierarchical Navigable Small World. It is a graph-based approximate nearest neighbour index
known for being very fast and accurate. According to the documents, both Chroma and Weaviate use HNSW as
their underlying index structure.

---

### q06 — ✅ PASS

**Question:** What is the Send API in LangGraph and what is it used for?  
**Source doc:** `langgraph_send_api.md`  

**Expected facts:**

- fan-out parallelism
- dynamically dispatch multiple tasks
- Send
- parallel

**Scores:**

| Dimension | Score (1–5) |
|---|---|
| Faithfulness | 5 |
| Relevance | 5 |
| Fact coverage | 5 |

**Judge explanation:** Send API correctly described; all expected facts present.

**Agent answer:**

The Send API is a LangGraph feature for fan-out parallelism. It allows you to dynamically dispatch multiple
tasks in parallel to the same or different nodes. You return a list of Send objects from a conditional edge
function, each targeting a node with its own input state. Common use cases include processing multiple
documents in parallel, running sub-agents concurrently, and map-reduce patterns over large datasets.

---

### q07 — ✅ PASS

**Question:** Explain chain-of-thought prompting and who introduced it.  
**Source doc:** `prompt_engineering.md`  

**Expected facts:**

- step-by-step reasoning
- Wei et al.
- 2022
- Chain-of-Thought Prompting Elicits Reasoning

**Scores:**

| Dimension | Score (1–5) |
|---|---|
| Faithfulness | 5 |
| Relevance | 5 |
| Fact coverage | 5 |

**Judge explanation:** All four facts present; paper title and authors cited correctly.

**Agent answer:**

Chain-of-thought (CoT) prompting asks the model to reason step-by-step before providing a final answer.
This technique was introduced in the 2022 paper "Chain-of-Thought Prompting Elicits Reasoning in Large
Language Models" by Wei et al. By showing intermediate reasoning steps, CoT significantly improves
performance on complex tasks like multi-step arithmetic and logical reasoning.

---

### q08 — ✅ PASS

**Question:** What are the main overfitting prevention techniques in machine learning?  
**Source doc:** `sample_ml.txt`  

**Expected facts:**

- cross-validation
- L1 regularization
- L2 regularization
- dropout
- early stopping

**Scores:**

| Dimension | Score (1–5) |
|---|---|
| Faithfulness | 5 |
| Relevance | 5 |
| Fact coverage | 5 |

**Judge explanation:** All five techniques correctly identified and briefly explained.

**Agent answer:**

To prevent overfitting, the documents list several techniques: cross-validation (specifically k-fold), L1
regularization (Lasso) which promotes sparsity, L2 regularization (Ridge) which penalises large weights,
dropout for neural networks, early stopping, and data augmentation. These help a model generalise to unseen
data rather than memorising training noise.

---

### q09 — ✅ PASS

**Question:** What is Constitutional AI and which company developed it?  
**Source doc:** `prompt_engineering.md`  

**Expected facts:**

- Anthropic
- helpful, harmless, and honest
- critique and revise
- constitution

**Scores:**

| Dimension | Score (1–5) |
|---|---|
| Faithfulness | 5 |
| Relevance | 5 |
| Fact coverage | 5 |

**Judge explanation:** Correct company, correct principles, correct mechanism all present.

**Agent answer:**

Constitutional AI (CAI) is an alignment technique developed by Anthropic. It trains models to be helpful,
harmless, and honest by having the model critique and revise its own outputs based on a set of principles
called a "constitution." A related technique, RLAIF (Reinforcement Learning from AI Feedback), uses AI-
generated feedback instead of human feedback, enabling scalable alignment.

---

### q10 — ✅ PASS

**Question:** What are the three main checkpointer options in LangGraph and what does each one do?  
**Source doc:** `langgraph_send_api.md`  

**Expected facts:**

- MemorySaver
- SqliteSaver
- PostgresSaver
- in-memory
- SQLite

**Scores:**

| Dimension | Score (1–5) |
|---|---|
| Faithfulness | 5 |
| Relevance | 5 |
| Fact coverage | 5 |

**Judge explanation:** All three checkpointers named with correct descriptions.

**Agent answer:**

LangGraph offers three built-in checkpointers: (1) MemorySaver — stores state in-memory, lost when the
process restarts; (2) SqliteSaver — persists state to a SQLite file for lightweight durability; (3)
PostgresSaver — a production-grade option for persistent, scalable state storage.

---

### q11 — ✅ PASS

**Question:** What is few-shot prompting and how does it differ from zero-shot prompting?  
**Source doc:** `prompt_engineering.md`  

**Expected facts:**

- few-shot provides examples
- zero-shot has no examples
- guides the model's behavior

**Scores:**

| Dimension | Score (1–5) |
|---|---|
| Faithfulness | 5 |
| Relevance | 5 |
| Fact coverage | 4 |

**Judge explanation:** Key distinction clear; "guides the model's behavior" phrasing slightly implicit but present.

**Agent answer:**

Zero-shot prompting asks the model to perform a task without any examples — it works for simple, well-
defined tasks. Few-shot prompting provides examples within the prompt to guide the model's behavior.
According to the documents, few-shot prompting dramatically improves performance on complex or ambiguous
tasks by showing the model the desired input-output pattern.

---

### q12 — ✅ PASS

**Question:** What are the benefits of using RAG over fine-tuning for injecting knowledge into LLMs?  
**Source doc:** `vector_databases.md`  

**Expected facts:**

- reduces hallucinations
- private or proprietary knowledge
- more cost-effective
- up-to-date information

**Scores:**

| Dimension | Score (1–5) |
|---|---|
| Faithfulness | 5 |
| Relevance | 5 |
| Fact coverage | 5 |

**Judge explanation:** All four benefits listed explicitly.

**Agent answer:**

The documents list four key benefits of RAG over fine-tuning: (1) It grounds responses in factual, up-to-
date information, reducing hallucinations; (2) It allows using private or proprietary knowledge that can't
be put in a public model; (3) It is more cost-effective than fine-tuning for knowledge injection; and (4)
the knowledge base can be updated without retraining.

---

### q13 — ✅ PASS

**Question:** What is the ReAct pattern in LangGraph and what does the loop look like?  
**Source doc:** `langgraph_send_api.md`  

**Expected facts:**

- Reasoning and Acting
- select a tool
- tool executes
- loop until done
- agent_node
- tool_node

**Scores:**

| Dimension | Score (1–5) |
|---|---|
| Faithfulness | 5 |
| Relevance | 5 |
| Fact coverage | 4 |

**Judge explanation:** Most facts covered; "Reasoning and Acting" expansion slightly informal but acceptable.

**Agent answer:**

The ReAct (Reasoning + Acting) pattern is the most common LangGraph agent pattern. The loop works as
follows: the agent_node reasons and selects a tool with its arguments; the tool_node executes the tool and
returns the result; the result is added to the conversation; and the agent reasons again. This cycle repeats
until the agent determines no more tool calls are needed.

---

### q14 — ❌ FAIL

**Question:** What are the three AI safety concerns mentioned in the documents?  
**Source doc:** `sample_ai.txt`  

**Expected facts:**

- alignment
- robustness
- interpretability
- fairness

**Scores:**

| Dimension | Score (1–5) |
|---|---|
| Faithfulness | 4 |
| Relevance | 4 |
| Fact coverage | 2 |

**Judge explanation:** Agent listed alignment and interpretability but missed robustness and fairness, covering only 2 of 4 expected facts.

**Agent answer:**

The documents mention two key AI safety concerns: alignment (ensuring AI pursues goals humans actually want)
and interpretability (understanding how AI systems make decisions). Anthropic and others are actively
researching these areas.

---
