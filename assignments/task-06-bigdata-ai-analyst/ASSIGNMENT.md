# Assignment Task 03 — Agentic Big Data Analyst (PySpark + LangGraph)

**Issued:** July 17, 2026
**Due:** July 24, 2026
**Difficulty:** Advanced
**Branch naming:** `feature/task-03-bigdata-ai-analyst`
**Prerequisite:** Task 01 & Task 02 completed and reviewed

---

## 🎯 Objective

Build an **AI Data Analyst agent** that answers natural-language questions about a **large dataset (1M+ rows)** by:

1. Processing raw data at scale with **Apache Spark (PySpark)** — cleaning, partitioning, and storing it as **Parquet**.
2. Using a **LangGraph agent** that translates the user's question into **Spark SQL**, executes it on the dataset, and explains the results in plain English.
3. Adding an **Insight Agent** that proactively detects anomalies/trends in the data and reports them.

This is the classic **Big Data + AI** combination used in industry: Spark does the heavy lifting on data, the LLM does the reasoning and language — neither can do the other's job.

> **Why not just load the data into the LLM?** Because 1M rows won't fit in any context window. The agent must learn to *query* data, not *read* it. That is the core lesson of this task.

---

## 🧠 What You Will Learn

- **PySpark fundamentals** — DataFrames, lazy evaluation, transformations vs. actions, `spark.sql()`.
- **Columnar storage** — writing/reading **Parquet**, partitioning by column, and why it's faster than CSV.
- **Text-to-SQL agents** — prompting an LLM with a schema so it generates *correct, runnable* SQL.
- **Tool-using agents on data** — the agent never sees raw data, only schemas + query results.
- **SQL guardrails** — validating LLM-generated SQL before execution (SELECT-only, row limits).
- **Self-correction loops** — when a generated query fails, feed the error back and let the agent retry.
- **ETL pipeline design** — raw → cleaned → aggregated (bronze/silver/gold layers, simplified).

---

## 🗂️ Folder Structure (Expected Output)

```
assignments/task-03-bigdata-ai-analyst/
│
├── ASSIGNMENT.md              ← this file (do not modify)
├── README.md                  ← your notes: architecture, setup, sample sessions
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                   ← generated CSV (gitignored — large!)
│   ├── warehouse/             ← cleaned Parquet, partitioned (gitignored)
│   └── .gitignore             ← ignore raw/ and warehouse/ contents
├── etl/
│   ├── __init__.py
│   ├── generate_data.py       ← synthetic dataset generator (1M+ rows)
│   └── pipeline.py            ← Spark ETL: raw CSV → cleaned Parquet
├── analyst/
│   ├── __init__.py
│   ├── tools.py               ← Spark-backed tools the agent calls
│   ├── guardrails.py          ← SQL validation before execution
│   ├── sql_agent.py           ← text-to-SQL agent node(s)
│   └── insight_agent.py       ← proactive anomaly/trend detector
├── state.py                   ← AnalystState TypedDict
├── graph.py                   ← LangGraph wiring
└── main.py                    ← CLI entry point (chat loop)
```

---

## ⚙️ Technical Requirements

### 1. Synthetic Dataset (`etl/generate_data.py`)

Generate an **e-commerce orders dataset** with **at least 1,000,000 rows** as CSV in `data/raw/`. Use `faker` + `numpy`/`random` — no API calls needed.

Required columns:

| Column | Type | Notes |
|---|---|---|
| `order_id` | string | unique |
| `order_ts` | timestamp | spread over the last 12 months |
| `customer_id` | string | ~50,000 distinct customers |
| `city` | string | 15–20 Indian cities |
| `category` | string | 8–10 product categories |
| `product_name` | string | |
| `quantity` | int | 1–5 |
| `unit_price` | float | |
| `discount_pct` | float | 0–30 |
| `payment_method` | string | UPI / Card / COD / Wallet |
| `status` | string | delivered / returned / cancelled |

**Deliberately inject dirty data** (the ETL must fix it):
- ~1% duplicate `order_id` rows
- ~2% rows with negative `quantity` or `unit_price`
- ~1% rows with null `city` or `category`
- At least **one planted anomaly** for the Insight Agent to find (e.g., one city's orders spike 5× in one specific week, or one category's return rate is abnormally high)

Document your planted anomaly in `README.md` — reviewers will check whether your Insight Agent finds it.

---

### 2. Spark ETL Pipeline (`etl/pipeline.py`)

Runs Spark in **local mode** (`local[*]`) — no cluster needed.

- Read raw CSV with an **explicit schema** (`StructType`, not `inferSchema` — explain why in README).
- Clean: drop duplicates by `order_id`, drop/repair invalid rows, fill or drop nulls (justify your choice).
- Derive columns: `revenue = quantity * unit_price * (1 - discount_pct/100)`, `order_date`, `order_month`.
- Write to `data/warehouse/orders/` as **Parquet, partitioned by `order_month`**.
- Print a small **data-quality report** at the end: rows in, rows dropped per rule, rows out.
- Must be **idempotent** — re-running overwrites cleanly (`mode="overwrite"`).

---

### 3. Agent Tools (`analyst/tools.py`)

The agent interacts with data **only through these tools** — it never sees more than ~50 rows at a time:

```python
@tool
def get_schema() -> str:
    """Return table name, columns, types, and 3 sample rows as text."""

@tool
def run_spark_sql(query: str) -> str:
    """Validate then execute a Spark SQL SELECT against the `orders` view.
    Returns at most 50 rows, formatted as a markdown table.
    On failure, returns the Spark error message (do NOT raise)."""

@tool
def get_column_stats(column: str) -> str:
    """Return distinct count + top-10 values (categorical) or
    min/max/mean/stddev (numeric) for one column."""
```

Register the Parquet data as a temp view once at startup:
`spark.read.parquet(...).createOrReplaceTempView("orders")`.

---

### 4. SQL Guardrails (`analyst/guardrails.py`)

LLM-generated SQL is **untrusted input**. Before executing, `validate_sql(query)` must reject:

- Anything that is not a single `SELECT` statement (no `DROP`, `INSERT`, `UPDATE`, `DELETE`, `CREATE`, no `;`-chained statements)
- Queries referencing tables other than `orders`
- Queries without a `LIMIT` → auto-append `LIMIT 50` instead of rejecting

Write **unit tests** for the guardrails in `tests/` (this is the one part of the task with required tests — at least 5 cases including a blocked `DROP TABLE`).

---

### 5. The SQL Agent (`analyst/sql_agent.py` + `graph.py`)

State:

```python
class AnalystState(TypedDict):
    messages:      Annotated[list, add_messages]
    question:      str        # user's natural-language question
    sql_query:     str        # generated SQL
    query_result:  str        # tool output (markdown table or error)
    error_count:   int        # retries used
    final_answer:  str
```

Graph flow:

```
[START] → planner_node → sql_generator_node → executor_node
                              ↑                    │
                              │   (error &&        │ (success)
                              │    retries < 3)    ↓
                              └──── on error   answer_node → [END]
```

- **`planner_node`** — calls `get_schema()` / `get_column_stats()` if needed, decides what to query.
- **`sql_generator_node`** — prompts the LLM with the schema + question → produces `sql_query` (use structured output, as in Task 02).
- **`executor_node`** — runs the query via the guarded tool. On error, increments `error_count` and routes **back to the generator with the error message in context** (self-correction). After 3 failures, answer honestly that the query could not be completed.
- **`answer_node`** — turns the result table into a clear plain-English answer, **including the SQL it ran** so the user can verify.

The agent must correctly answer questions like:
- "What was total revenue per category last quarter?"
- "Which city has the highest return rate?"
- "Show the monthly revenue trend for Electronics."
- "Who are the top 5 customers by lifetime spend?"

---

### 6. The Insight Agent (`analyst/insight_agent.py`)

A second entry point (`python main.py --insights`) that runs **without a user question**:

1. Runs a fixed set of ~5 diagnostic Spark SQL queries (weekly revenue by city, return rate by category, etc.).
2. Passes the aggregated results to the LLM with the prompt: *"You are a data analyst. Identify the 3 most significant anomalies or trends and explain each in 2–3 sentences with numbers."*
3. Prints a short **Insight Report**.

It must find your planted anomaly from step 1. If it doesn't, iterate on your diagnostic queries — that debugging loop *is* the exercise.

---

## ✅ Acceptance Criteria

- [ ] `generate_data.py` produces ≥ 1M rows with the required dirty data injected
- [ ] `pipeline.py` outputs partitioned Parquet + prints a data-quality report
- [ ] Raw/warehouse data is **gitignored** — no large files in the PR
- [ ] `validate_sql` blocks non-SELECT statements; unit tests pass (`pytest tests/`)
- [ ] Agent answers all 4 sample questions above with correct numbers (spot-check one by hand with a manual Spark query — show this in README)
- [ ] Self-correction works: a failed query is retried with the error in context (show one such session log in README)
- [ ] `python main.py --insights` finds the planted anomaly
- [ ] Final answers always show the SQL that was executed
- [ ] `.env.example` present, no real keys committed; `requirements.txt` installable
- [ ] `README.md` covers architecture, how to run each stage, and sample sessions

---

## 🌟 Bonus Challenges (optional)

| Stars | Bonus | Description |
|---|---|---|
| ⭐ | Query cache | Cache (question → SQL) pairs; on a repeated question, skip generation |
| ⭐⭐ | Chart output | Add a `plot_result` tool that renders the query result as a matplotlib PNG saved to `outputs/` |
| ⭐⭐ | Schema RAG | Store a data dictionary (column descriptions, business rules) in ChromaDB from Task 02; retrieve relevant entries into the SQL-generation prompt |
| ⭐⭐⭐ | Streaming ingest | Use Spark **Structured Streaming** to watch `data/raw/incoming/` and append new orders to the warehouse continuously |
| ⭐⭐⭐ | Multi-table | Split data into `orders` + `customers` tables; agent must generate correct JOINs |

---

## 📦 Starter Dependencies

```txt
# Big data
pyspark>=3.5.0

# Data generation
faker>=25.0.0
numpy>=1.26.0

# Agent (same stack as Task 01/02)
langgraph>=0.2.0
langchain>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.2.0          # or anthropic / google-genai / groq
python-dotenv>=1.0.0

# Tests
pytest>=8.0.0
```

> **PySpark needs Java.** Install a JDK first (`sudo apt install openjdk-17-jdk` on Ubuntu) and verify with `java -version`. If Spark fails to start, this is the first thing to check.

> **Memory tip:** 1M rows is small for Spark but big for pandas — that's the point. If your machine struggles, `local[2]` and `spark.driver.memory=2g` are fine.

---

## 📤 Submission

1. `git checkout -b feature/task-03-bigdata-ai-analyst`
2. Build inside `assignments/task-03-bigdata-ai-analyst/`.
3. Include in `README.md`: one full Q&A session log, one self-correction log, and the Insight Report output.
4. Open a Pull Request to `main` — tag the reviewer.

---

## 📚 Resources

- [PySpark — Getting Started](https://spark.apache.org/docs/latest/api/python/getting_started/index.html)
- [Spark SQL Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [Parquet & partitioning](https://spark.apache.org/docs/latest/sql-data-sources-parquet.html)
- [LangGraph — Agentic RAG / SQL patterns](https://langchain-ai.github.io/langgraph/tutorials/sql-agent/)
- [LangChain — Structured Output](https://python.langchain.com/docs/how_to/structured_output/)
- [Faker docs](https://faker.readthedocs.io/)

---

## 🗺️ Suggested Day-by-Day Plan

| Day | Focus |
|---|---|
| Day 1 | Install JDK + PySpark, run a hello-world Spark job, write `generate_data.py` |
| Day 2 | Build `pipeline.py`: schema, cleaning rules, Parquet output, quality report |
| Day 3 | Build `tools.py` + `guardrails.py` with unit tests — test tools in isolation |
| Day 4 | Build the SQL agent graph (generator → executor), no self-correction yet |
| Day 5 | Add the self-correction loop + `answer_node`; test the 4 sample questions |
| Day 6 | Build the Insight Agent; verify it finds your planted anomaly |
| Day 7 | Polish: README, session logs, attempt a bonus |

---

*Task 01 taught you agents. Task 02 taught you multi-agent RAG. Task 03 teaches you the pattern behind every "chat with your data" product: the LLM reasons, the engine computes. Keep the agent away from raw data — schemas in, results out.*
