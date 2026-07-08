# Prompt Engineering Techniques

Prompt engineering is the practice of crafting inputs to language models to reliably obtain high-quality outputs. As LLMs are sensitive to phrasing, structure, and context, prompt design is both a science and an art.

## Core Principles

**Be explicit about the task.** Ambiguous instructions produce inconsistent results. Instead of "summarize this," write "write a three-sentence summary that covers the main argument, the supporting evidence, and the conclusion."

**Provide examples (few-shot prompting).** Including one or more input-output pairs before the actual request dramatically improves performance on structured tasks. The model learns the expected format from the examples rather than from abstract instructions alone.

**Use chain-of-thought (CoT) prompting.** Appending "Think step by step" or providing a worked example with visible reasoning encourages the model to break down complex problems. CoT is especially effective for arithmetic, logical reasoning, and multi-step planning tasks.

## Structured Output

Modern LLMs support JSON-mode and tool-calling interfaces that force the output to conform to a schema. Using `with_structured_output(MyPydanticModel)` in LangChain wraps the LLM with an output parser that retries if the JSON is malformed, providing reliable typed responses.

## System Prompts

System prompts set the model's persona, constraints, and output format before the user message. They are ideal for injecting persistent instructions (e.g., "Always cite sources," "Never reveal internal reasoning") that apply to every turn in a conversation.

## Self-Reflection and Critique

A powerful advanced technique is to prompt the model to evaluate its own output before finalizing it. The model is asked to score or critique its draft, then revise it. This mirrors human proofreading and often catches errors or gaps the initial generation missed. In agentic pipelines, a separate Evaluator model can score the Writer's output and route the pipeline back for another attempt if quality is insufficient.

## Temperature and Sampling

Temperature controls output randomness. Temperature 0 produces deterministic, conservative outputs suitable for structured tasks. Higher temperatures (0.7–1.0) increase creativity and diversity, useful for brainstorming and creative writing. For research pipelines, planners and evaluators typically run at temperature 0 for reproducibility, while writers may use 0.3–0.5.
