# A-Paper-List-of-Awesome-Tabular-LLMs
Different types of tables are widely used to store and present information. To automatically process numerous tables and gain valuable insights, researchers have proposed a series of deep-learning models for various table-based tasks, e.g., table question answering (TQA), table-to-text (T2T), text-to-sql (NL2SQL) and table fact verification (TFV). Recently, the emerging [Large Language Models (LLMs)](https://github.com/Hannibal046/Awesome-LLM#chatgpt-evaluation) and more powerful [Multimodal Large Language Models (MLLMs)](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) have opened up new possibilities for processing the tabular data, i.e., we can use one general model to process diverse tables and fulfill different tabular tasks based on the user natural language instructions. We refer to these LLMs speciallized for tabular tasks as `Tabular LLMs`. **In this repository, we collect a paper list about recent Tabular (M)LLMs and divide them into the following categories based on their key idea.**

---

<font size=8><center><b> Table of Contents: </b> </center></font>
1. [**Survey of Tabular LLMs and table understanding**](#1-survey-of-tabular-llms-and-table-understanding)
2. [**Prompting LLMs for different tabular tasks**](#2-prompting-llms-for-different-tabular-tasks), e.g., in-context learning, prompt engineering and integrating external tools.
3. [**Training LLMs for better table understanding ability**](#3-training-llms-for-better-table-understanding-ability), e.g., training existing LLMs by instruction fine-tuning or post-pretraining.
4. [**Developing Agents for tabular data**](#4-developing-agents-for-processing-tabular-data), e.g., devolping copilot for processing excel tables.
5. [**RAG with tabular data**](#5-rag-with-tabular-data), e.g., devolping RAG systems for understanding long tables.
6. [**Empirical study for evaluating LLMs' table understanding ability**](#6-empirical-study-for-evaluating-llms-table-understanding-ability), e.g., exploring the influence of various table types or table formats.
7. [**Multimodal table understanding**](#7-multimodal-table-understanding), e.g., training MLLMs to understand diverse table images and textual user requests.
8. [**Table Understanding datasets and benchmarks**](#8-table-understanding-datasets-and-benchmarks), e.g., valuable datasets and benchmarks for model training and evaluation.
9. [**Evaluation Metrics for Table Understanding**](#9-designing-evaluation-metrics-for-table-understanding), e.g., devising better evaluation method for table understanding.
---

<font size=8><center><b> Task Names and  Abbreviations: </b> </center></font>

| Task Names | Abbreviations | Task Descriptions |
| :---: | :---: | :---: |
| Table Question Answering |  TQA | Answering questions based on the table(s), e.g., answer look-up or computation questions about table(s). |
| Table-to-Text | Table2Text or T2T | Generate a text based on the table(s), e.g., generate a analysis report given a financial statement. | 
| Text-to-Table | Text2Table | Generate structured tables based on input text, e.g., generate a statistical table based on the game summary. |
| Table Fact Verification | TFV | Judging if a statement is true or false (or not enough evidence) based on the table(s) |
| Text-to-SQL | NL2SQL | Generate a SQL statement to answer the user question based on the database schema |
| Tabular Mathematical Reasoning | TMR | Solving mathematical reasoning problems based on the table(s), e.g., solve math word problems related to a table |
| Table-and-Text Question Answering | TAT-QA | Answering questions based on both table(s) and their related texts, e.g., answer questions given wikipedia tables and their surrounding texts. |
| Table Interpretation | TI | Interpreting basic table content and structure information, e.g., column type annotation, entity linking, relation extraction, cell type classification et al. |
| Table Augmentation | TA | Augmenting existing tables with new data, e.g., schema augmentation, row population, et al. |
---


## 1. Survey of Tabular LLMs and Table Understanding
| Title | Source | Date  | Pages |
| ------ | :---: | :---: | :---: |
| [Table Question Answering in the Era of Large Language Models: A Comprehensive Survey of Tasks, Methods, and Evaluation](https://arxiv.org/abs/2510.09671) | arxiv | 2025-10-28 | 25 |
| [Toward Real-World Table Agents: Capabilities, Workflows, and Design Principles for LLM-based Table Intelligence](https://arxiv.org/abs/2507.10281) | arxiv | 2025-07-14 | 34 |
| [Language Modeling on Tabular Data: A Survey of Foundations, Techniques and Evolution](https://arxiv.org/abs/2408.10548) | arxiv | 2024-08-20 | 49 |
| [Large Language Model for Table Processing: A Survey](https://arxiv.org/abs/2402.05121) | arxiv | 2024-02-04  | 9 | 
| [A Survey of Table Reasoning with Large Language Models](https://arxiv.org/abs/2402.08259) | arxiv | 2024-02-13 | 9 |
| [Large Language Models(LLMs) on Tabular Data: Prediction, Generation, and Understanding -- A Survey](https://arxiv.org/abs/2402.17944) | arxiv | 2024-03-01 | 41 | 
| [Transformers for Tabular Data Representation: A Survey of Models and Applications](https://aclanthology.org/2023.tacl-1.14/) | TACL 2023 | | 23 |
| [Table Pre-training: A Survey on Model Architectures, Pre-training Objectives, and Downstream Tasks](https://arxiv.org/abs/2201.09745) | IJCAI 2022 | 2022-01-24 | 15 |

## 2. Prompting LLMs for Different Tabular Tasks
| Title | Source | Date |  Task | Code |
| --- | :---: | :---: | :---: | --- |
| [Map&Make: Schema Guided Text to Table Generation](https://arxiv.org/abs/2505.23174) | ACL 2025 | 2025-05-29 | Text-to-Table | |
| [Plugging Schema Graph into Multi-Table QA: A Human-Guided Framework for Reducing LLM Reliance](https://arxiv.org/abs/2506.04427) | EMNLP 2025 Findings | Multi-Table QA | [Github](https://github.com/hiXixi66/SGAM-Multi-table-QA) |
| [GRIT: Guided Relational Integration for Efficient Multi-Table Understanding](https://aclanthology.org/2025.emnlp-main.1118/) | EMNLP 2025 | | Multi-Table QA | |
| [RoT: Enhancing Table Reasoning with Iterative Row-Wise Traversals](https://arxiv.org/abs/2505.15110) | EMNLP 2025 | 2025-05-21 | TQA | |
| [Weaver: Interweaving SQL and LLM for Table Reasoning](https://arxiv.org/abs/2505.18961) | EMNLP 2025 | 2025-05-25 | TQA,TFV | |
| [Map&Make: Schema Guided Text to Table Generation](https://arxiv.org/pdf/2505.23174) | ACL 2025 | 2025-05-29 |Text2Table | |
| [Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning](https://arxiv.org/abs/2502.11799) | ACL 2025 | 2025-02-17 | TQA,TFV | [Github](https://github.com/Peiying-Yu/Table-Critic) |
| [Triples as the Key: Structuring Makes Decomposition and Verification Easier in LLM-based TableQA](https://openreview.net/forum?id=UwcZEoNP19) | ICLR 2025 | - | TQA | |
| [Piece of Table: A Divide-and-Conquer Approach for Selecting Subtables in Table Question Answering](https://arxiv.org/abs/2412.07629) | arxiv | 2024-12-10 | TQA | |
| [Tree-of-Table: Unleashing the Power of LLMs for Enhanced Large-Scale Table Understanding](https://arxiv.org/abs/2411.08516v1) | arxiv | 2024-11-13 | TQA,TFV,T2T | |
| [Retrieval & Fine-Tuning for In-Context Tabular Models](https://arxiv.org/abs/2406.05207) | NIPS 2024 | 2024-06-07 | Machine learning tasks with tabular data | |
| ![Star](https://img.shields.io/github/stars/JDing0521/GraphOTTER.svg?style=social&label=Star) <br> [GraphOTTER: Evolving LLM-based Graph Reasoning for Complex Table Question Answering](https://arxiv.org/abs/2412.01230) | COLING 2025 | 2024-12-02 | TQA | [Github](https://github.com/JDing0521/GraphOTTER) |
| [PoTable: Programming Standardly on Table-based Reasoning Like a Human Analyst](https://arxiv.org/abs/2412.04272) | arxiv | 2024-12-05 | TQA, TFV | |
| [Unveiling Implicit Table Knowledge with Question-Then-Pinpoint Reasoner for Insightful Table Summarization](https://arxiv.org/abs/2406.12269) | EMNLP 2024 Findings | 2024-06-18 | Table Summarization | | 
| [TKGT: Redefinition and A New Way of Text-to-Table Tasks Based on Real World Demands and Knowledge Graphs Augmented LLMs](https://openreview.net/forum?id=cfMRTgLYSf) | EMNLP 2024 | | Text2Table | |
| [Text-Tuple-Table: Towards Information Integration in Text-to-Table Generation via Global Tuple Extraction](https://arxiv.org/abs/2404.14215) | EMNLP 2024 | 2024-04-22 | Text2Table | [Github](https://github.com/HKUST-KnowComp/LiveSum) | 
| [TART: An Open-Source Tool-Augmented Framework for Explainable Table-based Reasoning](https://arxiv.org/abs/2409.11724) | arxiv | 2024-09-18 | TQA | [Github](https://github.com/XinyuanLu00/TART) |
| [SynTQA: Synergistic Table-based Question Answering via Mixture of Text-to-SQL and E2E TQA](https://arxiv.org/abs/2409.16682) | EMNLP 2024 | 2024-09-25 | TQA | |
| ![Star](https://img.shields.io/github/stars/zhxlia/FLEXTAF.svg?style=social&label=Star) <br> [FLEXTAF: Enhancing Table Reasoning with Flexible Tabular Formats](https://arxiv.org/abs/2408.08841) | arxiv | 2024-08-16 | TQA, TFV | [Github](https://github.com/zhxlia/FLEXTAF) |
| [Learning Relational Decomposition of Queries for Question Answering from Tables](https://aclanthology.org/2024.acl-long.564/) | ACL 2024 |  | TQA | | [Github](https://github.com/RaphaelMouravieff/Partial-Exec)
| [TaPERA: Enhancing Faithfulness and Interpretability in Long-Form Table QA by Content Planning and Execution-based Reasoning](https://aclanthology.org/2024.acl-long.692/) | ACL 2024 |  | TQA |  |
| [Enhancing Temporal Understanding in LLMs for Semi-structured Tables](https://arxiv.org/abs/2407.16030) | arxiv | 2024-07-22 | Temporal TQA | |
| ![Star](https://img.shields.io/github/stars/Hanzhang-lang/ALTER.svg?style=social&label=Star) <br> [ALTER: Augmentation for Large-Table-Based Reasoning](https://arxiv.org/abs/2407.03061) | arxiv | 2024-07-03 | TQA | [Github](https://github.com/Hanzhang-lang/ALTER) |
| [TrustUQA: A Trustful Framework for Unified Structured Data Question Answering](https://arxiv.org/abs/2406.18916) | arxiv | 2024-06-27 | TQA | |
| [Adapting Knowledge for Few-shot Table-to-Text Generation](https://arxiv.org/abs/2302.12468) | arxiv | 2024-03-27 |  T2T | |
| [Graph Reasoning Enhanced Language Models for Text-to-SQL](https://dl.acm.org/doi/abs/10.1145/3626772.3657961) | SIGIR 2024 | | NL2SQL | |
| [NormTab: Improving Symbolic Reasoning in LLMs Through Tabular Data Normalization](https://arxiv.org/abs/2406.17961) | arxiv | 2024-06-25 | TQA,TFV | |
| [Improving Factual Accuracy of Neural Table-to-Text Output by Addressing Input Problems in ToTTo](https://arxiv.org/abs/2404.04103) | NAACL 2024 | 2024-04-05 | T2T | |
| [TabSQLify: Enhancing Reasoning Capabilities of LLMs Through Table Decomposition](https://arxiv.org/abs/2404.10150) | NAACL 2024 |  |TQA,TFV |    |
| ![Star](https://img.shields.io/github/stars/zzh-SJTU/E5-Hierarchical-Table-Analysis.svg?style=social&label=Star) <br> [E5: Zero-shot Hierarchical Table Analysis using Augmented LLMs via Explain, Extract, Execute, Exhibit and Extrapolate](https://aclanthology.org/2024.naacl-long.68/) | NAACL 2024 |  | TQA on hierarchical tables | [Github](https://github.com/zzh-SJTU/E5-Hierarchical-Table-Analysis) |
| [OpenTE: Open-Structure Table Extraction From Text](https://ieeexplore.ieee.org/abstract/document/10448427) | ICASSP 2024 |  | Text-to-Table Extraction |  |
| [On Linearizing Structured Data in Encoder-Decoder Language Models: Insights from Text-to-SQL](https://arxiv.org/abs/2404.02389) |  NAACL 2024 | 2024-04-03 | NL2SQL |   | 
| [MFORT-QA: Multi-hop Few-shot Open Rich Table Question Answering](https://arxiv.org/abs/2403.19116) | arxiv | 2024-03-28 | TQA |  |
| ![Star](https://img.shields.io/github/stars/amazon-science/llm-open-domain-table-reasoner.svg?style=social&label=Star) <br> [OpenTab: Advancing Large Language Models as Open-domain Table Reasoners](https://arxiv.org/abs/2402.14361) | ICLR 2024 | 2024-02-22 | TQA,TFV | [Github](https://github.com/amazon-science/llm-open-domain-table-reasoner)  |
| [CABINET: Content Relevance based Noise Reduction for Table Question Answering](https://arxiv.org/abs/2402.01155) | ICLR 2024 | 2024-02-02 | TQA |   |
| ![Star](https://img.shields.io/github/stars/UCSB-NLP-Chang/Augment_tableQA.svg?style=social&label=Star) <br> [Augment before You Try: Knowledge-Enhanced Table Question Answering via Table Expansion](https://arxiv.org/abs/2401.15555) | EMNLP 2025 Findings | 2024-01-24 | TQA | [Github](https://github.com/UCSB-NLP-Chang/Augment_tableQA) |
| [Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding](https://arxiv.org/abs/2401.04398) | ICLR 2024 | 2024-01-09 | TQA,TFV | |
| [TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning](https://arxiv.org/abs/2312.09039) | EMNLP 2024 Findings | 2023-12-14 | TQA,TAT-QA,TFV,T2T | [Github](https://anonymous.4open.science/r/TableProvider-4CC3/README.md) |
| [Large Language Models are Complex Table Parsers](https://arxiv.org/abs/2312.11521) | EMNLP 2023 | 2023-12-13 | TQA  |   |
| [API-Assisted Code Generation for Question Answering on Varied Table Structures](https://arxiv.org/abs/2310.14687) | EMNLP 2023 | 2023-10-23 | TQA  |   |
| ![Star](https://img.shields.io/github/stars/lfy79001/TableQAKit.svg?style=social&label=Star) <br> [TableQAKit: A Comprehensive and Practical Toolkit for Table-based Question Answering](https://arxiv.org/abs/2310.15075) | arxiv | 2023-10-23 | TQA,NL2SQL  | [Github](https://github.com/lfy79001/TableQAKit)  |
| [Enhancing Few-shot Text-to-SQL Capabilities of Large Language Models: A Study on Prompt Design Strategies](https://arxiv.org/abs/2305.12586) | arxiv |  2023-05-21 | NL2SQL |  |
| ![Star](https://img.shields.io/github/stars/RUCAIBox/StructGPT.svg?style=social&label=Star) <br>[StructGPT: A General Framework for Large Language Model to Reason over Structured Data](https://arxiv.org/abs/2305.09645) | EMNLP 2023 | 2023-05-16 | TQA, TFV  |  [Github](https://github.com/RUCAIBox/StructGPT)   | 
| ![Star](https://img.shields.io/github/stars/lupantech/chameleon-llm.svg?style=social&label=Star) <br> [Chameleon：Plug-and-Play Compositional Reasoning with Large Language Models](https://arxiv.org/abs/2304.09842) | NIPS 2023 | 2023-04-19 | TMR | [Github](https://github.com/lupantech/chameleon-llm) |
| [Generate, Transform, Answer: Question Specific Tool Synthesis for Tabular Data](https://arxiv.org/abs/2303.10138) | EMNLP 2023 | 2023-03-17 | TQA,NL2SQL  |   |
| [DTT: An Example-Driven Tabular Transformer for Joinability by Leveraging Large Language Models](https://arxiv.org/abs/2303.06748) | SIGMOD 2024 | 2023-03-12 |  Table Transformation |   |
| ![Star](https://img.shields.io/github/stars/AlibabaResearch/DAMO-ConvAI.svg?style=social&label=Star) <br> [Large Language Models are Versatile Decomposers：Decompose Evidence and Questions for Table-based Reasoning](https://arxiv.org/abs/2301.13808) | SIGIR 2023 | 2023-01-13 |  TQA, TFV | [Github](https://github.com/AlibabaResearch/DAMO-ConvAI)    |
| ![Star](https://img.shields.io/github/stars/wenhuchen/Program-of-Thoughts.svg?style=social&label=Star) <br> [Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks](https://arxiv.org/abs/2211.12588) | TMLR 2023 | 2022-11-22  | TMR, TAT-QA | [Github](https://github.com/wenhuchen/Program-of-Thoughts)  |
| ![Star](https://img.shields.io/github/stars/wenhuchen/TableCoT.svg?style=social&label=Star) <br> [Large Language Models are few(1)-shot Table Reasoners](https://arxiv.org/abs/2210.06710) | EACL 2023 Findings | 2022-10-13  | TQA, TFV | [Github](https://github.com/wenhuchen/TableCoT)  |
| ![Star](https://img.shields.io/github/stars/xlang-ai/Binder.svg?style=social&label=Star) <br> [Binding Language Models in Symbolic Languages](https://arxiv.org/abs/2210.02875) | ICLR 2023 | 2022-10-06  |  TQA, TFV | [Github](https://github.com/xlang-ai/Binder)  |
| ![Star](https://img.shields.io/github/stars/lupantech/PromptPG.svg?style=social&label=Star) <br> [Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning](https://arxiv.org/abs/2209.14610) | ICLR 2023 | 2022-09-29 | TMR (Tabular Mathematical Reasoning)   |  [Github](https://github.com/lupantech/PromptPG)  |


## 3. Training LLMs for Better Table Understanding Ability

### 3.1 Supervised Fine-tuning (SFT) for Tabular LLMs
| Title | Source | Date | Task | LLM Backbone  | Code |
| --- | :---: | :---: | :---: | :---: | :---: |
| [QuASAR: A Question-Driven Structure-Aware Approach for Table-to-Text Generation](https://aclanthology.org/2025.acl-long.1300/) | ACL 2025 | | Table-to-text | T5-Base | [Github](https://github.com/weijieliu-cs/QuASAR) |
| [RelationalCoder: Rethinking Complex Tables via Programmatic Relational Transformation](https://aclanthology.org/2025.acl-long.89/) | ACL 2025 | | TQA with hierarchical tables | | [Github](https://github.com/haoyudong/RelationalCoder) |
| [Table-LLM-Specialist: Language Model Specialists for Tables using Iterative Generator-Validator Fine-tuning](https://arxiv.org/abs/2410.12164) | EMNLP 2025 | 2024-10-16 | Classification table tasks like Schema matching and Generative table tasks like TQA | GPT-3.5 and GPT-4 | [Github](https://github.com/microsoft/Table-Specialist) |
| [TableDreamer: Progressive and Weakness-guided Data Synthesis from Scratch for Table Instruction Tuning](https://arxiv.org/abs/2506.08646) | ACL 2025 Findings | 2025-06-10 | Synthesize diverse table instruction tuning data | Llama3.1-8B-Instruct | [Github](https://github.com/SpursGoZmy/TableDreamer) |
| [TableMoE: Neuro-Symbolic Routing for Structured Expert Reasoning in Multimodal Table Understanding](https://arxiv.org/abs/2506.21393) | arxiv | 2025-06-26 | Multimodal Table Understanding | LLaMA 3.1–8B–Instruct, Qwen2.5–VL–7B–Instruct | [Github](https://github.com/ai-agi/TableMoE) |
| [TableLoRA: Low-rank Adaptation on Table Structure Understanding for Large Language Models](https://arxiv.org/abs/2503.04396) | ACL 2025 | 2025-03-06 | TQA,TFV | DeepSeek, Llama2/3 with a specially designed LoRA module for table understanding | |
| [RePanda: Pandas-powered Tabular Verification and Reasoning](https://arxiv.org/abs/2503.11921) | arxiv | 2025-03-14 | TFV | DeepSeek-coder-7B-instruct-v1.5 | |
| [LaTeXNet: A Specialized Model for Converting Visual Tables and Equations to LaTeX Code](https://ieeexplore.ieee.org/abstract/document/10887698) | arxiv | ICASSP 2025 | Table-image-to-LaTeX |  |  |
| [General Table Question Answering via Answer-Formula Joint Generation](https://arxiv.org/abs/2503.12345) | arxiv | 2025-03-16 | TQA | Llama3.1, Qwen2.5-coder |  |
| [Rethinking Table Instruction Tuning](https://arxiv.org/abs/2501.14693) | ACL 2025 Findings | 2025-01-24 | TQA,TFV | Enhance OOD and general capacity of tabular LLMs | |
| [Bridging the Semantic Gap Between Text and Table: A Case Study on NL2SQL](https://openreview.net/forum?id=qmsX2R19p9) | ICLR 2025 | - | NL2SQL | LLMs with a specially trained table encoders.  | |
| [TableGPT2: A Large Multimodal Model with Tabular Data Integration](https://arxiv.org/abs/2411.02059v1) | arxiv | 2024-11-04 | TQA, TFV, et al. | Qwen2.5 model family with a special pre-trained table encoder. | [Github](https://github.com/tablegpt/tablegpt-agent) | 
| [Large Scale Transfer Learning for Tabular Data via Language Modeling](https://arxiv.org/abs/2406.12031) | NIPS 2024 | 2024-06-17 | tabular data prediction (classification and binned regression) | Llama 3-8B | |
| [ProTrix: Building Models for Planning and Reasoning over Tables with Sentence Context](https://arxiv.org/abs/2403.02177) | EMNLP 2024 Findings | 2024-03-04 | TQA, TFV | Llama-2 | [Github](https://github.com/WilliamZR/ProTrix) | 
| [UniTabNet: Bridging Vision and Language Models for Enhanced Table Structure Recognition](https://arxiv.org/abs/2409.13148) | EMNLP 2024 Findings | 2024-09-20 | Table Recognition | |
| [Table Question Answering for Low-resourced Indic Languages](https://arxiv.org/abs/2410.03576) | EMNLP 2024 | 2024-10-04 | Indian TQA | mBART | [Github](https://github.com/kolk/Low-Resource-TableQA-Indic-languages) | 
| [TabMoE: A General Framework for Diverse Table-Based Reasoning with Mixture-of-Experts](https://www.mdpi.com/2227-7390/12/19/3031) | Mathematics | 2024-08-16 | TQA, TFV, T2T | BART | |
| ![Star](https://img.shields.io/github/stars/rllm-team/rllm.svg?style=social&label=Star) <br/> [rLLM: Relational Table Learning with LLMs](https://arxiv.org/abs/2407.20157) | arxiv | 2024-07-29 | multi-table joint learning tasks | a PyTorch library designed for Relational Table Learning (RTL) with Large Language Models (LLMs).    |  [Github](https://github.com/rllm-team/rllm) |
| ![Star](https://img.shields.io/github/stars/basf/mamba-tabular.svg?style=social&label=Star) <br> [Mambular: A Sequential Model for Tabular Deep Learning](https://arxiv.org/abs/2408.06291) | arxiv | 2024-08-12 | ML Classification and Regression tasks like California Housing | Mamba | [Github](https://github.com/basf/mamba-tabular)
| [MambaTab: A Plug-and-Play Model for Learning Tabular Data](https://arxiv.org/abs/2401.08867) | MIPR 2024 | 2024-01-16 | ML Classification tasks | Mamba |  | 
| [SpreadsheetLLM: Encoding Spreadsheets for Large Language Models](https://arxiv.org/abs/2407.09025) | arxiv | 2024-07-12 | Excel Manipulation |   | |
| [Unleashing the Potential of Large Language Models for Predictive Tabular Tasks in Data Science](https://arxiv.org/abs/2403.20208) | arxiv  | 2024-03-29 | Predictive Tabular Tasks | Llama2 7B | [HuggingFace](https://huggingface.co/OldBirdAZ/itab-llm) |
| [HGT: Leveraging Heterogeneous Graph-enhanced Large Language Models for Few-shot Complex Table Understanding](https://arxiv.org/abs/2403.19723) | arxiv  | 2024-03-28 | TI,TQA | Vicuna-1.5 7B |  |
| ![Star](https://img.shields.io/github/stars/RUCKBReasoning/TableLLM.svg?style=social&label=Star) <br> [TableLLM: Enabling Tabular Data Manipulation by LLMs in Real Office Usage Scenarios](https://arxiv.org/abs/2403.19318) | ACL 2025 Findings | 2024-03-28 | Table Manipulation | CodeLlama 7B, 13B | [Github](https://github.com/RUCKBReasoning/TableLLM) |
| ![Star](https://img.shields.io/github/stars/TIGER-AI-Lab/StructLM.svg?style=social&label=Star) <br> [StructLM: Towards Building Generalist Models for Structured Knowledge Grounding](https://arxiv.org/abs/2402.16671) | CoLM 2024 | 2024-02-26 | TQA,TFV,T2T,NL2SQL  | CodeLlama 7B-34B   | [Github](https://github.com/TIGER-AI-Lab/StructLM)  |
| ![Star](https://img.shields.io/github/stars/fengbinzhu/TAT-LLM.svg?style=social&label=Star) <br> [TAT-LLM: A Specialized Language Model for Discrete Reasoning over Tabular and Textual Data](https://arxiv.org/abs/2401.13223) | arxiv | 2024-01-24 | TQA | Llama2 7B, 13B, 70B | [Github](https://github.com/fengbinzhu/TAT-LLM) |
| ![Star](https://img.shields.io/github/stars/OSU-NLP-Group/TableLlama.svg?style=social&label=Star) <br> [TableLlama: Towards Open Large Generalist Models for Tables](https://arxiv.org/abs/2311.09206) | NAACL 2024 | 2023-11-15 | TQA,TFV,T2T,TA,TI  | Llama2 7B | [Github](https://github.com/OSU-NLP-Group/TableLlama)  |
| [HELLaMA: LLaMA-based Table to Text Generation by Highlighting the Important Evidence](https://arxiv.org/abs/2311.08896)  | arxiv | 2023-11-15 | T2T | Llama2 7B-13B |  |
| [Table-GPT: Table-tuned GPT for Diverse Table Tasks](https://arxiv.org/abs/2310.09263)   | arxiv  | 2023-10-13  | TQA | GPT-3.5, ChatGPT |   |

### 3.2 Reinforcement Learning (RL) for Tabular LLM 
| Title | Source | Date | Task | LLM Backbone  | Code |
| --- | :---: | :---: | :---: | :---: | :---: |
| [Exploring Generative Process Reward Modeling for Semi-Structured Data: A Case Study of Table Question Answering](https://arxiv.org/abs/2510.20304) | arxiv | 2025-10-23 | Evalating current PRM for TQA | | |
| [STaR: Towards Cognitive Table Reasoning via Slow-Thinking Large Language Models](https://arxiv.org/abs/2511.11233) | arxiv | 2025-11-14 | TQA | Qwen3-0.5B/8B | [Github](https://github.com/zhjai/STaR) |
| [TaTToo: Tool-Grounded Thinking PRM for Test-Time Scaling in Tabular Reasoning](https://arxiv.org/abs/2510.06217) | arxiv | 2025-10-07 | Building a better PRM for tabular task |  Qwen-3-8B |  |
| [Can GRPO Boost Complex Multimodal Table Understanding?](https://arxiv.org/abs/2509.16889) | EMNLP 2025 | 2025-09-21 | Multimodal Table Understanding | Qwen2-VL-7B |  |
| [PPT: A Process-based Preference Learning Framework for Self Improving Table Question Answering Models](https://www.arxiv.org/abs/2505.17565) | arxiv | 2025-05-23 | Qwen2.5-7B, Llama3.1-8B | |
| [Fortune: Formula-Driven Reinforcement Learning for Symbolic Table Reasoning in Language Models](https://arxiv.org/abs/2505.23667) | arxiv | 2025-05-29 | TQA, TFV | Multiple LLMs | |
| [Reasoning-Table: Exploring Reinforcement Learning for Table Reasoning](https://arxiv.org/abs/2506.01710) | arxiv | 2025-06-02 | TQA,TFV,T2T,NL2SQL | Qwen2.5-7B | [Github](https://github.com/MJinXiang/Reasoning-Table) |
| [OpenTable-R1: A Reinforcement Learning Augmented Tool Agent for Open-Domain Table Question Answering](https://arxiv.org/abs/2507.03018) | arxiv | 2025-07-02 | Open-Domain TQA | | [Github](https://github.com/TabibitoQZP/OpenTableR1) |
| [Table-r1: Self-supervised and Reinforcement Learning for Program-based Table Reasoning in Small Language Models](https://arxiv.org/abs/2506.06137) | arxiv | 2025-06-06 | TQA | Qwen2.5-Coder-7B-Inst, LLaMA3.1-8B-Inst | [Github](https://github.com/AriKing11/Table_r1_public) |
| ![Star](https://img.shields.io/github/stars/Table-R1/Table-R1.svg?style=social&label=Star) <br/> [Table-R1: Inference-Time Scaling for Table Reasoning](https://arxiv.org/abs/2505.23621) | EMNLP 2025 | 2025-05-29 | TQA,TFV,T2T| Qwen2.5-7B | [Github](https://github.com/Table-R1/Table-R1) |
| [Table-R1: Region-based Reinforcement Learning for Table Understanding](https://arxiv.org/abs/2505.12415) | arxiv | 2025-05-18 | TQA | Multiple LLMs  |   |
| ![Star](https://img.shields.io/github/stars/NEUIR/HIPPO.svg?style=social&label=Star) <br/> [HIPPO: Enhancing the Table Understanding Capability of Large Language Models through Hybrid-Modal Preference Optimization](https://arxiv.org/abs/2502.17315) | arxiv | 2025-02-24 | TQA,TFV | MiniCPM-V-2.6 with DPO training | [Github](https://github.com/NEUIR/HIPPO)  |

### 3.3 Pre-trained Tabular Language Models (non-LLM)
| Title | Source | Date |  Task | Code |
| --- | :---: | :---: | :---: | :---: |
| [Structural Deep Encoding for Table Question Answering](https://arxiv.org/abs/2503.01457) | ACL 2025 Findings | 2025-03-03 | WTQ, WikiSQL | |
| ![Star](https://img.shields.io/github/stars/awslabs/hypergraph-tabular-lm.svg?style=social&label=Star) <br> [HYTREL: Hypergraph-enhanced Tabular Data Representation Learning](https://arxiv.org/abs/2307.08623) | NIPS 2023 | 2023-07-14 |  TA, TI | [Github](https://github.com/awslabs/hypergraph-tabular-lm) |
| [FLAME: A small language model for spreadsheet formulas](https://arxiv.org/abs/2301.13779)  | AAAI 2024 | 2023-01-31 | Generating Excel Formulas | [Github](https://github.com/microsoft/prose-benchmarks/tree/main/FLAME)  |

## 4. Developing Agents for Processing Tabular Data
| Title | Source | Date |  Task | Code |
| --- | :---: | :---: | :---: | :---: |
| [TST: A Schema-Based Top-Down and Dynamic-Aware Agent of Text-to-Table Tasks](https://aclanthology.org/2025.acl-long.829/) | ACL 2025 | | Text-to-Table | [Github](https://github.com/jiangpw41/TST) |
| [Beyond Summaries: Multi-Agent Generation of Investment Reports with Text, Tables, and Charts](https://aclanthology.org/2025.finnlp-2.18/) | EMNLP 2025 Findings | | Investment Reports Generation | [Github](https://github.com/RaphaelYangWJ/earnings2insights) |
| [TALON: A Multi-Agent Framework for Long-Table Exploration and Question Answering](https://aclanthology.org/2025.emnlp-main.1393/) | EMNLP 2025 |TQA and Text2SQL like WTQ and BirdQA | [Github](https://github.com/Wwestmoon/TALON) |
| [SheetAgent: A Generalist Agent for Spreadsheet Reasoning and Manipulation via Large Language Models](https://arxiv.org/abs/2403.03636) | arxiv | 2024-03-06 | Manipulating Excels with LLM | [Github](https://github.com/sheetagent/sheetagent.github.io) |
| ![Star](https://img.shields.io/github/stars/wshi83/EhrAgent.svg?style=social&label=Star) <br> [EHRAgent: Code Empowers Large Language Models for Few-shot Complex Tabular Reasoning on Electronic Health Records](https://arxiv.org/pdf/2401.07128.pdf) | arxiv | 2024-01-13 | TQA | [Github](https://github.com/wshi83/EhrAgent) |
| ![Star](https://img.shields.io/github/stars/InfiAgent/InfiAgent.svg?style=social&label=Star) <br> [InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks](https://arxiv.org/abs/2401.05507) | arxiv | 2024-01-10 | Data Analysis | [Github](https://github.com/InfiAgent/InfiAgent) |
| ![Star](https://img.shields.io/github/stars/eosphoros-ai/DB-GPT.svg?style=social&label=Star) <br> [DB-GPT: Empowering Database Interactions with Private Large Language Models](https://arxiv.org/abs/2312.17449) | arxiv | 2023-12-29 | Data Analysis | [Github](https://github.com/eosphoros-ai/DB-GPT) |
| [ReAcTable: Enhancing ReAct for Table Question Answering](https://arxiv.org/abs/2310.00815) | arxiv | 2023-10-01 | TQA | |
| ![Star](https://img.shields.io/github/stars/BraveGroup/SheetCopilot.svg?style=social&label=Star) <br>[SheetCopilot: Bringing Software Productivity to the Next Level through Large Language Models](https://arxiv.org/abs/2305.19308) | NIPS 2023  |2023-05-30 | Manipulating Excels with LLM | [Github](https://github.com/BraveGroup/SheetCopilot)  |
| [TableGPT: Towards Unifying Tables, Nature Language and Commands into One GPT](https://arxiv.org/abs/2307.08674) | arxiv | 2023-07-17 | Manipulating CSV table with LLM | |

## 5. RAG with Tabular Data
| Title | Source | Date |  Task | Code |
| --- | :---: | :---: | :---: | :---: |
| [TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning](https://arxiv.org/abs/2506.10380) | EMNLP 2025 | 2025-06-12 | TQA | [Github](https://github.com/yxh-y/TableRAG/tree/main) |
| [HD-RAG: Retrieval-Augmented Generation for Hybrid Documents Containing Text and Hierarchical Tables](https://arxiv.org/abs/2504.09554) | arxiv | 2025-04-13 | TQA | |
| [GTR: Graph-Table-RAG for Cross-Table Question Answering](https://arxiv.org/abs/2504.01346) | arxiv | 2025-04-02 | Cross-table Question Answering | |
| [TableRAG: Million-Token Table Understanding with Language Models](https://arxiv.org/abs/2410.04739) | NIPS 2024 | 2024-10-07 | TQA for extremely long tables | |
| [Evaluation of Table Representations to Answer Questions from Tables in Documents : A Case Study using 3GPP Specifications](https://arxiv.org/abs/2408.17008) | arxiv | 2024-08-30 | how to represent tables for better retrieval within RAG systems | |
| [THoRR: Complex Table Retrieval and Refinement for RAG](https://ceur-ws.org/Vol-3784/short2.pdf) | IR-RAG 2024 workshop |  | RAG with large and complex tables | |

## 6. Empirical Study for Evaluating LLMs' Table Understanding Ability
| Title | Source | Date |  Task | Code |
| --- | :---: | :---: | :---: | :---: |
| ![Stars](https://img.shields.io/github/stars/socialfoundations/folktexts) <br> [Evaluating language models as risk scores](https://openreview.net/pdf?id=qrZxL3Bto9) | NeurIPS 2024 | 2024-12-10 | TQA | [Github](https://github.com/socialfoundations/folktexts) |
| [Rethinking Tabular Data Understanding with Large Language Models](https://arxiv.org/abs/2312.16702) | NAACL 2024 | 2023-12-27 | TQA | |
| [On the Robustness of Language Models for Tabular Question Answering](https://arxiv.org/abs/2406.12719) | arxiv | 2024-06-18 | TQA | |
| [FREB-TQA: A Fine-Grained Robustness Evaluation Benchmark for Table Question Answering](https://arxiv.org/abs/2404.18585) | NAACL 2024 | 2024-04-29 | TQA | |
| [How Robust are the Tabular QA Models for Scientific Tables? A Study using Customized Dataset](https://arxiv.org/abs/2404.00401) | arxiv | 2024-03-20 | TQA | |
| ![Star](https://img.shields.io/github/stars/microsoft/InstructExcel.svg?style=social&label=Star) <br> [InstructExcel: A Benchmark for Natural Language Instruction in Excel](https://arxiv.org/abs/2310.14495) | Findings of EMNLP 2023 |  2023-10-23 | Excel operations | [Github](https://github.com/microsoft/InstructExcel) |
| [Tabular Representation, Noisy Operators, and Impacts on Table Structure Understanding Tasks in LLMs](https://arxiv.org/abs/2310.10358) | arxiv | 2023-10-16 | Fact-Finding Tasks, Transformation Tasks  |    |
|  ![Star](https://img.shields.io/github/stars/yale-nlp/LLM-T2T.svg?style=social&label=Star) <br> [Investigating Table-to-Text Generation Capabilities of LLMs in Real-World Information Seeking Scenarios](https://arxiv.org/abs/2305.14987) | EMNLP 2023 | 2023-05-24 | T2T | [Github](https://github.com/yale-nlp/LLM-T2T)   |
| ![Star](https://img.shields.io/github/stars/dylan-slack/Tablet.svg?style=social&label=Star) <br> [TABLET: Learning From Instructions For Tabular Data](https://arxiv.org/abs/2304.13188) | arxiv | 2023-04-25 |    |  [Github](https://github.com/dylan-slack/Tablet)   |
| [Table Meets LLM: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study](https://arxiv.org/abs/2305.13062) | WSDM 2024 | 2023-05-22 | TQA,TFV,T2T | |
| [Evaluating the Text-to-SQL Capabilities of Large Language Models](https://arxiv.org/abs/2204.00498) | arxiv  | 2022-03-15 | NL2SQL | |
| ![Star](https://img.shields.io/github/stars/THU-BPM/chatgpt-sql.svg?style=social&label=Star) <br> [A comprehensive evaluation of ChatGPT's zero-shot Text-to-SQL capability](https://arxiv.org/abs/2303.13547) | arxiv |  2023-03-12 | NL2SQL |  [Github](https://github.com/THU-BPM/chatgpt-sql) |
| ![Star](https://img.shields.io/github/stars/yilunzhao/RobuT.svg?style=social&label=Star) <br> [RobuT: A Systematic Study of Table QA Robustness Against Human-Annotated Adversarial Perturbations](https://arxiv.org/abs/2306.14321) | ACL 2023 | 2023-06-25  | TQA | [Github](https://github.com/yilunzhao/RobuT) |


## 7. Multimodal Table Understanding
| Title | Source | Date |  Task | Code |
| --- | :---: | :---: | :---: | :---: |
| [Texts or Images? A Fine-grained Analysis on the Effectiveness of Input Representations and Models for Table Question Answering](https://arxiv.org/abs/2505.14131) | ACL 2025 Findings | 2025-05-20 | TQA | [Github](https://github.com/boschresearch/FRES) | 
| [Compositional Condition Question Answering in Tabular Understanding](https://openreview.net/forum?id=aXU48nrA2v) | ICML 2025 |  |  | [Github](https://github.com/LAMDA-Tabular/MMTU) |
| [Enhancing Large Vision-Language Models with Layout Modality for Table Question Answering on Japanese Annual Securities Reports](https://arxiv.org/abs/2505.17625) | IIAI AAI 2025 | 2025-05-23 | |
| [TableMoE: Neuro-Symbolic Routing for Structured Expert Reasoning in Multimodal Table Understanding](https://arxiv.org/abs/2506.21393) | arxiv | 2025-06-26 | Multimodal Table Understanding | LLaMA 3.1–8B–Instruct, Qwen2.5–VL–7B–Instruct | [Github](https://github.com/ai-agi/TableMoE) |
| [Multimodal Tabular Reasoning with Privileged Structured Information](https://arxiv.org/abs/2506.04088) | arxiv | 2025-06-04 | |
| [SynTab-LLaVA: Enhancing Multimodal Table Understanding with Decoupled Synthesis](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_SynTab-LLaVA_Enhancing_Multimodal_Table_Understanding_with_Decoupled_Synthesis_CVPR_2025_paper.html) | CVPR 2025 | | Understanding table images | |
| [MMTBENCH: A Unified Benchmark for Complex Multimodal Table Reasoning](https://arxiv.org/abs/2505.21771) | arxiv | 2025-05-27 | Complex Multimodal Table Reasoning | |
| ![Star](https://img.shields.io/github/stars/Bernard-Yang/MMSci_Table.svg?style=social&label=Star) <br> [Does Table Source Matter? Benchmarking and Improving Multimodal Scientific Table Understanding and Reasoning](https://arxiv.org/abs/2501.13042) | arxiv | 2025-01-22 | Understanding Scientific Table Images | |
| [Knowledge-Aware Reasoning over Multimodal Semi-structured Tables](https://arxiv.org/abs/2408.13860) | EMNLP 2024 Findings | 2024-08-25 | Understanding table images with visual elements like symbols and icons | |
| [Leopard: A Vision Language Model For Text-Rich Multi-Image Tasks](https://github.com/tencent-ailab/Leopard) | arxiv | 2024-10-02 | Multi Table Image QA | [Github](https://github.com/tencent-ailab/Leopard) |
| ![Star](https://img.shields.io/github/stars/alonsoapp/PixT3.svg?style=social&label=Star) <br> [PixT3: Pixel-based Table-To-Text Generation](https://arxiv.org/abs/2311.09808) | ACL 2024 | 2023-11-16 | T2T | [Github](https://github.com/alonsoapp/PixT3) |
| [TabPedia: Towards Comprehensive Visual Table Understanding with Concept Synergy](https://arxiv.org/abs/2406.01326) | NIPS 2024 | 2024-06-03 | TQA,TI | |
| ![Star](https://img.shields.io/github/stars/naver-ai/tablevqabench.svg?style=social&label=Star) <br> [TableVQA-Bench: A Visual Question Answering Benchmark on Multiple Table Domains](https://arxiv.org/abs/2404.19205) | arxiv  | 2024-04-30 | TQA, TFV | [Github](https://github.com/naver-ai/tablevqabench) |
| [Tables as Texts or Images: Evaluating the Table Reasoning Ability of LLMs and MLLMs](https://arxiv.org/abs/2402.12424) | ACL 2024 | 2024-02-19 | TQA,TFV,T2T | |
| ![Star](https://img.shields.io/github/stars/SpursGoZmy/Table-LLaVA.svg?style=social&label=Star) <br> [Multimodal Table Understanding](https://arxiv.org/abs/2406.08100) | ACL 2024  | 2024-02-15 | TQA, TFV, T2T, TI, TAT-QA, TMR | [Github](https://github.com/SpursGoZmy/Table-LLaVA)   |

## 8. Table Understanding Datasets and Benchmarks
### 8.1 Recent Datasets and Benchmarks for LLMs
| Title | Source | Date | Task | Data Volume | Domain | Table Type | Data and Code |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts](https://arxiv.org/abs/2506.15594) | ACL 2025 Findings | 2025-06-18 | QA over over Tables and Charts | 1,000 multiple-choice questions | diverse domains like Economy, Geography, History, Politics, Science, Sport | [Github](https://github.com/negar-foroutan/WikiMixQA) |
| [TabXEval: Why this is a Bad Table? An eXhaustive Rubric for Table Evaluation](https://arxiv.org/abs/2505.22176) | ACL 2025 Findings | 2025-05-28 | evaluate generated tables | | | | [Github](https://coral-lab-asu.github.io/tabxeval/) |
| [GRI-QA: a Comprehensive Benchmark for Table Question Answering over Environmental Data](https://aclanthology.org/2025.findings-acl.814/) | ACL 2025 Findings | | TQA | 4089 questions, 204 tables | environmental | flat and hierarchical tables | [Github](https://github.com/softlab-unimore/gri_qa) | 
| [2Columns1Row: A Russian Benchmark for Textual and Multimodal Table Understanding and Reasoning](https://aclanthology.org/2025.findings-emnlp.721/) | EMNLP 2025 Findings | | Textual and Multimodal TQA in Russian | 28,800 instances | | |
| [LongTableBench: Benchmarking Long-Context Table Reasoning across Real-World Formats and Domains](https://aclanthology.org/2025.findings-emnlp.638/) | EMNLP 2025 Findings | | Long-table QA | 5,950 QA instances spanning 7 table format, and input lengths up to 128K tokens, including multi-turn and multi-table settings |  18 domains | flat tables | [Github](https://github.com/liyaooi/LongTableBench) |
| [Table-Text Alignment: Explaining Claim Verification Against Tables in Scientific Papers](https://arxiv.org/abs/2506.10486) | EMNLP 2025 Findings | 2025-06-12 | scientific table-based verification | 372 samples | scientic | flat | [Github](https://github.com/Alab-NII/SciTabAlign) |
| [SportReason: Evaluating Retrieval-Augmented Reasoning across Tables and Text for Sports Question Answering](https://aclanthology.org/2025.emnlp-main.34/) | EMNLP 2025 | - | RAG over table and text data for Sports QA | 3,000 QA pairs | Sports | flat table | [Github](https://github.com/kaiyuef/SportReason) | 
| [T2R-bench: A Benchmark for Generating Article-Level Reports from Real World Industrial Tables](https://www.arxiv.org/abs/2508.19813) | EMNLP 2025 | 2025-08-27 | Table2Reports | 457 real-world industrial tables | 19 industry domains | four table types | [Github](https://github.com/Tele-AI/TeleTableBench) |
| [MTabVQA: Evaluating Multi-Tabular Reasoning of Language Models in Visual Space](https://arxiv.org/abs/2506.11684) | arxiv | 2025-06-13 | Multi-Tabular Reasoning | 3,745 complex question-answer pairs | | | [huggingface](https://huggingface.co/datasets/mtabvqa/MTabVQA-Eval) |
| [TReB: A Comprehensive Benchmark for Evaluating Table Reasoning Capabilities of Large Language Models](https://arxiv.org/abs/2506.18421) | arxiv | 2025-06-23 |  26 table-related tasks such as data analysis | 7,790 samples | | | |
| [TableEval: A Real-World Benchmark for Complex, Multilingual, and Multi-Structured Table Question Answering](https://arxiv.org/abs/2506.03949) | EMNLP 2025 | 2025-06-11 | Data Analysis, Information Retrieval, Numerical Analysis | 617 tables and 2,325 QA pairs | financial reports, industry/stock research reports, academic papers and goverment reports | Flat, hierarchical and complex tables | [Github](https://github.com/wenge-research/TableEval) |
| [RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis](https://arxiv.org/abs/2506.13405) | ACL 2025 | 2025-06-19 | Table analysis over complex tables | 708 tables,  3,752 QA pairs | 24 domains like economy, society, science | complex tables in image and textual format | [Github](https://github.com/cspzyy/RealHiTBench) |
| [Automated Text-to-Table for Reasoning-Intensive Table QA: Pipeline Design and Benchmarking Insights](https://arxiv.org/abs/2505.19563) | arxiv | 2025-05-26 | Text2Table | | | | [Github](https://github.com/jokersio-tsy/AutoT2T) |
| [MULTITAT: Benchmarking Multilingual Table-and-Text Question Answering](https://arxiv.org/abs/2502.17253) | EMNLP 2025 | 2025-02-24 | Multilingual Table-and-Text Question Answering | 250 samples | | | [Github](https://github.com/zhxlia/MULTITAT) |
| [MT-RAIG: Novel Benchmark and Evaluation Framework for Retrieval-Augmented Insight Generation over Multiple Tables](https://arxiv.org/abs/2502.11735) | ACL 2025 | 2025-02-17 | Insight Generation over Mulitple-Tables | 19,563 tables and 18,532 questions | Tables from SPIDER and Wikipedia | Flat tables | [Github](https://github.com/KWONDU/mt-raig) |
| [TransientTables: Evaluating LLMs' Reasoning on Temporally Evolving Semi-structured Tables](https://arxiv.org/abs/2504.01879) | arxiv | 2025-04-02 | TQA over temporally evolving semi-structured tables | 3,971 questions, 14,000 tables | Wikipedia |  Infobox tables | [Github](https://transienttables.github.io/) |
| [SCITAT: A Question Answering Benchmark for Scientific Tables and Text Covering Diverse Reasoning Types](https://arxiv.org/abs/2412.11757) | ACL 2025 Findings | 2024-12-16 | lookup, numerical reasoning, analysis and tabulation | 953 samples | | | [Github](https://github.com/zhxlia/SciTaT) |
| [MMQA: Evaluating LLMs with Multi-Table Multi-Hop Complex Questions](https://openreview.net/forum?id=GGlpykXDCa) | ICLR 2025 | - | Multi-table retrieval, NL2SQL, Multi-table QA, and Key Selection (primary key and foreign key) | 3,312 tables | Wikipedia | Flat tables |  |
| [SpreadsheetBench: Towards Challenging Real World Spreadsheet Manipulation](https://arxiv.org/abs/2406.14991) | NIPS 2024 | 2024-06-21 | Spreadsheet Manipulation | 2729 spreadsheets, 912 instructions | Excel Forum & Blog | Flat tables, hierarchical tables, multi-tables | [Github](https://spreadsheetbench.github.io/) |
| [MiMoTable: A Multi-scale Spreadsheet Benchmark with Meta Operations for Table Reasoning](https://arxiv.org/abs/2412.11711) | COLING 2024 | 2024-12-16 | TQA,T2T,Table manipulation, Data analysis |  1,719 (spreadsheet, question, answer) triplets from 428 different spreadsheets | Multiple domains | Flat and hierarchical tables | [Github](https://github.com/jasonNLP/MiMoTable) |
| [ENTRANT: A Large Financial Dataset for Table Understanding](https://www.nature.com/articles/s41597-024-03605-5) | Sci Data | 2024-07-04 | Cell Type Classification, Header Extraction, et al | Millions of tables with cell attributes, as well as positional and hierarchical information | Financial | Flat tables and hierarchical tables | [Github](https://github.com/iit-Demokritos/entrant?tab=readme-ov-file#data) |
| [TableBench: A Comprehensive and Complex Benchmark for Table Question Answering](https://arxiv.org/abs/2408.09174) | arxiv | 2024-08-17 | TMR, TFV, Trend Forecasting and Chart Generation | 3681 tables and 20K samples | Collect tables from academic datasets like WTQ and FeTaQA | Flat tables and a small number of hierarchical tables | [Github](https://tablebench.github.io/) |
| [DocTabQA: Answering Questions from Long Documents Using Tables](https://arxiv.org/abs/2408.11490) | arxiv | 2024-08-21 | Table Generation based on question and document | 300 documents and 1.5k question-table pairs | Financial | Flat tables and hierarchical tables | [Github](https://github.com/SmileWHC/DocTabQA) |

### 8.2 Classic Datasets of Downstream Table Tasks

## 9. Designing Evaluation Metrics for Table Understanding
| Title | Source | Date |  Task | Code |
| --- | :---: | :---: | :---: | :---: |
| [Revisiting Automated Evaluation for Long-form Table Question Answering in the Era of Large Language Models](https://openreview.net/forum?id=3PABAHvV6H) | EMNLP 2024 | | TQA | |
| [Is This a Bad Table? A Closer Look at the Evaluation of Table Generation from Text](https://arxiv.org/abs/2406.14829) | EMNLP 2024 | 2024-06-21 | Text2Table | |



