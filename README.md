# A-Paper-List-of-Awesome-Tabular-LLMs
Different types of tables are widely used to store and present information. To automatically process numerous tables and gain valuable insights, researchers have proposed a series of deep-learning models for various table-based tasks, e.g., table question answering (TQA), table-to-text (T2T), text-to-sql (NL2SQL) and table fact verification (TFV). Recently, the emerging [Large Language Models (LLMs)](https://github.com/Hannibal046/Awesome-LLM#chatgpt-evaluation) and more powerful [Multimodal Large Language Models (MLLMs)](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) have opened up new possibilities for processing the tabular data, i.e., we can use one general model to process diverse tables and fulfill different tabular tasks based on the user natural language instructions. We refer to these LLMs speciallized for tabular tasks as `Tabular LLMs`. **In this repository, we collect a paper list about Tabular (M)LLMs.**

---
<font size=8><center><b> Table of Contents: </b> </center></font>
1. [**Survey of Tabular LLMs**](#1-survey-of-tabular-llms)
2. [**Prompting LLMs for different tabular tasks**](#2-prompting-llms-for-tabular-tasks), e.g., in-context learning, prompt engineering and integrating external tools.
3. [**Training LLMs for better table understanding**](#3-training-llms-for-better-table-understanding), e.g., training existing LLMs by instruction fine-tuning or post-pretraining.
4. [**Developing agents for diverse tabular data**](#4-developing-agents-for-diverse-tabular-data), e.g., devolping copilot for processing excel tables.
5. [**Empirical study of LLMs' table understanding ability**](#5-empirical-study-of-llms-table-understanding-ability), e.g., exploring the influence of various table types or table formats.
6. [**Datasets for Tabular LLMs**](#6-datasets-for-tabular-llms), e.g., instruction tuning data for table-based tasks.
---

## 1. Survey of Tabular LLMs
| Title | Conference | Date  | Pages |
| ------ | :---: | :---: | :---: |
| [Large Language Model for Table Processing: A Survey](https://arxiv.org/abs/2402.05121) | arxiv | 2024-02-04  | 9 | 
| [A Survey of Table Reasoning with Large Language Models](https://arxiv.org/abs/2402.08259) | arxiv | 2024-02-13 | 9 |
| [Large Language Models(LLMs) on Tabular Data: Prediction, Generation, and Understanding -- A Survey](https://arxiv.org/abs/2402.17944) | arxiv | 2024-03-01 | 41 | 

## 2. Prompting LLMs for tabular tasks
| Title | Conference and Time |  Task | Summary |
| --- | :---: | :---: | :---: |
| [Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning](https://arxiv.org/abs/2209.14610) | ICLR 2023 |   Tabular Mathematical Reasoning |   |
| [Large Language Models are few(1)-shot Table Reasoners](https://arxiv.org/abs/2210.06710) | EACL 2023 Findings |  TQA, TFV |  |
| [Large Language Models are Versatile Decomposers：Decompose Evidence and Questions for Table-based Reasoning](https://arxiv.org/abs/2301.13808) | SIGIR 2023 |   TQA, TFV |   |
| [Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks](https://arxiv.org/abs/2211.12588) | 2022.11.22 on arxiv |  Tabular Mathematical Reasoning |  |
| [Chameleon：Plug-and-Play Compositional Reasoning with Large Language Models](https://arxiv.org/abs/2304.09842) | 2023.4.19 on arxiv |  Tabular Mathematical Reasoning with external API tools |  |
| [Binding Language Models in Symbolic Languages](https://arxiv.org/abs/2210.02875) | ICLR 2023 |   TQA, TFV |  |
| [Large Language Models are Effective Table-to-Text Generators, Evaluators, and Feedback Providers](https://arxiv.org/abs/2305.14987) | 2023.5.24 on arxiv | Table-to-text |  |
| [Enhancing Few-shot Text-to-SQL Capabilities of Large Language Models: A Study on Prompt Design Strategies](https://arxiv.org/abs/2305.12586) | 2023.5.21 on arxiv | Text-to-SQL | |
| [StructGPT: A General Framework for Large Language Model to Reason over Structured Data](https://arxiv.org/abs/2305.09645) | EMNLP 2023 | TQA, TFV  |

## 3. Training LLMs for better table understanding
| Title | Conference and Time |  Task | Summary |
| --- | :---: | :---: | :---: |
|[FLAME: A small language model for spreadsheet formulas](https://arxiv.org/abs/2301.13779) | 2023.1.31 on arxiv | Generating Excel Formulas | |
| [Table-GPT: Table-tuned GPT for Diverse Table Tasks](https://arxiv.org/abs/2310.09263)   | All kinds of table task!!  | **A Must Read work from Microsoft**   |

### Excellent pre-trained Tabular Language Models (non-LLM)
| Title | Conference and Time |  Task | Summary |
| --- | :---: | :---: | :---: |
|[HYTREL: Hypergraph-enhanced Tabular Data Representation Learning](https://arxiv.org/abs/2307.08623) | NIPS 2023 | Column Type Annotation,Column Property Annotation,Table Type Detection,Table Similarity Prediction | |

## 4. Developing agents for diverse tabular data
| Title | Conference and Time |  Task | Summary |
| --- | :---: | :---: | :---: |
| [SheetCopilot: Bringing Software Productivity to the Next Level through Large Language Models](https://arxiv.org/abs/2305.19308) | 20230.5.30 on arxiv | Manipulating SpreedSheets with LLM | |
| [TableGPT: Towards Unifying Tables, Nature Language and Commands into One GPT](https://arxiv.org/abs/2307.08674) | 2023.7.17 on arxiv | Manipulating CSV table with LLM | |


## 5. Empirical study of LLMs' table understanding ability
| Title | Conference and Time |  Task | Summary |
| --- | :---: | :---: | :---: |
| [Evaluating and Enhancing Structural Understanding Capabilities of Large Language Models on Tables via Input Designs](https://arxiv.org/abs/2305.13062) | 2023.5.22 on arxiv | TQA, Table Fact Verfication, Table-to-text | |
|[Evaluating the Text-to-SQL Capabilities of Large Language Models](https://arxiv.org/abs/2204.00498) | 2022.3.15 on arxiv | Text-to-SQL | |
|[A comprehensive evaluation of ChatGPT's zero-shot Text-to-SQL capability](https://arxiv.org/abs/2303.13547) | 2023.3.12 on arxiv | Text-to-SQL | |
|[RobuT: A Systematic Study of Table QA Robustness Against Human-Annotated Adversarial Perturbations](https://arxiv.org/abs/2306.14321) | ACL 2023 | TQA | |

## 6. Datasets for Tabular LLMs
| Title | Conference and Time |  Task | Summary |
| --- | :---: | :---: | :---: |
| [InstructExcel: A Benchmark for Natural Language Instruction in Excel](https://arxiv.org/abs/2310.14495) | Findings of EMNLP 2023 | Excel operations | |


 
