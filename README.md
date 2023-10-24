# A-Paper-List-of-Awesome-Tabular-LLMs
Different types of tables are widely used by individual users or companies to store information. To automatically process amounts of tables and gain valuable insights, researchers have proposed a series of deep-learning models for various table-related tasks, e.g., table (+text) question answering (TQA), table-to-text, text-to-sql (NL2SQL) and table fact verification (TFV). Recently, the emerging [Large Language Models (LLMs)](https://github.com/Hannibal046/Awesome-LLM#chatgpt-evaluation) and more powerful [Multimodal Large Language Models (MLLMs)](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) have opened up new possibilities for processing the tabular data. **In this repository, we collect papers about Tabular LLMs.** 

We divide related works into the following directions:
1. [**Prompting off-the-shelf LLMs for table tasks**](#1-prompting-off-the-shelf-llms-for-table-tasks), e.g., in-context learning, prompt engineering and tool learning.
2. [**Developing Tabular LLMs by fine-tuning or post-pretraining**](#2-developing-tabular-llms-by-fine-tuning-or-post-pretraining), e.g., training Table-GPTs or Table-Copilots for table processing.
3. [**Evaluating the table understanding ability of current LLMs**](#3-evaluating-the-table-understanding-ability-of-current-llms), e.g., exploring the influence of various table types or table representations.
4. [**Constructing Datasets for training LLMs**](#4-constructing-datasets-for-training-llms).e.g., instruction tuning data for table-related tasks.

## 1. Prompting off-the-shelf LLMs for table tasks
| Title | Conference and Time |  Task | Summary |
| --- | --- | --- | --- |
| [Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning](https://arxiv.org/abs/2209.14610) | ICLR 2023 |   Tabular Mathematical Reasoning |   |
| [Large Language Models are few(1)-shot Table Reasoners](https://arxiv.org/abs/2210.06710) | EACL 2023 Findings |  TQA, TFV |  |
| [Large Language Models are Versatile Decomposers：Decompose Evidence and Questions for Table-based Reasoning](https://arxiv.org/abs/2301.13808) | SIGIR 2023 |   TQA, TFV |   |
| [Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks](https://arxiv.org/abs/2211.12588) | 2022.11.22 on arxiv |  Tabular Mathematical Reasoning |  |
| [Chameleon：Plug-and-Play Compositional Reasoning with Large Language Models](https://arxiv.org/abs/2304.09842) | 2023.4.19 on arxiv |  Tabular Mathematical Reasoning with external API tools |  |
| [Binding Language Models in Symbolic Languages](https://arxiv.org/abs/2210.02875) | ICLR 2023 |   TQA, TFV |  |
| [Large Language Models are Effective Table-to-Text Generators, Evaluators, and Feedback Providers](https://arxiv.org/abs/2305.14987) | 2023.5.24 on arxiv | Table-to-text |  |
| [Enhancing Few-shot Text-to-SQL Capabilities of Large Language Models: A Study on Prompt Design Strategies](https://arxiv.org/abs/2305.12586) | 2023.5.21 on arxiv | Text-to-SQL | |
| [StructGPT: A General Framework for Large Language Model to Reason over Structured Data](https://arxiv.org/abs/2305.09645) | EMNLP 2023 | TQA, TFV  |

## 2. Developing Tabular LLMs by fine-tuning or post-pretraining
| Title | Conference and Time |  Task | Summary |
| --- | --- | --- | --- |
| [SheetCopilot: Bringing Software Productivity to the Next Level through Large Language Models](https://arxiv.org/abs/2305.19308) | 20230.5.30 on arxiv | Manipulating SpreedSheets with LLM | |
|[FLAME: A small language model for spreadsheet formulas](https://arxiv.org/abs/2301.13779) | 2023.1.31 on arxiv | Generating Excel Formulas | |
|[TableGPT: Towards Unifying Tables, Nature Language and Commands into One GPT](https://arxiv.org/abs/2307.08674) | 2023.7.17 on arxiv | Manipulating CSV table with LLM | |
| [Table-GPT: Table-tuned GPT for Diverse Table Tasks](https://arxiv.org/abs/2310.09263)   | All kinds of table task!!  | **A Must Read work from Microsoft**   |

### Excellent pre-trained Tabular Language Models
| Title | Conference and Time |  Task | Summary |
| --- | --- | --- | --- |
|[HYTREL: Hypergraph-enhanced Tabular Data Representation Learning](https://arxiv.org/abs/2307.08623) | NIPS 2023 | Column Type Annotation,Column Property Annotation,Table Type Detection,Table Similarity Prediction | |


## 3. Evaluating the table understanding ability of current LLMs
| Title | Conference and Time |  Task | Summary |
| --- | --- | --- | --- |
| [Evaluating and Enhancing Structural Understanding Capabilities of Large Language Models on Tables via Input Designs](https://arxiv.org/abs/2305.13062) | 2023.5.22 on arxiv | TQA, Table Fact Verfication, Table-to-text | |
|[Evaluating the Text-to-SQL Capabilities of Large Language Models](https://arxiv.org/abs/2204.00498) | 2022.3.15 on arxiv | Text-to-SQL | |
|[A comprehensive evaluation of ChatGPT's zero-shot Text-to-SQL capability](https://arxiv.org/abs/2303.13547) | 2023.3.12 on arxiv | Text-to-SQL | |
|[RobuT: A Systematic Study of Table QA Robustness Against Human-Annotated Adversarial Perturbations](https://arxiv.org/abs/2306.14321) | ACL 2023 | TQA | |

## 4. Constructing Datasets for training LLMs
| Title | Conference and Time |  Task | Summary |
| --- | --- | --- | --- |
| [InstructExcel: A Benchmark for Natural Language Instruction in Excel](https://arxiv.org/abs/2310.14495) | Findings of EMNLP 2023 | Excel operations | |


 
