# A-Paper-List-of-Awesome-Tabular-LLMs
Different types of tables are widely used to store and present information. To automatically process numerous tables and gain valuable insights, researchers have proposed a series of deep-learning models for various table-based tasks, e.g., table question answering (TQA), table-to-text (T2T), text-to-sql (NL2SQL) and table fact verification (TFV). Recently, the emerging [Large Language Models (LLMs)](https://github.com/Hannibal046/Awesome-LLM#chatgpt-evaluation) and more powerful [Multimodal Large Language Models (MLLMs)](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) have opened up new possibilities for processing the tabular data, i.e., we can use one general model to process diverse tables and fulfill different tabular tasks based on the user natural language instructions. We refer to these LLMs speciallized for tabular tasks as `Tabular LLMs`. **In this repository, we collect a paper list about recent Tabular (M)LLMs and divide them into the following categories based on their key idea.**

---

<font size=8><center><b> Table of Contents: </b> </center></font>
1. [**Survey of Tabular LLMs and table understanding**](#1-survey-of-tabular-llms-and-table-understanding)
2. [**Prompting LLMs for different tabular tasks**](#2-prompting-llms-for-different-tabular-tasks), e.g., in-context learning, prompt engineering and integrating external tools.
3. [**Training LLMs for better table understanding ability**](#3-training-llms-for-better-table-understanding-ability), e.g., training existing LLMs by instruction fine-tuning or post-pretraining.
4. [**Developing agents for processing tabular data**](#4-developing-agents-for-processing-tabular-data), e.g., devolping copilot for processing excel tables.
5. [**Empirical study or benchmarks for evaluating LLMs' table understanding ability**](#5-empirical-study-or-benchmarks-for-evaluating-llms-table-understanding-ability), e.g., exploring the influence of various table types or table formats.
6. [**Multimodal table understanding**](#6-multimodal-table-understanding), e.g., training MLLMs to understand diverse table images and textual user requests.
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
| Title | Conference | Date  | Pages |
| ------ | :---: | :---: | :---: |
| [Large Language Model for Table Processing: A Survey](https://arxiv.org/abs/2402.05121) | arxiv | 2024-02-04  | 9 | 
| [A Survey of Table Reasoning with Large Language Models](https://arxiv.org/abs/2402.08259) | arxiv | 2024-02-13 | 9 |
| [Large Language Models(LLMs) on Tabular Data: Prediction, Generation, and Understanding -- A Survey](https://arxiv.org/abs/2402.17944) | arxiv | 2024-03-01 | 41 | 
| [Transformers for Tabular Data Representation: A Survey of Models and Applications](https://aclanthology.org/2023.tacl-1.14/) | TACL 2023 | | 23 |
| [Table Pre-training: A Survey on Model Architectures, Pre-training Objectives, and Downstream Tasks](https://arxiv.org/abs/2201.09745) | IJCAI 2022 | 2022-01-24 | 15 |

## 2. Prompting LLMs for Different Tabular Tasks
| Title | Conference | Date |  Task | Code |
| --- | :---: | :---: | :---: | --- |
| ![Star](https://img.shields.io/github/stars/rllm-team/rllm.svg?style=social&label=Star) <br/>[rLLM: Relational Table Learning with LLMs](https://arxiv.org/abs/2407.20157) | arxiv | 2024-07-29 |  | [Github](https://github.com/rllm-team/rllm) |
| [Learning Relational Decomposition of Queries for Question Answering from Tables](https://aclanthology.org/2024.acl-long.564/) | ACL 2024 |  | TQA | |
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
| ![Star](https://img.shields.io/github/stars/UCSB-NLP-Chang/Augment_tableQA.svg?style=social&label=Star) <br> [Augment before You Try: Knowledge-Enhanced Table Question Answering via Table Expansion](https://arxiv.org/abs/2401.15555) | arxiv | 2024-01-24 | TQA | [Github](https://github.com/UCSB-NLP-Chang/Augment_tableQA) |
| [Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding](https://arxiv.org/abs/2401.04398) | ICLR 2024 | 2024-01-09 | TQA,TFV | |
| [TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning](https://arxiv.org/abs/2312.09039) | arxiv | 2023-12-14 | TQA,TAT-QA,TFV,T2T | [Github](https://anonymous.4open.science/r/TableProvider-4CC3/README.md) |
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


| Title | Conference | Date |  Task | LLM Backbone  | Code |
| --- | :---: | :---: | :---: | :---: | :---: |
| ![Star](https://img.shields.io/github/stars/basf/mamba-tabular.svg?style=social&label=Star) <br> [Mambular: A Sequential Model for Tabular Deep Learning](https://arxiv.org/abs/2408.06291) | arxiv | 2024-08-12 | ML Classification and Regression tasks like California Housing | Mamba | [Github](https://github.com/basf/mamba-tabular)
| [MambaTab: A Plug-and-Play Model for Learning Tabular Data](https://arxiv.org/abs/2401.08867) | MIPR 2024 | 2024-01-16 | ML Classification tasks | Mamba |  | 
| [SpreadsheetLLM: Encoding Spreadsheets for Large Language Models](https://arxiv.org/abs/2407.09025) | arxiv | 2024-07-12 | Excel Manipulation |   | |
| [Unleashing the Potential of Large Language Models for Predictive Tabular Tasks in Data Science](https://arxiv.org/abs/2403.20208) | arxiv  | 2024-03-29 | Predictive Tabular Tasks | Llama2 7B | [HuggingFace](https://huggingface.co/OldBirdAZ/itab-llm) |
| [HGT: Leveraging Heterogeneous Graph-enhanced Large Language Models for Few-shot Complex Table Understanding](https://arxiv.org/abs/2403.19723) | arxiv  | 2024-03-28 | TI,TQA | Vicuna-1.5 7B |  |
| ![Star](https://img.shields.io/github/stars/RUCKBReasoning/TableLLM.svg?style=social&label=Star) <br> [TableLLM: Enabling Tabular Data Manipulation by LLMs in Real Office Usage Scenarios](https://arxiv.org/abs/2403.19318) | arxiv  | 2024-03-28 | Table Manipulation | CodeLlama 7B, 13B | [Github](https://github.com/RUCKBReasoning/TableLLM) |
| ![Star](https://img.shields.io/github/stars/TIGER-AI-Lab/StructLM.svg?style=social&label=Star) <br> [StructLM: Towards Building Generalist Models for Structured Knowledge Grounding](https://arxiv.org/abs/2402.16671) | CoLM 2024 | 2024-02-26 | TQA,TFV,T2T,NL2SQL  | CodeLlama 7B-34B   | [Github](https://github.com/TIGER-AI-Lab/StructLM)  |
| ![Star](https://img.shields.io/github/stars/fengbinzhu/TAT-LLM.svg?style=social&label=Star) <br> [TAT-LLM: A Specialized Language Model for Discrete Reasoning over Tabular and Textual Data](https://arxiv.org/abs/2401.13223) | arxiv | 2024-01-24 | TQA | Llama2 7B, 13B, 70B | [Github](https://github.com/fengbinzhu/TAT-LLM) |
| ![Star](https://img.shields.io/github/stars/OSU-NLP-Group/TableLlama.svg?style=social&label=Star) <br> [TableLlama: Towards Open Large Generalist Models for Tables](https://arxiv.org/abs/2311.09206) | NAACL 2024 | 2023-11-15 | TQA,TFV,T2T,TA,TI  | Llama2 7B | [Github](https://github.com/OSU-NLP-Group/TableLlama)  |
| [HELLaMA: LLaMA-based Table to Text Generation by Highlighting the Important Evidence](https://arxiv.org/abs/2311.08896)  | arxiv | 2023-11-15 | T2T | Llama2 7B-13B |  |
| [Table-GPT: Table-tuned GPT for Diverse Table Tasks](https://arxiv.org/abs/2310.09263)   | arxiv  | 2023-10-13  | TQA | GPT-3.5, ChatGPT |   |

### Pre-trained Tabular Language Models (non-LLM)
| Title | Conference | Date |  Task | Code |
| --- | :---: | :---: | :---: | :---: |
| ![Star](https://img.shields.io/github/stars/awslabs/hypergraph-tabular-lm.svg?style=social&label=Star) <br> [HYTREL: Hypergraph-enhanced Tabular Data Representation Learning](https://arxiv.org/abs/2307.08623) | NIPS 2023 | 2023-07-14 |  TA, TI | [Github](https://github.com/awslabs/hypergraph-tabular-lm) |
| [FLAME: A small language model for spreadsheet formulas](https://arxiv.org/abs/2301.13779)  | AAAI 2024 | 2023-01-31 | Generating Excel Formulas | [Github](https://github.com/microsoft/prose-benchmarks/tree/main/FLAME)  |

## 4. Developing Agents for Processing Tabular Data
| Title | Conference | Date |  Task | Code |
| --- | :---: | :---: | :---: | :---: |
| [SheetAgent: A Generalist Agent for Spreadsheet Reasoning and Manipulation via Large Language Models](https://arxiv.org/abs/2403.03636) | arxiv | 2024-03-06 | Manipulating Excels with LLM | [Github](https://github.com/sheetagent/sheetagent.github.io) |
| ![Star](https://img.shields.io/github/stars/wshi83/EhrAgent.svg?style=social&label=Star) <br> [EHRAgent: Code Empowers Large Language Models for Few-shot Complex Tabular Reasoning on Electronic Health Records](https://arxiv.org/pdf/2401.07128.pdf) | arxiv | 2024-01-13 | TQA | [Github](https://github.com/wshi83/EhrAgent) |
| ![Star](https://img.shields.io/github/stars/InfiAgent/InfiAgent.svg?style=social&label=Star) <br> [InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks](https://arxiv.org/abs/2401.05507) | arxiv | 2024-01-10 | Data Analysis | [Github](https://github.com/InfiAgent/InfiAgent) |
| ![Star](https://img.shields.io/github/stars/eosphoros-ai/DB-GPT.svg?style=social&label=Star) <br> [DB-GPT: Empowering Database Interactions with Private Large Language Models](https://arxiv.org/abs/2312.17449) | arxiv | 2023-12-29 | Data Analysis | [Github](https://github.com/eosphoros-ai/DB-GPT) |
| [ReAcTable: Enhancing ReAct for Table Question Answering](https://arxiv.org/abs/2310.00815) | arxiv | 2023-10-01 | TQA | |
| ![Star](https://img.shields.io/github/stars/BraveGroup/SheetCopilot.svg?style=social&label=Star) <br>[SheetCopilot: Bringing Software Productivity to the Next Level through Large Language Models](https://arxiv.org/abs/2305.19308) |NIPS 2023  |2023-05-30 | Manipulating Excels with LLM | [Github](https://github.com/BraveGroup/SheetCopilot)  |
| [TableGPT: Towards Unifying Tables, Nature Language and Commands into One GPT](https://arxiv.org/abs/2307.08674) | arxiv | 2023-07-17 | Manipulating CSV table with LLM | |


## 5. Empirical Study or Benchmarks for Evaluating LLMs' Table Understanding Ability
| Title | Conference | Date |  Task | Code |
| --- | :---: | :---: | :---: | :---: |
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


## 6. Multimodal Table Understanding
| Title | Conference | Date |  Task | Code |
| --- | :---: | :---: | :---: | :---: |
| ![Star](https://img.shields.io/github/stars/alonsoapp/PixT3.svg?style=social&label=Star) <br> [PixT3: Pixel-based Table-To-Text Generation](https://arxiv.org/abs/2311.09808) | ACL 2024 | 2023-11-16 | T2T | [Github](https://github.com/alonsoapp/PixT3) |
| [TabPedia: Towards Comprehensive Visual Table Understanding with Concept Synergy](https://arxiv.org/abs/2406.01326) | arxiv | 2024-06-03 | TQA,TI |
| ![Star](https://img.shields.io/github/stars/naver-ai/tablevqabench.svg?style=social&label=Star) <br> [TableVQA-Bench: A Visual Question Answering Benchmark on Multiple Table Domains](https://arxiv.org/abs/2404.19205) | arxiv  | 2024-04-30 | TQA, TFV | [Github](https://github.com/naver-ai/tablevqabench) |
| [Tables as Texts or Images: Evaluating the Table Reasoning Ability of LLMs and MLLMs](https://arxiv.org/abs/2402.12424) | ACL 2024 | 2024-02-19 | TQA,TFV,T2T | |
| ![Star](https://img.shields.io/github/stars/SpursGoZmy/Table-LLaVA.svg?style=social&label=Star) <br> [Multimodal Table Understanding](https://arxiv.org/abs/2406.08100) | ACL 2024  | 2024-02-15 | TQA, TFV, T2T, TI, TAT-QA, TMR | [Github](https://github.com/SpursGoZmy/Table-LLaVA)   |



