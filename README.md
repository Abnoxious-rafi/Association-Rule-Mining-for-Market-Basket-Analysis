# Association-Rule-Mining-for-Market-Basket-Analysis
A reproducible, controlled comparative study of Apriori and FP-Growth algorithms for Market Basket Analysis on the UCI Online Retail dataset (541k+ transactions) in Python, evaluating runtime, peak memory, and association rule quality (Support, Confidence, Lift).

# Association Rule Mining for Market Basket Analysis: A Comparative Study of Apriori and FP-Growth

[![Python](https://img.shields.io/badge/Python-3.10.0-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.3.3-150458.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.7-orange.svg)](https://matplotlib.org/)
[![MLxtend](https://img.shields.io/badge/MLxtend-0.23.4-green.svg)](https://rasbt.github.io/mlxtend/)

A controlled, reproducible empirical comparison of **Apriori** and **FP-Growth** algorithms implemented from scratch in Python for Market Basket Analysis using the **UCI Online Retail** dataset.

> **Course:** ICT-1004 Data Mining Sessional  
> **Institution:** Department of Information and Communication Technology, Comilla University  
> **Supervised By:** Dr. Md. Saifur Rahman, Professor, Dept. of ICT, Comilla University

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset & Preprocessing](#-dataset--preprocessing)
- [Experimental Setup](#-experimental-setup)
- [Key Findings & Benchmark Results](#-key-findings--benchmark-results)
- [Discovered Product Associations](#-discovered-product-associations)
- [Repository Structure](#-repository-structure)
- [Installation & Usage](#-installation--usage)
- [Team Members](#-team-members)
- [References](#-references)

---

## 🔍 Overview

Retail businesses generate massive transactional logs daily, making manual discovery of purchasing patterns infeasible. While **Apriori** relies on level-wise candidate generation and repeated database scans, **FP-Growth** compresses transactions into a prefix tree (FP-Tree) to mine frequent patterns recursively without candidate generation.

This project runs both custom-implemented algorithms on identical preprocessed baskets under uniform thresholds to evaluate:
1. **Computational Efficiency:** Miner-only execution time and Python peak memory allocation (`tracemalloc`).
2. **Scalability:** Performance behavior across varying support thresholds and dataset sizes.
3. **Rule Quality:** Actionable retail associations ranked by **Support**, **Confidence**, and **Lift**, cross-checked against `MLxtend` for exact itemset parity.

---

## 📊 Dataset & Preprocessing

The study uses the [UCI Machine Learning Repository Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) containing UK-based non-store online retail transactions (Dec 2010 – Dec 2011).

| Preprocessing Step | Record Count | Notes |
| :--- | :---: | :--- |
| **Raw Dataset Rows** | 541,909 | 8 attributes (`InvoiceNo`, `StockCode`, `Description`, etc.) |
| Missing/Blank Required Fields Removed | -1,454 | Missing `InvoiceNo` or `Description` |
| Cancelled Invoices Removed | -9,288 | Invoices starting with `'C'` |
| Invalid Quantity / UnitPrice Removed | -1,063 | Non-positive values (`<= 0`) |
| Duplicate Line Items Removed | -5,226 | Exact duplicates dropped |
| **Final Cleaned Product Lines** | **524,878** | Preserved 135,080 rows with missing `CustomerID` |
| **Total Baskets (Invoices)** | **19,960** | Grouped by `InvoiceNo` (presence/absence encoding) |
| **Distinct Products** | **3,922** | Unique `StockCode` items |

---

## 🧪 Experimental Setup

Both algorithms were benchmarked in **Python 3.10.0** (`random_seed = 42`, `max_itemset_length = 3`) and verified against `MLxtend` (confirming an exact match of **142 frequent itemsets** at `min_support = 0.03`):

- **Experiment A — Support Sensitivity:** Full cleaned dataset (`100%` / 19,960 baskets) tested across `min_support` $\in \{0.05, 0.03, 0.02\}$.
- **Experiment B — Dataset Scalability:** Fixed `min_support = 0.03` tested across random basket subsets of $\{25\%, 50\%, 75\%, 100\%\}$.
- **Rule Filtering:** Minimum Confidence $\ge 0.60$ and $\text{Lift} > 1.0$.

---

## 📈 Key Findings & Benchmark Results

### 1. Support Sensitivity (`100%` Dataset = 19,960 Baskets)
| Min Support | Frequent Itemsets | Rules Found | Apriori Time (s) | FP-Growth Time (s) | Apriori Peak Mem (MB) | FP-Growth Peak Mem (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.050** | 36 | 0 | 0.476 | 1.013 | 0.31 | 10.34 |
| **0.030** | 142 | 8 | 1.600 | 2.608 | 1.96 | 27.94 |
| **0.020** | 385 | 31 | **21.002** | **5.448** | 8.23 | 49.41 |

### 2. Dataset Scalability (`min_support = 0.030`)
| Dataset Size | Itemsets | Rules | Apriori Time (s) | FP-Growth Time (s) | Apriori Peak Mem (MB) | FP-Growth Peak Mem (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **25%** | 133 | 12 | 1.033 | 0.547 | 1.81 | 7.02 |
| **50%** | 142 | 9 | 2.352 | 1.267 | 1.91 | 14.32 |
| **75%** | 142 | 8 | 1.274 | 1.878 | 1.95 | 21.14 |
| **100%** | 142 | 8 | 1.600 | 2.608 | 1.96 | 27.94 |

> **Takeaway:** At high support thresholds (`0.03–0.05`), Apriori has lower overhead and uses significantly less memory (`~1.96 MB` vs `27.94 MB`). However, as minimum support drops to `0.02` and candidate itemsets explode, **FP-Growth outperforms Apriori by ~3.85× in speed** (`5.45s` vs `21.00s`), trading higher RAM usage for faster tree-based mining.

---

## 🛒 Discovered Product Associations

At `min_support = 0.03` and `min_confidence = 0.60`, **8 strong positive association rules** were discovered (maximum Lift = **16.28**):

| Antecedent (If Bought) | Consequent (Also Bought) | Support | Confidence | Lift |
| :--- | :--- | :---: | :---: | :---: |
| `22698` \| PINK REGENCY TEACUP AND SAUCER | `22697` \| GREEN REGENCY TEACUP AND SAUCER | 0.032 | **82.6%** | **16.28** |
| `22697` \| GREEN REGENCY TEACUP AND SAUCER | `22698` \| PINK REGENCY TEACUP AND SAUCER | 0.032 | 62.4% | **16.28** |
| `22697` \| GREEN REGENCY TEACUP AND SAUCER | `22699` \| ROSES REGENCY TEACUP AND SAUCER | 0.038 | 75.7% | 14.19 |
| `22699` \| ROSES REGENCY TEACUP AND SAUCER | `22697` \| GREEN REGENCY TEACUP AND SAUCER | 0.038 | 72.0% | 14.19 |
| `22726` \| ALARM CLOCK BAKELIKE GREEN | `22727` \| ALARM CLOCK BAKELIKE RED | 0.032 | 65.3% | 12.40 |
| `22727` \| ALARM CLOCK BAKELIKE RED | `22726` \| ALARM CLOCK BAKELIKE GREEN | 0.032 | 60.9% | 12.40 |
| `22386` \| JUMBO BAG PINK POLKADOT | `85099B` \| JUMBO BAG RED RETROSPOT | 0.041 | 67.7% | 6.47 |
| `21931` \| JUMBO STORAGE BAG SUKI | `85099B` \| JUMBO BAG RED RETROSPOT | 0.036 | 61.1% | 5.84 |

---

## 📁 Repository Structure

```text
├── data/
│   └── Online Retail.xlsx              # Raw UCI Online Retail dataset
├── notebooks/ / src/
│   └── market_basket_analysis.ipynb    # Preprocessing, Apriori & FP-Growth implementations, benchmarking
├── results/
│   ├── performance_comparison.csv      # Runtime & tracemalloc memory metrics
│   ├── frequent_itemsets.csv           # Extracted frequent itemsets
│   ├── association_rules.csv           # Filtered rules with Support, Confidence, Lift
│   ├── top_products.csv                # Most frequent individual items
│   ├── pipeline_summary.json           # Reproducibility metadata & environment versions
│   ├── useful_knowledge.txt            # Summary report of findings
│   ├── support_sensitivity.png         # Runtime & memory vs. min_support chart
│   ├── dataset_scalability.png         # Runtime & memory vs. dataset size chart
│   ├── top_products.png                # Top 10 most frequent products bar chart
│   └── rule_quality.png                # Support vs. Confidence (colored by Lift) scatter plot
├── Data_Mining_Project_Proposal.pdf    # Formal academic project proposal
└── README.md
