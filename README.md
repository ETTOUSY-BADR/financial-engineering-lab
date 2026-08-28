#  Financial Engineering Lab

Welcome to my **Financial Engineering Lab** — a personal space where I connect my deep passion for **applied mathematics** with the dynamic world of **finance**.  

This repository documents my journey of mastering the mathematical foundations of finance and applying them through real data, coding projects, and research-inspired experiments.  

---

##  My Motivation

Mathematics has always been my strongest asset.  
I believe that **mathematical thinking is the most powerful tool for solving financial problems** — from pricing complex derivatives to managing portfolio risk and designing trading strategies.  

To build this bridge between math and finance, I am following two complementary learning paths:
-  **MIT OpenCourseWare**: *From Mathematics to Finance* — grounding myself in the rigorous math behind financial models.  
-  **Columbia University (Coursera)**: *Financial Engineering and Risk Management Specialization* — applying advanced concepts to real-world markets.  

This lab is where **theory meets practice**.  

---

##  Objectives

- Implement **core models** of financial engineering:  
  - Derivatives pricing (Black–Scholes, binomial trees, Monte Carlo).  
  - Risk management (VaR, CVaR, stress testing).  
  - Portfolio optimization and factor models.  

- Build **tools & utilities** to make financial experiments reproducible.  

- Apply models to **real market data** using Python.  

- Document progress in clear, organized **notebooks and projects**.  

- Lay the foundation for future **quantitative trading research** — with the ultimate long-term vision of building a **hedge fund powered by mathematics, data, and AI**.  

---

## 📂 Repo Structure

- `notebooks/` → Lecture-aligned notebooks (MIT + Columbia), with explanations and code.  
- `data/` → Example datasets (market prices, options, bonds, indices).  
- `projects/` → Mini-projects that combine theory + data for real insights.  
- `utils/` → Python modules for pricing, risk metrics, and backtesting.  

---

# Financial Engineering Lab

A research-oriented workspace for studying quantitative finance from first principles: probability, stochastic calculus, asset pricing, econometrics, macro-finance, derivatives, portfolio construction, and risk.

## What is here

- `course/quant_finance_book.tex` is the master source for the compiled course book, with separate chapters for `01_mathematics`, `02_macroeconomics`, `03_microeconometrics`, and `04_financial_markets`.
- `course/quant_finance_book.pdf` is the current compiled edition.
- `utils/quant_models.py` contains small, auditable primitives: log returns, annualized and rolling volatility, drawdown, historical VaR/CVaR, Black-Scholes, and Monte Carlo pricing.
- `projects/regime_risk/` is the first empirical project. It downloads adjusted prices for `SPY`, `TLT`, `GLD`, and `^VIX`, computes tail-risk diagnostics, and records provenance.
- `data/` is reserved for explicitly sourced datasets; generated project outputs live beside their project.
- `notebooks/` is reserved for lecture-aligned experiments that grow out of the book and projects.

## Reproduce the research

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python projects/regime_risk/run_research.py
```

The default data source is Yahoo Finance via `yfinance`. Without network access, the pipeline uses a deterministic geometric-Brownian-motion fallback and labels it in `projects/regime_risk/output/provenance.txt`; synthetic results are for testing the workflow, not market conclusions.

## Build the book

From the repository root, run twice so the table of contents and cross-references settle:

```powershell
Push-Location course
pdflatex -interaction=nonstopmode -halt-on-error quant_finance_book.tex
pdflatex -interaction=nonstopmode -halt-on-error quant_finance_book.tex
Pop-Location
```

## Research standards

Every model should state its probability space, information set, assumptions, estimand, calibration sample, validation design, uncertainty, transaction costs, and failure modes. Results are educational research and are not investment advice.
