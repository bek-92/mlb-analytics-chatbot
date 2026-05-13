# MLB Analytics Chatbot

An interactive Streamlit chatbot for MLB performance, payroll, and player-value analysis.

## Overview

This project was developed as part of a graduate business analytics capstone at George Washington University. The chatbot allows users to ask natural-language questions about MLB player performance, payroll constraints, free-agent value, pitching, batting, fielding, and roster decision scenarios.

## Features

- Natural-language MLB analytics chatbot
- Player performance, payroll, and value analysis
- Pitching, batting, fielding, and salary-aware recommendations
- Budget-constrained roster decision support
- Streamlit interface for interactive exploration
- LLM-powered response generation
- Table, prose, and chart-style outputs

## Tech Stack

- Python
- Streamlit
- pandas
- OpenAI / Azure OpenAI-compatible API
- LangGraph
- Altair
- Matplotlib

## Example Questions

- Who are the best 2027 free-agent starting pitchers with ERA+ above 115 and salary under $18M?
- Which two relief pitchers provide the best value under a $20M budget?
- Which catchers had the best defensive value?
- Which players are underpaid based on WAR and salary?
- Explain in prose why two recommended pitchers are a strong package.

## Data Note

The full dataset is not included in this repository. The app expects MLB performance and payroll files inside a local `Data/` folder.

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py

Author

Bek Rustamov
Global MBA & M.S. in Business Analytics
George Washington University
