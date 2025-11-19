# WindFarm
This repository contains a Python-based tool for simulating wind farm energy production and generating detailed LaTeX reports. The tool analyzes wind data, turbine performance, wake effects, and other factors to estimate yearly, quarterly, and monthly energy production (P50, P75, P90).

## Repository Structure
📦 WindFarm  
├── 📂 data/                  # Input weather data and turbine models  
│   ├── WeatherData_*.csv     # Historical wind data 
│   └── wind_turbines_models.json  # Wind turbine specifications  
├── 📂 outputs/  
│   ├── 📂 figures/           # Generated plots (wind rose, AEP, power curves)  
│   ├── 📂 tables/            # CSV tables (monthly/quarterly production, percentiles)  
│   └── 📂 reports/           # Final LaTeX report (output.pdf)  
├── 📂 src/  
│   └── run_windfarmsimulation_latex.py  # Main simulation script  
├── 📂 templates/  
│   └── main.tex              # LaTeX template for report generation  
├── 📜 build.yaml             # GitHub Actions workflow for automation  
└── 📜 README.md              # This file  

## Features

✅ Wind Data Analysis

  Interpolates wind speeds at hub heights.
  
  Generates wind roses, Weibull distributions, and correlation heatmaps.

✅ Energy Production Simulation

  Computes Annual Energy Production (AEP) with/without wake effects.
  
  Estimates P50/P75/P90 percentiles for risk assessment.

✅ Automated Reporting

  Compiles LaTeX reports with dynamic placeholders.
  
  Includes tables, plots, and turbine specifications.

✅ CI/CD Integration

  GitHub Actions (build.yaml) automates:
  
  Python dependency installation.
  
  LaTeX compilation (via xelatex).
  
  Artifact upload (PDF, figures, tables).

## Setup & Usage

1️⃣ Prerequisites
Python 3.9+

LaTeX distribution (e.g., texlive-xetex, texlive-latex-extra)

Required Python packages:
bash
pip install pandas numpy matplotlib py_wake scipy seaborn windrose

2️⃣ Running the Simulation
Configure Inputs in run_windfarmsimulation_latex.py:

  python
  plantname = "Your-Wind-Farm"  
  modelname = "Turbine-Model-Name"  
  hub_height = 150  # meters  
  json_file = "../data/wind_turbines_models.json"  

Execute the Script:

  bash
  cd src  
  python run_windfarmsimulation_latex.py  
  Generated Outputs:
  
  Plots (e.g., AEP_*.png, wind_rose_*.png) → outputs/figures/
  
  Tables (e.g., *_summary.csv) → outputs/tables/
  
  LaTeX report → outputs/reports/output.pdf

3️⃣ GitHub Actions (Optional)
The workflow in build.yaml automates:

Simulation + LaTeX report generation on git push.

Uploads artifacts (PDF, figures, tables).


