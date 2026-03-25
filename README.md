# School Abandonment Analysis

## Description
This project aims to analyze school abandonment (dropout) rates and identify factors that influence students leaving school. Using historical school and student data, we explore patterns, correlations, and predictive modeling to understand which factors most affect dropout rates.

## Project Structure (not complete yet)
The repository is organized as follows:

- **data/** → Contains datasets used for analysis.
- **scripts/** → Python or R scripts for data preprocessing, analysis, and modeling.
- **README.md** → Project documentation.

## Key Variables
- **localizacao** → School location, to capture regional differences in dropout rates.
- **rede** → School network (public, private, or other), impacting resources and dropout.
- **atu_em** → Average number of students per class; large classes can increase dropout.
- **had_em** → Daily class hours; extreme workloads can affect dropout risk.
- **tdi_em** → Age-grade distortion; more students out of the correct grade increases risk.
- **taxa_aprovacao_em / taxa_reprovacao_em** → Historical approval and failure rates; included carefully to avoid target leakage.
- **dsu_em** → Percentage of teachers with higher education; more qualified teachers can reduce dropout.
- **afd_em_grupo_1** → Teacher adequacy (best-trained group); higher quality teaching reduces abandonment risk.

## Workflow
1. Work in the `dev` branch.
2. Add, commit, and push your changes to `dev`.
3. Merge tested and stable changes from `dev` into `main`.

## Objective
The ultimate goal is to build a predictive model that estimates the probability of students dropping out based on school and student characteristics, enabling better intervention strategies.