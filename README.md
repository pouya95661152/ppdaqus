Overview
This repository contains a Python implementation of a novel Data Envelopment Analysis (DEA) model based on an epsilon constraint for classifying flexible measures and identifying the most efficient Decision Making Units (DMUs) in the Istanbul Stock Exchange (BIST). The model addresses limitations of traditional DEA approaches by incorporating flexible measures and ensuring all inputs, outputs, and flexible measures contribute to the efficiency evaluation process.

Research Context
The research focuses on evaluating the efficiency of 19 companies listed on the BIST using financial data from 2016 to 2021. The model classifies flexible measures (e.g., inventory turnover and asset turnover) as either inputs or outputs and ranks DMUs based on their efficiency scores. The proposed approach introduces an epsilon constraint to prevent zero weights, ensuring all measures are considered in the evaluation, unlike traditional models where some measures might be ignored.

Model Description
The model is a mixed-integer linear programming formulation that:
Uses binary variables to classify flexible measures as inputs or outputs.
Applies an epsilon constraint (ε=0.002) to ensure non-zero weights for all inputs, outputs, and flexible measures.
Maximizes the efficiency gap (t) to identify the most efficient DMU while ensuring other DMUs have efficiency scores less than one.
Iteratively ranks efficient DMUs by excluding previously identified efficient units.

