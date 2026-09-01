# Wound Healing Analysis

This repository contains analysis code for studying wound healing dynamics in epithelial tissue using time-lapse imaging data. The workflows process cell tracking, wound geometry, migration, division behaviour, and tissue-level statistics to generate quantitative summaries and figures used in research analyses.

## Project overview

The code in this repository is designed to:

- analyse wound area and tissue response over time
- quantify cell migration and shape changes near wounds
- measure division orientation and timing relative to wound position
- compare control and perturbation conditions across datasets
- generate publication-ready figures and summary statistics

Key analysis scripts include:

- `healingWound.py` — wound growth and tissue response analysis
- `woundShape.py` — shape and geometry metrics around the wound edge
- `wingVelocity.py` and `woundVelocity.py` — migration velocity analyses
- `division.py` and `divisionOrientation.py` — division timing and orientation analysis
- `shapeDivision.py` and `shapeNucleus.py` — morphological and nuclear feature analyses

## Data availability

All data generated or analysed during this study has been deposited in Zenodo at this link https://zenodo.org/records/13819609.

## Repository structure

- `dat/` — raw and processed imaging datasets grouped by condition/timepoint
- `databases/` — generated database tables and summarised analysis outputs
- `results/` — figures and output plots from the analysis pipeline
- `paper_*.py` — scripts linked to manuscript-specific analyses
- `utils.py` and `utils2.py` — shared helper functions for loading data and processing files

## Papers and key scripts

This repository was used in the analyses for the following papers.

### 1. Deep learning for rapid analysis of cell divisions in vivo during epithelial morphogenesis and repair
https://journals-biologists-com.libproxy1.nus.edu.sg/dev/article/151/18/dev202943/362123

- Related script: `paper_divisionDL.py`
- Key supporting scripts: `division.py`, `divisionOrientation.py`, `divisionGM.py`, `shapeDivision.py`, `shapeNucleus.py`
- Notes: This analysis focuses on division timing, orientation, and cell morphology, including image-based and learning-related workflows.

### 2. Deep learning reveals a damage signalling hierarchy that coordinates different cell behaviours driving wound re-epithelialisation
https://elifesciences.org/articles/87949#content

- Related script: `paper_BiologyWound.py`
- Key supporting scripts: `healingWound.py`, `healingWoundGM.py`, `division.py`, `woundShape.py`, `woundVelocity.py`, `divisionGM.py`, `woundShapeGM.py`, `woundVelocityGM.py`
- Notes: This analysis focuses on wound closure dynamics, wound geometry, and tissue-level responses over time. Also analysis of genetic modefied tissues and altered would closing behaviours

### 3. Quantifying cell shape and density fluctuations in epithelial tissue in vivo

- Related script: `paper_MathsTheory.py`, `paper_applyTheory.py`
- Key supporting scripts: `correlationDatabases.py`, `correlationDatabasesWound.py`
- Notes: This work models the underlying tissue dynamics and compares experimental measurements with theoretical predictions.

## Notes

This project is designed for reproducible analysis of wound-healing image data and is structured around manuscript-specific scripts and shared computational utilities. The generated outputs are stored under the `results/` and `databases/` folders.
