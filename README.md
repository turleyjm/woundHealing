# Wound healing analysis

This repo contains the code used to analysis wound healing in Drosophila pupal wings for the paper "AI reveals a damage signalling hierarchy that coordinates different cell behaviours driving wound re-epithelialisation" on [biorxiv]

In particular code and analysis for the paper is in "paper_BiologyWound.py" and "paper_BiologyWound_rebuttal"

## Data and processing

The data can be found at [Zenodo]. This data has been processed using a workflow in [git_processData] from the origan raw 5D stacks from the confocal microscope. The raw 5D stack data is large can be made available upon request. This workflow process the data quantifying the cell behaviours such as cell migration, shape changes and divisions.

The deep learning algorithms used in this workflow are discussed in a previous [paper] and the [cell-division-dl-plugin] repo has the code and data.

"databases.py" collects the data about cell behaviours from individual videos and pools them with others in the same condition.

[biorxiv]: https://www.biorxiv.org/content/10.1101/2024.04.10.588842v2.abstract
[Zenodo]: https://zenodo.org/records/10846684
[git_processData]: https://github.com/turleyjm/processData
[cell-division-dl-plugin]: https://github.com/turleyjm/processData
[paper]: https://www.biorxiv.org/content/10.1101/2023.03.20.533343v3.abstract
