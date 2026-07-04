# Quantitative Pathway-Resolved Kinetics from Neural Network-Guided Weighted Ensemble Simulations

Repository for "Quantitative Pathway-Resolved Kinetics from Neural Network-Guided Weighted Ensemble Simulations".

## Overview

This repository contains notebooks, scripts, and supporting simulation assets used to analyze pathway-resolved transition kinetics from weighted ensemble simulations guided by neural-network-refined reaction pathways. The workflow combines path-generation, path collective variables, trajectory clustering, flux analysis, and rate estimation across toy model systems and molecular systems.

The included examples cover:

- Toy potential models, including Muller-Brown and three-hole potentials.
- OAMe-G2 host-guest unbinding path generation and metadynamics setup files.
- 3PTB ligand-binding pathway and flux analysis.
- 1OPJ wild-type and N368S pathway analyses, including AC and P-loop pathways.

## NoteBooks

This directory contains the notebooks used to reproduce the analysis and figures:

- `MB_Figure.ipynb`: Muller-Brown path-generation and figure analysis.
- `Three_Hole_Potential_Figures.ipynb`: Three-hole potential rate and dimensionality analyses.
- `3PTB_Analysis.ipynb`: Pathway-specific flux and rate analysis for 3PTB.
- `1OPJ_Rates.ipynb`: Weighted ensemble rate analysis for 1OPJ WT and N368S systems.
- `1OPJ_Path_CLustering.ipynb`: Path clustering and PCA-space analysis for the 1OPJ N368S system.

## Scripts

`NoteBooks/Scripts` contains reusable Python utilities for:

- Path collective variable construction.
- PathGennie-style molecular dynamics path generation.
- Neural-network-based ensemble path refinement.
- Toy-potentials.
- Simulation object utilities used by the notebooks.

## Assets

The `assets` directory contains the supporting data required by the notebooks:

- `Toy/`: Flux files for toy-potential weighted ensemble analyses.
- `OAMe-G2/`: Host-guest topology, coordinates, path-generation inputs, metadynamics files, and refined path representatives.
- `3PTB/`: Protein-ligand structures, trajectories, PCA pipeline, and pathway-specific flux files.
- `1OPJ/`: WT and N368S structures, PCA pipelines, path files and weighted ensemble flux files.

## Requirements

Python 3.8+ with common scientific Python packages such as `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `torch`, `MDAnalysis`, `openmm`, and `tqdm`. Some molecular dynamics workflows also require system-specific OpenMM, PLUMED, and trajectory-analysis dependencies.

## Notes

Please use the Software PathGennie to perform the run/analysis using the parameters. This repository is only for the reproducibility purpose of the paper.
