# Neutron star mode surrogate

Code for the MSc dissertation *Decreasing the Computation Time of the Neutron Star Mode Spectrum via Neural Surrogates* (Isaac Dodds, University of Bath, 2026).

The forward model turns ten nuclear parameters into a labelled quadrupolar mode spectrum of a neutron star in four stages: equation of state, stellar structure, mode scan, labelling. It is a reimplementation in JAX of the published pipeline of Neill, Newton and Tsang, written from the physics with no code taken from the original release. The surrogates are neural networks placed at each stage of that chain and scored on one accuracy against time axis.

## Layout

| path | what it is |
|---|---|
| `forward_model/forward_model.py` | the pipeline: metamodel and Fermi gas foundations, compressible liquid drop crust, beta equilibrated outer core, five inner core families, TOV structure with mass inversion, oscillation scan and refinement, eigenfunction classifier, batch driver |
| `forward_model/relabel_spectrum.py` | recounts nodes against a local envelope and relabels a stored star from its eigenfunctions |
| `surrogates/variants.py` | data loader, network, loss and trainer shared by every seat |
| `surrogates/architecture_scan.py` | the scan protocol: candidates on validation, top three at three seeds, one test |
| `surrogates/dense_search.py` | the deep emulator network search of Kasim et al. for related inputs |
| `surrogates/front_seats.py` | the equation of state and structure seats |
| `surrogates/warm_start.py` | window and polish scoring for the solver warm start |
| `surrogates/curves_and_importance.py` | training curves, permutation importance, learning curves |
| `surrogates/baselines.py` | slot mean and kernel ridge baselines |
| `surrogates/count_analysis.py` | mode count and mask diagnostics |
| `surrogates/train_starter.py` | minimal self contained trainer |
| `figures/dataset_figures.py` | dataset figures |
| `results/` | the numbers behind the dissertation's figures and tables |

## Install

Python 3.11 or later.

```
pip install -r requirements.txt
```

## Reproduce one star

```
python forward_model/forward_model.py --batch 1 --start 0 --seed 1 --tier training --out dataset
```

Each accepted star is written as one archive holding every stage; every rejected draw is named in a manifest, and rejected draws that reach a stellar sequence keep their tables.

## Train a surrogate

```
python surrogates/architecture_scan.py results.csv dataset --seat c2
python surrogates/train_starter.py dataset
```

Set `FM3_VARIANTS` to point at `surrogates/variants.py` if the scripts are run from another directory.

## Data

The training set (1,584 unique stars, all four stages, 821 MB) exceeds what this repository holds. It is retained on university storage and available on request; a public archive with a DOI will be linked here.

## Licence

MIT. See `LICENSE`.
