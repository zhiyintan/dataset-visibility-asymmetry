# Language Dataset Visibility and Low-Resource NLP

Code and data for the LREC 2026 paper `Beyond Catalogue Counts: the Dataset Visibility Asymmetry in Low-Resource Multilingual NLP`.

> How do we know whether a language is low-resource, and what evidence are we using?
>
> This project compares catalogue records with paper-traced dataset evidence, and maintains a browsable dataset collection for multilingual NLP researchers.

## Start Here

- 🧭 **[Project page](https://zhiyintan.github.io/dataset-visibility-asymmetry/)** — read the story, figures, and main claims.
- 🔎 **[Dataset browser](https://zhiyintan.github.io/dataset-visibility-asymmetry/data.html)** — search checked dataset records by language, modality, task, and access state.


## How This Repository Fits In

This repository is the analysis and release layer for the paper. The upstream discovery layer lives in [M3D](https://github.com/Fireblossom/citation-context-dataset-discovery).

- **Discovery layer: M3D**  
  Start there if you want to input language names, query Semantic Scholar, extract citation contexts, and generate candidate dataset records.

- **Analysis and release layer: this repository**  
  Use this repo to inspect the paper snapshot, catalogue baselines, checked tables, analysis scripts, generated figures, GitHub Pages site, and public dataset browser built from the candidate records.

## Contribute a dataset

Missing a dataset? Please [open an issue](https://github.com/zhiyintan/dataset-visibility-asymmetry/issues/new) with the language, dataset name, source paper, and access link if available. The online dataset browser is actively maintained and may differ from the paper snapshot as records are corrected, consolidated, or added.

## Citation

If you use this work, please cite the paper:

```bibtex
@inproceedings{tan-etal-2026-beyond,
  series = {LREC},
  title = {Beyond Catalogue Counts: The Dataset Visibility Asymmetry in Low-Resource Multilingual NLP},
  ISSN = {2522-2686},
  url = {http://dx.doi.org/10.63317/3bep4yiomtp2},
  DOI = {10.63317/3bep4yiomtp2},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  publisher = {European Language Resources Association (ELRA)},
  author = {Tan,  Zhiyin and Duan,  Changxu},
  year = {2026},
  month = May,
  pages = {6068–6079},
  collection = {LREC}
}
```
