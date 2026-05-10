# Beyond Catalogue Counts: the Dataset Visibility Asymmetry in Low-Resource Multilingual NLP

Code and data for the LREC 2026 paper `Beyond Catalogue Counts: the Dataset Visibility Asymmetry in Low-Resource Multilingual NLP`.

## Overview

Multilingual NLP often labels languages as resource-rich or resource-poor based on dataset counts in centralized catalogues. However, catalogues only capture one layer of visibility (what has been registered or institutionally distributed) and may miss datasets that are actually created, cited, or reused in research practice.

This work combines two complementary perspectives to quantify that gap:

1. **Resource Density Index (RDI)**: a population-normalized metric defined as the number of catalogued datasets per one million speakers, computed for the 200 most widely spoken languages in *Ethnologue* (2025), using entries from the LRE Map and the LDC.
2. **Citation-based dataset audit**: an LLM-assisted citation-mining pipeline over Semantic Scholar, applied to the 141 low-visibility languages whose average catalogue RDI falls below 0.1.

## Key findings

- 118 of the top-200 languages (59%) have an average RDI of 0 across LRE Map and LDC; another 23 fall below 0.1.
- After manual validation and consolidation, the citation-based audit yields **609 unique datasets across 53 languages**, of which **356 remain openly accessible** through working public links.
- Many high-population languages (e.g., Indonesian, Marathi, Assamese, Setswana, Nepali) appear data-poor in catalogues but show clear evidence of dataset activity in the literature.
- Multilingual data scarcity should be understood not only as a production problem, but also as a question of documentation, discoverability, and long-term accessibility.

## Repository structure

```
data/        # catalogue sources (LRE Map, LDC) and metadata-fetching scripts
scripts/     # citation-mining, modality extraction, and plotting pipelines
tables/      # per-language RDI counts, validated dataset inventory, modality/usage tables
plots/       # figures used in the paper (RDI distribution, emergence vs. usage, modality Sankey)
paper.tex    # LaTeX source of the paper
```

## Citation

If you use this work, please cite the paper (LREC 2026). BibTeX entry will be added upon publication.