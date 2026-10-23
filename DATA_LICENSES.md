# Data licenses

## Derived artifacts (this release)

The artifacts deposited on Zenodo alongside this repository — pattern breakdowns, extracted feature matrices, per-corpus label CSVs, and pre-computed permutation-importance tables — are released under:

**Creative Commons Attribution 4.0 International (CC-BY 4.0)**
<https://creativecommons.org/licenses/by/4.0/>

Attribution must credit the paper's authors and the Zenodo record DOI.

## Source corpora

The derived artifacts are computed from three publicly available folk-music corpora, each retaining its original license. Redistribution of *raw* score files falls under those terms, not CC-BY 4.0. Only the derived features and labels are re-licensed.

### The Session
- **License**: CC0 1.0 Universal (Public Domain Dedication)
- **Source**: <https://github.com/adactio/TheSession-data>
- **Citation**: Keith, J. (2020). The Session. <https://thesession.org>.
- **Files used**: 20,747 tunes covering all tune types, converted from ABC to MusicXML.

### SymbTr (Symbolic Turkish Makam Music Corpus)
- **License**: CC BY-NC-SA 4.0 International
  <https://creativecommons.org/licenses/by-nc-sa/4.0/>
- **Source**: <https://github.com/MTG/SymbTr>
- **Citation**: Karaosmanoğlu, M. K. (2012). *A Turkish makam music symbolic database for music information retrieval: SymbTr*. ISMIR.
- **Files used**: 1,192 MusicXML scores (non-empty after cleaning).

### Essen Folksong Collection
- **License**: Distributed via the EsAC database under public-domain terms for academic use.
- **Source**: SQLite export used here originated from the Essen Associative Code / EsAC project.
- **Citation**: Schaffrath, H. (1995). *The Essen Folksong Collection*. D. Huron, ed.
- **Files used**: 2,924 melodies across seven regional collections.

## How to attribute in derivative work

Please cite:

> Gutiérrez Álvarez, E. "Viewpoint–Task Alignment: When and Why Pattern-Derived Features Improve Symbolic Folk-Music Classification." *Transactions of the International Society for Music Information Retrieval*, 2026 (in press).

> Gutiérrez Álvarez, E. *Viewpoint–Task Alignment: Derived Artifacts*. Zenodo. DOI: [10.5281/zenodo.21129773](https://doi.org/10.5281/zenodo.21129773).

If you redistribute raw SymbTr scores as part of your own work, SymbTr's non-commercial share-alike terms carry through — this release distributes only *derived features* and *breakdowns*, which are documented here as a transformative academic use.
