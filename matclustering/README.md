# MAT-clustering: Clustering Methods for Multiple Aspect Trajectory Data Mining \[MAT-Tools Framework\]
---

\[[Publication](#)\] \[[Bibtex](https://github.com/mat-analysis/mat-tools/blob/main/references/mat-tools.bib)\] \[[GitHub](https://github.com/mat-analysis/mat-clustering)\] \[[PyPi](https://pypi.org/project/mat-clustering/)\]

The present application offers a tool, to support the user in the data mining task of multiple aspect trajectories, specifically for clustering its complex data. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. 

Created on Apr, 2024
Copyright (C) 2024, License GPL Version 3 or superior (see LICENSE file)

### Main Modules

- Core Classes:
    1. **TrajectoryClustering** - Base class for trajectory clustering
    2. **HSTrajectoryClustering** - Hyperparameter search model for trajectory clustering
    3. **SimilarityClustering** - Similarity-based clustering for trajectory data

- Similarity-based clustering methods:
    1. **TSAgglomerative** - MAT Hierarchical Agglomerative Clustering
    2. **TSBirch** - MAT BIRCH Clustering
    3. **TSDBSCAN** - MAT DBSCAN Clustering    
    4. **TSKMeans** - MAT K-Means Clustering
    5. **TSKMedoids** - MAT K-Medoids Clustering
    6. **TSpectral** - MAT Spectral Clustering
    
- CoClustering clustering methods: **Under Development**

- Hierarchical clustering methods: **Under Development**

### Installation

Install directly from PyPi repository, or, download from github. (python >= 3.7 required)

```bash
    pip install mat-clustering
```

### Citing

If you use `mat-clustering` please cite the following paper:

 - Portela, T. T.; Machado, V. L.; Renso, C. Unified Approach to Trajectory Data Mining and Multi-Aspect Trajectory Analysis with MAT-Tools Framework. In: SIMPÓSIO BRASILEIRO DE BANCO DE DADOS (SBBD), 39. , 2024, Florianópolis/SC. \[[Bibtex](https://github.com/mat-analysis/mat-tools/blob/main/references/mat-tools.bib)\]

### Collaborate with us

Any contribution is welcome. This is an active project and if you would like to include your algorithm in `matclustering`, feel free to fork the project, open an issue and contact us.

Feel free to contribute in any form, such as scientific publications referencing `matclustering`, teaching material and workshop videos.

### Related packages

This package is part of _MAT-Tools Framework_ for Multiple Aspect Trajectory Data Mining, check the guide project:

- **[mat-tools](https://github.com/mat-analysis/mat-tools)**: Reference guide for MAT-Tools Framework repositories


### Change Log

This is a package under construction, see [CHANGELOG.md](https://github.com/mat-analysis/mat-clustering/blob/main/CHANGELOG.md)
