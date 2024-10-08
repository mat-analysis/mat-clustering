# -*- coding: utf-8 -*-
"""
MAT-Tools: Python Framework for Multiple Aspect Trajectory Data Mining

The present application offers a tool, to support the user in the clustering of multiple aspect trajectory data.It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Apr, 2024
Copyright (C) 2024, License GPL Version 3 or superior (see LICENSE file)

Authors:
    - Tarlis Portela
    - Yuri Santos
"""
def get_similarity_matrix(cls, dataset, sim_measure):
    """
      Creates the distance matrix of the trajectories of a given cluster using
      the given similarity metric.

      Parameters
      ----------
      dataset : pandas.DataFrame
        Dataset of trajectories of a given cluster
      sim_measure : str
        Similarity metric [MUITAS, MSM, EDR, LCSS].

      Returns
      -------
      pandas.DataFrame
        A dataframe of [MUITAS, MSM, EDR, LCSS] similarity metric.
    """

    df_tid = dataset.tid.unique()
    cols = []
    # Select columns according to trajectories
    for e in df_tid:
        cols.append(str(e))

    df_aux = None
    sim_matrix = None
    if sim_measure == 'MUITAS':
        sim_matrix = df_muitas[cols]
        df_aux = df_muitas.copy()
    elif sim_measure == 'MSM':
        sim_matrix = df_msm[cols]
        df_aux = df_msm.copy()
    elif sim_measure == 'EDR':
        sim_matrix = df_edr[cols]
        df_aux = df_edr.copy()
    elif sim_measure == 'LCSS':
        sim_matrix = df_lcss[cols]
        df_aux = df_lcss.copy()

    idx = []
    # Select rows according to trajectories
    for c in sim_matrix:
        idx.append(df_aux.columns.get_loc(c))
    sim_matrix = sim_matrix[sim_matrix.index.isin(idx)]
    return sim_matrix.set_index(sim_matrix.columns)