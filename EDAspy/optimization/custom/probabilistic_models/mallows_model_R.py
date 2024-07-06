#!/usr/bin/env python
# coding: utf-8

import math
import random
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

from ._probabilistic_model import ProbabilisticModel


class MallowsModelR(ProbabilisticModel):

    def __init__(self, variables: list, model: str, distance: str, estimation: str):
        super().__init__(variables)

        robjects.r('library(Rcpp)')
        robjects.r('library(PerMallows)')
        
        self.n_variables = len(variables)
        self.model = model
        self.distance = distance
        self.estimation = estimation

        self.sigma0 = 0
        if self.model == 'mm':
            self.theta = 0
        else:
            self.theta = []

    def sample(self, size: int) -> np.array:
        if self.model == 'mm':
            output = robjects.r('rmm')(size, self.sigma0, self.theta, self.distance, "gibbs")
        else:
            output = robjects.r('rgmm')(size, self.sigma0, self.theta, self.distance, "gibbs")
        
        samples = np.array(output)
        samples = samples.astype(int)

        return samples

    def learn(self, dataset: np.array):
        df = pd.DataFrame(dataset)
        pandas2ri.activate()
        dfr = pandas2ri.py2rpy(df)
        matrix = robjects.r('as.matrix')(dfr)
        indentity = robjects.r('identity.permutation')(len(dataset[0]))
        
        if self.model == 'mm':
            output = robjects.r('lmm')(matrix, indentity, self.distance, self.estimation)
        else:
            output = robjects.r('lgmm')(matrix, indentity, self.distance, self.estimation)

        self.sigma0 = output[0]
        self.theta = output[1]

    def print_structure(self) -> list:
        return list()