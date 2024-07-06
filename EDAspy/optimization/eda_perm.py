#!/usr/bin/env python
# coding: utf-8
import numpy as np
from typing import Union, List

from .eda import EDA
from .custom.probabilistic_models import MallowsModelR
from .custom.initialization_models import RandomGenInit


class EDAPerm(EDA):

    def __init__(self,
                 size_gen: int,
                 max_iter: int,
                 dead_iter: int,
                 n_variables: int,
                 alpha: float = 0.5,
                 elite_factor: float = 0.4,
                 disp: bool = True,
                 parallelize: bool = False,
                 init_data: np.array = None,
                 w_noise: float = 0,
                 model: str = 'mm',
                 distance: str = 'kendall',
                 estimation: str = 'approx'):
        r"""
        :param size_gen: Population size. Number of individuals in each generation.
        :param max_iter: Maximum number of iterations during runtime.
        :param dead_iter: Stopping criteria. Number of iterations with no improvement after which, the algorithm finish.
        :param n_variables: Number of variables to be optimized.
        :param alpha: Percentage of population selected to update the probabilistic model.
        :param elite_factor: Percentage of previous population selected to add to new generation (elite approach).
        :param disp: Set to True to print convergence messages.
        :param parallelize: True if the evaluation of the solutions is desired to be parallelized in multiple cores.
        :param init_data: Numpy array containing the data the EDA is desired to be initialized from. By default, an initializer is used.

        :param distance: The distance that the Mallows Model uses to learn the probabilistic distribution (kendall, cayley, hamming or ulam)
        :param estimation: The type of estimation used by the Mallows model (approx or exact)
        """
        super().__init__(size_gen=size_gen, max_iter=max_iter, dead_iter=dead_iter,
                         n_variables=n_variables, alpha=alpha, elite_factor=elite_factor, disp=disp,
                         parallelize=parallelize, init_data=init_data, w_noise=w_noise)
        
        self.model = model
        self.distance = distance
        self.estimation = estimation

        if self.model != 'mm' and self.model != 'gmm':
            print("ERROR. Modelo mal introducido.")
        if (self.distance != 'kendall' and self.distance != 'cayley' and self.distance != 'hamming' and self.distance != 'ulam') or (self.estimation != 'approx' and self.estimation != 'exact'):
            print("ERROR. Distancia o estimacioón mal introducidas.")
        if self.model == 'mm' and self.estimation == 'approx' and self.distance == 'hamming':
            print("ERROR. El aprendizaje aproximado no disponible para la distancia hamming en el modelo de mallows (MM)")
        if self.model == 'mm' and self.estimation == 'exact' and (self.distance == 'kendall' or self.distance == 'ulam'):
            print("ERROR. El aprendizaje exacto no está disponible para las distancias kendall o ulam en el modelo de mallows (MM)")
        if self.model == 'gmm' and self.estimation == 'approx' and self.distance == 'ulam':
            print("ERROR. El aprendizaje aproximado no está disponible para la distancia ulam en el modelo de mallows generalizado (GMM).")
        if self.model == 'gmm' and self.estimation == 'exact' and (self.distance == 'kendall' or self.distance == 'hamming' or self.distance == 'ulam'):
            print("ERROR. El aprendizaje exacto no está disponible para las distancias kendall, hamming o ulam en el modelo de mallows generalizado (GMM).")
        
        self.pm = MallowsModelR(list(range(self.n_variables)), self.model, self.distance, self.estimation)
        self.init = RandomGenInit(self.n_variables)