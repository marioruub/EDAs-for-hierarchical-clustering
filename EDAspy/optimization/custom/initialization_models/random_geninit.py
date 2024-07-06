#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math

from ._generation_init import GenInit


class RandomGenInit(GenInit):

    def __init__(self, n_variables: int):
        """
        :param n_variables: Number of variables.
        """
        super().__init__(n_variables)

        self.n_variables = n_variables
        
        self.id = 7

    def sample(self, size: int) -> np.array:
        """
        size: tamaño de la población a generar
        n_variables: número de variables por muestra
        """

        perm_list = []
        for _ in range(size):
            perm = np.arange(1, self.n_variables+1)
            np.random.shuffle(perm)
            perm_list.append(perm)
        
        return np.array(perm_list)
        