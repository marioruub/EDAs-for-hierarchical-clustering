#!/usr/bin/env python
# coding: utf-8

import math
import random
import numpy as np
from ._probabilistic_model import ProbabilisticModel


class MallowsModel(ProbabilisticModel):

    def __init__(self, variables: list, sigma_0_ini=None, distance_type=0, estimation_type=0, sampling_method=0):
        super().__init__(variables)
        """
        sigma_0_ini: Initial consensus permutation
        distance_type: 0->Cayley (default), 1->Kendall, 2->Hamming, 3->Ulam
        estimation_type: 0->exact (default), 1->aprox
        sampling_method: 0->gibbs (default), 1->distances, 2->multistage
        """

        self.pm_sigma0 = np.zeros(len(variables), dtype=int)  
        self.pm_theta = np.empty(len(variables))

        self.distance_type = distance_type
        self.estimation_type = estimation_type
        self.sampling_method = sampling_method

        self.sigma_0_ini = sigma_0_ini
        self.estimation_type = estimation_type
        self.sampling_method = sampling_method
        self.m = 0
        self.n = 0

        self.xbound = np.empty(self.n, dtype=int)

        #Newton_raphson (for learning theta)
        self.dist_avg = 0.0
        self.initial_guess = -10.001
        self.upper_theta = 5
        self.xacc = 0.000001

        self.id = 8

        if self.sigma_0_ini is None:
            self.sigma_0_ini = np.arange(1, len(variables)+1)

        if (distance_type < 0 or distance_type > 3): return
        if (distance_type == 2 and estimation_type == 0): return
        if ((distance_type != 1 and distance_type != 2) and estimation_type == 1): return
        if (estimation_type < 0 or estimation_type > 1): return
        if (sampling_method < 0 or sampling_method > 2): return

    def sample(self, size: int) -> np.array:
        if (self.sampling_method == 0): #gibbs
            sample = self.gibbs_sampling(size)
            #print(sample)

            if (not np.array_equal(self.pm_sigma0, np.arange(1, self.n + 1))):
                # compose(sample, sigma0)
                #########################################################################################################
                #if (not self.is_permutation(sample) or not self.is_permutation(self.pm_sigma0)):
                    #return None
                if (self.is_matrix(sample) and self.is_matrix(self.pm_sigma0)):
                    return None
                if (self.is_vector(sample) and self.is_vector(self.pm_sigma0) and len(sample) == len(self.pm_sigma0)):
                    sample = self.compose_perms(sample, self.pm_sigma0)
                if (self.is_matrix(sample) and (sample.ndim == 2 and sample.shape[1] == len(self.pm_sigma0))):
                    sample = np.apply_along_axis(lambda x: self.compose_perms(x, self.pm_sigma0), axis=1, arr=sample)
                    #sample = np.transpose(resultado)
                if (self.is_matrix(self.pm_sigma0) and (self.pm_sigma0.ndim == 2 and self.pm_sigma0.shape[1] == len(sample))):
                    sample = np.apply_along_axis(lambda x: self.compose_perms(sample, x), axis=1, arr=self.pm_sigma0)
                    #sample = np.transpose(resultado)
                #########################################################################################################
            
            return sample
        else:
            return None

    def learn(self, dataset: np.array):
        if (self.estimation_type == 0): #cayley
            self.pm_sigma0 = self.estimate_consensus_exact(dataset, self.sigma_0_ini, self.distance_type)
            self.pm_theta = self.estimate_theta(dataset, self.distance_type)
        else:
            return None

    def print_structure(self) -> list:
        """
        Prints the arcs between the nodes that represent the variables in the dataset. This function
        must be used after the learning process. Univariate approaches generate no-edged graphs.

        :return: list of arcs between variables
        :rtype: list
        """
        return list()
    


    


    #Learning (sigma)
    def estimate_consensus_exact(self, samples: np.array, sigma_0_ini: np.array, distance_type: int) -> np.array:
        if (self.distance_type == 0):
            self.m = len(samples) #Número de filas (muestras)
            self.n = len(samples[0]) #Número de columnas (variables)

            self.estimate_consensus_approx(samples) #(samples, sigma_0)

            visited_nodes = self.estimate_consensus_exact_mm(samples, sigma_0_ini) #(samples, sigma_0_ini, sigma_0)

            return self.pm_sigma0

    def estimate_consensus_exact_mm(self, samples: np.array, sigma_0_ini: np.array): #(self, samples: np.array, sigma_0_ini: np.array, sigma_0: np.array)
        samples_inv = np.empty((self.m, self.n), dtype=int)
        x_acum = np.empty(self.n, dtype=int)
        sigma_0_aux = np.full(self.n, -1)
        sigma_0_inv_aux = np.full(self.n, -1)

        best_distance = (self.n - 1) * self.m

        for i in range(self.m):
            for j in range(self.n):
                samples_inv[i][samples[i][j] - 1] = j + 1
        
        self.estimate_consensus_approx(samples) #(samples, sigma_0)

        best_distance = self.distance_to_sample(samples, self.pm_sigma0)

        if sigma_0_ini is not None:
            # obtener la verosimilitud de sigma_0_ini propuesto
            dist_ini = self.distance_to_sample(samples, sigma_0_ini)
            # comparar ambas soluciones: sigma_0_ini propuesto vs. el mejor aproximado. Descartar el peor.
            if dist_ini < best_distance:
                best_distance = dist_ini
                self.pm_sigma0[:] = sigma_0_ini[:]
        
        visited_nodes = self.estimate_consensus_exact_mm_core(0, samples, samples_inv, x_acum, sigma_0_aux, sigma_0_inv_aux, 0, best_distance) #(0, samples, samples_inv, x_acum, sigma_0_aux, sigma_0_inv_aux, 0, sigma_0, best_distance)

        return visited_nodes
        
    def estimate_consensus_exact_mm_core(self, pos: int, samples: np.array, samples_inv: np.array, x_acum: np.array, current_sigma: np.array, current_sigma_inv: np.array, current_dist_bound: float, best_dist: int):
        #(self, pos: int, samples: np.array, samples_inv: np.array, x_acum: np.array, current_sigma: np.array, current_sigma_inv: np.array, current_dist_bound: float, best_sigma: np.array, best_dist: int)
        if pos == self.n and current_dist_bound <= best_dist:
            #print(self.pm_sigma0, current_sigma)
            self.pm_sigma0[:] = current_sigma[:]
            best_dist = current_dist_bound
            return 1
        
        visited_nodes = 0
        trace = True
        enc = False
        x_acum_var = np.empty(self.n, dtype=int)
        candVec = np.empty(self.n, dtype=int)
        cand = self.n
        freq = np.zeros(self.n, dtype=int)

        for s in range(self.m):
            if not enc and (freq[samples_inv[s][pos] - 1] + 1) > (self.m / 2):
                candVec[0] = samples_inv[s][pos] - 1
                enc = True
                cand = 1
            freq[samples_inv[s][pos] - 1] += 1

        if not enc:
            candVec = np.arange(0, self.n)
            cand = self.n
            
        for index in range(cand):
            it = candVec[index]
            if current_sigma[it] == -1:
                x_incr = 0
                current_sigma_inv[pos] = it + 1
                current_sigma[it] = pos + 1
                pos_swaps = np.empty(self.m, dtype=int)

                for s in range(self.m):
                    pos_swaps[s] = -1
                    if samples[s][it] != current_sigma[it]:
                        x = samples[s][it]
                        y = samples_inv[s][pos] - 1
                        samples[s][it] = pos + 1
                        samples[s][y] = x
                        samples_inv[s][pos] = it + 1
                        samples_inv[s][x - 1] = y + 1
                        pos_swaps[s] = y
                        x_incr += 1
                
                distance_bound = 0
                self.xbound = np.zeros(self.n, dtype=int)
                self.get_x_lower_bound(samples_inv, pos + 1)
                for i in range(pos + 1, self.n):
                    distance_bound += self.xbound[i]
                for i in range(self.n):
                    x_acum_var[i] = x_acum[i]
                    try:
                        distance_bound += x_acum_var[i] # <------------------
                    except OverflowError:
                        print()
                x_acum_var[pos] += x_incr
                distance_bound += x_incr

                #print("distance_bound <= best_dist", distance_bound, best_dist)
                if distance_bound <= best_dist:
                    visited_nodes += self.estimate_consensus_exact_mm_core(pos + 1, samples, samples_inv, x_acum_var, current_sigma, current_sigma_inv, distance_bound, best_dist)
                current_sigma_inv[pos] = -1
                current_sigma[it] = -1
                for s in range(self.m):
                    if pos_swaps[s] != -1:
                        y = pos_swaps[s]
                        x = samples[s][y]
                        samples[s][it] = x
                        samples[s][y] = pos + 1
                        samples_inv[s][pos] = y + 1
                        samples_inv[s][x - 1] = it + 1
                        pos_swaps[s] = -1
        
        return visited_nodes + 1

    def get_x_lower_bound(self, sample: np.array, ini_pos: int):
        freq = np.zeros(self.n, dtype=int)
        max_freq = 0

        for j in range(ini_pos, self.n - 1):
            for s in range(self.m):
                freq[sample[s][j] - 1] += 1
                if freq[sample[s][j] - 1] > max_freq:
                    max_freq = freq[sample[s][j] - 1]

            self.xbound[j] = self.m - max_freq
            if self.xbound[j] < 0:
                self.xbound[j] = 0         

    def distance_to_sample(self, samples: np.array, sigma: np.array):
        distance = 0
        comp = np.empty(self.n, dtype=int)
        sigma_inv = np.empty(self.n, dtype=int)

        for j in range(self.n):
            sigma_inv[sigma[j] - 1] = j + 1
        #print("SIGMA", sigma)
        #print("SIGMA_INV", sigma_inv)
        
        for s in range(self.m):
            for i in range(self.n):
                comp[i] = samples[s][sigma_inv[i] - 1]
            distance += self.perm2dist_decomp_vector(comp, None)
    
        return distance

    def perm2dist_decomp_vector(self, sigma: np.array, vec: np.array):
        if vec is not None:
            for i in range(self.n):
                vec[i] = 1

        num_cycles = 0
        num_visited = 0
        item = 0
        visited = np.full(self.n, False)

        while num_visited < self.n:
            item = num_cycles
            while visited[item]:
                item += 1
            num_cycles += 1
            max_item_in_cycle = 0

            while not visited[item]:
                if item > max_item_in_cycle:
                    max_item_in_cycle = item
                visited[item] = True
                num_visited += 1
                item = sigma[item] - 1

            if vec is not None:
                vec[max_item_in_cycle] = 0

        return (self.n - num_cycles)

    def estimate_consensus_approx(self, samples: np.array): #(self, samples: np.array, sigma_0: np.array)
        samples_inv = np.empty((self.m, self.n), dtype=int)
        samples_copy = np.empty((self.m, self.n), dtype=int)

        for i in range(self.m):
            for j in range(self.n):
                samples_inv[i][samples[i][j] - 1] = j + 1
                samples_copy[i][j] = samples[i][j]

        best_likeli = [0.0]
        self.estimate_consensus_approx_mm(samples_copy, samples_inv, best_likeli) #(samples_copy, samples_inv, sigma_0, best_likeli)
        #print("SIGMA antes\t", self.pm_sigma0)
        self.variable_neighborhood_search(samples, best_likeli) #(samples, sigma_0, best_likeli)
        #print("SIGMA despues\t", self.pm_sigma0)

    def estimate_consensus_approx_mm(self, samples_copy: np.array, samples_inv: np.array, best_distance: np.array): #(self, samples_copy: np.array, samples_inv: np.array, sigma_0: np.array, best_distance: np.array)
        distance_increase = 0
        remaining = self.n
        freq = np.empty((self.n, self.n), dtype=int)

        self.pm_sigma0 = np.full(self.n, -1)

        while remaining > 0:
            freq = np.zeros((self.n, self.n), dtype=int)
            max_freq = 0
            pi = -1
            pj = -1
            dirty_items = True

            for s in range(self.m):
                if (dirty_items):
                    for i in range(self.n):
                        if (dirty_items):
                            if self.pm_sigma0[i] == -1:
                                freq[i][samples_copy[s][i] - 1] += 1
                            if freq[i][samples_copy[s][i] - 1] > max_freq:
                                max_freq = freq[i][samples_copy[s][i] - 1]
                                pi = i
                                pj = samples_copy[s][i] - 1
                                if max_freq > self.m / 2:
                                    dirty_items = False
            #print("max_freq > self.m .....", pi, pj)
            
            self.pm_sigma0[pi] = pj + 1
            for s in range(self.m):
                if samples_copy[s][pi] != pj + 1:
                    x = samples_copy[s][pi]
                    y = samples_inv[s][pj] - 1
                    samples_copy[s][pi] = pj + 1
                    samples_copy[s][y] = x
                    samples_inv[s][pj] = pi + 1
                    samples_inv[s][x - 1] = y + 1
                    distance_increase += 1

            remaining -= 1
            #print("IT:", remaining, " AQUI:", self.pm_sigma0)

        best_distance[0] = distance_increase


    def variable_neighborhood_search(self, samples: np.array, f_eval: np.array): #(self, samples: np.array, sigma: np.array, f_eval: np.array)
        improve = True
        while improve:
            f_eval_ini = f_eval[0]
            #print("ANTES feval", f_eval[0])
            self.local_search_swap_mm(samples, f_eval) #(samples, sigma, f_eval)
            #print("DESPUES feval", f_eval[0])
            self.local_search_insert(samples, f_eval) #(samples, sigma, f_eval)
            improve = False
            if f_eval_ini > f_eval[0]:
                improve = True

    def local_search_swap_mm(self, samples: np.array, f_eval: np.array): #(self, samples: np.array, sigma_0: np.array, f_eval: np.array)
        samples_comp = np.empty((self.m, self.n), dtype=int)
        sigma_0_inv = np.empty(self.n, dtype=int)
        cycle_items = np.empty(self.n, dtype=int)
        cycle_index = np.empty(self.n, dtype=int)
        same_cycle = np.zeros((self.n, self.n), dtype=int)

        distance_variation = 0
        index_i = 0
        index_j = 0
        for i in range(self.n):
            sigma_0_inv[self.pm_sigma0[i] - 1] = i + 1
        
        improve = True
        while improve:
            max_freq = 0
            same_cycle = np.zeros((self.n, self.n), dtype=int)
            for s in range(self.m):
                for j in range(self.n):
                    samples_comp[s][j] = samples[s][sigma_0_inv[j] - 1]
                self.get_cycles(samples_comp[s], cycle_items, cycle_index)

                for i in range(self.n):
                    for j in range(i + 1, self.n):
                        if cycle_index[i] == cycle_index[j]:
                            if(cycle_index[i] > cycle_index[j]):
                                max_val = cycle_items[i] - 1
                                min_val = cycle_items[j] - 1
                            else:
                                min_val = cycle_items[i] - 1
                                max_val = cycle_items[j] - 1
                            same_cycle[min_val][max_val] += 1

                            if max_freq < same_cycle[min_val][max_val]:
                                max_freq = same_cycle[min_val][max_val]
                                index_i = min_val
                                index_j = max_val

            distance_variation = self.m - 2 * max_freq
            improve = False
            #print("Valores...", self.m, max_freq)
            #print("distance...", distance_variation)
            if distance_variation < 0:
                improve = True
                sigma_0_inv[index_i], sigma_0_inv[index_j] = sigma_0_inv[index_j], sigma_0_inv[index_i]
                f_eval[0] += distance_variation # <----------------

                for i in range(self.n):
                    self.pm_sigma0[sigma_0_inv[i] - 1] = i + 1

    def get_cycles(self, sigma: np.array, cycle_items: np.array, cycle_indices: np.array):
        visited = np.full(self.n, False)
        cont = 0
        cycle_index = 0

        while cont < self.n:
            item_index = 0
            while visited[item_index]:
                item_index += 1
            while not visited[item_index]:
                visited[item_index] = True
                cycle_items[cont] = item_index + 1
                cycle_indices[cont] = cycle_index
                item_index = sigma[item_index] - 1
                cont += 1
            cycle_index += 1

        return cycle_index
    
    def local_search_insert(self, samples: np.array, f_eval: np.array): #(self, samples: np.array, sigma: np.array, f_eval: np.array)
        best_sol = np.empty(self.n, dtype=int)
        next_sol = np.empty(self.n, dtype=int)
        x_acumul = np.empty(self.n, dtype=int)
        x = np.empty(self.n, dtype=int)
        theta = np.empty(self.n, dtype=int)
        best_eval = 0

        better = True
        while better:
            better = False
            best_eval = 0

            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        ##############################################
                        if i < j:
                            for s in range(i):
                                next_sol[s] = self.pm_sigma0[s]
                            for s in range(i, j):
                                next_sol[s] = self.pm_sigma0[s + 1]
                            next_sol[j] = self.pm_sigma0[i]
                            for s in range(j + 1, self.n):
                                next_sol[s] = self.pm_sigma0[s]
                        else:
                            for s in range(j):
                                next_sol[s] = self.pm_sigma0[s]
                            next_sol[j] = self.pm_sigma0[i]
                            for s in range(j + 1, i + 1):
                                next_sol[s] = self.pm_sigma0[s - 1]
                            for s in range(i + 1, self.n):
                                next_sol[s] = self.pm_sigma0[s]
                        ##############################################

                        dist = self.distance_to_sample(samples, next_sol)
                        if dist < best_eval or best_eval == 0:
                            best_eval = dist
                            best_sol[:] = next_sol[:]
            
            if (best_eval < f_eval[0]):
                f_eval[0] = best_eval
                self.pm_sigma0[:] = best_sol[:]
                better = True
    

    # Learning (theta)
    def estimate_theta(self, samples: np.array, distance_type: int):
        if (distance_type == 0):
            self.m = len(samples)
            self.n = len(samples[0])
            self.pm_theta = np.full(self.n, 0.0)

            dist = self.distance_to_sample(samples, self.pm_sigma0)

            # Newton_raphson_method
            ###################################################

            self.dist_avg = dist/self.m
           
            self.pm_theta[0] = self.rtsafe(self.initial_guess, self.upper_theta, self.xacc)

            ###################################################

            return self.pm_theta

    def rtsafe(self, x1: float, x2: float, xacc: float):
        maxit = 200
        j = 1
        dxold = 0.0
        
        fl, df = self.funcd(x1)
        fh, df = self.funcd(x2)
        
        if fl == 0.0:
            return x1
        if fh == 0.0:
            return x2
        if fl < 0.0:
            xl = x1
            xh = x2
        else:
            xh = x1
            xl = x2
        
        rts = x1
        dxold = abs(x2 - x1)
        dx = dxold
        f, df = self.funcd(rts)
        
        while j <= maxit:
            if (((rts - xh) * df - f) * ((rts - xl) * df - f) > 0.0) or (abs(2.0 * f) > abs(dxold * df)):
                dxold = dx
                dx = 0.5 * (xh - xl)
                rts = xl + dx
                if xl == rts:
                    return rts
            else:
                dxold = dx
                dx = f / df
                temp = rts
                rts -= dx
                if temp == rts:
                    return rts
            
            if abs(dx) < xacc:
                return rts
            
            f, df = self.funcd(rts)
            
            if f < 0.0:
                xl = rts
            else:
                xh = rts
            
            j += 1
        
        return 0.0 # Never get here

    def funcd(self, theta: float):
        f = self.f(theta)
        fdev = self.fdev(theta)

        return f, fdev

    def f(self, theta: float):
        sum_val = 0
        for j in range(1, self.n):
            ex = math.exp(theta)
            denom = j + ex
            sum_val += j / denom
        
        return (sum_val - self.dist_avg)

    def fdev(self, theta: float):
        sum_val = 0
        for j in range(1, self.n):
            sum_val += (-j * math.exp(theta)) / (math.pow(math.exp(theta) + j, 2))

        return sum_val


    #Sampling
    def gibbs_sampling(self, size: int):
        if (self.distance_type == 0):
            self.m = size
            self.n = len(self.pm_sigma0)

            samples = np.empty((self.m, self.n), dtype=int)
            burning_period_samples = int(self.n * np.log(self.n))

            # Generate a random permutation
            sigma = np.arange(1, self.n + 1)
            for i in range(self.n - 1):
                pos = random.randint(i, self.n - 1)
                sigma[i], sigma[pos] = sigma[pos], sigma[i]
            
            for sample in range(self.m + burning_period_samples):
                i, j = random.sample(range(self.n), 2)
                max_i, max_j = -1, -1
                
                while i == j:
                    i, j = random.sample(range(self.n), 2)
                
                make_swap = False
                if self.same_cycle(i, j, sigma):
                    make_swap = True
                else:
                    rand_double = random.random()
                    if rand_double < math.exp(-self.pm_theta[0]):
                        make_swap = True

                if make_swap:
                    sigma[i], sigma[j] = sigma[j], sigma[i]
                
                if sample >= burning_period_samples:
                    samples[sample - burning_period_samples] = sigma[:]

            return samples

    def same_cycle(self, i: int, j: int, sigma: np.array):
        index = sigma[i] - 1
        while index != i and index != j:
            index = sigma[index] - 1

        return index == j

    def is_permutation(self, perm: np.array):
        if isinstance(perm, list):
            return sorted(perm) == list(range(1, len(perm) + 1))
        elif isinstance(perm, list) and all(isinstance(row, list) for row in perm):
            return all((sorted(perm) == list(range(1, len(perm) + 1))) for row in perm)

    def is_matrix(self, matrix: np.array):
        return isinstance(matrix, np.ndarray) and len(matrix.shape) == 2

    def is_vector(self, vector: np.array):
        return isinstance(vector, np.ndarray) and vector.ndim == 1
    
    def compose_perms(self, perm1, perm2):
        return [perm1[i-1] for i in perm2] # COMPROBAR ESTO