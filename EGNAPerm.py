import numpy as np
import pandas as pd
import csv
from scipy.spatial.distance import pdist, squareform
from EDAspy.optimization import EGNA
import time
import argparse

def cod2tree(order: np.array, structure: np.array, n: int):
    tree = np.full((n - 1, 2), 9999)
    aux_structure = np.zeros(n - 1, dtype=int)
    aux_order = np.zeros(n, dtype=int)
    aux_structure[:] = structure[:]
    aux_order[:] = order[:]
    for i in range(n - 1):
        if 2 not in aux_structure:
            print("Codificación no válida")
            return
        for j in range(n - 1):
            if aux_structure[j] == 2:
                if aux_order[j] != 0 and aux_order[j+1] != 0:
                    tree[i] = np.array([-aux_order[j], -aux_order[j+1]])
                    aux_order[j] = 0
                    aux_order[j+1] = 0
                elif aux_order[j] != 0 and aux_order[j+1] == 0:
                    row = np.where(tree == -order[j+1])[0][0]
                    while row in tree:
                        row = np.where(tree == row)[0][0]
                    tree[i] = np.array([-aux_order[j], row])
                    aux_order[j] = 0
                elif aux_order[j] == 0 and aux_order[j+1] != 0:
                    row = np.where(tree == -order[j])[0][0]
                    while row in tree:
                        row = np.where(tree == row)[0][0]
                    tree[i] = np.array([row, -aux_order[j+1]])
                    aux_order[j+1] = 0
                elif aux_order[j] == 0 and aux_order[j+1] == 0:
                    row1 = np.where(tree == -order[j])[0][0]
                    while row1 in tree:
                        row1 = np.where(tree == row1)[0][0]
                    row2 = np.where(tree == -order[j+1])[0][0]
                    while row2 in tree:
                        row2 = np.where(tree == row2)[0][0]
                    tree[i] = np.array([row1, row2])
                aux_structure[j] = 0
                k = j
                if k-1 > -1:
                    while aux_structure[k-1] == 0:
                        k -= 1
                        if k-1 <= -1:
                            break
                    if k-1 > -1:
                        aux_structure[k-1] -= 1
                k = j
                if k+1 < n - 1:
                    while aux_structure[k+1] == 0:
                        k += 1
                        if k+1 >= n - 1:
                            break
                    if k+1 < n - 1:
                        aux_structure[k+1] -= 1
                break
        
    return tree

def get_u_dist_matrix(structure, n: int, order, dissimilarity_matrix, u_dist_matrix_variables, cod):
    # Ordenar la matriz ultramétrica con el orden
    sorted_u_dist_matrix_variables = np.zeros((n, n))
    for i in range(n):
        pos1 = np.where(order == i+1)[0][0]
        for j in range(n):
            pos2 = np.where(order == j+1)[0][0]
            sorted_u_dist_matrix_variables[i][j] = u_dist_matrix_variables[pos1][pos2]
    sorted_u_dist_matrix_variables = sorted_u_dist_matrix_variables.astype(int)
    
    # Calculo de cada variable
    u_dist_matrix = np.zeros((n, n))
    vars_values = np.empty(n-1)
    cont = np.full(n-1, 0, dtype=int)
    acum = np.full(n-1, 0.0)
    for i in range(n):
        for j in range(i):
            cont[sorted_u_dist_matrix_variables[i][j]] += 1
            acum[sorted_u_dist_matrix_variables[i][j]] += dissimilarity_matrix[i][j]
    for i in range(n-1):
        vars_values[i] = acum[i] / cont[i]
    for i in range(n):
        for j in range(i):
            u_dist_matrix[i][j] = vars_values[sorted_u_dist_matrix_variables[i][j]]
            u_dist_matrix[j][i] = vars_values[sorted_u_dist_matrix_variables[i][j]]

    # Compruebo si es una matriz ultramétrica válida
    validation = False
    for i in range(n-2, -1, -1):
        if cod[i][0] >= 0 and cod[i][1] >= 0: #+ +
            if vars_values[cod[i][0]] > vars_values[i]:
                vars_values[cod[i][0]] = vars_values[i]
                validation = True
            if vars_values[cod[i][1]] > vars_values[i]:
                vars_values[cod[i][1]] = vars_values[i]
                validation = True
        elif cod[i][0] < 0 and cod[i][1] >= 0: #- +
            if vars_values[cod[i][1]] > vars_values[i]:
                vars_values[cod[i][1]] = vars_values[i]
                validation = True
        elif cod[i][0] >= 0 and cod[i][1] < 0: #+ -
            if vars_values[cod[i][0]] > vars_values[i]:
                vars_values[cod[i][0]] = vars_values[i]
                validation = True
    
    if(validation):
        for k in range(n-1):
            rows = np.where(sorted_u_dist_matrix_variables == k)[0]
            rows = np.unique(rows)
            for i in rows:
                row = np.where(sorted_u_dist_matrix_variables[i] == k)[0]
                if len(row) <= 1:
                    j = row[0]
                    u_dist_matrix[i][j] = vars_values[k]
                else:
                    for j in row:
                        u_dist_matrix[i][j] = vars_values[k]
    
    return u_dist_matrix

def root_nodes(structure, n: int, order, cod):
    # Calculo de los nodos raiz de cada dos objetos
    u_dist_matrix_variables = np.full((n, n), -1, dtype=int)
    for i in range(n):
        for j in range(i):
            root = len(cod)-1
            nodes_i = []
            row = np.where(cod == -order[i])[0][0]
            nodes_i.append(row)
            while root != row:
                row = np.where(cod == row)[0][0]
                nodes_i.append(row)
            nodes_j = []
            row = np.where(cod == -order[j])[0][0]
            nodes_j.append(row)
            while root != row:
                row = np.where(cod == row)[0][0]
                nodes_j.append(row)
            
            root_tree = 0
            for k in range(len(nodes_i)):
                if nodes_i[k] in nodes_j:
                    root_tree = nodes_i[k]
                    break
            u_dist_matrix_variables[i][j] = root_tree
            u_dist_matrix_variables[j][i] = root_tree

    return u_dist_matrix_variables

def objective_function(dissimilarity_matrix: np.array, dist_matrix: np.array, n: int):
    sum_fitness = 0
    for i in range(n):
        for j in range(i):
            sum_fitness += (dissimilarity_matrix[i][j] - dist_matrix[i][j]) ** 2
    sum_fitness = round(sum_fitness, 2)
    
    return sum_fitness

def discrete_order(order):
    sorted_indices = np.argsort(order)
    order = np.empty_like(sorted_indices)
    order[sorted_indices] = np.arange(len(order))
    order += 1

    return order


# MAIN #
parser = argparse.ArgumentParser(description="EDAPerm")
parser.add_argument("n_objects", type=int, help="Number of objects")
parser.add_argument("dataset_name", type=str, help="The dataset number")
args = parser.parse_args()

n_objects = args.n_objects
dataset_name = args.dataset_name

url = "C:\\Users\\usuario\\OneDrive\\Escritorio"

# Importar estructuras válidas para los EDAs
csv_file = url + "\\individuals\\n" + str(n_objects) + ".csv"
structures_list = []
with open(csv_file, mode='r') as file:
    reader = csv.reader(file)
    for structure in reader:
        row = [int(value) for value in structure]
        structures_list.append(row)

if len(structures_list) < n_objects * 50:
    n_samples = len(structures_list)
else:
    n_samples = n_objects * 50


# Crear matriz disimilar del dataset
dataset = pd.read_csv(url + "\\clean_data\\" + dataset_name + str(n_objects) + ".csv")
dissimilarity_matrix = squareform(pdist(dataset, metric='euclidean'))
dissimilarity_matrix = np.round(dissimilarity_matrix, decimals=2)

#Bucle de EDAs
size_gen = n_objects * 50
max_iter = 40
dead_iter = 20
lower_bound = -n_objects
upper_bound = n_objects

edas_results = []
egna = EGNA(size_gen=size_gen, max_iter=max_iter, dead_iter=dead_iter, n_variables=n_objects, lower_bound=lower_bound, upper_bound=upper_bound)
start_time = time.time()
for index, structure in enumerate(structures_list[:n_samples]):
    order = np.arange(1, n_objects+1)
    cod = cod2tree(order, structure, n_objects)
    u_dist_matrix_variables = root_nodes(structure, n_objects, order, cod)

    def fitness_function(order: list):
        order = discrete_order(order)
        u_dist_matrix = get_u_dist_matrix(structure, n_objects, order, dissimilarity_matrix, u_dist_matrix_variables, cod)
        fitness = objective_function(dissimilarity_matrix, u_dist_matrix, n_objects)
        return fitness
    
    print("EDA número", index+1, structure)
    edas_results.append(egna.minimize(fitness_function, True))
end_time = time.time()
time = end_time - start_time

fitness_best_result = 9999999999999999999999999999999999999999999
best_individual = []
best_individual_index = 0
for index, eda_result in enumerate(edas_results):
    if eda_result.best_cost < fitness_best_result:
        fitness_best_result = eda_result.best_cost
        best_individual = discrete_order(eda_result.best_ind)
        best_individual_index = index

with open(url + "\\NewExecN" + str(n_objects) + ".txt", 'a') as f:
    f.write("n_objects: {}, n_samples: {}, size_gen: {}, max_iter: {}, dead_iter: {}\n".format(n_objects, n_samples, size_gen, max_iter, dead_iter))
    f.write("{} {} {} {}\n".format(fitness_best_result, time, best_individual, structures_list[best_individual_index]))
    f.write("----------------------------------------------------\n")