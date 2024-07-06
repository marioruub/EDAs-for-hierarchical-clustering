import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, minmax_scale
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import cvxpy as cp
import csv

def clean_dataset(csv_file: str, n_sample: int):
    # Preprocesamiento del dataset
    dataset = pd.read_csv(csv_file)

    # 1.Eliminar los valores vacios (si los hay)
    {att : dataset[dataset[att].isnull()].shape[0] for att in dataset.columns}

    # 2.Para todos los valores no numéricos, comprobar cuantas instancias por etiqueta hay y eliminar las instancias con pocas etiquetas
    #print (dataset['Engine Information.Driveline'].value_counts())
    #print (dataset['Engine Information.Engine Type'].value_counts())
    cont_labels = dataset['Engine Information.Engine Type'].value_counts()
    labels_to_remove = cont_labels[cont_labels < 10].index
    dataset = dataset[~dataset['Engine Information.Engine Type'].isin(labels_to_remove)]
    #print (dataset['Engine Information.Hybrid'].value_counts())
    dataset = dataset.drop('Engine Information.Hybrid', axis=1)
    #print (dataset['Engine Information.Transmission'].value_counts())
    #print (dataset['Fuel Information.Fuel Type'].value_counts())
    dataset.drop(dataset[dataset['Fuel Information.Fuel Type']=="Diesel fuel"].index,inplace=True)
    #print (dataset['Identification.Classification'].value_counts())
    #print (dataset['Identification.ID'].value_counts())
    dataset = dataset.drop('Identification.ID', axis=1)
    #print (dataset['Identification.Make'].value_counts())
    cont_labels = dataset['Identification.Make'].value_counts()
    labels_to_remove = cont_labels[cont_labels < 20].index
    dataset = dataset[~dataset['Identification.Make'].isin(labels_to_remove)]
    #print (dataset['Identification.Model Year'].value_counts())
    cont_labels = dataset['Identification.Model Year'].value_counts()
    labels_to_remove = cont_labels[cont_labels < 5].index
    dataset = dataset[~dataset['Identification.Model Year'].isin(labels_to_remove)]

    # 3.Barajar el dataset
    dataset=dataset.sample(frac=1)
    dataset=dataset.sample(frac=1)
    dataset=dataset.sample(frac=1).reset_index(drop=True)

    # 4.Codificar los valores discretos
    engine_info_driveline = LabelEncoder()
    dataset['Engine Information.Driveline'] = engine_info_driveline.fit_transform(dataset['Engine Information.Driveline'].values)
    engine_info_engine_type = LabelEncoder()
    dataset['Engine Information.Engine Type'] = engine_info_engine_type.fit_transform(dataset['Engine Information.Engine Type'].values)
    engine_info_transmission = LabelEncoder()
    dataset['Engine Information.Transmission'] = engine_info_transmission.fit_transform(dataset['Engine Information.Transmission'].values)
    engine_info_fuel_type = LabelEncoder()
    dataset['Fuel Information.Fuel Type'] = engine_info_fuel_type.fit_transform(dataset['Fuel Information.Fuel Type'].values)
    identification_classification = LabelEncoder()
    dataset['Identification.Classification'] = identification_classification.fit_transform(dataset['Identification.Classification'].values)
    identification_make = LabelEncoder()
    dataset['Identification.Make'] = identification_make.fit_transform(dataset['Identification.Make'].values)
    identification_model_year = LabelEncoder()
    dataset['Identification.Model Year'] = identification_model_year.fit_transform(dataset['Identification.Model Year'].values)

    samples = dataset.sample(n=n_sample, random_state=None).reset_index(drop=True)
    return samples

def normalize_matrix(matrix: np.array, n: int):
    norm_matrix = np.zeros((n, n))
    max_num = np.max(matrix)
    min_num = np.min(matrix)
    for i in range(n):
        for j in range(n):
            norm_matrix[i][j] = (matrix[i][j] - min_num)/(max_num - min_num)
    
    return norm_matrix

def objective_function(dissimilarity_matrix: np.array, dist_matrix: np.array, n: int):
    sum_fitness = 0
    for i in range(n):
        for j in range(i):
            sum_fitness += (dissimilarity_matrix[i][j] - dist_matrix[i][j]) ** 2
    sum_fitness = round(sum_fitness, 2)

    return sum_fitness

def get_order(order: np.array, indexes: np.array, num: int):
    if indexes[num][0] < 0 and indexes[num][1] >= 0:# izq - der +
        order = np.append(order, -indexes[num][0])
        order = get_order(order, indexes, indexes[num][1])
        return order
    elif indexes[num][0] >= 0 and indexes[num][1] < 0:# izq + der -
        order = get_order(order, indexes, indexes[num][0])
        order = np.append(order, -indexes[num][1])
        return order
    elif indexes[num][0] >= 0 and indexes[num][1] >= 0: # izq + der +
        order = get_order(order, indexes, indexes[num][0])
        order = get_order(order, indexes, indexes[num][1])
        return order
    else: # izq - der -
        order = np.append(order, -indexes[num][0])
        order = np.append(order, -indexes[num][1])
        return order

def get_u_dist_matrix(cod, n: int, order, dissimilarity_matrix):
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

    # Ordenar la matriz ultramétrica con el orden
    sorted_u_dist_matrix_variables = np.zeros((n_objects, n_objects), dtype=int)
    for i in range(n_objects):
        pos1 = np.where(order == i+1)[0][0]
        for j in range(n_objects):
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


# MAIN #

n_objects = 10

# Crear matriz disimilar del dataset
#csv_file = "C:\\Users\\usuario\\OneDrive\\Escritorio\\cars.csv"
#dataset = clean_dataset(csv_file, n_objects)
dataset = pd.read_csv('C:\\Users\\usuario\\OneDrive\\Escritorio\\clean_data\\covid_mobility' + str(n_objects) +'.csv')
dissimilarity_matrix = squareform(pdist(dataset, metric='euclidean'))
dissimilarity_matrix = np.round(dissimilarity_matrix, decimals=2)
#norm_dissimilarity_matrix = normalize_matrix(dissimilarity_matrix, n_objects)

# Algoritmo clustering jerárquico
linkage_matrix = linkage(dataset, method='ward', metric='euclidean')
cod = np.full((n_objects - 1, 2), 9999)
for i in range(n_objects-1):
    for j in range(2):
        if linkage_matrix[i][j] < n_objects:
            cod[i][j] = -(linkage_matrix[i][j] + 1)
        else:
            cod[i][j] = linkage_matrix[i][j] - n_objects

# Calcular matriz de distancias ultramétrica
order = np.array([], dtype=int)
order = get_order(order, cod, n_objects-2)
u_dist_matrix = get_u_dist_matrix(cod, n_objects, order, dissimilarity_matrix)

# Calcular fitness
fitness = objective_function(dissimilarity_matrix, u_dist_matrix, n_objects)
print("fitness:", fitness)

plt.figure(figsize=(5, 5))
dendrogram(Z=linkage_matrix, color_threshold=0, above_threshold_color='black')
plt.show()


"""
# Definir las variables de optimización
delta = cp.Variable(4, nonneg=True)

# Definir los términos constantes en la función objetivo
a1 = (4 + 3 + 5 + 1) / 4
a3 = (8 + 3 + 6 + 2) / 4

# Definir la función objetivo
objective = cp.Minimize((delta[0] - a1)**2 + (delta[1] - 2)**2 + (delta[2] - a3)**2 + (delta[3] - 9)**2)

# Definir las restricciones
constraints = [
    delta[2] - delta[0] <= 0,
    delta[1] - delta[2] <= 0,
    delta[3] - delta[2] <= 0,
]

# Formular el problema
problem = cp.Problem(objective, constraints)

# Resolver el problema
problem.solve()

# Imprimir los resultados
print("Estado del problema:", problem.status)
print("Valor óptimo de la función objetivo:", problem.value)
print("Solución óptima delta:", delta.value)
"""
