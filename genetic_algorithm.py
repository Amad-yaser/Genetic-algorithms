import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def initialize_population(num_features, population_size):
    return np.random.randint(2, size=(population_size, num_features))

def calculate_fitness(chromosome, X_train, X_test, y_train, y_test):
    selected_features_indices = np.where(chromosome == 1)[0]
    
    if len(selected_features_indices) == 0:
        return 0
        
    X_train_subset = X_train.iloc[:, selected_features_indices]
    X_test_subset = X_test.iloc[:, selected_features_indices]
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_subset, y_train)
    y_pred = model.predict(X_test_subset)
    accuracy = accuracy_score(y_test, y_pred)
    
    feature_penalty = 0.01 * (len(selected_features_indices) / X_train.shape[1])
    
    return accuracy - feature_penalty

def selection(population, fitness_scores):
    top_indices = np.argsort(fitness_scores)[-len(population)//2:]
    return population[top_indices]

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    return child

def mutation(chromosome, mutation_rate=0.05):
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

def run_genetic_algorithm(X, y, population_size=50, generations=20, mutation_rate=0.05, progress_callback=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    num_features = X.shape[1]
    
    population = initialize_population(num_features, population_size)
    
    best_chromosome_overall = None
    best_fitness_overall = -1

    for gen in range(generations):
        fitness_scores = np.array([calculate_fitness(chrom, X_train, X_test, y_train, y_test) for chrom in population])
        
        current_best_fitness = np.max(fitness_scores)
        if current_best_fitness > best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_chromosome_overall = population[np.argmax(fitness_scores)]

        if progress_callback:
            progress_callback(gen + 1, generations, best_fitness_overall)
        
        parents = selection(population, fitness_scores)
        
        next_population = []
        while len(next_population) < population_size:
            parent1, parent2 = parents[np.random.choice(len(parents), 2, replace=False)]
            
            child = crossover(parent1, parent2)
            
            child = mutation(child, mutation_rate)
            
            next_population.append(child)
        
        population = np.array(next_population)

    selected_features = list(X.columns[np.where(best_chromosome_overall == 1)[0]])
    return selected_features, best_fitness_overall