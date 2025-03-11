"""
MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) implementation for timetable scheduling.

This algorithm uses the following key components:
1. Decomposition of multi-objective problem into single-objective subproblems
2. Weight vectors to define subproblems
3. Neighborhood-based selection and replacement
4. Tchebycheff scalarizing function to convert multiple objectives into a single objective

The algorithm optimizes for multiple objectives simultaneously:
- Room overbooking
- Slot conflicts
- Professor conflicts
- Student group conflicts
- Unassigned activities
"""
import random
import numpy as np
from utils import (
    Space, Group, Activity, Lecturer, 
    evaluate_hard_constraints, evaluate_soft_constraints, 
    evaluate, get_classsize, dominates, load_data
)

# Algorithm parameters
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
NUM_OBJECTIVES = 5  # Number of objectives in the optimization
T = 10  # Neighborhood size
NEIGHBORHOOD_SELECTION_PROB = 0.9  # Probability of selecting from neighborhood

def generate_initial_population(slots, spaces_dict, activities_dict):
    """
    Generate an initial random population of timetables.
    
    Args:
        slots: List of available time slots
        spaces_dict: Dictionary of available spaces/rooms
        activities_dict: Dictionary of activities to be scheduled
    
    Returns:
        List of randomly initialized timetable dictionaries
    """
    population = []
    
    for _ in range(POPULATION_SIZE):
        # Initialize empty timetable
        timetable = {}
        for slot in slots:
            timetable[slot] = {}
            for space in spaces_dict:
                timetable[slot][space] = ""
        
        # Randomly assign activities to slots and rooms
        activities_list = list(activities_dict.values())
        random.shuffle(activities_list)
        
        for activity in activities_list:
            # Randomly choose a slot and a room
            slot = random.choice(slots)
            space = random.choice(list(spaces_dict.keys()))
            
            # Assign the activity to the timetable
            timetable[slot][space] = activity
        
        population.append(timetable)
    
    return population

def evaluate_population(population, activities_dict, groups_dict, spaces_dict):
    """
    Evaluate the fitness of each timetable in the population.
    
    Args:
        population: List of timetables to evaluate
        activities_dict, groups_dict, spaces_dict: Data dictionaries for evaluation
    
    Returns:
        List of fitness tuples for each timetable
    """
    fitness_values = []
    for timetable in population:
        fitness_values.append(evaluate_hard_constraints(timetable, activities_dict, groups_dict, spaces_dict))
    return fitness_values

def generate_weight_vectors(num_vectors, num_objectives):
    """
    Generate evenly distributed weight vectors for decomposition.
    
    Args:
        num_vectors: Number of weight vectors to generate
        num_objectives: Number of objectives
    
    Returns:
        Array of weight vectors
    """
    if num_objectives == 2:
        # Simple method for bi-objective case
        weights = []
        for i in range(num_vectors):
            weight = i / (num_vectors - 1) if num_vectors > 1 else 0.5
            weights.append([weight, 1 - weight])
        return np.array(weights)
    else:
        # Simplified method for more than 2 objectives
        weights = []
        for _ in range(num_vectors):
            weight = np.random.dirichlet(np.ones(num_objectives))
            weights.append(weight)
        return np.array(weights)

def compute_euclidean_distances(weight_vectors):
    """
    Compute Euclidean distances between weight vectors.
    
    Args:
        weight_vectors: Array of weight vectors
    
    Returns:
        Distances between weight vectors and their indices sorted by distance
    """
    num_vectors = len(weight_vectors)
    distances = np.zeros((num_vectors, num_vectors))
    sorted_indices = np.zeros((num_vectors, num_vectors), dtype=int)
    
    for i in range(num_vectors):
        for j in range(num_vectors):
            distances[i, j] = np.linalg.norm(weight_vectors[i] - weight_vectors[j])
        
        # Sort indices by distance (excluding self)
        sorted_indices[i] = np.argsort(distances[i])
    
    return distances, sorted_indices

def initialize_neighborhoods(sorted_indices, t):
    """
    Initialize neighborhoods for each subproblem.
    
    Args:
        sorted_indices: Indices of weight vectors sorted by distance
        t: Neighborhood size
    
    Returns:
        List of neighborhoods (each a list of indices)
    """
    num_vectors = len(sorted_indices)
    neighborhoods = []
    
    for i in range(num_vectors):
        # Exclude the first index (self) and take the next t indices
        neighborhood = sorted_indices[i, 1:t+1].tolist()
        neighborhoods.append(neighborhood)
    
    return neighborhoods

def tchebycheff_distance(fitness, weight, ideal_point):
    """
    Calculate the Tchebycheff distance for a fitness vector.
    
    Args:
        fitness: Fitness vector
        weight: Weight vector
        ideal_point: Ideal point (best objective values)
    
    Returns:
        Tchebycheff distance
    """
    max_val = 0
    for i in range(len(fitness)):
        # Prevent division by zero
        if weight[i] < 1e-6:
            val = 1e6  # Large value to penalize
        else:
            val = abs(fitness[i] - ideal_point[i]) / weight[i]
        
        max_val = max(max_val, val)
    
    return max_val

def crossover(parent1, parent2):
    """
    Perform crossover by swapping time slots between two parents.
    
    Args:
        parent1, parent2: Two parent timetable dictionaries
    
    Returns:
        Offspring timetable dictionary
    """
    if random.random() > CROSSOVER_RATE:
        return parent1.copy()
    
    child = parent1.copy()
    slots = list(parent1.keys())
    
    # One-point crossover
    split = random.randint(0, len(slots) - 1)
    
    for i in range(split, len(slots)):
        child[slots[i]] = parent2[slots[i]]
    
    return child

def mutate(individual, slots, spaces_dict):
    """
    Perform mutation by randomly swapping activities in the timetable.
    
    Args:
        individual: Timetable dictionary to mutate
        slots: List of available time slots
        spaces_dict: Dictionary of available spaces/rooms
    
    Returns:
        Mutated timetable dictionary
    """
    if random.random() > MUTATION_RATE:
        return individual
    
    # Choose two random slots and spaces
    slot1, slot2 = random.sample(slots, 2)
    space1, space2 = random.choice(list(spaces_dict.keys())), random.choice(list(spaces_dict.keys()))
    
    # Swap activities
    individual[slot1][space1], individual[slot2][space2] = individual[slot2][space2], individual[slot1][space1]
    
    return individual

def select_parent_indexes(index, neighborhoods, neighborhood_selection_prob):
    """
    Select parent indices for reproduction.
    
    Args:
        index: Index of the current subproblem
        neighborhoods: List of neighborhoods for each subproblem
        neighborhood_selection_prob: Probability of selecting from neighborhood
    
    Returns:
        Two indices for parents
    """
    if random.random() < neighborhood_selection_prob:
        # Select from neighborhood
        neighborhood = neighborhoods[index]
        return random.sample(neighborhood, 2)
    else:
        # Select from entire population
        return random.sample(range(POPULATION_SIZE), 2)

def update_reference_point(fitness_values, ideal_point):
    """
    Update the ideal point with the best objective values found.
    
    Args:
        fitness_values: List of fitness vectors
        ideal_point: Current ideal point
    
    Returns:
        Updated ideal point
    """
    for fitness in fitness_values:
        for i in range(len(ideal_point)):
            ideal_point[i] = min(ideal_point[i], fitness[i])
    
    return ideal_point

def moead(activities_dict, groups_dict, spaces_dict, slots, generations=NUM_GENERATIONS):
    """
    Run the MOEA/D algorithm to optimize timetable scheduling.
    
    Args:
        activities_dict, groups_dict, spaces_dict: Data dictionaries
        slots: List of available time slots
        generations: Number of generations to run
    
    Returns:
        Final population of optimized timetables and their fitness values
    """
    # Initialize weight vectors
    weight_vectors = generate_weight_vectors(POPULATION_SIZE, NUM_OBJECTIVES)
    
    # Compute distances between weight vectors and initialize neighborhoods
    _, sorted_indices = compute_euclidean_distances(weight_vectors)
    neighborhoods = initialize_neighborhoods(sorted_indices, T)
    
    # Initialize population
    population = generate_initial_population(slots, spaces_dict, activities_dict)
    
    # Evaluate initial population
    fitness_values = evaluate_population(population, activities_dict, groups_dict, spaces_dict)
    
    # Initialize ideal point
    ideal_point = [min(f[i] for f in fitness_values) for i in range(NUM_OBJECTIVES)]
    
    for generation in range(generations):
        # For each subproblem
        for i in range(POPULATION_SIZE):
            # Select parents
            parent_indexes = select_parent_indexes(i, neighborhoods, NEIGHBORHOOD_SELECTION_PROB)
            parent1, parent2 = population[parent_indexes[0]], population[parent_indexes[1]]
            
            # Generate offspring
            offspring = crossover(parent1, parent2)
            offspring = mutate(offspring, slots, spaces_dict)
            
            # Evaluate offspring
            offspring_fitness = evaluate_hard_constraints(offspring, activities_dict, groups_dict, spaces_dict)
            
            # Update ideal point
            for j in range(NUM_OBJECTIVES):
                ideal_point[j] = min(ideal_point[j], offspring_fitness[j])
            
            # Update neighborhood solutions
            for neighbor_idx in neighborhoods[i]:
                weight = weight_vectors[neighbor_idx]
                
                # Calculate Tchebycheff distances
                neighbor_dist = tchebycheff_distance(fitness_values[neighbor_idx], weight, ideal_point)
                offspring_dist = tchebycheff_distance(offspring_fitness, weight, ideal_point)
                
                # Replace if offspring is better
                if offspring_dist <= neighbor_dist:
                    population[neighbor_idx] = offspring
                    fitness_values[neighbor_idx] = offspring_fitness
        
        # Print progress
        if generation % 10 == 0:
            print(f"Generation {generation}: Population size {len(population)}")
    
    return population, fitness_values

def find_best_solution(final_population, fitness_values):
    """Find the best solution in the final population based on the sum of constraint violations."""
    best_solution = None
    min_violations = float('inf')
    
    for idx, fitness in enumerate(fitness_values):
        total_violations = sum(fitness[1:])  # Exclude vacant rooms count
        
        if total_violations < min_violations:
            min_violations = total_violations
            best_solution = final_population[idx]
    
    return best_solution

def run_moead_optimization(data_file):
    """
    Main function to run the MOEA/D optimization for timetable scheduling.
    
    Args:
        data_file: Path to the JSON data file
    
    Returns:
        Best timetable solution
    """
    # Load data
    spaces_dict, groups_dict, lecturers_dict, activities_dict, slots = load_data(data_file)
    
    print("Running MOEA/D optimization...")
    final_population, fitness_values = moead(activities_dict, groups_dict, spaces_dict, slots)
    
    print("Finding best solution...")
    best_timetable = find_best_solution(final_population, fitness_values)
    
    print("Evaluating best solution:")
    evaluate(best_timetable, groups_dict, lecturers_dict, activities_dict, spaces_dict, slots)
    
    return best_timetable

if __name__ == "__main__":
    # Run the MOEA/D optimization
    best_timetable = run_moead_optimization("sliit_computing_dataset.json")
    
    # You can save the best timetable to a file if needed
    print("MOEA/D optimization completed.")
