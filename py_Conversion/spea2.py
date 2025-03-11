"""
SPEA2 (Strength Pareto Evolutionary Algorithm 2) implementation for timetable scheduling.

This algorithm uses the following key components:
1. Fine-grained fitness assignment strategy based on dominated and dominating solutions
2. Nearest neighbor density estimation for breaking ties between individuals with equal fitness
3. Environmental selection method that preserves boundary solutions
4. Archive of non-dominated solutions

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
ARCHIVE_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
NUM_OBJECTIVES = 5  # Number of objectives in the optimization

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

def calculate_strength(fitness_values):
    """
    Calculate strength for each individual, which is the number of solutions it dominates.
    
    Args:
        fitness_values: List of fitness tuples for the population
    
    Returns:
        List of strength values
    """
    n = len(fitness_values)
    strength = [0] * n
    
    for i in range(n):
        for j in range(n):
            if i != j and dominates(fitness_values[i], fitness_values[j]):
                strength[i] += 1
    
    return strength

def calculate_raw_fitness(fitness_values, strength):
    """
    Calculate raw fitness for each individual, which is the sum of the strengths of its dominators.
    
    Args:
        fitness_values: List of fitness tuples for the population
        strength: List of strength values
    
    Returns:
        List of raw fitness values (lower is better)
    """
    n = len(fitness_values)
    raw_fitness = [0] * n
    
    for i in range(n):
        for j in range(n):
            if i != j and dominates(fitness_values[j], fitness_values[i]):
                raw_fitness[i] += strength[j]
    
    return raw_fitness

def calculate_distances(fitness_values):
    """
    Calculate distances between individuals in objective space.
    
    Args:
        fitness_values: List of fitness tuples for the population
    
    Returns:
        2D array of distances
    """
    n = len(fitness_values)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(sum((fitness_values[i][k] - fitness_values[j][k])**2 for k in range(NUM_OBJECTIVES)))
            distances[i, j] = distances[j, i] = dist
    
    return distances

def calculate_density(distances, k=10):
    """
    Calculate density for each individual using k-nearest neighbors.
    
    Args:
        distances: 2D array of distances between individuals
        k: Number of nearest neighbors to consider
    
    Returns:
        List of density values (lower is better)
    """
    n = len(distances)
    density = [0] * n
    
    for i in range(n):
        # Sort distances to the i-th individual
        kth_distance = sorted(distances[i])[min(k, n-1)]
        # Density is defined as 1 / (kth_distance + 2) to avoid division by zero
        density[i] = 1 / (kth_distance + 2)
    
    return density

def calculate_fitness(raw_fitness, density):
    """
    Calculate the final fitness by adding raw fitness and density.
    
    Args:
        raw_fitness: List of raw fitness values
        density: List of density values
    
    Returns:
        List of final fitness values (lower is better)
    """
    return [raw_fitness[i] + density[i] for i in range(len(raw_fitness))]

def environmental_selection(combined_population, combined_fitness, combined_fitness_values, archive_size):
    """
    Perform environmental selection to create the archive for the next generation.
    
    Args:
        combined_population: Combined list of current population and archive
        combined_fitness: Combined list of fitness values for current population and archive
        combined_fitness_values: Combined list of objective values
        archive_size: Maximum size of the archive
    
    Returns:
        Selected archive of non-dominated solutions
    """
    n = len(combined_population)
    # Sort by fitness (lower is better)
    sorted_indices = np.argsort(combined_fitness)
    
    # Count number of non-dominated solutions (fitness < 1.0)
    non_dominated_count = 0
    for idx in sorted_indices:
        if combined_fitness[idx] < 1.0:
            non_dominated_count += 1
        else:
            break
    
    if non_dominated_count <= archive_size:
        # If we have fewer non-dominated solutions than the archive size,
        # fill the rest with dominated solutions
        archive_indices = sorted_indices[:archive_size]
    else:
        # If we have more non-dominated solutions than the archive size,
        # use truncation operator to remove similar solutions
        archive_indices = []
        for idx in sorted_indices[:non_dominated_count]:
            archive_indices.append(idx)
        
        # Calculate distances between non-dominated solutions
        non_dominated_fitness_values = [combined_fitness_values[i] for i in archive_indices]
        distances = calculate_distances(non_dominated_fitness_values)
        
        # Truncate archive
        while len(archive_indices) > archive_size:
            # For each solution, find its nearest neighbor
            min_distances = np.inf * np.ones(len(archive_indices))
            for i in range(len(archive_indices)):
                for j in range(len(archive_indices)):
                    if i != j and distances[i, j] < min_distances[i]:
                        min_distances[i] = distances[i, j]
            
            # Remove the solution with the smallest distance to its nearest neighbor
            idx_to_remove = np.argmin(min_distances)
            archive_indices.pop(idx_to_remove)
            
            # Update distances
            distances = np.delete(distances, idx_to_remove, 0)
            distances = np.delete(distances, idx_to_remove, 1)
    
    # Create archive from selected indices
    archive = [combined_population[i] for i in archive_indices]
    return archive

def tournament_selection(population, fitness, tournament_size=2):
    """
    Perform tournament selection to select parents for crossover.
    
    Args:
        population: List of timetables
        fitness: List of fitness values (lower is better)
        tournament_size: Number of candidates to consider in each tournament
    
    Returns:
        Selected parent timetable
    """
    # Select random candidates for tournament
    candidates = random.sample(range(len(population)), tournament_size)
    
    # Find the candidate with the lowest fitness (best)
    best_candidate = min(candidates, key=lambda x: fitness[x])
    
    return population[best_candidate]

def crossover(parent1, parent2):
    """
    Perform crossover by swapping time slots between two parents.
    
    Args:
        parent1, parent2: Two parent timetable dictionaries
    
    Returns:
        Two offspring timetable dictionaries
    """
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()
    
    child1, child2 = parent1.copy(), parent2.copy()
    slots = list(parent1.keys())
    
    # One-point crossover
    split = random.randint(0, len(slots) - 1)
    
    for i in range(split, len(slots)):
        child1[slots[i]], child2[slots[i]] = parent2[slots[i]], parent1[slots[i]]
    
    return child1, child2

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

def spea2(activities_dict, groups_dict, spaces_dict, slots, generations=NUM_GENERATIONS):
    """
    Run the SPEA2 algorithm to optimize timetable scheduling.
    
    Args:
        activities_dict, groups_dict, spaces_dict: Data dictionaries
        slots: List of available time slots
        generations: Number of generations to run
    
    Returns:
        Archive of non-dominated solutions
    """
    # Initialize population and empty archive
    population = generate_initial_population(slots, spaces_dict, activities_dict)
    archive = []
    
    for generation in range(generations):
        # Evaluate the current population
        population_fitness_values = evaluate_population(population, activities_dict, groups_dict, spaces_dict)
        archive_fitness_values = evaluate_population(archive, activities_dict, groups_dict, spaces_dict) if archive else []
        
        # Combine population and archive
        combined_population = population + archive
        combined_fitness_values = population_fitness_values + archive_fitness_values
        
        # Calculate strength
        strength = calculate_strength(combined_fitness_values)
        
        # Calculate raw fitness
        raw_fitness = calculate_raw_fitness(combined_fitness_values, strength)
        
        # Calculate distances between solutions
        distances = calculate_distances(combined_fitness_values)
        
        # Calculate density
        density = calculate_density(distances)
        
        # Calculate final fitness
        fitness = calculate_fitness(raw_fitness, density)
        
        # Environmental selection
        archive = environmental_selection(combined_population, fitness, combined_fitness_values, ARCHIVE_SIZE)
        
        # Generate offspring
        offspring = []
        while len(offspring) < POPULATION_SIZE:
            # Select parents
            parent1 = tournament_selection(archive, fitness[:len(archive)])
            parent2 = tournament_selection(archive, fitness[:len(archive)])
            
            # Perform crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Perform mutation
            child1 = mutate(child1, slots, spaces_dict)
            child2 = mutate(child2, slots, spaces_dict)
            
            offspring.extend([child1, child2])
        
        # Truncate offspring to maintain population size
        population = offspring[:POPULATION_SIZE]
        
        # Print progress
        if generation % 10 == 0:
            print(f"Generation {generation}: Archive size {len(archive)}")
    
    return archive

def find_best_solution(archive, activities_dict, groups_dict, spaces_dict):
    """Find the best solution in the archive based on the sum of constraint violations."""
    best_solution = None
    min_violations = float('inf')
    
    for timetable in archive:
        result = evaluate_hard_constraints(timetable, activities_dict, groups_dict, spaces_dict)
        total_violations = sum(result[1:])  # Exclude vacant rooms count
        
        if total_violations < min_violations:
            min_violations = total_violations
            best_solution = timetable
    
    return best_solution

def run_spea2_optimization(data_file):
    """
    Main function to run the SPEA2 optimization for timetable scheduling.
    
    Args:
        data_file: Path to the JSON data file
    
    Returns:
        Best timetable solution
    """
    # Load data
    spaces_dict, groups_dict, lecturers_dict, activities_dict, slots = load_data(data_file)
    
    print("Running SPEA2 optimization...")
    final_archive = spea2(activities_dict, groups_dict, spaces_dict, slots)
    
    print("Finding best solution...")
    best_timetable = find_best_solution(final_archive, activities_dict, groups_dict, spaces_dict)
    
    print("Evaluating best solution:")
    evaluate(best_timetable, groups_dict, lecturers_dict, activities_dict, spaces_dict, slots)
    
    return best_timetable

if __name__ == "__main__":
    # Run the SPEA2 optimization
    best_timetable = run_spea2_optimization("sliit_computing_dataset.json")
    
    # You can save the best timetable to a file if needed
    print("SPEA2 optimization completed.")
