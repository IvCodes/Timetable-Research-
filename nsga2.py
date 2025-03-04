"""
NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation for timetable scheduling.

This algorithm uses the following key components:
1. Non-dominated sorting for ranking solutions
2. Crowding distance for diversity preservation
3. Tournament selection based on rank and crowding distance
4. Standard crossover and mutation operators

The algorithm optimizes for multiple objectives simultaneously:
- Room overbooking
- Slot conflicts
- Professor conflicts
- Student group conflicts
- Unassigned activities
"""
import random
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

def fast_nondominated_sort(fitness_values):
    """
    Perform non-dominated sorting to rank solutions based on Pareto dominance.
    
    Args:
        fitness_values: List of fitness tuples for the population
    
    Returns:
        List of fronts, where each front is a list of indices to the original population
    """
    n = len(fitness_values)
    domination_count = [0] * n  # Number of solutions that dominate solution i
    dominated_solutions = [[] for _ in range(n)]  # List of solutions that solution i dominates
    
    # First front
    first_front = []
    
    # Calculate domination for each solution
    for i in range(n):
        for j in range(n):
            if i != j:
                if dominates(fitness_values[i], fitness_values[j]):
                    # i dominates j
                    dominated_solutions[i].append(j)
                elif dominates(fitness_values[j], fitness_values[i]):
                    # j dominates i
                    domination_count[i] += 1
        
        # If no solution dominates i, it belongs to the first front
        if domination_count[i] == 0:
            first_front.append(i)
    
    # Initialize the set of fronts with the first front
    fronts = [first_front]
    
    # Generate subsequent fronts
    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front += 1
        if next_front:
            fronts.append(next_front)
    
    return fronts

def calculate_crowding_distance(front, fitness_values):
    """
    Calculate crowding distance for solutions in a front to maintain diversity.
    
    Args:
        front: List of indices representing a Pareto front
        fitness_values: List of fitness tuples for the population
    
    Returns:
        List of crowding distances for each solution in the front
    """
    if len(front) <= 2:
        return [float('inf')] * len(front)
    
    n = len(front)
    distances = [0] * n
    
    # Calculate crowding distance for each objective
    for obj in range(NUM_OBJECTIVES):
        # Sort front by objective value
        sorted_front = sorted(front, key=lambda x: fitness_values[x][obj])
        
        # Set infinite distance to boundary points
        distances[sorted_front.index(sorted_front[0])] = float('inf')
        distances[sorted_front.index(sorted_front[-1])] = float('inf')
        
        # Calculate distance for intermediate points
        if fitness_values[sorted_front[-1]][obj] != fitness_values[sorted_front[0]][obj]:
            norm = fitness_values[sorted_front[-1]][obj] - fitness_values[sorted_front[0]][obj]
            for i in range(1, n - 1):
                distances[sorted_front.index(sorted_front[i])] += (
                    (fitness_values[sorted_front[i + 1]][obj] - fitness_values[sorted_front[i - 1]][obj]) / norm
                )
    
    return distances

def select_parents(population, fitness_values, fronts, crowding_distances):
    """
    Select parents for crossover using tournament selection based on rank and crowding distance.
    
    Args:
        population: List of timetables
        fitness_values: List of fitness tuples
        fronts: List of Pareto fronts
        crowding_distances: Dictionary mapping indices to crowding distances
    
    Returns:
        Two selected parent timetables
    """
    def tournament_selection():
        # Randomly select two candidates for tournament
        candidates = random.sample(range(len(population)), 2)
        
        # Find the front each candidate belongs to
        candidate_fronts = []
        for candidate in candidates:
            for i, front in enumerate(fronts):
                if candidate in front:
                    candidate_fronts.append((candidate, i, crowding_distances.get(candidate, 0)))
                    break
        
        # Sort candidates by front rank (lower is better)
        candidate_fronts.sort(key=lambda x: x[1])
        
        # If candidates are in the same front, select based on crowding distance
        if candidate_fronts[0][1] == candidate_fronts[1][1]:
            if candidate_fronts[0][2] > candidate_fronts[1][2]:
                return population[candidate_fronts[0][0]]
            else:
                return population[candidate_fronts[1][0]]
        
        # Otherwise, select the candidate from the better front
        return population[candidate_fronts[0][0]]
    
    parent1 = tournament_selection()
    parent2 = tournament_selection()
    
    return parent1, parent2

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

def nsga2(activities_dict, groups_dict, spaces_dict, slots, generations=NUM_GENERATIONS):
    """
    Run the NSGA-II algorithm to optimize timetable scheduling.
    
    Args:
        activities_dict, groups_dict, spaces_dict: Data dictionaries
        slots: List of available time slots
        generations: Number of generations to run
    
    Returns:
        Final population of optimized timetables
    """
    # Initialize population
    population = generate_initial_population(slots, spaces_dict, activities_dict)
    
    for generation in range(generations):
        # Evaluate the current population
        fitness_values = evaluate_population(population, activities_dict, groups_dict, spaces_dict)
        
        # Create combined population (parents + offspring)
        combined_population = population.copy()
        combined_fitness = fitness_values.copy()
        
        # Non-dominated sorting
        fronts = fast_nondominated_sort(combined_fitness)
        
        # Calculate crowding distance for each front
        crowding_distances = {}
        for front in fronts:
            distances = calculate_crowding_distance(front, combined_fitness)
            for i, idx in enumerate(front):
                crowding_distances[idx] = distances[i]
        
        # Create new population
        new_population = []
        
        # Add solutions from each front until population is filled
        front_index = 0
        while len(new_population) + len(fronts[front_index]) <= POPULATION_SIZE:
            for idx in fronts[front_index]:
                new_population.append(combined_population[idx])
            front_index += 1
            if front_index >= len(fronts):
                break
        
        # If needed, fill the rest of the population with solutions from the current front
        # sorted by crowding distance
        if len(new_population) < POPULATION_SIZE and front_index < len(fronts):
            last_front = fronts[front_index]
            sorted_last_front = sorted(last_front, key=lambda x: crowding_distances.get(x, 0), reverse=True)
            remaining = POPULATION_SIZE - len(new_population)
            for idx in sorted_last_front[:remaining]:
                new_population.append(combined_population[idx])
        
        # Generate offspring for the next generation
        offspring = []
        while len(offspring) < POPULATION_SIZE:
            parent1, parent2 = select_parents(new_population, fitness_values, fronts, crowding_distances)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, slots, spaces_dict)
            child2 = mutate(child2, slots, spaces_dict)
            offspring.extend([child1, child2])
        
        # Truncate offspring to maintain population size
        population = offspring[:POPULATION_SIZE]
        
        # Print progress
        if generation % 10 == 0:
            print(f"Generation {generation}: Population size {len(population)}")
    
    return population

def find_best_solution(final_population, activities_dict, groups_dict, spaces_dict):
    """Find the best solution in the final population based on the sum of constraint violations."""
    best_solution = None
    min_violations = float('inf')
    
    for timetable in final_population:
        result = evaluate_hard_constraints(timetable, activities_dict, groups_dict, spaces_dict)
        total_violations = sum(result[1:])  # Exclude vacant rooms count
        
        if total_violations < min_violations:
            min_violations = total_violations
            best_solution = timetable
    
    return best_solution

def run_nsga2_optimization(data_file):
    """
    Main function to run the NSGA-II optimization for timetable scheduling.
    
    Args:
        data_file: Path to the JSON data file
    
    Returns:
        Best timetable solution
    """
    # Load data
    spaces_dict, groups_dict, lecturers_dict, activities_dict, slots = load_data(data_file)
    
    print("Running NSGA-II optimization...")
    final_population = nsga2(activities_dict, groups_dict, spaces_dict, slots)
    
    print("Finding best solution...")
    best_timetable = find_best_solution(final_population, activities_dict, groups_dict, spaces_dict)
    
    print("Evaluating best solution:")
    evaluate(best_timetable, groups_dict, lecturers_dict, activities_dict, spaces_dict, slots)
    
    return best_timetable

if __name__ == "__main__":
    # Run the NSGA-II optimization
    best_timetable = run_nsga2_optimization("sliit_computing_dataset.json")
    
    # You can save the best timetable to a file if needed
    print("NSGA-II optimization completed.")
