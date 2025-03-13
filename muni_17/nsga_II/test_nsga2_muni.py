"""
Testing NSGA-II algorithm on the munifspsspr17 dataset.

This script adapts the NSGA-II implementation to work with the munifspsspr17.json dataset structure.
It loads the data, runs the optimization, and evaluates the results.
"""
import json
import random
import time
import numpy as np
from scipy.spatial import ConvexHull
from collections import defaultdict
import pandas as pd
import os
import sys

# Add parent directory to sys.path to import modules from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from parent directory
from solution import Solution
from utils_1 import TupleKeyEncoder, save_results,print_solution_stats
from data_loader_muni import load_muni_data
from evaluate import evaluate_solution
from metrics import calculate_hypervolume, track_constraint_violations, calculate_spacing, calculate_igd, analyze_constraint_violations
from plots import plot_metrics, plot_constraint_violations, plot_pareto_size
from validate import get_valid_rooms, get_valid_times

# Algorithm parameters
POPULATION_SIZE = 10
NUM_GENERATIONS = 10
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8

# Reference point for hypervolume calculation (worst possible values)
# These will be updated dynamically based on the actual objectives
REFERENCE_POINT = [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')]


def initialize_population(data, size=POPULATION_SIZE):
    """
    Initialize a population of random solutions.
    
    Args:
        data: Dictionary with problem data
        size: Population size
    
    Returns:
        List of Solution objects
    """
    population = []
    
    for _ in range(size):
        solution = Solution()
        
        # Assign each class to a time and room
        for class_id, class_info in data['classes'].items():
            # Skip if class has no time or room options
            if not class_info['times'] or not class_info['rooms']:
                continue
            
            # Randomly select a time and room
            time_choice = random.choice(class_info['times'])
            room_choice = random.choice(class_info['rooms'])
            
            # Store the assignment
            solution.assignments[class_id] = (time_choice['time_id'], room_choice['room_id'])
        
        population.append(solution)
    
    return population

def evaluate_population(population, data):
    """
    Evaluate all solutions in the population.
    
    Args:
        population: List of Solution objects
        data: Dictionary with problem data
    
    Returns:
        List of fitness values for each solution
    """
    fitness_values = []
    
    for solution in population:
        fitness = evaluate_solution(solution, data)
        
        # Ensure fitness values are properly assigned
        if solution.fitness is None:
            solution.fitness = fitness
        elif isinstance(solution.fitness, list):
            solution.fitness = fitness
        else:
            raise ValueError("Invalid fitness value type")
        
        fitness_values.append(fitness)
    
    return fitness_values

def compute_domination_info(population):
    """
    Compute domination relationships between solutions.
    
    Args:
        population: List of Solution objects
        
    Returns:
        Tuple of (dominated_solutions, domination_counts)
    """
    dominated_solutions = {i: [] for i in range(len(population))}
    domination_counts = [0] * len(population)
    
    for i in range(len(population)):
        for j in range(len(population)):
            if i == j:
                continue
                
            if dominates(population[i], population[j]):
                # i dominates j
                dominated_solutions[i].append(j)
            elif dominates(population[j], population[i]):
                # j dominates i
                domination_counts[i] += 1
                
    return dominated_solutions, domination_counts

def find_first_front(population, domination_counts):
    """Find solutions that belong to the first front."""
    first_front = []
    for i in range(len(population)):
        if domination_counts[i] == 0:
            first_front.append(population[i])
    return first_front

def fast_nondominated_sort(population):
    """
    Perform non-dominated sorting on a population of Solution objects.

    Args:
        population: List of Solution objects

    Returns:
        List of fronts, where each front is a list of Solution objects
    """
    # Compute domination relationships
    dominated_solutions, domination_counts = compute_domination_info(population)
    
    # Initialize fronts list and add first front
    fronts = []
    first_front = find_first_front(population, domination_counts)
    fronts.append(first_front)
    
    # Find subsequent fronts
    i = 0
    while i < len(fronts):
        current_front = fronts[i]
        next_front = []
        
        # For each solution in current front
        for solution in current_front:
            # Find its index in the population
            solution_index = population.index(solution)
            
            # Update dominated solutions
            for dominated_index in dominated_solutions[solution_index]:
                domination_counts[dominated_index] -= 1
                if domination_counts[dominated_index] == 0:
                    next_front.append(population[dominated_index])
        
        # Add next front if not empty
        if next_front:
            fronts.append(next_front)
        
        i += 1
        
    return fronts

def dominates(solution1, solution2):
    """
    Check if solution1 dominates solution2.
    
    Args:
        solution1, solution2: Two Solution objects
    
    Returns:
        True if solution1 dominates solution2, False otherwise
    """
    fitness1 = solution1.fitness
    fitness2 = solution2.fitness
    
    if fitness1 is None or fitness2 is None:
        raise ValueError("Fitness values are not set")
    
    # Check if solution1 is better in at least one objective
    # and not worse in any other objective
    better_in_at_least_one = False
    worse_in_any = False
    
    for i in range(len(fitness1)):
        if fitness1[i] < fitness2[i]:
            better_in_at_least_one = True
        elif fitness1[i] > fitness2[i]:
            worse_in_any = True
    
    return better_in_at_least_one and not worse_in_any

def crowding_distance_assignment(front):
    """
    Assign crowding distance to solutions in a front.
    
    Args:
        front: List of Solution objects in the front
    """
    if not front:
        return
    
    n = len(front)
    
    # Initialize crowding distance for all solutions
    for solution in front:
        solution.crowding_distance = 0
    
    # Calculate crowding distance for each objective
    num_objectives = len(front[0].fitness)
    
    for obj_index in range(num_objectives):
        # Sort by current objective
        front.sort(key=lambda x: x.fitness[obj_index])
        
        # Set boundary points to infinity
        front[0].crowding_distance = float('inf')
        front[n-1].crowding_distance = float('inf')
        
        # Calculate crowding distance for middle points
        if n > 2:
            f_min = front[0].fitness[obj_index]
            f_max = front[n-1].fitness[obj_index]
            
            # Avoid division by zero
            if f_max == f_min:
                continue
            
            # Calculate crowding distance
            for i in range(1, n-1):
                front[i].crowding_distance += (front[i+1].fitness[obj_index] - front[i-1].fitness[obj_index]) / (f_max - f_min)

def adjust_mutation_rate(solution, base_rate=0.1):
    """Adjust mutation rate based on solution generation."""
    if hasattr(solution, 'generation'):
        generation_factor = min(0.5, solution.generation / 20.0)
        return base_rate + generation_factor
    return base_rate

def mutate_existing_assignment(solution, class_id, data, mutation_strength):
    """Mutate a specific class assignment."""
    if random.random() < mutation_strength * 0.3:  # 30% chance to remove
        del solution.assignments[class_id]
        return
        
    # Reassign to a different room or time
    valid_rooms = get_valid_rooms(data, class_id)
    if not valid_rooms:
        return
        
    room_id = random.choice(valid_rooms)
    
    # Copy the solution temporarily to avoid conflicts with current class
    temp_solution = solution.copy()
    if class_id in temp_solution.assignments:
        del temp_solution.assignments[class_id]
    
    valid_times = get_valid_times(data, class_id, room_id, temp_solution)
    if valid_times:
        solution.assignments[class_id] = (room_id, random.choice(valid_times))

def add_new_assignment(solution, class_id, data):
    """Add a new assignment for an unassigned class."""
    valid_rooms = get_valid_rooms(data, class_id)
    if not valid_rooms:
        return
        
    room_id = random.choice(valid_rooms)
    valid_times = get_valid_times(data, class_id, room_id, solution)
    
    if valid_times:
        solution.assignments[class_id] = (room_id, random.choice(valid_times))

def mutate(solution, data, mutation_rate=0.1, mutation_strength=0.2):
    """
    Mutate a solution by reassigning classes or removing assignments.
    
    Args:
        solution: Solution object to mutate
        data: Dictionary with problem data
        mutation_rate: Probability of a class being selected for mutation
        mutation_strength: Controls how aggressively to mutate
    
    Returns:
        None (modifies solution in-place)
    """
    # Adjust mutation rate dynamically
    mutation_rate = adjust_mutation_rate(solution, mutation_rate)
    
    # Get list of all class IDs
    class_ids = list(data['classes'].keys())
    
    # Mutate existing assignments
    for class_id in list(solution.assignments.keys()):
        if random.random() < mutation_rate:
            mutate_existing_assignment(solution, class_id, data, mutation_strength)
    
    # Add new assignments for unassigned classes
    unassigned_classes = [c_id for c_id in class_ids if c_id not in solution.assignments]
    num_to_add = int(len(unassigned_classes) * mutation_rate * mutation_strength)
    
    for class_id in random.sample(unassigned_classes, min(num_to_add, len(unassigned_classes))):
        add_new_assignment(solution, class_id, data)

def create_offspring(population, generation, data, population_size, crossover_rate, mutation_rate, metrics):
    """Create offspring through selection, crossover, and mutation."""
    offspring = []
    
    # Adaptive tournament size based on generation progress
    base_tournament_size = 2
    max_tournament_size = 4
    max_generations = 100  # Using NUM_GENERATIONS default
    tournament_size = base_tournament_size + int((max_tournament_size - base_tournament_size) * 
                                                (generation / max_generations))
    
    # Determine adaptive crossover rate based on population diversity
    adaptive_crossover_rate = get_adaptive_crossover_rate(crossover_rate, metrics)
    
    for _ in range(0, population_size, 2):
        # Select parents using tournament selection
        parent1 = tournament_selection(population, tournament_size)
        parent2 = tournament_selection(population, tournament_size)
        
        # Create children
        child1, child2 = create_children(parent1, parent2, data, adaptive_crossover_rate, 
                                         generation, mutation_rate)
        
        # Add children to offspring
        offspring.append(child1)
        offspring.append(child2)
    
    # Ensure we have exactly population_size offspring
    return offspring[:population_size]

def get_adaptive_crossover_rate(base_rate, metrics):
    """Calculate adaptive crossover rate based on population diversity."""
    if len(metrics['spacing']) > 0:
        current_spacing = metrics['spacing'][-1]
        if current_spacing < 10:  # Low diversity
            return min(base_rate + 0.2, 0.95)  # Increase crossover rate
    return base_rate

def create_children(parent1, parent2, data, crossover_rate, generation, mutation_rate):
    """Create and mutate children from parents."""
    # Perform crossover with adaptive rate
    if random.random() < crossover_rate:
        child1, child2 = crossover(parent1, parent2, data)
    else:
        child1, child2 = parent1.copy(), parent2.copy()
    
    # Set generation number for dynamic mutation rate
    child1.generation = generation
    child2.generation = generation
    
    # Perform mutation with adaptive rate
    if random.random() < mutation_rate:
        mutate(child1, data, mutation_rate=mutation_rate, mutation_strength=0.2)
    if random.random() < mutation_rate:
        mutate(child2, data, mutation_rate=mutation_rate, mutation_strength=0.2)
    
    return child1, child2

def select_next_generation(combined_fronts, population_size):
    """Select solutions for the next generation based on fronts and crowding distance."""
    next_population = []
    front_index = 0
    
    # Add complete fronts until we reach population_size
    while front_index < len(combined_fronts) and len(next_population) + len(combined_fronts[front_index]) <= population_size:
        # Add all solutions from this front
        next_population.extend(combined_fronts[front_index])
        front_index += 1
    
    # If we need more solutions, add from the next front based on crowding distance
    if len(next_population) < population_size and front_index < len(combined_fronts):
        # Calculate crowding distance for the current front
        crowding_distance_assignment(combined_fronts[front_index])
        
        # Sort by crowding distance (descending)
        current_front = sorted(combined_fronts[front_index], 
                              key=lambda x: float('-inf') if x.crowding_distance is None else x.crowding_distance, 
                              reverse=True)
        
        # Add solutions until we reach population_size
        remaining_slots = population_size - len(next_population)
        next_population.extend(current_front[:remaining_slots])
    
    return next_population

def handle_diversity_preservation(next_population, metrics, data, population_size, verbose=False):
    """Inject random solutions if stagnation is detected."""
    if len(metrics['hypervolume']) > 5:
        # Check for stagnation in hypervolume
        last_5_hv = metrics['hypervolume'][-5:]
        hv_improvement = (last_5_hv[-1] - last_5_hv[0]) / max(abs(last_5_hv[0]), 1e-10)
        
        if hv_improvement < 0.01:  # Less than 1% improvement in 5 generations
            num_random = int(population_size * 0.1)  # Replace 10% of population
            random_solutions = initialize_population(data, num_random)
            evaluate_population(random_solutions, data)
            
            # Replace worst solutions in the population
            next_population.sort(key=lambda x: sum(x.fitness))  
            next_population = next_population[:-num_random] + random_solutions
            
            if verbose:
                print(f"Injecting {num_random} random solutions to prevent stagnation")
    
    return next_population

def update_metrics(fronts, population, metrics, reference_point, best_solution, data, verbose=False):
    """Update all metrics for the current generation."""
    if not fronts or len(fronts[0]) == 0:
        return best_solution, metrics
    
    # Calculate metrics for Pareto front
    pareto_front_fitness = [s.fitness for s in fronts[0]]
    
    # Update metrics
    metrics['hypervolume'].append(calculate_hypervolume(pareto_front_fitness, reference_point))
    metrics['spacing'].append(calculate_spacing(pareto_front_fitness))
    metrics['pareto_front_size'].append(len(fronts[0]))
    
    # Calculate IGD if we have a reference front
    if 'reference_front' in data:
        igd_value = calculate_igd(pareto_front_fitness, data['reference_front'])
        metrics['igd'].append(igd_value)
    
    # Analyze constraint violations
    violations = analyze_constraint_violations(population, data)
    
    # Add the total_counts field that's expected by plot_metrics
    violations['total_counts'] = {
        'room_conflicts': violations['room_conflicts']['total'] / len(population),
        'time_conflicts': violations['time_conflicts']['total'] / len(population),
        'distribution_conflicts': violations['distribution_conflicts']['total'] / len(population),
        'student_conflicts': violations['student_conflicts']['total'] / len(population),
        'capacity_violations': violations['capacity_violations']['total'] / len(population),
        'total_weighted_score': violations['total']['total'] / len(population)
    }
    
    metrics['constraint_violations'].append(violations)
    
    # Update best solution
    current_best = find_best_solution(population)
    if best_solution is None or dominates(current_best, best_solution):
        best_solution = current_best.copy()
    
    # Track fitness values
    metrics['best_fitness'].append(min([sum(s.fitness) for s in population]))
    metrics['average_fitness'].append(sum([sum(s.fitness) for s in population]) / len(population))
    
    if verbose:
        print("\nGeneration Metrics:")
        print("  Best fitness: {:.2f}".format(metrics['best_fitness'][-1]))
        print("  Average fitness: {:.2f}".format(metrics['average_fitness'][-1]))
        print("  Pareto front size: {}".format(metrics['pareto_front_size'][-1]))
        print("  Hypervolume: {:.2f}".format(metrics['hypervolume'][-1]))
        if 'igd' in metrics and metrics['igd']:
            print("  IGD: {:.4f}".format(metrics['igd'][-1]))
        print("\nConstraint Violations:")
        print("  Room conflicts: {:.2f}".format(violations['total_counts']['room_conflicts']))
        print("  Time conflicts: {:.2f}".format(violations['total_counts']['time_conflicts']))
        print("  Distribution conflicts: {:.2f}".format(violations['total_counts']['distribution_conflicts']))
        print("  Student conflicts: {:.2f}".format(violations['total_counts']['student_conflicts']))
        print("  Capacity violations: {:.2f}".format(violations['total_counts']['capacity_violations']))
        print("  Total weighted score: {:.2f}".format(violations['total_counts']['total_weighted_score']))
    
    return best_solution, metrics

def calculate_mixing_rate(parent1, parent2):
    """Calculate mixing rate based on similarity between parents."""
    all_classes = set(parent1.assignments.keys()) | set(parent2.assignments.keys())
    common_assignments = set(parent1.assignments.keys()) & set(parent2.assignments.keys())
    similarity = len(common_assignments) / len(all_classes) if all_classes else 0
    
    if similarity > 0.8:
        return 0.7  # High similarity - higher mixing rate
    elif similarity > 0.5:
        return 0.5  # Moderate similarity - standard mixing
    return 0.3  # Low similarity - conservative mixing

def assign_class_to_children(class_id, parent1, parent2, child1, child2, mixing_rate):
    """Assign a class to children based on parents' assignments."""
    in_parent1 = class_id in parent1.assignments
    in_parent2 = class_id in parent2.assignments
    
    # Both parents have this class assigned
    if in_parent1 and in_parent2:
        if random.random() < mixing_rate:
            child1.assignments[class_id] = parent1.assignments[class_id]
            child2.assignments[class_id] = parent2.assignments[class_id]
        else:
            child1.assignments[class_id] = parent2.assignments[class_id]
            child2.assignments[class_id] = parent1.assignments[class_id]
    # Only parent1 has this class assigned
    elif in_parent1:
        if random.random() < 0.5:
            child1.assignments[class_id] = parent1.assignments[class_id]
        else:
            child2.assignments[class_id] = parent1.assignments[class_id]
    # Only parent2 has this class assigned
    elif in_parent2:
        if random.random() < 0.5:
            child1.assignments[class_id] = parent2.assignments[class_id]
        else:
            child2.assignments[class_id] = parent2.assignments[class_id]

def crossover(parent1, parent2, data):
    """
    Perform crossover between two parent solutions to create two offspring.
    
    Args:
        parent1: First parent Solution
        parent2: Second parent Solution
        data: Dictionary with problem data
    
    Returns:
        (child1, child2) - Two new Solution objects
    """
    child1 = Solution()
    child2 = Solution()
    
    # Get all class IDs that are assigned in either parent
    all_classes = set(parent1.assignments.keys()) | set(parent2.assignments.keys())
    
    # Determine mixing rate based on parents' similarity
    mixing_rate = calculate_mixing_rate(parent1, parent2)
    
    # Perform crossover for each class
    for class_id in all_classes:
        assign_class_to_children(class_id, parent1, parent2, child1, child2, mixing_rate)
    
    # Schedule repair to handle conflicts
    repair_schedule(child1, data)
    repair_schedule(child2, data)
    
    return child1, child2

def detect_conflicts(solution):
    """
    Detect conflicts in a solution's schedule.
    
    Args:
        solution: Solution to check for conflicts
    
    Returns:
        tuple of (room_time_assignments, conflicts)
    """
    room_time_assignments = {}  # (room_id, time_id) -> class_id
    conflicts = set()
    
    for class_id, assignment in solution.assignments.items():
        # Skip invalid assignments
        if not assignment or len(assignment) < 2:
            continue
            
        # Unpack assignment safely
        try:
            room_id, time_id = assignment
        except (ValueError, TypeError):
            continue
        
        if not room_id or not time_id:
            continue
            
        # Check for room conflicts
        key = (room_id, time_id)
        if key in room_time_assignments:
            # Room conflict found
            conflicting_class = room_time_assignments[key]
            conflicts.add(class_id)
            conflicts.add(conflicting_class)
        else:
            room_time_assignments[key] = class_id
    
    return room_time_assignments, conflicts

def try_reassign_class(class_id, solution, data):
    """
    Try to reassign a class to a different time/room.
    
    Args:
        class_id: ID of the class to reassign
        solution: Solution object
        data: Dictionary with problem data
        
    Returns:
        bool: True if reassignment was successful, False otherwise
    """
    # Temporarily remove this class to avoid detecting conflicts with itself
    temp_solution = solution.copy()
    if class_id in temp_solution.assignments:
        del temp_solution.assignments[class_id]
    
    # Get valid alternatives
    valid_rooms = get_valid_rooms(data, class_id)
    
    # Try to find a valid assignment
    for room_id in valid_rooms:
        valid_times = get_valid_times(data, class_id, room_id, temp_solution)
        if valid_times:
            # Found a valid alternative
            solution.assignments[class_id] = (room_id, random.choice(valid_times))
            return True
    
    return False

def repair_schedule(solution, data):
    """
    Repair a schedule by resolving conflicts.
    
    Args:
        solution: Solution to repair
        data: Dictionary with problem data
    
    Returns:
        None (modifies solution in-place)
    """
    # Detect conflicts
    _, conflicts = detect_conflicts(solution)
    
    # Resolve conflicts
    for class_id in conflicts:
        if class_id not in solution.assignments:
            continue
            
        # Try to reassign or remove
        if random.random() < 0.5:
            success = try_reassign_class(class_id, solution, data)
            if not success and class_id in solution.assignments:
                del solution.assignments[class_id]
        else:
            # Remove this assignment
            del solution.assignments[class_id]

def find_best_solution(population):
    """
    Find the best solution in the population based on total violations.
    
    Args:
        population: List of Solution objects
    
    Returns:
        Best Solution object
    """
    return min(population, key=lambda sol: sol.fitness[0])


    """
    Calculate the spacing metric for a Pareto front.
    
    The spacing metric measures how evenly the solutions are distributed along the front.
    Lower values indicate more uniform spacing.
    
    Args:
        front: List of fitness values for solutions in the Pareto front
        
    Returns:
        Spacing metric value
    """
    if not front or len(front) < 2:
        return 0.0
    
    # Convert to numpy array
    points = np.array(front)
    n = len(points)
    
    # Calculate distances between consecutive points
    distances = []
    
    for i in range(n):
        # Find minimum distance to any other point
        min_dist = float('inf')
        for j in range(n):
            if i != j:
                # Euclidean distance
                dist = np.sqrt(np.sum((points[i] - points[j])**2))
                min_dist = min(min_dist, dist)
        distances.append(min_dist)
    
    # Calculate mean distance
    mean_dist = np.mean(distances)
    
    # Calculate standard deviation of distances
    spacing = np.sqrt(np.sum((distances - mean_dist)**2) / (n - 1))
    
    return spacing

def calculate_crowding_distance(front, fitness_values):
    """
    Calculate crowding distance for solutions in a front.
    
    Args:
        front: List of Solution objects in the front
        fitness_values: List of fitness value lists for each solution
        
    Returns:
        None (updates crowding_distance attribute of each solution)
    """
    if len(front) <= 2:
        # If the front has only 1 or 2 solutions, set their crowding distances to infinity
        for solution in front:
            solution.crowding_distance = float('inf')
        return
    
    # Initialize crowding distances to zero
    for solution in front:
        solution.crowding_distance = 0
    
    # Number of objectives
    num_objectives = len(fitness_values[0])
    
    # For each objective
    for objective_index in range(num_objectives):
        # Sort solutions by the current objective
        sorted_indices = sorted(range(len(front)), key=lambda i: fitness_values[i][objective_index])
        
        # Set boundary points to infinity
        front[sorted_indices[0]].crowding_distance = float('inf')
        front[sorted_indices[-1]].crowding_distance = float('inf')
        
        # Calculate crowding distance for middle points
        objective_min = fitness_values[sorted_indices[0]][objective_index]
        objective_max = fitness_values[sorted_indices[-1]][objective_index]
        
        # If min and max are the same, skip this objective
        if objective_max == objective_min:
            continue
        
        # Calculate normalized distances
        for i in range(1, len(front) - 1):
            front[sorted_indices[i]].crowding_distance += (
                (fitness_values[sorted_indices[i+1]][objective_index] - 
                 fitness_values[sorted_indices[i-1]][objective_index]) / 
                (objective_max - objective_min)
            )

def tournament_selection(population, tournament_size=2):
    """
    Select a solution using tournament selection.
    
    Args:
        population: List of Solution objects
        tournament_size: Number of solutions to compare in the tournament
        
    Returns:
        Selected Solution object
    """
    # Randomly select tournament_size solutions
    candidates = random.sample(population, min(tournament_size, len(population)))
    
    # Find the best solution based on rank and crowding distance
    best_candidate = candidates[0]
    
    for candidate in candidates[1:]:
        # If either solution has no fitness, skip comparison
        if best_candidate.fitness is None or candidate.fitness is None:
            continue
            
        # Compare based on non-domination rank first (assuming first fitness value is total violations)
        # Lower value is better for minimization problem
        if candidate.fitness[0] < best_candidate.fitness[0]:
            best_candidate = candidate
        elif candidate.fitness[0] == best_candidate.fitness[0]:
            # If same rank, choose the one with larger crowding distance
            if hasattr(candidate, 'crowding_distance') and hasattr(best_candidate, 'crowding_distance'):
                if candidate.crowding_distance > best_candidate.crowding_distance:
                    best_candidate = candidate
    
    return best_candidate

def nsga2_muni(data, population_size=POPULATION_SIZE, max_generations=NUM_GENERATIONS, crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE, verbose=False):
    """
    Run NSGA-II algorithm on the MUni dataset.
    
    Args:
        data: Dictionary with problem data
        population_size: Size of the population
        max_generations: Maximum number of generations
        crossover_rate: Probability of crossover
        mutation_rate: Probability of mutation
        verbose: If True, print progress information
    
    Returns:
        (best_solution, fronts, metrics)
    """
    # Initialize population and evaluate
    population = initialize_population(data, population_size)
    evaluate_population(population, data)
    fronts = fast_nondominated_sort(population)
    
    # Initialize metrics tracking
    metrics = {
        'hypervolume': [],
        'spacing': [],
        'igd': [],
        'best_fitness': [],
        'average_fitness': [],
        'pareto_front_size': [],
        'constraint_violations': []
    }
    
    # Track best solution and set reference point for hypervolume
    best_solution = None
    reference_point = [1000, 100, 100, 100, 100, 100]  # High values for each objective
    
    # Main loop
    for generation in range(max_generations):
        if verbose:
            print("\nGeneration {}/{}".format(generation+1, max_generations))
        
        # Calculate crowding distance for each front
        for front in fronts:
            crowding_distance_assignment(front)
        
        # Create offspring
        offspring = create_offspring(population, generation, data, population_size, 
                                    crossover_rate, mutation_rate, metrics)
        
        # Evaluate offspring
        evaluate_population(offspring, data)
        
        # Combine parent and offspring populations and do non-dominated sorting
        combined_population = population + offspring
        combined_fronts = fast_nondominated_sort(combined_population)
        
        # Select next generation
        next_population = select_next_generation(combined_fronts, population_size)
        
        # Handle diversity preservation
        next_population = handle_diversity_preservation(
            next_population, metrics, data, population_size, verbose)
        
        # Update population and fronts
        population = next_population
        fronts = fast_nondominated_sort(population)
        
        # Update metrics and best solution
        best_solution, metrics = update_metrics(
            fronts, population, metrics, reference_point, best_solution, data, verbose)
    
    # Return best solution, final fronts, and metrics
    if best_solution is None and population:
        best_solution = find_best_solution(population)
    
    return best_solution, fronts, metrics

def run_muni_optimization(verbose=False):
    """Run the optimization on the MUni dataset."""
    print("Loading munifspsspr17 dataset...")
    data = load_muni_data(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'munifspsspr17.json'))
    
    # Create a reference front for IGD calculation
    # Using realistic reference points based on typical violation ranges
    data['reference_front'] = [
        [0, 0, 0, 0, 0, 0],      # Perfect solution (no violations)
        [100, 10, 10, 10, 10, 10],  # Moderate room conflicts
        [10, 100, 10, 10, 10, 10],  # Moderate time conflicts
        [10, 10, 100, 10, 10, 10],  # Moderate distribution conflicts
        [10, 10, 10, 100, 10, 10],  # Moderate student conflicts
        [10, 10, 10, 10, 100, 10],  # Moderate capacity violations
    ]
    
    print("\nRunning NSGA-II optimization...")
    best_solution, fronts, metrics = nsga2_muni(data, verbose=True)
    
    # Print best solution
    print("\nBest solution:\n")
    print_solution_stats(best_solution, data)
    
    # Process metrics to ensure no tuple keys remain
    processed_metrics = process_metrics_for_serialization(metrics)
    
    # Plot metrics
    plot_metrics(processed_metrics)
    
    # Plot constraint violations over generations
    violations = [v['total_counts'] for v in processed_metrics['constraint_violations']]
    
    # Add debug print to check if student_conflicts exists in the data
    if violations:
        print("\nDebug - Constraint violation keys in first generation:")
        for key in violations[0].keys():
            print(f"  - {key}")
    
    plot_constraint_violations(violations)
    
    # Plot Pareto front size over generations
    plot_pareto_size(processed_metrics['pareto_front_size'])
    
    # Convert any tuple keys in solution assignments before saving
    try:
        # Create a deep copy of the best solution with string keys
        modified_solution = best_solution.copy()
        modified_solution.assignments = convert_tuple_keys_in_dict(best_solution.assignments)
        
        # Process constraint violations if they exist
        if hasattr(modified_solution, 'constraint_violations') and modified_solution.constraint_violations:
            modified_solution.constraint_violations = convert_tuple_keys_in_dict(modified_solution.constraint_violations)
        
        # Process fronts to ensure no tuple keys
        processed_fronts = []
        for front in fronts:
            processed_front = []
            for solution in front:
                s_copy = solution.copy()
                s_copy.assignments = convert_tuple_keys_in_dict(solution.assignments)
                if hasattr(s_copy, 'constraint_violations') and s_copy.constraint_violations:
                    s_copy.constraint_violations = convert_tuple_keys_in_dict(s_copy.constraint_violations)
                processed_front.append(s_copy)
            processed_fronts.append(processed_front)
        
        save_results(modified_solution, processed_fronts, processed_metrics, data, 'nsga2_muni_results.json')
        print("\nNSGA-II optimization on munifspsspr17 dataset completed successfully.")
        print("Results saved to nsga2_muni_results.json")
    except Exception as e:
        print(f"Error saving detailed results: {e}")
        # Try direct JSON serialization with custom encoder
        try:
            import json
            with open('nsga2_muni_results_direct.json', 'w') as f:
                # Create a simplified result structure
                simple_results = {
                    'best_solution': {
                        'fitness': best_solution.fitness,
                        'num_assigned': len(best_solution.assignments)
                    },
                    'metrics': {
                        'hypervolume': processed_metrics['hypervolume'],
                        'spacing': processed_metrics['spacing'],
                        'pareto_front_size': processed_metrics['pareto_front_size'],
                        'best_fitness': processed_metrics['best_fitness'],
                        'average_fitness': processed_metrics['average_fitness']
                    }
                }
                json.dump(simple_results, f, indent=2, cls=TupleKeyEncoder)
            print("Simplified results saved to nsga2_muni_results_direct.json")
        except Exception as e2:
            print(f"All saving attempts failed: {e2}")
    
    return best_solution, fronts, metrics, data

def convert_value(v):
    """Helper function to convert a single value for JSON serialization."""
    if isinstance(v, dict):
        return convert_tuple_keys_in_dict(v)
    elif isinstance(v, tuple):
        return str(v)
    elif isinstance(v, list):
        return [convert_value(item) for item in v]
    else:
        return v

def convert_tuple_keys_in_dict(d):
    """Convert tuple keys in a dictionary to string keys recursively."""
    if not isinstance(d, dict):
        return d
    
    result = {}
    for k, v in d.items():
        # Convert key if it's a tuple
        new_key = str(k) if isinstance(k, tuple) else k
        # Process value using helper function
        result[new_key] = convert_value(v)
    
    return result

def process_metrics_for_serialization(metrics):
    """Process metrics to ensure they can be serialized to JSON."""
    processed_metrics = {}
    
    for key, value in metrics.items():
        if key == 'constraint_violations':
            # Process constraint violations list
            processed_violations = []
            for violation_data in value:
                processed_violation = convert_tuple_keys_in_dict(violation_data)
                processed_violations.append(processed_violation)
            processed_metrics[key] = processed_violations
        else:
            # Other metrics can be copied directly
            processed_metrics[key] = value
    
    return processed_metrics

if __name__ == "__main__":
    # Run the NSGA-II optimization on munifspsspr17 dataset
    best_solution, fronts, metrics, data = run_muni_optimization()
