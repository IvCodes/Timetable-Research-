"""
SPEA2 (Strength Pareto Evolutionary Algorithm 2) implementation for the munifspsspr17 dataset.

This implementation adapts the SPEA2 algorithm to the university course timetabling problem
using the standard UniTime data format.
"""
import json
import time
import random
import numpy as np
from datetime import datetime
import copy
import os
import sys

# Add parent directory to sys.path to import modules from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import common modules
from solution import Solution
from data_loader_muni import load_muni_data
from evaluate import evaluate_solution
from utils_1 import TupleKeyEncoder, save_results, print_solution_stats, repair_schedule, detect_conflicts
from metrics import calculate_hypervolume, track_constraint_violations, calculate_spacing, calculate_igd, analyze_constraint_violations, update_reference_point
from plots import plot_metrics, plot_constraint_violations, plot_pareto_size

# Import NSGA-II helper functions
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nsga_II'))
from test_nsga2_muni import (
    initialize_population, evaluate_population, mutate, crossover, 
    tournament_selection, find_best_solution, 
    mutate_existing_assignment, add_new_assignment
)

# Algorithm parameters
POPULATION_SIZE = 10
ARCHIVE_SIZE = 5
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
NUM_OBJECTIVES = 5  # Number of objectives in the optimization

REFERENCE_POINT = [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')]

def get_pareto_front(population):
    """
    Get the Pareto front from a population of solutions.
    
    Args:
        population: List of Solution objects
    
    Returns:
        List of non-dominated Solution objects
    """
    pareto_front = []
    for i, solution in enumerate(population):
        if solution.fitness is None:
            continue
        
        is_dominated = False
        for j, other in enumerate(population):
            if i == j or other.fitness is None:
                continue
            
            if all(f2 <= f1 for f1, f2 in zip(solution.fitness, other.fitness)) and \
               any(f2 < f1 for f1, f2 in zip(solution.fitness, other.fitness)):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_front.append(solution)
    
    return pareto_front

def _analyze_violations(solution, data):
    """Helper function to analyze different types of violations."""
    violations = {
        "room_capacity": 0,
        "room_feature": 0,
        "instructor": 0,
        "class_time": 0
    }
    
    # This is simplified as we're using the analyze_constraint_violations function
    # from metrics for detailed analysis
    
    return violations

def calculate_strength(population):
    """
    Calculate the strength of each individual in the population.
    
    Args:
        population: List of Solution objects
    
    Returns:
        List of strength values (number of solutions dominated by each solution)
    """
    strength = [0] * len(population)
    
    for i in range(len(population)):
        if population[i].fitness is None:
            continue
            
        for j in range(len(population)):
            if i == j or population[j].fitness is None:
                continue
                
            # Count how many solutions this solution dominates
            if all(f1 <= f2 for f1, f2 in zip(population[i].fitness, population[j].fitness)) and \
               any(f1 < f2 for f1, f2 in zip(population[i].fitness, population[j].fitness)):
                strength[i] += 1
    
    return strength

def calculate_raw_fitness(population, strength):
    """
    Calculate the raw fitness for each individual based on strength.
    
    Args:
        population: List of Solution objects
        strength: List of strength values
    
    Returns:
        List of raw fitness values (sum of strengths of all dominators)
    """
    raw_fitness = [0] * len(population)
    
    # Helper function to check if solution j dominates solution i
    def is_dominated_by(i, j):
        if population[i].fitness is None or population[j].fitness is None:
            return False
        return (all(f2 <= f1 for f1, f2 in zip(population[i].fitness, population[j].fitness)) and
                any(f2 < f1 for f1, f2 in zip(population[i].fitness, population[j].fitness)))
    
    for i in range(len(population)):
        if population[i].fitness is None:
            # Penalize solutions without fitness
            raw_fitness[i] = float('inf')
            continue
            
        # Sum strengths of all solutions that dominate this one
        raw_fitness[i] = sum(strength[j] for j in range(len(population)) 
                           if i != j and is_dominated_by(i, j))
    
    return raw_fitness

def calculate_density(population, k=1):
    """
    Calculate density based on k-nearest neighbor.
    
    Args:
        population: List of Solution objects
        k: Number of nearest neighbors to consider
    
    Returns:
        List of density values
    """
    if len(population) <= k:
        return [0] * len(population)
    
    # Calculate distances between all pairs of solutions
    distances = compute_distances(population)
    
    # Calculate density for each solution
    return compute_density_values(population, distances, k)

def compute_distances(population):
    """Helper function to compute distances between all solutions."""
    distances = np.zeros((len(population), len(population)))
    
    for i in range(len(population)):
        if population[i].fitness is None:
            continue
            
        for j in range(i+1, len(population)):
            if population[j].fitness is None:
                continue
                
            # Euclidean distance in objective space
            dist = np.sqrt(sum((f1 - f2) ** 2 for f1, f2 in 
                           zip(population[i].fitness, population[j].fitness)))
            distances[i, j] = distances[j, i] = dist
    
    return distances

def compute_density_values(population, distances, k):
    """Helper function to compute density values from distances."""
    density = [0] * len(population)
    
    for i in range(len(population)):
        if population[i].fitness is None:
            density[i] = float('inf')  # Penalize solutions without fitness
            continue
            
        # Sort distances to other solutions
        dist_to_others = sorted(d for j, d in enumerate(distances[i]) if j != i and d > 0)
        
        # Get distance to kth nearest neighbor
        if len(dist_to_others) > k:
            kth_distance = dist_to_others[k]
            density[i] = 1 / (kth_distance + 2)  # Add 2 to avoid division by very small numbers
        else:
            density[i] = 0  # Not enough neighbors
    
    return density

def calculate_fitness(raw_fitness, density):
    """
    Calculate the final fitness by combining raw fitness and density.
    
    Args:
        raw_fitness: List of raw fitness values
        density: List of density values
    
    Returns:
        List of final fitness values (lower is better)
    """
    return [r + d for r, d in zip(raw_fitness, density)]

def environmental_selection(population, archive_size):
    """
    Perform environmental selection to update the archive.
    
    Args:
        population: Combined population of current population and archive
        archive_size: Maximum size of the archive
    
    Returns:
        Selected individuals as the new archive
    """
    if not population:
        return []
    
    # Remove solutions without fitness
    valid_population = [s for s in population if s.fitness is not None]
    
    if not valid_population:
        return []
    
    # Calculate strength for all solutions
    strength = calculate_strength(valid_population)
    
    # Calculate raw fitness for all solutions
    raw_fitness = calculate_raw_fitness(valid_population, strength)
    
    # Find all non-dominated solutions (raw fitness = 0)
    non_dominated = [i for i, r in enumerate(raw_fitness) if r == 0]
    
    # If we have fewer non-dominated solutions than archive size, just return them all
    if len(non_dominated) <= archive_size:
        return select_non_dominated_solutions(valid_population, non_dominated)
    
    # Otherwise, truncate using density
    return truncate_by_density(valid_population, raw_fitness, archive_size)

def select_non_dominated_solutions(population, non_dominated_indices):
    """Helper function to select non-dominated solutions from population."""
    return [population[i] for i in non_dominated_indices]

def truncate_by_density(population, raw_fitness, archive_size):
    """Helper function to truncate population based on density."""
    # Calculate density to use for truncation
    density = calculate_density(population)
    fitness = calculate_fitness(raw_fitness, density)
    
    # Sort by fitness and return the best archive_size solutions
    indices = sorted(range(len(fitness)), key=lambda i: fitness[i])
    return [population[i] for i in indices[:archive_size]]

def select_parents(population, tournament_size=2):
    """
    Select parents for reproduction using tournament selection.
    
    Args:
        population: List of Solution objects
        tournament_size: Size of tournament selection
    
    Returns:
        Two selected parents
    """
    # Select first parent
    parent1 = tournament_selection(population, tournament_size)
    
    # Select second parent (ensure it's different from the first)
    parent2 = None
    attempts = 0
    while parent2 is None or parent2 == parent1:
        parent2 = tournament_selection(population, tournament_size)
        attempts += 1
        if attempts > 5:  # Prevent infinite loop in small populations
            parent2 = random.choice(population)
            break
    
    return parent1, parent2

def update_metrics_from_archive(archive, metrics, ideal_point, data, verbose):
    """
    Update metrics based on the current archive.
    
    Args:
        archive: Current archive of solutions
        metrics: Dictionary of metrics to update
        ideal_point: Current ideal point
        data: Problem data
        verbose: Whether to print progress information
        
    Returns:
        Updated metrics, updated ideal point
    """
    if not archive:
        return metrics, ideal_point
    
    # Calculate metrics from pareto front
    pareto_front = get_pareto_front(archive)
    metrics['hypervolume'].append(calculate_hypervolume(pareto_front))
    metrics['spacing'].append(calculate_spacing(pareto_front))
    metrics['igd'].append(calculate_igd(pareto_front, ideal_point))
    metrics['pareto_front_size'].append(len(pareto_front))
    
    # Update constraint violations
    violations = analyze_constraint_violations(archive, data)
    metrics['constraint_violations'].append(violations)
    
    # Update ideal point
    for solution in archive:
        if solution.fitness:
            update_reference_point(solution.fitness, ideal_point)
            
    # Update fitness metrics
    update_fitness_metrics(archive, metrics)
    
    # Print progress if needed
    if verbose and len(metrics['hypervolume']) % 10 == 0:
        print(f"Generation metrics - HV: {metrics['hypervolume'][-1]:.4f}, " 
              f"Pareto size: {metrics['pareto_front_size'][-1]}")
              
    return metrics, ideal_point

def update_fitness_metrics(archive, metrics):
    """Update best and average fitness metrics from archive."""
    fitnesses = [s.fitness[0] for s in archive if s.fitness]
    if fitnesses:
        metrics['best_fitness'].append(min(fitnesses))
        metrics['average_fitness'].append(sum(fitnesses) / len(fitnesses))
    return metrics

def process_generation(population, archive, generation, metrics, ideal_point, data, verbose):
    """
    Process a single generation of the SPEA2 algorithm.
    
    Args:
        population: Current population
        archive: Current archive
        generation: Current generation number
        metrics: Dictionary of metrics to update
        ideal_point: Current ideal point
        data: Problem data
        verbose: Whether to print progress information
        
    Returns:
        Updated population, archive, best solution, metrics, ideal point
    """
    gen_start_time = time.time()
    
    # Print progress information
    if verbose and generation % 10 == 0:
        print(f"Generation {generation}/{generations}")
    
    # Environmental selection
    archive = perform_environmental_selection(population, archive)
    
    # Find best solution in archive
    current_best = find_best_solution(archive)
    best_solution = copy.deepcopy(current_best) if is_better_solution(current_best, None) else None
    
    # Update metrics
    metrics, ideal_point = update_metrics_from_archive(archive, metrics, ideal_point, data, verbose)
    
    # Update execution time
    gen_time = time.time() - gen_start_time
    metrics['execution_time'].append(gen_time)
    
    # Generate offspring
    if generation < NUM_GENERATIONS - 1:
        population = generate_offspring(population, archive, generation, data)
        # Update best solution from population
        best_solution = update_best_solution(population, best_solution)
    
    return population, archive, best_solution, metrics, ideal_point

def spea2_muni(data, generations=NUM_GENERATIONS, verbose=False):
    """
    Run the SPEA2 algorithm to optimize university course timetabling.
    
    Args:
        data: Dictionary containing dataset information
        generations: Number of generations to run
        verbose: Whether to print progress information
    
    Returns:
        Final archive of optimized solutions, metrics dict
    """
    start_time = time.time()
    
    # Print initial information
    print_initial_info(data, verbose)
    
    # Initialize metrics, population, and archive
    metrics = initialize_metrics()
    population, archive = initialize_algorithm_components(data)
    best_solution = None
    ideal_point = [float('inf')] * NUM_OBJECTIVES
    
    # Main evolutionary loop
    for generation in range(generations):
        population, archive, best_solution, metrics, ideal_point = process_generation(
            population, archive, generation, metrics, ideal_point, data, verbose
        )
    
    # Print final information
    total_time = time.time() - start_time
    if verbose:
        print(f"\nOptimization completed in {total_time:.2f} seconds")
    
    return archive, metrics

def print_initial_info(data, verbose):
    """Print initial dataset information."""
    if verbose:
        print(f"Dataset loaded. {len(data['classes'])} classes, {len(data['rooms'])} rooms")
        fixed_classes = count_fixed_classes(data)
        print(f"Fixed classes: {fixed_classes}")

def count_fixed_classes(data):
    """Count the number of classes with fixed assignments."""
    fixed_count = 0
    for class_id, class_info in data['classes'].items():
        if len(class_info.get('rooms', [])) == 1 and len(class_info.get('times', [])) == 1:
            fixed_count += 1
    return fixed_count

def initialize_metrics():
    """Initialize metrics dictionary."""
    return {
        'hypervolume': [],
        'spacing': [],
        'igd': [],
        'best_fitness': [],
        'average_fitness': [],
        'pareto_front_size': [],
        'constraint_violations': [],
        'execution_time': []
    }

def initialize_algorithm_components(data):
    """Initialize population and archive."""
    # Initialize population
    population = initialize_population(data, size=POPULATION_SIZE)
    
    # Evaluate initial population
    fitnesses = evaluate_population(population, data)
    for i, solution in enumerate(population):
        solution.fitness = fitnesses[i]
    
    # Initialize empty archive
    archive = []
    
    return population, archive

def perform_environmental_selection(population, archive):
    """Perform environmental selection to update the archive."""
    # Combine population and archive
    combined_population = population + archive
    
    # Environmental selection to update archive
    return environmental_selection(combined_population, ARCHIVE_SIZE)

def generate_offspring(population, archive, generation, data):
    """Generate offspring population through selection, crossover, and mutation."""
    combined_population = population + archive
    offspring = []
    
    for _ in range(POPULATION_SIZE):
        # Create and evaluate a new offspring
        child = create_offspring(combined_population, generation, data)
        offspring.append(child)
    
    return offspring

def create_offspring(population, generation, data):
    """Create a single offspring through selection, crossover, and mutation."""
    # Select parents using tournament selection
    idx1 = random.randrange(len(population))
    idx2 = random.randrange(len(population))
    if idx2 == idx1:
        idx2 = (idx1 + 1) % len(population)
    
    parent1 = population[idx1]
    parent2 = population[idx2]
    
    # Create offspring through crossover and mutation
    child, _ = crossover(parent1, parent2, data)
    
    # Initialize generation for mutation rate calculation
    if not hasattr(child, 'generation') or child.generation is None:
        child.generation = generation
        
    # Mutate the child
    mutate(child, data, mutation_rate=MUTATION_RATE)
    
    # Apply repair operator
    repair_schedule(child, data)
    
    # Evaluate the child
    child.fitness = evaluate_solution(child, data)
    
    return child

def update_best_solution(population, best_solution):
    """Update the best solution if a better one is found."""
    current_best = find_best_solution(population)
    
    # Check if the current best is better than the stored best solution
    if is_better_solution(current_best, best_solution):
        return copy.deepcopy(current_best)
    return best_solution

def is_better_solution(solution1, solution2):
    """Check if solution1 is better than solution2."""
    if solution1 is None:
        return False
    if solution2 is None:
        return True  # solution1 is not None, so it's better
    
    # Check if both solutions have valid fitness values
    if not hasattr(solution1, 'fitness') or solution1.fitness is None:
        return False
    if not hasattr(solution2, 'fitness') or solution2.fitness is None:
        return True  # solution1 has fitness, solution2 doesn't
    
    # Compare the actual fitness values (first objective)
    if len(solution1.fitness) > 0 and len(solution2.fitness) > 0:
        return solution1.fitness[0] < solution2.fitness[0]
    
    return False

def run_muni_optimization(verbose=False):
    """
    Main function to run the SPEA2 optimization for the munifspsspr17 dataset.
    
    Returns:
        Best timetable solution, formatted solution, data, metrics
    """
    # Load dataset with proper path
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'munifspsspr17.json')
    data = load_muni_data(data_path)
    
    if verbose:
        print(f"Dataset loaded. {len(data['classes'])} classes, {len(data['rooms'])} rooms, {len(data['time_patterns'])} time patterns")
        fixed_classes = count_fixed_classes(data)
        print(f"Fixed classes: {fixed_classes}")
    
    # Run SPEA2 algorithm
    archive, metrics = spea2_muni(data, verbose=verbose)
    
    # Find the best solution
    best_solution = find_best_solution(archive)
    
    # Add constraint violations to the best solution for compatibility with print_solution_stats
    violations = track_constraint_violations(best_solution, data)
    best_solution.constraint_violations = {
        'room_conflicts': violations['total_by_type']['room_conflicts'],
        'time_conflicts': violations['total_by_type']['time_conflicts'],
        'distribution_conflicts': violations['total_by_type']['distribution_conflicts'],
        'student_conflicts': violations['total_by_type']['student_conflicts'],
        'capacity_violations': violations['total_by_type']['capacity_violations']
    }
    
    # Evaluate the best solution
    if verbose:
        print("\nBest Solution Statistics:")
        print_solution_stats(best_solution, data)
    
    # Format solution for frontend
    formatted_solution = format_solution_for_output(best_solution, data)
    
    # Save solution and results
    save_solution_for_frontend(formatted_solution)
    
    # Create a pareto front structure expected by save_results
    pareto_front = get_pareto_front(archive)
    fronts = [pareto_front]  # List of fronts, where each front is a list of solutions
    
    # Save results with the correctly structured fronts parameter
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spea2_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_results(best_solution, fronts, metrics, data, os.path.join(output_dir, "spea2_results.json"))
    
    return best_solution, formatted_solution, data, metrics

def save_solution_for_frontend(solution):
    """
    Save the formatted solution to a JSON file for the frontend.
    
    Args:
        solution: Formatted solution dictionary
    """
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spea2_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, "spea2_solution.json")
    
    with open(output_file, 'w') as f:
        json.dump(solution, f, indent=2, cls=TupleKeyEncoder)
    
    print(f"Solution saved to {output_file}")

def create_structured_entity(entity_type, data):
    """
    Create a properly structured entity for frontend compatibility.
    
    Args:
        entity_type: Type of entity (day, period, room)
        data: The data to structure
        
    Returns:
        A dictionary with structured entity data
    """
    if entity_type == 'day':
        # Create a day entity
        if isinstance(data, dict) and 'name' in data:
            # Already formatted
            return data
        else:
            # Format as needed
            return {
                'name': str(data),
                'code': str(data)
            }
    elif entity_type == 'period':
        # Create a period entity
        if isinstance(data, dict) and 'name' in data:
            # Already formatted
            return data
        else:
            # Format as needed
            return {
                'name': str(data),
                'code': str(data)
            }
    elif entity_type == 'room':
        # Create a room entity
        if isinstance(data, dict):
            # Already has some structure
            return {
                'name': data.get('name', data.get('_id', '')),
                'code': data.get('_id', ''),
                'capacity': data.get('_capacity', 0)
            }
        else:
            # Just ID provided
            return {
                'name': str(data),
                'code': str(data),
                'capacity': 0
            }
    return {}

def create_room_entity(room_id, data):
    """Helper function to create a properly structured room entity."""
    if room_id in data['rooms']:
        room_data = data['rooms'][room_id]
        return {
            'name': room_id,
            'code': room_id,
            'capacity': room_data.get('capacity', 0)
        }
    else:
        return {
            'name': room_id,
            'code': room_id,
            'capacity': 0
        }

def format_solution_for_output(solution, data):
    """
    Format the solution for output in a structured format compatible with frontend.
    
    Args:
        solution: Solution object
        data: Dataset information
    
    Returns:
        Dictionary with properly formatted solution
    """
    formatted_solution = {
        'algorithm': 'SPEA2',
        'activities': []
    }
    
    # Process each assignment
    for assignment in solution.assignments:
        # Extract assignment details
        class_id, room_id, time_id = extract_assignment_details(assignment)
        
        # Get entity data
        day_entity, period_entity, room_entity, activity_name = get_entity_data(class_id, room_id, time_id, data)
        
        # Create activity entry
        activity = create_activity_entry(activity_name, day_entity, period_entity, room_entity)
        
        formatted_solution['activities'].append(activity)
    
    return formatted_solution

def extract_assignment_details(assignment):
    """Extract basic assignment details."""
    class_id = assignment['class_id']
    room_id = assignment['room_id']
    time_id = assignment['time_id']
    return class_id, room_id, time_id

def get_entity_data(class_id, room_id, time_id, data):
    """Get entity data for an assignment."""
    # Get time pattern info
    time_pattern = data['time_patterns'][time_id]
    days = time_pattern['days']
    start_time = time_pattern['start']
    
    # Get class and course info
    class_info = data['classes'][class_id]
    course_id = class_info['course_id']
    course_name = data['courses'][course_id]['name'] if course_id in data['courses'] else course_id
    
    # Create structured entities
    day_entity = create_structured_entity('day', days)
    period_entity = create_structured_entity('period', start_time)
    room_entity = create_room_entity(room_id, data)
    
    # Create activity name
    activity_name = f"{course_name} - {class_id}"
    
    return day_entity, period_entity, room_entity, activity_name

def create_activity_entry(name, day, period, room):
    """Create a properly structured activity entry for the frontend."""
    # Ensure the activity has the algorithm field set for proper frontend display
    # This is critical according to the memories about standardizing output formats
    return {
        'name': name,
        'day': day,
        'period': period,
        'room': room,
        'algorithm': 'SPEA2'  # Explicitly setting algorithm field for frontend compatibility
    }

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    try:
        # Run optimization
        best_solution, formatted_solution, data, metrics = run_muni_optimization(verbose=True)
        
        # Generate plots - using correct parameter count
        plot_metrics(metrics)
        plot_constraint_violations(metrics)
        plot_pareto_size(metrics)
        
        print("\nSPEA2 optimization completed successfully!")
        print(f"Solution saved to: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spea2_results', 'spea2_solution.json')}")
        print(f"Results saved to: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spea2_results', 'spea2_results.json')}")
    except Exception as e:
        print(f"Error during SPEA2 optimization: {e}")
        import traceback
        traceback.print_exc()