"""
MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) implementation for the munifspsspr17 dataset.

This implementation adapts the MOEA/D algorithm to the university course timetabling problem
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
from utils_1 import TupleKeyEncoder, save_results, print_solution_stats
from metrics import calculate_hypervolume, track_constraint_violations, calculate_spacing, calculate_igd, analyze_constraint_violations
from plots import plot_metrics, plot_constraint_violations, plot_pareto_size

# Import NSGA-II helper functions
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nsga_II'))
from test_nsga2_muni import (
    initialize_population, evaluate_population, mutate, crossover, 
    detect_conflicts, repair_schedule, find_best_solution, 
    tournament_selection, mutate_existing_assignment, add_new_assignment
)

# Algorithm parameters
POPULATION_SIZE = 50
NUM_GENERATIONS = 10
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
NUM_OBJECTIVES = 6  # Change from 5 to 6 to match evaluate_solution's return value
T = 10  # Neighborhood size
NEIGHBORHOOD_SELECTION_PROB = 0.9  # Probability of selecting from neighborhood

# Reference point for hypervolume calculation (worst possible values)
REFERENCE_POINT = [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')]

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

def calculate_tchebycheff(fitness, weight, ideal_point):
    """
    Calculate Tchebycheff approach for decomposition.
    
    Args:
        fitness: Fitness values
        weight: Weight vector
        ideal_point: Ideal point
    
    Returns:
        Tchebycheff value (smaller is better)
    """
    # Ensure the arrays are the same length
    min_len = min(len(fitness), len(weight), len(ideal_point))
    fitness = fitness[:min_len]
    weight = weight[:min_len]
    ideal_point = ideal_point[:min_len]
    
    # Calculate weighted distance from ideal point
    max_diff = float('-inf')
    for i in range(min_len):
        # Skip invalid fitness values
        if not np.isfinite(fitness[i]) or not np.isfinite(ideal_point[i]):
            continue
            
        # Weight must be positive
        weight_i = max(0.0001, weight[i])
        diff = abs(fitness[i] - ideal_point[i]) * weight_i
        max_diff = max(max_diff, diff)
    
    return max_diff if max_diff > float('-inf') else float('inf')

# Make tchebycheff_distance a synonym for calculate_tchebycheff for backward compatibility
tchebycheff_distance = calculate_tchebycheff

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
    # Use neighborhood or entire population based on probability
    if random.random() < neighborhood_selection_prob:
        # Select from neighborhood
        neighborhood = neighborhoods[index]
        idx1 = random.choice(neighborhood)
        idx2 = random.choice(neighborhood)
        
        # Ensure idx1 and idx2 are different
        while idx2 == idx1:
            idx2 = random.choice(neighborhood)
    else:
        # Select from entire population
        population_size = len(neighborhoods)
        idx1 = random.randrange(population_size)
        idx2 = random.randrange(population_size)
        
        # Ensure idx1 and idx2 are different
        while idx2 == idx1:
            idx2 = random.randrange(population_size)
    
    return idx1, idx2

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
            # Objectives are minimization problems
            ideal_point[i] = min(ideal_point[i], fitness[i])
    
    return ideal_point

def initialize_algorithm(data, population_size, verbose=False):
    """
    Initialize the MOEA/D algorithm components.
    
    Args:
        data: Dictionary containing dataset information
        population_size: Size of the population
        verbose: Whether to print progress information
    
    Returns:
        Tuple of (population, fitness_values, ideal_point, weight_vectors, neighborhoods)
    """
    # For large datasets, reduce population size
    adjusted_population_size = population_size
    if len(data['classes']) > 300:
        if verbose:
            print(f"Large dataset detected ({len(data['classes'])} classes). Reducing population size.")
        adjusted_population_size = min(population_size, 30)
    
    # Initialize population using NSGA-II helper function
    population = initialize_population(data, size=adjusted_population_size)
    
    # Evaluate initial population
    fitness_values = evaluate_population(population, data)
    
    # Initialize ideal point (minimize all objectives)
    ideal_point = [float('inf')] * NUM_OBJECTIVES
    ideal_point = update_reference_point(fitness_values, ideal_point)
    
    # Generate weight vectors and compute neighborhoods
    weight_vectors = generate_weight_vectors(adjusted_population_size, NUM_OBJECTIVES)
    
    return population, fitness_values, ideal_point, weight_vectors

def setup_neighborhoods(weight_vectors, t):
    """
    Set up the neighborhood structure for MOEA/D.
    
    Args:
        weight_vectors: List of weight vectors
        t: Neighborhood size
    
    Returns:
        List of neighborhoods
    """
    # Adjusted t for large datasets
    adjusted_t = min(t, 5) if len(weight_vectors) > 50 else t
    
    # Compute distances and sort indices
    _, sorted_indices = compute_euclidean_distances(weight_vectors)
    
    # Initialize neighborhoods
    neighborhoods = initialize_neighborhoods(sorted_indices, adjusted_t)
    
    return neighborhoods

def moead_muni(data, generations=NUM_GENERATIONS, population_size=POPULATION_SIZE, t=T, 
              neighborhood_selection_prob=NEIGHBORHOOD_SELECTION_PROB, verbose=False):
    """
    Run the MOEA/D algorithm to optimize university course timetabling.
    
    Args:
        data: Dictionary containing dataset information
        generations: Number of generations to run
        population_size: Size of the population
        t: Neighborhood size
        neighborhood_selection_prob: Probability of selecting from neighborhood
        verbose: Whether to print progress information
    
    Returns:
        Final population of optimized timetables, fitness values, metrics
    """
    start_time = time.time()
    
    # Initialize algorithm components
    population, fitness_values, ideal_point, weight_vectors = initialize_algorithm(
        data, population_size, verbose)
    
    # Set up neighborhoods
    neighborhoods = setup_neighborhoods(weight_vectors, t)
    
    # Initialize metrics tracking
    metrics = {
        'hypervolume': [],
        'spacing': [],
        'igd': [],
        'best_fitness': [],
        'average_fitness': [],
        'constraint_violations': [],
        'pareto_front_size': []
    }
    
    # Initialize best solution tracking
    best_solution = None
    
    # Main evolutionary loop
    for gen in range(generations):
        gen_start_time = time.time()
        if verbose:
            print(f"Generation {gen}/{generations}")
        
        # Process each subproblem
        _process_generation(population, neighborhoods, weight_vectors, ideal_point, 
                           fitness_values, data, neighborhood_selection_prob)
        
        # Update metrics
        update_metrics(population, metrics, ideal_point, best_solution, data, verbose)
        
        # Print generation time to detect slow performance
        if verbose:
            gen_time = time.time() - gen_start_time
            print(f"  Generation {gen} completed in {gen_time:.2f} seconds")
            
        # Early stopping if we're making good progress
        if gen > 10 and metrics['best_fitness'][-1] < 10:
            if verbose:
                print("Early stopping criteria met - good solution found")
            break
    
    end_time = time.time()
    
    if verbose:
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    
    # Generate plots
    try:
        plot_metrics(metrics)
        plot_constraint_violations([v['total_counts'] for v in metrics['constraint_violations']])
        plot_pareto_size(metrics['pareto_front_size'])
    except Exception as e:
        if verbose:
            print(f"Error generating plots: {e}")
    
    return population, fitness_values, ideal_point, metrics

def _select_and_create_offspring(i, neighborhoods, neighborhood_selection_prob, population, data):
    """
    Select parents and create offspring for a given subproblem.
    
    Args:
        i: Index of the current subproblem
        neighborhoods: List of neighborhoods
        neighborhood_selection_prob: Probability of selecting from neighborhood
        population: List of Solution objects
        data: Dictionary containing dataset information
    
    Returns:
        The created child solution
    """
    # Select parents
    p1_idx, p2_idx = select_parent_indexes(i, neighborhoods, neighborhood_selection_prob)
    parent1, parent2 = population[p1_idx], population[p2_idx]
    
    # Create offspring via crossover
    child1, _ = crossover(parent1, parent2, data)
    
    # Initialize generation for mutation rate calculation
    if not hasattr(child1, 'generation') or child1.generation is None:
        child1.generation = 0
        
    # Mutate the child in-place (mutate doesn't return anything)
    mutate(child1, data, mutation_rate=MUTATION_RATE)
    
    # Apply repair operator to fix constraint violations
    repair_schedule(child1, data)
    
    return child1

def _update_neighborhood_with_child(i, neighborhoods, child, child_fitness, 
                                  population, fitness_values, weight_vectors, ideal_point):
    """
    Update neighboring solutions with the child if it improves the subproblem.
    
    Args:
        i: Index of the current subproblem
        neighborhoods: List of neighborhoods
        child: Child solution
        child_fitness: Fitness values of the child
        population: List of Solution objects
        fitness_values: List of fitness values for each solution
        weight_vectors: List of weight vectors
        ideal_point: The ideal point for normalization
    """
    # Check each neighbor in the neighborhood
    for j in neighborhoods[i]:
        # Calculate tchebycheff fitness for both neighbor and child
        neighbor_tcheby = calculate_tchebycheff(fitness_values[j], weight_vectors[j], ideal_point)
        child_tcheby = calculate_tchebycheff(child_fitness, weight_vectors[j], ideal_point)
        
        # Replace neighbor if child is better
        if child_tcheby < neighbor_tcheby:
            population[j] = copy_solution(child)
            population[j].fitness = child_fitness
            fitness_values[j] = child_fitness

def _process_generation(population, neighborhoods, weight_vectors, ideal_point, 
                       fitness_values, data, neighborhood_selection_prob):
    """Helper function to process a single generation of MOEA/D to reduce complexity."""
    # For each subproblem
    for i in range(len(population)):
        if i % 10 == 0:  # Add debug print to track progress
            print(f"  Processing subproblem {i}/{len(population)}")
            
        # Select parents and create offspring
        child = _select_and_create_offspring(i, neighborhoods, 
                                           neighborhood_selection_prob, population, data)
        
        # Evaluate offspring
        offspring_fitness = evaluate_solution(child, data)
        child.fitness = offspring_fitness
        
        # Update ideal point
        for j in range(len(ideal_point)):
            ideal_point[j] = min(ideal_point[j], offspring_fitness[j])
        
        # Update neighboring solutions
        _update_neighborhood_with_child(i, neighborhoods, child, offspring_fitness, 
                                      population, fitness_values, weight_vectors, ideal_point)

def update_metrics(population, metrics, ideal_point, best_solution, data, verbose=False):
    """
    Update all metrics for the current generation.
    """
    # Calculate hypervolume
    try:
        # Filter out None fitness values and ensure they are valid numpy arrays
        fitness_arrays = []
        for s in population:
            if s.fitness is not None:
                # Make sure all fitness values are finite
                fitness = np.array([min(1e6, max(-1e6, f)) for f in s.fitness])
                fitness_arrays.append(fitness)
        
        if fitness_arrays and len(fitness_arrays) > 1:
            # Ensure all fitness arrays have the same dimension
            min_dim = min(len(f) for f in fitness_arrays)
            fitness_arrays = [f[:min_dim] for f in fitness_arrays]
            
            # Use a reference point slightly worse than the worst observed values
            # This is more reliable than using infinity
            ref_point = np.array([max(f[i] for f in fitness_arrays) * 1.1 for i in range(min_dim)])
            
            # Calculate hypervolume with the adaptive reference point
            hv = calculate_hypervolume(fitness_arrays, ref_point)
            metrics['hypervolume'].append(hv)
            
            # Calculate spacing
            spacing = calculate_spacing(fitness_arrays)
            metrics['spacing'].append(spacing)
            
            # Calculate IGD (if we have a reference front)
            # For simplicity, we'll use the current non-dominated front as reference
            igd = calculate_igd(fitness_arrays, fitness_arrays)
            metrics['igd'].append(igd)
        else:
            # Placeholder if calculation fails
            if verbose:
                print("Not enough valid fitness values for hypervolume calculation")
            metrics['hypervolume'].append(0.0)
            metrics['spacing'].append(0.0)
            metrics['igd'].append(0.0)
    except Exception as e:
        if verbose:
            print(f"Error calculating diversity metrics: {e}")
        metrics['hypervolume'].append(0.0)
        metrics['spacing'].append(0.0)
        metrics['igd'].append(0.0)
    
    # Calculate fitness statistics
    valid_fitness = [s.fitness[0] if s.fitness else float('inf') for s in population]
    if valid_fitness:
        best_fitness = min(valid_fitness)
        avg_fitness = sum(valid_fitness) / len(valid_fitness)
        metrics['best_fitness'].append(best_fitness)
        metrics['average_fitness'].append(avg_fitness)
    else:
        metrics['best_fitness'].append(0.0)
        metrics['average_fitness'].append(0.0)
    
    # Track constraint violations - FIXED HERE
    try:
        # Use analyze_constraint_violations which already supports populations
        violations = analyze_constraint_violations([s for s in population if hasattr(s, 'assignments')], data)
        metrics['constraint_violations'].append(violations)
    except Exception as e:
        if verbose:
            print(f"Error tracking constraint violations: {e}")
        # Add empty violation data
        metrics['constraint_violations'].append({
            'total_counts': {
                'room_conflicts': 0,
                'time_conflicts': 0,
                'distribution_conflicts': 0,
                'student_conflicts': 0,
                'capacity_violations': 0,
                'total_weighted_score': 0
            }
        })
    
    # Estimate Pareto front size by counting non-dominated solutions
    pareto_front = get_pareto_front(population)
    metrics['pareto_front_size'].append(len(pareto_front))
    
    # Update best solution if needed
    current_best = find_best_solution(population)
    if best_solution is None or (current_best and current_best.fitness and best_solution and 
                                best_solution.fitness and current_best.fitness[0] < best_solution.fitness[0]):
        best_solution = copy.deepcopy(current_best)

def get_pareto_front(population):
    """
    Extract the Pareto front from the population.
    
    Args:
        population: Population of Solution objects
    
    Returns:
        List of non-dominated solutions
    """
    # First, filter out solutions with None fitness
    valid_population = [sol for sol in population if sol.fitness is not None]
    
    # Early return if not enough solutions
    if len(valid_population) <= 1:
        return valid_population
        
    pareto_front = []
    max_check = 500  # Limit the number of solutions to check to avoid very long runtime
    
    for i, solution in enumerate(valid_population[:max_check]):
        dominated = False
        
        # Only check against a reasonable number of other solutions
        for other in valid_population[:max_check]:
            if solution != other and dominates(other, solution):
                dominated = True
                break
                
        if not dominated:
            pareto_front.append(solution)
            
    return pareto_front

def dominates(solution1, solution2):
    """
    Check if solution1 dominates solution2.
    
    Args:
        solution1, solution2: Two Solution objects
    
    Returns:
        True if solution1 dominates solution2, False otherwise
    """
    # Safety checks first
    if (solution1 is None or solution2 is None or 
        solution1.fitness is None or solution2.fitness is None):
        return False
    
    # Make sure both fitness vectors have the same length
    fitness1 = solution1.fitness
    fitness2 = solution2.fitness
    
    if len(fitness1) != len(fitness2):
        # Trim to the shorter length
        min_len = min(len(fitness1), len(fitness2))
        fitness1 = fitness1[:min_len]
        fitness2 = fitness2[:min_len]
    
    # Must be better in at least one objective
    better_in_one = False
    # Must not be worse in any objective
    not_worse = True
    
    for i in range(len(fitness1)):
        # Skip any non-numeric values or NaN
        if not isinstance(fitness1[i], (int, float)) or not isinstance(fitness2[i], (int, float)):
            continue
            
        if fitness1[i] < fitness2[i]:
            better_in_one = True
        elif fitness1[i] > fitness2[i]:
            not_worse = False
            break
    
    return better_in_one and not_worse

def count_fixed_classes(data):
    """Count the number of classes with fixed assignments."""
    fixed_count = 0
    for class_id, class_info in data['classes'].items():
        if class_info.get('fixed', False):
            fixed_count += 1
    return fixed_count

def create_structured_entity(entity_value, entity_type="unknown"):
    """
    Create a structured entity (day, period, room) in the format expected by frontend.
    
    Args:
        entity_value: The entity value, either a dict or a string
        entity_type: Type of entity (for generating default code)
    
    Returns:
        Dictionary with id, name, and code properties
    """
    if isinstance(entity_value, dict):
        return {
            "id": entity_value.get('id', 'UNK'),
            "name": entity_value.get('name', f'Unknown {entity_type.capitalize()}'),
            "code": entity_value.get('code', 'UNK')
        }
    else:
        # It's a string or other type, create a dictionary from it
        str_value = str(entity_value)
        return {
            "id": str_value,
            "name": str_value,
            "code": str_value[:3].upper() if str_value else 'UNK'
        }

def create_room_entity(room_id, room_info):
    """
    Create a structured room entity in the format expected by frontend.
    
    Args:
        room_id: The room ID
        room_info: Room information dictionary
    
    Returns:
        Dictionary with room details
    """
    return {
        "id": room_id,
        "name": room_info.get('name', 'Unknown Room'),
        "capacity": room_info.get('capacity', 0),
        "features": room_info.get('features', []),
        "building": room_info.get('building', 'Unknown Building'),
        "code": room_info.get('code', 'UNK')
    }

def format_solution_for_output(solution, data):
    """
    Format solution for output in a way that the frontend expects.
    
    Args:
        solution: Solution object with assignments
        data: Dictionary with dataset information
    
    Returns:
        Dictionary with properly formatted solution
    """
    formatted_activities = []
    
    # Convert Solution object assignments to structured format
    for class_id, assignment in solution.assignments.items():
        # Make sure we can handle both (time_id, room_id) and (room_id, time_id) formats
        if len(assignment) == 2:
            # Determine which item is the room_id and which is the time_id
            # Typically room_id is a key in data['rooms'] and time_id in data['time_patterns']
            item1, item2 = assignment
            
            if str(item1) in data['rooms']:
                room_id, time_id = item1, item2
            else:
                time_id, room_id = item1, item2
            
            if class_id in data['classes']:
                class_info = data['classes'][class_id]
                
                # Get room information
                room_info = data['rooms'].get(room_id, {})
                room = create_room_entity(room_id, room_info)
                
                # Get time pattern information
                time_info = data['time_patterns'].get(time_id, {})
                
                # Create day and period entities
                day = create_structured_entity(time_info.get('days', 'Unknown Day'), 'day')
                period = create_structured_entity(time_info.get('period', 'Unknown Period'), 'period')
                
                # Create activity
                activity = {
                    "id": class_id,
                    "name": class_info.get('name', 'Unknown Class'),
                    "instructor": class_info.get('instructor', 'Unknown Instructor'),
                    "enrollment": class_info.get('enrollment', 0),
                    "day": day,
                    "period": period,
                    "room": room,
                    "algorithm": "MOEAD"  # Mark source algorithm
                }
                
                formatted_activities.append(activity)
    
    return {"activities": formatted_activities}

def run_muni_optimization(verbose=False):
    """
    Main function to run the MOEA/D optimization for the munifspsspr17 dataset.
    
    Returns:
        Best timetable solution
    """
    # Load dataset with proper path
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'munifspsspr17.json')
    data = load_muni_data(data_path)
    
    if verbose:
        print(f"Dataset loaded. {len(data['classes'])} classes, {len(data['rooms'])} rooms, {len(data['time_patterns'])} time patterns")
        fixed_classes = count_fixed_classes(data)
        print(f"Fixed classes: {fixed_classes}")
    
    # Run MOEA/D algorithm - use proper variable naming for metrics
    population, _, _, metrics = moead_muni(data, verbose=verbose)
    
    # Find the best solution
    best_solution = find_best_solution(population)
    
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
    pareto_front = get_pareto_front(population)
    fronts = [pareto_front]  # List of fronts, where each front is a list of solutions
    
    # Save results with the correctly structured fronts parameter
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moead_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_results(best_solution, fronts, metrics, data, os.path.join(output_dir, "moead_results.json"))
    
    return best_solution, formatted_solution, data, metrics

def save_solution_for_frontend(formatted_solution, algorithm="MOEAD"):
    """
    Save the solution in a format compatible with the frontend application.
    
    Args:
        formatted_solution: Solution formatted for output
        algorithm: Algorithm name used for the solution
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"timetable_{algorithm}_{timestamp}.json")
    
    # Save solution to file
    with open(output_file, 'w') as f:
        json.dump(formatted_solution, f, indent=2)
    
    print(f"Solution saved to {output_file}")

def analyze_solution(solution, data):
    """
    Analyze a solution and return statistics.
    
    Args:
        solution: Solution object
        data: Dataset information
    
    Returns:
        Dictionary with analysis results
    """
    # Evaluate solution
    fitness = evaluate_solution(solution, data)
    
    # Count assignments
    total_classes = len(data['classes'])
    assigned_classes = len(solution.assignments)
    assignment_rate = assigned_classes / total_classes if total_classes > 0 else 0
    
    # Detect conflicts
    _, conflicts = detect_conflicts(solution)  
    
    # Split analysis into smaller functions to reduce complexity
    violations = _analyze_violations(solution, data)
    
    # Use the analyze_constraint_violations function from metrics
    # Wrap the single solution in a list since the function expects a population
    detailed_violations = analyze_constraint_violations([solution], data)
    
    analysis = {
        "fitness": fitness,
        "total_classes": total_classes,
        "assigned_classes": assigned_classes,
        "assignment_rate": assignment_rate,
        "total_conflicts": len(conflicts),
        "room_capacity_violations": violations["room_capacity"],
        "room_feature_violations": violations["room_feature"],
        "instructor_conflicts": violations["instructor"],
        "class_time_violations": violations["class_time"],
        "detailed_violations": detailed_violations
    }
    
    return analysis

def _analyze_violations(solution, data):
    """Helper function to analyze constraint violations to reduce complexity."""
    # Initialize counters
    room_capacity_violations = 0
    room_feature_violations = 0
    instructor_conflicts = 0
    class_time_violations = 0
    
    # Track instructor assignments to detect conflicts
    instructor_assignments = {}
    
    # Count different types of violations
    for class_id, (time_id, room_id) in solution.assignments.items():
        if class_id not in data['classes']:
            continue
            
        class_info = data['classes'][class_id]
        
        # Check room capacity violations
        if _check_room_capacity(class_info, room_id, data):
            room_capacity_violations += 1
        
        # Check room feature violations
        if _check_room_features(class_info, room_id, data):
            room_feature_violations += 1
            
        # Track instructor conflicts
        instructor = class_info.get('instructor')
        if instructor:
            if (instructor, time_id) in instructor_assignments:
                instructor_conflicts += 1
            instructor_assignments[(instructor, time_id)] = class_id
    
    return {
        "room_capacity": room_capacity_violations,
        "room_feature": room_feature_violations,
        "instructor": instructor_conflicts,
        "class_time": class_time_violations
    }

def _check_room_capacity(class_info, room_id, data):
    """Check if there's a room capacity violation."""
    if room_id in data['rooms']:
        room_capacity = data['rooms'][room_id].get('capacity', 0)
        enrollment = class_info.get('enrollment', 0)
        return enrollment > room_capacity
    return False

def _check_room_features(class_info, room_id, data):
    """Check if there's a room feature violation."""
    if 'required_features' in class_info and room_id in data['rooms']:
        required_features = class_info['required_features']
        room_features = data['rooms'][room_id].get('features', [])
        for feature in required_features:
            if feature not in room_features:
                return True
    return False

def copy_solution(solution):
    """
    Create a deep copy of a Solution object.
    
    Args:
        solution: The Solution object to copy
    
    Returns:
        A new Solution object with the same properties
    """
    new_solution = Solution()
    
    # Copy assignments (deep copy to avoid reference issues)
    new_solution.assignments = copy.deepcopy(solution.assignments)
    
    # Copy fitness if available
    if hasattr(solution, 'fitness') and solution.fitness is not None:
        new_solution.fitness = copy.deepcopy(solution.fitness)
    
    # Copy other attributes if they exist
    if hasattr(solution, 'rank'):
        new_solution.rank = solution.rank
    
    if hasattr(solution, 'crowding_distance'):
        new_solution.crowding_distance = solution.crowding_distance
    
    if hasattr(solution, 'generation'):
        new_solution.generation = solution.generation
    
    return new_solution

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run optimization with metrics
    best_solution, formatted_solution, data, metrics = run_muni_optimization(verbose=True)
    
    # Analyze the solution
    analysis_results = analyze_solution(best_solution, data)
    
    # Print analysis results
    print("\nAnalysis Results:")
    print(f"Fitness: {analysis_results['fitness']}")
    print(f"Assignment Rate: {analysis_results['assignment_rate']:.2%} ({analysis_results['assigned_classes']} of {analysis_results['total_classes']} classes assigned)")
    print(f"Total Conflicts: {analysis_results['total_conflicts']}")
    print(f"Room Capacity Violations: {analysis_results['room_capacity_violations']}")
    print(f"Room Feature Violations: {analysis_results['room_feature_violations']}")
    print(f"Instructor Conflicts: {analysis_results['instructor_conflicts']}")
    
    # Print detailed constraint violations
    print("\nDetailed Constraint Violations:")
    for violation_type, count in analysis_results['detailed_violations'].items():
        print(f"{violation_type}: {count}")
    
    # Create a custom solution to demonstrate Solution class usage
    print("\nCreating a custom solution using Solution class:")
    custom_solution = Solution()
    # Copy a few assignments from the best solution to demonstrate
    for i, (class_id, assignment) in enumerate(best_solution.assignments.items()):
        if i < 5:  # Just copy a few for demonstration
            custom_solution.assignments[class_id] = assignment
            print(f"Added assignment for class {class_id}: {assignment}")
    
    # Save the solution for the frontend
    save_solution_for_frontend(formatted_solution)
    
    # Save final metrics as JSON
    with open('moead_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, cls=TupleKeyEncoder)
    
    print("\nOptimization complete. Results saved to output directory.")