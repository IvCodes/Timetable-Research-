"""
Testing NSGA-II algorithm on the munifspsspr17 dataset.

This script adapts the NSGA-II implementation to work with the munifspsspr17.json dataset structure.
It loads the data, runs the optimization, and evaluates the results.
"""
import json
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.spatial import ConvexHull
from collections import defaultdict
import pandas as pd
import os

# Algorithm parameters
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8

# Reference point for hypervolume calculation (worst possible values)
# These will be updated dynamically based on the actual objectives
REFERENCE_POINT = [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')]

class Solution:
    """Class to represent a timetable solution."""
    def __init__(self):
        self.assignments = {}  # Maps class_id to (time_id, room_id)
        self.fitness = None
        self.constraint_violations = None  # Will store detailed constraint violation info
        self.crowding_distance = None  # Crowding distance for diversity preservation
        self.generation = None  # Generation number for dynamic mutation rate
    
    def copy(self):
        """Create a deep copy of the solution."""
        new_solution = Solution()
        new_solution.assignments = deepcopy(self.assignments)
        new_solution.fitness = deepcopy(self.fitness) if self.fitness else None
        new_solution.constraint_violations = deepcopy(self.constraint_violations) if self.constraint_violations else None
        new_solution.crowding_distance = self.crowding_distance
        new_solution.generation = self.generation
        return new_solution

def load_muni_data(file_path):
    """
    Load data from the munifspsspr17.json file.
    
    Returns:
        Dictionary with rooms, courses, classes, time patterns, etc.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print("Parsing munifspsspr17.json dataset...")
    
    # Extract key components
    rooms = data['problem']['rooms']['room']
    courses = data['problem']['courses']['course']
    
    # Process optimization weights
    optimization = data['problem'].get('optimization', {})
    weights = {
        'TIME': int(optimization.get('_time', 25)),
        'ROOM': int(optimization.get('_room', 1)),
        'DISTRIBUTION': int(optimization.get('_distribution', 15)),
        'STUDENT': int(optimization.get('_student', 100))
    }
    
    # Process rooms
    room_data = {}
    for room in rooms:
        room_id = room.get('_id', '')
        room_capacity = int(room.get('_capacity', 0))
        room_data[room_id] = {
            'capacity': room_capacity,
            'travel': {},
            'features': []
        }
        
        # Add travel times
        if 'travel' in room:
            travels = room['travel']
            if not isinstance(travels, list):
                travels = [travels]
            for travel in travels:
                dest_room = travel.get('_room', '')
                value = int(travel.get('_value', 0))
                room_data[room_id]['travel'][dest_room] = value
    
    # Process courses and classes
    class_data = {}
    course_data = {}
    time_patterns = {}
    class_count = 0
    
    # Track relationships between courses and classes
    course_to_classes = {}
    
    # Process all courses and their classes
    for course in courses:
        course_id = course.get('_id', '')
        course_data[course_id] = {
            'configs': []
        }
        
        course_to_classes[course_id] = []
        
        configs = course.get('config', [])
        if not isinstance(configs, list):
            configs = [configs]
        
        for config in configs:
            config_id = config.get('_id', '')
            course_data[course_id]['configs'].append(config_id)
            
            subparts = config.get('subpart', [])
            if not isinstance(subparts, list):
                subparts = [subparts]
            
            for subpart in subparts:
                subpart_id = subpart.get('_id', '')
                
                classes = subpart.get('class', [])
                if not isinstance(classes, list):
                    classes = [classes]
                
                for cls in classes:
                    class_id = cls.get('_id', '')
                    class_limit = int(cls.get('_limit', 0))
                    class_count += 1
                    
                    # Add to course-class relationship
                    course_to_classes[course_id].append(class_id)
                    
                    # Process rooms for this class
                    rooms_for_class = []
                    class_rooms = cls.get('room', [])
                    if not isinstance(class_rooms, list):
                        class_rooms = [class_rooms]
                    
                    for room in class_rooms:
                        if not room:  # Skip if room is None or empty
                            continue
                        room_id = room.get('_id', '')
                        penalty = int(room.get('_penalty', 0))
                        rooms_for_class.append({
                            'room_id': room_id,
                            'penalty': penalty
                        })
                    
                    # Process time patterns for this class
                    times_for_class = []
                    class_times = cls.get('time', [])
                    if not isinstance(class_times, list):
                        class_times = [class_times]
                    
                    for time in class_times:
                        if not time:  # Skip if time is None or empty
                            continue
                        days = time.get('_days', '')
                        start = int(time.get('_start', 0))
                        length = int(time.get('_length', 0))
                        weeks = time.get('_weeks', '')
                        penalty = int(time.get('_penalty', 0)) if '_penalty' in time else 0
                        
                        time_id = f"{days}_{start}_{length}_{weeks}"
                        time_patterns[time_id] = {
                            'days': days,
                            'start': start,
                            'length': length,
                            'weeks': weeks
                        }
                        
                        times_for_class.append({
                            'time_id': time_id,
                            'penalty': penalty
                        })
                    
                    # Store class data
                    class_data[class_id] = {
                        'course_id': course_id,
                        'config_id': config_id,
                        'subpart_id': subpart_id,
                        'limit': class_limit,
                        'rooms': rooms_for_class,
                        'times': times_for_class
                    }
    
    # Track distribution constraints
    distribution_constraints = []
    if 'distributions' in data['problem']:
        distributions = data['problem']['distributions'].get('distribution', [])
        if not isinstance(distributions, list):
            distributions = [distributions]
        
        for dist in distributions:
            constraint = {
                'id': dist.get('_id', ''),
                'type': dist.get('_type', ''),
                'required': dist.get('_required', 'true') == 'true',
                'penalty': int(dist.get('_penalty', 0)),
                'classes': []
            }
            
            # Extract classes involved in this constraint
            classes = dist.get('class', [])
            if not isinstance(classes, list):
                classes = [classes]
                
            for cls in classes:
                if cls and '_id' in cls:
                    constraint['classes'].append(cls.get('_id', ''))
            
            distribution_constraints.append(constraint)
    
    # Process students and enrollments
    students = {}
    if 'students' in data['problem']:
        student_data = data['problem']['students'].get('student', [])
        if not isinstance(student_data, list):
            student_data = [student_data]
            
        for student in student_data:
            student_id = student.get('_id', '')
            students[student_id] = {
                'courses': []
            }
            
            # Get courses this student is enrolled in
            courses = student.get('course', [])
            if not isinstance(courses, list):
                courses = [courses]
                
            for course in courses:
                if course and '_id' in course:
                    students[student_id]['courses'].append(course.get('_id', ''))
    
    # Print summary of loaded data
    print(f"Loaded {len(course_data)} courses, {len(class_data)} classes, {len(room_data)} rooms, {len(students)} students, {len(distribution_constraints)} distributions")
    
    return {
        'rooms': room_data,
        'courses': course_data,
        'classes': class_data,
        'time_patterns': time_patterns,
        'course_to_classes': course_to_classes,
        'distribution_constraints': distribution_constraints,
        'students': students,
        'weights': weights
    }

def evaluate_solution(solution, data):
    """
    Evaluate a solution based on hard and soft constraints.
    
    Args:
        solution: A Solution object
        data: Dictionary with problem data
        
    Returns:
        List of objective values [total_violations, room_conflicts, time_conflicts, 
                                 distribution_conflicts, student_conflicts, capacity_violations]
    """
    # Extract data
    class_data = data.get('class_data', {})
    room_data = data.get('room_data', {})
    distribution_constraints = data.get('distribution_constraints', [])
    course_to_classes = data.get('course_to_classes', {})
    weights = data.get('weights', {
        'ROOM': 1.0,
        'TIME': 1.0,
        'DISTRIBUTION': 1.0,
        'STUDENT': 1.0,
        'CAPACITY': 1.0
    })
    
    # Count conflicts
    room_conflicts = 0
    time_conflicts = 0
    distribution_conflicts = 0
    student_conflicts = 0
    capacity_violations = 0
    
    # Check for room conflicts (same room, same time)
    room_time_assignments = {}
    for class_id, (time_id, room_id) in solution.assignments.items():
        if room_id and time_id:
            key = (room_id, time_id)
            if key in room_time_assignments:
                room_conflicts += 1
            else:
                room_time_assignments[key] = class_id
    
    # Check for time conflicts (same course, same time)
    course_time_assignments = {}
    for class_id, (time_id, _) in solution.assignments.items():
        if time_id and class_id in class_data:
            course_id = class_data[class_id].get('course_id')
            if course_id:
                key = (course_id, time_id)
                if key in course_time_assignments:
                    time_conflicts += 1
                else:
                    course_time_assignments[key] = class_id
    
    # Check distribution constraints
    for constraint in distribution_constraints:
        if constraint.get('required', False):  # Only check hard constraints
            classes = constraint.get('classes', [])
            constraint_type = constraint.get('type')
            
            # Check if all classes in the constraint are assigned
            assigned_classes = [c for c in classes if c in solution.assignments]
            
            if len(assigned_classes) >= 2:  # Need at least 2 classes to check constraints
                # Different constraint types require different checks
                if constraint_type == 'SameTime':
                    # All classes should be at the same time
                    times = [solution.assignments[c][0] for c in assigned_classes if solution.assignments[c][0]]
                    if len(set(times)) > 1:  # More than one unique time
                        distribution_conflicts += 1
                
                elif constraint_type == 'DifferentTime':
                    # All classes should be at different times
                    times = [solution.assignments[c][0] for c in assigned_classes if solution.assignments[c][0]]
                    if len(set(times)) < len(times):  # Duplicate times exist
                        distribution_conflicts += 1
                
                # Add more constraint types as needed
    
    # Check student conflicts (same student, overlapping times)
    # This is a simplified version - in reality, you'd need student enrollment data
    for course_id, class_list in course_to_classes.items():
        assigned_classes = [c for c in class_list if c in solution.assignments]
        for i in range(len(assigned_classes)):
            for j in range(i+1, len(assigned_classes)):
                class1 = assigned_classes[i]
                class2 = assigned_classes[j]
                time1 = solution.assignments[class1][0] if class1 in solution.assignments else None
                time2 = solution.assignments[class2][0] if class2 in solution.assignments else None
                
                if time1 and time2 and time1 == time2:  # Same time slot
                    student_conflicts += 1
    
    # Check room capacity violations
    for class_id, (_, room_id) in solution.assignments.items():
        if room_id and class_id in class_data and room_id in room_data:
            class_limit = class_data[class_id].get('limit', 0)
            room_capacity = room_data[room_id].get('capacity', 0)
            
            if class_limit > room_capacity:
                capacity_violations += 1
    
    # Calculate total weighted violations
    total_violations = (
        weights.get('ROOM', 1.0) * room_conflicts +
        weights.get('TIME', 1.0) * time_conflicts +
        weights.get('DISTRIBUTION', 1.0) * distribution_conflicts +
        weights.get('CAPACITY', 1.0) * capacity_violations +
        weights.get('STUDENT', 1.0) * student_conflicts
    )
    
    # Get detailed constraint violations for analysis
    detailed_violations = track_constraint_violations(solution, data)
    
    # Store detailed constraint violation information
    solution.constraint_violations = {
        'room_conflicts': room_conflicts,
        'time_conflicts': time_conflicts,
        'distribution_conflicts': distribution_conflicts,
        'student_conflicts': student_conflicts,
        'capacity_violations': capacity_violations,
        'total_by_type': {
            'room_conflicts': room_conflicts,
            'time_conflicts': time_conflicts,
            'distribution_conflicts': distribution_conflicts,
            'student_conflicts': student_conflicts,
            'capacity_violations': capacity_violations
        },
        'detailed': detailed_violations
    }
    
    return [
        total_violations,
        room_conflicts,
        time_conflicts,
        distribution_conflicts,
        student_conflicts,
        capacity_violations
    ]

def print_solution_stats(solution, data):
    """
    Print statistics for a solution.
    
    Args:
        solution: A Solution object
        data: Dictionary with problem data
    """
    # Count assigned classes
    total_classes = len(data['classes'])
    assigned_classes = len(solution.assignments)
    assignment_rate = (assigned_classes / total_classes) * 100 if total_classes > 0 else 0
    
    print(f"\nSolution Statistics:")
    print(f"Total Classes: {total_classes}")
    print(f"Assigned Classes: {assigned_classes} ({assignment_rate:.2f}%)")
    print(f"Fitness: {solution.fitness}")
    
    # Count conflicts by type
    print(f"\nConstraint Violations:")
    print(f"Room Conflicts: {solution.constraint_violations['room_conflicts']}")
    print(f"Time Conflicts: {solution.constraint_violations['time_conflicts']}")
    print(f"Distribution Conflicts: {solution.constraint_violations['distribution_conflicts']}")
    print(f"Student Conflicts: {solution.constraint_violations['student_conflicts']}")
    print(f"Capacity Violations: {solution.constraint_violations['capacity_violations']}")
    
    # Calculate total weighted violation score
    print(f"\nTotal Weighted Violation Score: {solution.fitness[0]}")
    
    # Additional statistics
    fixed_classes = 0
    for class_id, class_info in data['classes'].items():
        if len(class_info['rooms']) == 1 and len(class_info['times']) == 1:
            fixed_classes += 1
    
    print(f"\nDataset Statistics:")
    print(f"Courses: {len(data['courses'])}")
    print(f"Classes: {len(data['classes'])}")
    print(f"Fixed Classes: {fixed_classes}")
    print(f"Rooms: {len(data['rooms'])}")
    print(f"Students: {len(data['students'])}")
    print(f"Distribution Constraints: {len(data['distribution_constraints'])}")

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

def fast_nondominated_sort(population):
    """
    Perform non-dominated sorting on a population of Solution objects.
    
    Args:
        population: List of Solution objects
        
    Returns:
        List of fronts, where each front is a list of Solution objects
    """
    fronts = [[]]
    dominated_solutions = {i: [] for i in range(len(population))}
    domination_counts = [0] * len(population)
    
    # For each solution, determine which solutions it dominates
    # and which solutions dominate it
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
        
        # If no solution dominates i, it belongs to the first front
        if domination_counts[i] == 0:
            fronts[0].append(population[i])
    
    # Find subsequent fronts
    front_index = 0
    while fronts[front_index]:
        next_front = []
        
        for solution_index, solution in enumerate(population):
            if solution in fronts[front_index]:
                for dominated_index in dominated_solutions[solution_index]:
                    domination_counts[dominated_index] -= 1
                    if domination_counts[dominated_index] == 0:
                        next_front.append(population[dominated_index])
        
        front_index += 1
        fronts.append(next_front)
    
    # Remove the empty front at the end
    return fronts[:-1]

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
    
    num_solutions = len(front)
    
    # Initialize crowding distance for all solutions
    for solution in front:
        solution.crowding_distance = 0
    
    # Calculate crowding distance for each objective
    num_objectives = len(front[0].fitness)
    
    for objective_index in range(num_objectives):
        # Sort by current objective
        front.sort(key=lambda sol: sol.fitness[objective_index])
        
        # Set boundary points to infinity
        front[0].crowding_distance = float('inf')
        front[num_solutions - 1].crowding_distance = float('inf')
        
        # Calculate crowding distance for middle points
        if num_solutions > 2:
            min_fitness = front[0].fitness[objective_index]
            max_fitness = front[num_solutions - 1].fitness[objective_index]
            
            # Avoid division by zero
            if max_fitness == min_fitness:
                continue
            
            # Calculate crowding distance
            for i in range(1, num_solutions - 1):
                next_fitness = front[i + 1].fitness[objective_index]
                prev_fitness = front[i - 1].fitness[objective_index]
                front[i].crowding_distance += (next_fitness - prev_fitness) / (max_fitness - min_fitness)

def get_valid_rooms(data, class_id):
    """
    Get valid rooms for a class.
    
    Args:
        data: Dictionary with problem data
        class_id: ID of the class
    
    Returns:
        List of valid room IDs
    """
    class_info = data['classes'].get(class_id, {})
    valid_rooms = []
    
    for room in class_info.get('rooms', []):
        room_id = room.get('room_id')
        if room_id and room_id in data['rooms']:
            valid_rooms.append(room_id)
    
    return valid_rooms

def get_valid_times(data, class_id, room_id, solution):
    """
    Get valid times for a class in a specific room.
    
    Args:
        data: Dictionary with problem data
        class_id: ID of the class
        room_id: ID of the room
        solution: Current solution (to check for conflicts)
    
    Returns:
        List of valid time IDs
    """
    class_info = data['classes'].get(class_id, {})
    valid_times = []
    
    for time in class_info.get('times', []):
        time_id = time.get('time_id')
        if not time_id:
            continue
            
        # Check for conflicts with other classes in the same room
        conflict = False
        for other_class, assignment in solution.assignments.items():
            if other_class != class_id and assignment and len(assignment) >= 2:
                if assignment[0] == room_id and assignment[1] == time_id:
                    conflict = True
                    break
        
        if not conflict:
            valid_times.append(time_id)
    
    return valid_times

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
    # Dynamically adjust mutation based on generation
    if hasattr(solution, 'generation'):
        # Increase mutation rate for older solutions to promote diversity
        generation_factor = min(0.5, solution.generation / 20.0)
        mutation_rate = mutation_rate + generation_factor
    
    # Get list of all class IDs
    class_ids = list(data['classes'].keys())
    
    # Mutate existing assignments
    for class_id in list(solution.assignments.keys()):
        if random.random() < mutation_rate:
            # Decide whether to remove or reassign
            if random.random() < mutation_strength * 0.3:  # 30% chance to remove
                del solution.assignments[class_id]
            else:
                # Reassign to a different room or time
                valid_rooms = get_valid_rooms(data, class_id)
                
                if valid_rooms:
                    room_id = random.choice(valid_rooms)
                    
                    # Copy the solution temporarily to avoid conflicts with current class
                    temp_solution = solution.copy()
                    if class_id in temp_solution.assignments:
                        del temp_solution.assignments[class_id]
                    
                    valid_times = get_valid_times(data, class_id, room_id, temp_solution)
                    if valid_times:
                        solution.assignments[class_id] = (room_id, random.choice(valid_times))
    
    # Add new assignments for unassigned classes
    unassigned_classes = [c_id for c_id in class_ids if c_id not in solution.assignments]
    num_to_add = int(len(unassigned_classes) * mutation_rate * mutation_strength)
    
    for class_id in random.sample(unassigned_classes, min(num_to_add, len(unassigned_classes))):
        valid_rooms = get_valid_rooms(data, class_id)
        if valid_rooms:
            room_id = random.choice(valid_rooms)
            valid_times = get_valid_times(data, class_id, room_id, solution)
            if valid_times:
                solution.assignments[class_id] = (room_id, random.choice(valid_times))

def nsga2_muni(data, population_size=POPULATION_SIZE, max_generations=NUM_GENERATIONS, crossover_rate=0.8, mutation_rate=0.2, verbose=False):
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
    # Initialize population
    population = initialize_population(data, population_size)
    
    # Evaluate initial population
    evaluate_population(population, data)
    
    # Fast non-dominated sort
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
    
    # Track best solution
    best_solution = None
    
    # Reference point for hypervolume calculation
    # Use worst possible fitness values based on problem constraints
    reference_point = [1000, 100, 100, 100, 100, 100]  # High values for each objective
    
    # Main loop
    for generation in range(max_generations):
        if verbose:
            print(f"\nGeneration {generation+1}/{max_generations}")
        
        # Calculate crowding distance for each front
        for front in fronts:
            crowding_distance_assignment(front)
        
        # Create combined population (current population + offspring)
        combined_population = []
        
        # Create offspring through selection, crossover, and mutation
        offspring = []
        
        # Adaptive tournament size based on generation progress
        # Start with smaller tournaments for more exploration, gradually increase for exploitation
        base_tournament_size = 2
        max_tournament_size = 4
        tournament_size = base_tournament_size + int((max_tournament_size - base_tournament_size) * (generation / max_generations))
        
        # Select parents and create offspring
        for _ in range(0, population_size, 2):
            # Select parents using tournament selection
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            
            # Determine adaptive crossover rate based on population diversity
            adaptive_crossover_rate = crossover_rate
            if len(metrics['spacing']) > 0:
                current_spacing = metrics['spacing'][-1]
                if current_spacing < 10:  # Low diversity
                    adaptive_crossover_rate = min(crossover_rate + 0.2, 0.95)  # Increase crossover rate
            
            # Perform crossover with adaptive rate
            if random.random() < adaptive_crossover_rate:
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
            
            # Add children to offspring
            offspring.append(child1)
            offspring.append(child2)
        
        # Ensure we have exactly population_size offspring
        offspring = offspring[:population_size]
        
        # Evaluate offspring
        evaluate_population(offspring, data)
        
        # Combine parent and offspring populations
        combined_population = population + offspring
        
        # Non-dominated sorting on combined population
        combined_fronts = fast_nondominated_sort(combined_population)
        
        # Select the best solutions for the next generation
        next_population = []
        front_index = 0
        
        # Add complete fronts until we reach population_size
        while len(next_population) + len(combined_fronts[front_index]) <= population_size:
            # Add all solutions from this front
            for solution in combined_fronts[front_index]:
                next_population.append(solution)
            front_index += 1
            
            # Break if we've added all fronts
            if front_index >= len(combined_fronts):
                break
        
        # If we have room for more solutions, add solutions from the next front
        # based on crowding distance
        if len(next_population) < population_size and front_index < len(combined_fronts):
            # Calculate crowding distance for the current front
            crowding_distance_assignment(combined_fronts[front_index])
            
            # Sort by crowding distance (descending)
            current_front = sorted(combined_fronts[front_index], 
                                   key=lambda x: float('-inf') if x.crowding_distance is None else x.crowding_distance, 
                                   reverse=True)
            
            # Add solutions until we reach population_size
            for solution in current_front:
                if len(next_population) >= population_size:
                    break
                next_population.append(solution)
        
        # Diversity preservation: if we're stagnating, inject random solutions
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
        
        # Update population
        population = next_population
        
        # Perform non-dominated sorting to update fronts
        fronts = fast_nondominated_sort(population)
        
        # Track metrics
        if len(fronts) > 0:
            # Calculate hypervolume
            pareto_front_fitness = [s.fitness for s in fronts[0]]
            hypervolume_value = calculate_hypervolume(pareto_front_fitness, reference_point)
            metrics['hypervolume'].append(hypervolume_value)
            
            # Calculate spacing
            spacing_value = calculate_spacing(pareto_front_fitness)
            metrics['spacing'].append(spacing_value)
            
            # Track Pareto front size
            metrics['pareto_front_size'].append(len(fronts[0]))
            
            # Analyze constraint violations
            violations = analyze_constraint_violations(population, data)
            metrics['constraint_violations'].append(violations)
            
            # Find best solution in current population
            current_best = find_best_solution(population)
            if best_solution is None or dominates(current_best, best_solution):
                best_solution = current_best.copy()
            
            # Track fitness values
            metrics['best_fitness'].append(min([sum(s.fitness) for s in population]))
            metrics['average_fitness'].append(sum([sum(s.fitness) for s in population]) / len(population))
            
            if verbose:
                print(f"  Best fitness: {metrics['best_fitness'][-1]:.2f}")
                print(f"  Pareto front size: {metrics['pareto_front_size'][-1]}")
                print(f"  Hypervolume: {metrics['hypervolume'][-1]:.2f}")
                print(f"  Constraint violations: {violations['total']}")
    
    # Return best solution, final fronts, and metrics
    if best_solution is None and len(population) > 0:
        best_solution = find_best_solution(population)
    
    return best_solution, fronts, metrics

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
    
    # Determine crossover strategy based on parents' similarity
    common_assignments = set(parent1.assignments.keys()) & set(parent2.assignments.keys())
    similarity = len(common_assignments) / len(all_classes) if all_classes else 0
    
    if similarity > 0.8:
        # Parents are very similar - use uniform crossover with higher mixing rate
        mixing_rate = 0.7
    elif similarity > 0.5:
        # Parents moderately similar - use standard uniform crossover
        mixing_rate = 0.5
    else:
        # Parents quite different - use more conservative mixing
        mixing_rate = 0.3
    
    # Enhanced uniform crossover with partial schedule repair
    for class_id in all_classes:
        if class_id in parent1.assignments and class_id in parent2.assignments:
            # Both parents have this class assigned
            if random.random() < mixing_rate:
                child1.assignments[class_id] = parent1.assignments[class_id]
                child2.assignments[class_id] = parent2.assignments[class_id]
            else:
                child1.assignments[class_id] = parent2.assignments[class_id]
                child2.assignments[class_id] = parent1.assignments[class_id]
        elif class_id in parent1.assignments:
            # Only parent1 has this class assigned
            if random.random() < 0.5:
                child1.assignments[class_id] = parent1.assignments[class_id]
            else:
                child2.assignments[class_id] = parent1.assignments[class_id]
        else:
            # Only parent2 has this class assigned
            if random.random() < 0.5:
                child1.assignments[class_id] = parent2.assignments[class_id]
            else:
                child2.assignments[class_id] = parent2.assignments[class_id]
    
    # Schedule repair to handle conflicts
    repair_schedule(child1, data)
    repair_schedule(child2, data)
    
    return child1, child2

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
    room_time_assignments = {}  # (room_id, time_id) -> class_id
    conflicts = set()
    
    for class_id, assignment in solution.assignments.items():
        # Defensive programming: Make sure assignment exists and has right structure
        if not assignment or len(assignment) < 2:
            continue
            
        # Unpack assignment safely
        try:
            room_id, time_id = assignment
        except (ValueError, TypeError):
            continue  # Skip if assignment is not properly structured
        
        if not room_id or not time_id:
            continue  # Skip if room_id or time_id is None or empty
            
        # Check for room conflicts
        key = (room_id, time_id)
        if key in room_time_assignments:
            # Room conflict found
            conflicting_class = room_time_assignments[key]
            conflicts.add(class_id)
            conflicts.add(conflicting_class)
        else:
            room_time_assignments[key] = class_id
    
    # Resolve conflicts by removing random assignments
    for class_id in conflicts:
        if random.random() < 0.5 and class_id in solution.assignments:
            # Try to reassign to a different time/room
            current_room, current_time = solution.assignments[class_id]
            
            # Get valid alternatives
            valid_rooms = get_valid_rooms(data, class_id)
            
            # Temporarily remove this class to avoid detecting conflicts with itself
            temp_solution = solution.copy()
            if class_id in temp_solution.assignments:
                del temp_solution.assignments[class_id]
            
            # Try to find an alternative assignment
            found_alternative = False
            for room_id in valid_rooms:
                valid_times = get_valid_times(data, class_id, room_id, temp_solution)
                if valid_times:
                    # Found a valid alternative
                    solution.assignments[class_id] = (room_id, random.choice(valid_times))
                    found_alternative = True
                    break
            
            if not found_alternative:
                # Couldn't find a valid alternative, remove the assignment
                del solution.assignments[class_id]
        else:
            # Remove this assignment
            if class_id in solution.assignments:
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

def calculate_hypervolume(front, reference_point=None):
    """
    Calculate the hypervolume indicator for a Pareto front.
    
    Args:
        front: List of fitness values for solutions in the Pareto front
        reference_point: Reference point for hypervolume calculation (worst possible values)
        
    Returns:
        Hypervolume value
    """
    if not front:
        return 0.0
    
    # Use provided reference point or update the global one
    if reference_point is None:
        reference_point = REFERENCE_POINT
    
    # Convert to numpy array for easier manipulation
    points = np.array(front)
    
    # Check which dimensions have variation
    dimension_ranges = np.ptp(points, axis=0)
    active_dimensions = dimension_ranges > 1e-10
    
    # If we have less than 2 active dimensions, use a simple approach
    if np.sum(active_dimensions) < 2:
        # Use product of ranges as an approximate hypervolume
        hypervolume = 1.0
        for i in range(points.shape[1]):
            dimension_min = np.min(points[:, i])
            hypervolume *= (reference_point[i] - dimension_min)
        return hypervolume
    
    # For 2D fronts, use a simpler calculation
    if points.shape[1] == 2:
        # Sort points by first objective
        sorted_points = points[points[:, 0].argsort()]
        
        # Calculate hypervolume as sum of rectangles
        hypervolume = 0.0
        for i in range(len(sorted_points)):
            if i == 0:
                # First point forms rectangle with reference point
                width = reference_point[0] - sorted_points[i, 0]
                height = reference_point[1] - sorted_points[i, 1]
            else:
                # Other points form rectangles with previous point's y-value
                width = sorted_points[i-1, 0] - sorted_points[i, 0]
                height = reference_point[1] - sorted_points[i, 1]
            
            if width > 0 and height > 0:
                hypervolume += width * height
        
        return hypervolume
    
    # For higher dimensions, use a custom calculation approach
    try:
        # Only keep active dimensions to avoid qhull errors
        active_dim_indices = np.where(active_dimensions)[0]
        
        if len(active_dim_indices) < 2:  # Need at least 2 dimensions
            # Fall back to simple approach
            hypervolume = 1.0
            for i in range(points.shape[1]):
                dimension_min = np.min(points[:, i])
                hypervolume *= (reference_point[i] - dimension_min)
            return hypervolume
        
        # Filter points to only include active dimensions
        filtered_points = points[:, active_dim_indices]
        filtered_reference = np.array(reference_point)[active_dim_indices]
        
        # Calculate dominated hypervolume using Monte Carlo sampling
        # This is an approximation but more robust than qhull for degenerate cases
        n_samples = 10000
        count_dominated = 0
        
        # Define bounds for sampling
        lower_bounds = np.min(filtered_points, axis=0)
        upper_bounds = filtered_reference
        
        # Generate random points within the reference volume
        samples = np.random.uniform(
            low=lower_bounds,
            high=upper_bounds,
            size=(n_samples, len(active_dim_indices))
        )
        
        # Count points dominated by the Pareto front
        for sample in samples:
            # A sample is dominated if any point in the front dominates it
            is_dominated = False
            for point in filtered_points:
                # Check if point dominates sample
                if np.all(point <= sample) and np.any(point < sample):
                    is_dominated = True
                    break
            
            if is_dominated:
                count_dominated += 1
        
        # Calculate hypervolume as proportion of dominated points times reference volume
        dominated_ratio = count_dominated / n_samples
        reference_volume = np.prod(upper_bounds - lower_bounds)
        estimated_hypervolume = dominated_ratio * reference_volume
        
        return estimated_hypervolume
        
    except Exception as e:
        print(f"Error calculating hypervolume: {e}")
        # Fallback: use product of ranges in each dimension
        hypervolume = 1.0
        for i in range(points.shape[1]):
            dimension_min = np.min(points[:, i])
            hypervolume *= (reference_point[i] - dimension_min)
        return hypervolume

def calculate_spacing(front):
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

def calculate_igd(front, reference_front):
    """
    Calculate the Inverted Generational Distance (IGD) metric.
    
    IGD measures how far the approximated Pareto front is from the true Pareto front.
    Lower values indicate better convergence to the true front.
    
    Args:
        front: List of fitness values for solutions in the approximated Pareto front
        reference_front: List of fitness values for solutions in the true/reference Pareto front
        
    Returns:
        IGD value
    """
    if not front or not reference_front:
        return float('inf')
    
    # Convert to numpy arrays
    points = np.array(front)
    ref_points = np.array(reference_front)
    
    # Calculate minimum distance from each reference point to any point in the front
    total_dist = 0.0
    for ref_point in ref_points:
        min_dist = float('inf')
        for point in points:
            # Euclidean distance
            dist = np.sqrt(np.sum((ref_point - point)**2))
            min_dist = min(min_dist, dist)
        total_dist += min_dist
    
    # Average distance
    igd = total_dist / len(ref_points)
    
    return igd

def track_constraint_violations(solution, data):
    """
    Track detailed constraint violations in a solution.
    
    Args:
        solution: A Solution object
        data: Dictionary with problem data
    
    Returns:
        Dictionary with constraint violation details
    """
    violations = {
        'room_conflicts': {},
        'time_conflicts': {},
        'distribution_conflicts': {},
        'student_conflicts': {},
        'capacity_violations': {},
        'total_by_type': {
            'room_conflicts': 0,
            'time_conflicts': 0,
            'distribution_conflicts': 0,
            'student_conflicts': 0,
            'capacity_violations': 0
        }
    }
    
    # Get class data for quick lookups
    class_data = data.get('classes', {})
    distribution_constraints = data.get('distribution_constraints', [])
    
    # Check for room conflicts (same room, same time)
    room_time_assignments = {}  # (room_id, time_id) -> class_id
    
    for class_id, assignment in solution.assignments.items():
        # Defensive programming: Make sure assignment exists and has right structure
        if not assignment or len(assignment) < 2:
            continue
            
        # Unpack assignment safely
        try:
            room_id, time_id = assignment
        except (ValueError, TypeError):
            continue  # Skip if assignment is not properly structured
        
        if not room_id or not time_id:
            continue  # Skip if room_id or time_id is None or empty
            
        # Check for room conflicts
        key = (room_id, time_id)
        if key in room_time_assignments:
            # Room conflict found
            conflicting_class = room_time_assignments[key]
            violations['room_conflicts'][(class_id, conflicting_class)] = 1
            violations['total_by_type']['room_conflicts'] += 1
        else:
            room_time_assignments[key] = class_id
    
    # Check for time conflicts (same course, same time)
    course_time_assignments = {}
    for class_id, assignment in solution.assignments.items():
        # Defensive programming: Make sure assignment exists and has right structure
        if not assignment or len(assignment) < 2:
            continue
            
        # Unpack assignment safely
        try:
            room_id, time_id = assignment
        except (ValueError, TypeError):
            continue  # Skip if assignment is not properly structured
            
        if not time_id or class_id not in class_data:
            continue  # Skip if time_id is None or class not found
            
        course_id = class_data[class_id].get('course_id')
        if course_id:
            key = (course_id, time_id)
            if key in course_time_assignments:
                # Time conflict found
                conflicting_class = course_time_assignments[key]
                violations['time_conflicts'][(class_id, conflicting_class)] = 1
                violations['total_by_type']['time_conflicts'] += 1
            else:
                course_time_assignments[key] = class_id
    
    # Check for distribution constraints
    for constraint in distribution_constraints:
        constraint_type = constraint.get('type')
        classes = constraint.get('classes', [])
        
        # Defensive programming: Skip constraints without required fields
        if not constraint_type or not classes:
            continue
            
        # Convert class IDs to strings to ensure consistent lookup
        class_ids = [str(c_id) for c in classes for c_id in ([c.get('class_id')] if isinstance(c, dict) else [c])]
        
        # Only check constraints for classes that are assigned
        assigned_classes = [c_id for c_id in class_ids if c_id in solution.assignments]
        
        # For SameTime constraint
        if constraint_type == 'SameTime' and len(assigned_classes) > 1:
            times = set()
            for c_id in assigned_classes:
                assignment = solution.assignments.get(c_id)
                if assignment and len(assignment) >= 2:
                    times.add(assignment[1])  # time_id is the second element
            
            if len(times) > 1:  # Classes are not at the same time
                for i in range(len(assigned_classes)):
                    for j in range(i+1, len(assigned_classes)):
                        violations['distribution_conflicts'][(assigned_classes[i], assigned_classes[j])] = 1
                        violations['total_by_type']['distribution_conflicts'] += 1
        
        # For DifferentTime constraint
        elif constraint_type == 'DifferentTime' and len(assigned_classes) > 1:
            time_classes = {}
            for c_id in assigned_classes:
                assignment = solution.assignments.get(c_id)
                if assignment and len(assignment) >= 2:
                    time_id = assignment[1]
                    if time_id in time_classes:
                        time_classes[time_id].append(c_id)
                    else:
                        time_classes[time_id] = [c_id]
            
            for time_id, classes in time_classes.items():
                if len(classes) > 1:  # Multiple classes at same time
                    for i in range(len(classes)):
                        for j in range(i+1, len(classes)):
                            violations['distribution_conflicts'][(classes[i], classes[j])] = 1
                            violations['total_by_type']['distribution_conflicts'] += 1
    
    # Check for student conflicts
    student_enrollments = data.get('student_enrollments', {})
    for student_id, enrolled_classes in student_enrollments.items():
        assigned_classes = [c_id for c_id in enrolled_classes if c_id in solution.assignments]
        
        # Check pairs of classes
        for i in range(len(assigned_classes)):
            for j in range(i+1, len(assigned_classes)):
                class1 = assigned_classes[i]
                class2 = assigned_classes[j]
                
                # Safely get assignments
                assignment1 = solution.assignments.get(class1)
                assignment2 = solution.assignments.get(class2)
                
                # Defensive programming: Make sure assignments exist and have right structure
                if not assignment1 or not assignment2 or len(assignment1) < 2 or len(assignment2) < 2:
                    continue
                    
                time1 = assignment1[1]  # time_id is the second element
                time2 = assignment2[1]
                
                if time1 and time2 and time1 == time2:  # Same time slot
                    violations['student_conflicts'][(class1, class2)] = 1
                    violations['total_by_type']['student_conflicts'] += 1
    
    # Check for capacity violations
    rooms = data.get('rooms', {})
    for class_id, assignment in solution.assignments.items():
        # Defensive programming: Make sure assignment exists and has right structure
        if not assignment or len(assignment) < 2:
            continue
            
        room_id = assignment[0]
        if room_id in rooms and class_id in class_data:
            room_capacity = rooms[room_id].get('capacity', 0)
            enrolled_students = len(class_data[class_id].get('students', []))
            
            if enrolled_students > room_capacity:
                violations['capacity_violations'][class_id] = enrolled_students - room_capacity
                violations['total_by_type']['capacity_violations'] += 1
    
    # Calculate total weighted violation score
    total_weighted_score = (
        violations['total_by_type']['room_conflicts'] * 10 +
        violations['total_by_type']['time_conflicts'] * 20 +
        violations['total_by_type']['distribution_conflicts'] * 10 +
        violations['total_by_type']['student_conflicts'] * 5 +
        violations['total_by_type']['capacity_violations'] * 2
    )
    
    # Add the total counts for easy access
    violations['total_counts'] = {
        'room_conflicts': violations['total_by_type']['room_conflicts'],
        'time_conflicts': violations['total_by_type']['time_conflicts'],
        'distribution_conflicts': violations['total_by_type']['distribution_conflicts'],
        'student_conflicts': violations['total_by_type']['student_conflicts'],
        'capacity_violations': violations['total_by_type']['capacity_violations'],
        'total_weighted_score': total_weighted_score
    }
    
    return violations

def analyze_constraint_violations(population, data):
    """
    Analyze constraint violations across the population.
    
    Args:
        population: List of Solution objects
        data: Dictionary with problem data
        
    Returns:
        Dictionary with constraint violation statistics
    """
    # Initialize statistics
    violation_stats = {
        'room_conflicts': {
            'min': float('inf'),
            'max': 0,
            'avg': 0,
            'total': 0
        },
        'time_conflicts': {
            'min': float('inf'),
            'max': 0,
            'avg': 0,
            'total': 0
        },
        'distribution_conflicts': {
            'min': float('inf'),
            'max': 0,
            'avg': 0,
            'total': 0
        },
        'student_conflicts': {
            'min': float('inf'),
            'max': 0,
            'avg': 0,
            'total': 0
        },
        'capacity_violations': {
            'min': float('inf'),
            'max': 0,
            'avg': 0,
            'total': 0
        },
        'total': {
            'min': float('inf'),
            'max': 0,
            'avg': 0,
            'total': 0
        }
    }
    
    # If population is empty, return default values
    if not population:
        for key in violation_stats:
            violation_stats[key]['min'] = 0
        return violation_stats
    
    # Track violations for each solution
    for solution in population:
        # Track constraint violations for this solution
        violations = track_constraint_violations(solution, data)
        
        # Extract totals for each violation type
        room_conflicts = violations['total_by_type']['room_conflicts']
        time_conflicts = violations['total_by_type']['time_conflicts']
        distribution_conflicts = violations['total_by_type']['distribution_conflicts']
        student_conflicts = violations['total_by_type']['student_conflicts']
        capacity_violations = violations['total_by_type']['capacity_violations']
        
        # Calculate total weighted violations
        total_weighted = violations.get('total_counts', {}).get('total_weighted_score', 
            room_conflicts * 10 + 
            time_conflicts * 20 + 
            distribution_conflicts * 10 + 
            student_conflicts * 5 + 
            capacity_violations * 2)
        
        # Update room conflicts stats
        violation_stats['room_conflicts']['min'] = min(violation_stats['room_conflicts']['min'], room_conflicts)
        violation_stats['room_conflicts']['max'] = max(violation_stats['room_conflicts']['max'], room_conflicts)
        violation_stats['room_conflicts']['total'] += room_conflicts
        
        # Update time conflicts stats
        violation_stats['time_conflicts']['min'] = min(violation_stats['time_conflicts']['min'], time_conflicts)
        violation_stats['time_conflicts']['max'] = max(violation_stats['time_conflicts']['max'], time_conflicts)
        violation_stats['time_conflicts']['total'] += time_conflicts
        
        # Update distribution conflicts stats
        violation_stats['distribution_conflicts']['min'] = min(violation_stats['distribution_conflicts']['min'], distribution_conflicts)
        violation_stats['distribution_conflicts']['max'] = max(violation_stats['distribution_conflicts']['max'], distribution_conflicts)
        violation_stats['distribution_conflicts']['total'] += distribution_conflicts
        
        # Update student conflicts stats
        violation_stats['student_conflicts']['min'] = min(violation_stats['student_conflicts']['min'], student_conflicts)
        violation_stats['student_conflicts']['max'] = max(violation_stats['student_conflicts']['max'], student_conflicts)
        violation_stats['student_conflicts']['total'] += student_conflicts
        
        # Update capacity violations stats
        violation_stats['capacity_violations']['min'] = min(violation_stats['capacity_violations']['min'], capacity_violations)
        violation_stats['capacity_violations']['max'] = max(violation_stats['capacity_violations']['max'], capacity_violations)
        violation_stats['capacity_violations']['total'] += capacity_violations
        
        # Update total stats
        violation_stats['total']['min'] = min(violation_stats['total']['min'], total_weighted)
        violation_stats['total']['max'] = max(violation_stats['total']['max'], total_weighted)
        violation_stats['total']['total'] += total_weighted
    
    # Calculate averages
    n = len(population)
    violation_stats['room_conflicts']['avg'] = violation_stats['room_conflicts']['total'] / n
    violation_stats['time_conflicts']['avg'] = violation_stats['time_conflicts']['total'] / n
    violation_stats['distribution_conflicts']['avg'] = violation_stats['distribution_conflicts']['total'] / n
    violation_stats['student_conflicts']['avg'] = violation_stats['student_conflicts']['total'] / n
    violation_stats['capacity_violations']['avg'] = violation_stats['capacity_violations']['total'] / n
    violation_stats['total']['avg'] = violation_stats['total']['total'] / n
    
    # Handle case where there were no violations (min was not updated)
    for key in violation_stats:
        if violation_stats[key]['min'] == float('inf'):
            violation_stats[key]['min'] = 0
    
    return violation_stats

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

class TupleKeyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles dictionaries with tuple keys by converting them to strings.
    Also handles any other non-serializable types by converting them to strings.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)
            
    def encode(self, obj):
        def _preprocess_for_json(o):
            if isinstance(o, dict):
                # Process dictionaries
                result = {}
                for k, v in o.items():
                    # Convert tuple keys to strings
                    if isinstance(k, tuple):
                        new_key = str(k)
                    else:
                        new_key = k
                    
                    # Recursively process values
                    result[new_key] = _preprocess_for_json(v)
                return result
            elif isinstance(o, (list, tuple)):
                # Process lists and tuples
                return [_preprocess_for_json(item) for item in o]
            elif hasattr(o, 'keys') and callable(getattr(o, 'keys')):
                # Handle dict-like objects that aren't directly instance of dict
                result = {}
                for k in o.keys():
                    if isinstance(k, tuple):
                        result[str(k)] = _preprocess_for_json(o[k])
                    else:
                        result[k] = _preprocess_for_json(o[k])
                return result
            else:
                # Return other objects as is
                return o
        
        return super().encode(_preprocess_for_json(obj))

def save_results(best_solution, fronts, metrics, data, output_file='nsga2_muni_results.json'):
    """
    Save the optimization results to a JSON file.
    
    Args:
        best_solution: The best Solution object
        fronts: List of fronts (lists of Solution objects)
        metrics: Dictionary with performance metrics
        data: Dictionary with problem data
        output_file: Path to output JSON file
    """
    # Convert complex datastructures to JSON-serializable format
    
    # Convert the best solution
    best_solution_dict = {
        'fitness': best_solution.fitness,
        'assignments': best_solution.assignments,
        'num_assigned': len(best_solution.assignments),
        'crowding_distance': best_solution.crowding_distance
    }
    
    # Process constraint violations safely
    if hasattr(best_solution, 'constraint_violations') and best_solution.constraint_violations:
        violations_dict = {}
        for key, value in best_solution.constraint_violations.items():
            # Handle nested dictionaries with tuple keys
            if isinstance(value, dict):
                processed_dict = {}
                for sub_key, sub_value in value.items():
                    # Convert tuple keys to strings
                    if isinstance(sub_key, tuple):
                        processed_dict[str(sub_key)] = sub_value
                    else:
                        processed_dict[sub_key] = sub_value
                violations_dict[key] = processed_dict
            else:
                violations_dict[key] = value
        best_solution_dict['constraint_violations'] = violations_dict
    
    # Convert the Pareto fronts
    pareto_fronts = []
    for i, front in enumerate(fronts):
        front_dict = []
        for solution in front:
            solution_dict = {
                'fitness': solution.fitness,
                'num_assigned': len(solution.assignments),
                'crowding_distance': solution.crowding_distance
            }
            front_dict.append(solution_dict)
        pareto_fronts.append(front_dict)
    
    # Convert metric history to lists
    metric_history = {}
    for metric, values in metrics.items():
        if isinstance(values, list):
            metric_history[metric] = values
    
    # Create final results dictionary
    results = {
        'best_solution': best_solution_dict,
        'pareto_fronts': pareto_fronts,
        'metrics': metric_history,
        'dataset': {
            'num_courses': len(data.get('courses', {})),
            'num_classes': len(data.get('classes', {})),
            'num_rooms': len(data.get('rooms', {})),
            'num_students': data.get('num_students', 0),
            'num_distribution_constraints': len(data.get('distribution_constraints', []))
        }
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, cls=TupleKeyEncoder)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def plot_metrics(metrics):
    """Plot the metrics from the optimization."""
    plt.figure(figsize=(15, 10))
    
    # Plot hypervolume
    plt.subplot(2, 2, 1)
    plt.plot(metrics['hypervolume'])
    plt.title('Hypervolume')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    
    # Plot spacing
    plt.subplot(2, 2, 2)
    plt.plot(metrics['spacing'])
    plt.title('Spacing')
    plt.xlabel('Generation')
    plt.ylabel('Spacing')
    
    # Plot IGD
    plt.subplot(2, 2, 3)
    plt.plot(metrics['igd'])
    plt.title('Inverted Generational Distance')
    plt.xlabel('Generation')
    plt.ylabel('IGD')
    
    # Plot fitness
    plt.subplot(2, 2, 4)
    plt.plot(metrics['best_fitness'], label='Best')
    plt.plot(metrics['average_fitness'], label='Average')
    plt.title('Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (lower is better)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('nsga2_metrics.png')
    plt.close()
    
    # Plot constraint violations
    plt.figure(figsize=(15, 8))
    violations = [v['total_counts'] for v in metrics['constraint_violations']]
    
    for violation_type in ['room_conflicts', 'time_conflicts', 'distribution_conflicts', 
                          'student_conflicts', 'capacity_violations']:
        values = [v[violation_type] for v in violations]
        plt.plot(values, label=violation_type)
    
    plt.title('Constraint Violations')
    plt.xlabel('Generation')
    plt.ylabel('Number of Violations')
    plt.legend()
    plt.tight_layout()
    plt.savefig('nsga2_violations.png')
    plt.close()
    
    # Plot Pareto front size
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['pareto_front_size'])
    plt.title('Pareto Front Size')
    plt.xlabel('Generation')
    plt.ylabel('Number of Solutions')
    plt.tight_layout()
    plt.savefig('nsga2_pareto_size.png')
    plt.close()

def plot_constraint_violations(violations):
    """
    Plot the constraint violations over generations.
    
    Args:
        violations: List of dictionaries containing violation counts per generation
    """
    plt.figure(figsize=(15, 8))
    
    for violation_type in ['room_conflicts', 'time_conflicts', 'distribution_conflicts', 
                          'student_conflicts', 'capacity_violations']:
        values = [v[violation_type] for v in violations]
        plt.plot(values, label=violation_type)
    
    plt.title('Constraint Violations')
    plt.xlabel('Generation')
    plt.ylabel('Number of Violations')
    plt.legend()
    plt.tight_layout()
    plt.savefig('nsga2_violations.png')
    plt.close()

def plot_pareto_size(pareto_sizes):
    """
    Plot the size of the Pareto front over generations.
    
    Args:
        pareto_sizes: List of integers representing the size of the Pareto front per generation
    """
    plt.figure(figsize=(10, 6))
    plt.plot(pareto_sizes)
    plt.title('Pareto Front Size')
    plt.xlabel('Generation')
    plt.ylabel('Number of Solutions')
    plt.tight_layout()
    plt.savefig('nsga2_pareto_size.png')
    plt.close()

def run_muni_optimization(verbose=False):
    """Run the optimization on the MUni dataset."""
    print("Loading munifspsspr17 dataset...")
    data = load_muni_data(os.path.join(os.path.dirname(__file__), "munifspsspr17.json"))
    
    print("\nRunning NSGA-II optimization...")
    best_solution, fronts, metrics = nsga2_muni(data, verbose=True)
    
    # Print best solution
    print("\nBest solution:\n")
    print_solution_stats(best_solution, data)
    
    # Plot metrics
    plot_metrics(metrics)
    
    # Plot constraint violations over generations
    violations = [v['total_counts'] for v in metrics['constraint_violations']]
    plot_constraint_violations(violations)
    
    # Plot Pareto front size over generations
    plot_pareto_size(metrics['pareto_front_size'])
    
    # Save results to file with custom JSON encoder that handles tuple keys
    try:
        save_results(best_solution, fronts, metrics, data, 'nsga2_muni_results.json')
        print("\nNSGA-II optimization on munifspsspr17 dataset completed successfully.")
    except Exception as e:
        print(f"Error saving detailed results: {e}")
    
    return best_solution, fronts, metrics, data

if __name__ == "__main__":
    # Run the NSGA-II optimization on munifspsspr17 dataset
    best_solution, fronts, metrics, data = run_muni_optimization()
    print("\nNSGA-II optimization on munifspsspr17 dataset completed successfully.")
    print("Results saved to nsga2_muni_results.json")
