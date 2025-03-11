"""
Testing NSGA-II algorithm on the munifspsspr17 dataset.

This script adapts the NSGA-II implementation to work with the munifspsspr17.json dataset structure.
It loads the data, runs the optimization, and evaluates the results.
"""
import json
import random
import time
import numpy as np
from copy import deepcopy
from nsga2 import (
    fast_nondominated_sort, calculate_crowding_distance
)

# Algorithm parameters
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8

class Solution:
    """Class to represent a timetable solution."""
    def __init__(self):
        self.assignments = {}  # Maps class_id to (time_id, room_id)
        self.fitness = None
    
    def copy(self):
        """Create a deep copy of the solution."""
        new_solution = Solution()
        new_solution.assignments = deepcopy(self.assignments)
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
        solution: A Solution object with class assignments
        data: Dictionary with problem data
    
    Returns:
        List of objective values [total, room_conflicts, time_conflicts, distribution_conflicts, student_conflicts, capacity_violations]
    """
    room_conflicts = 0
    time_conflicts = 0
    distribution_conflicts = 0
    student_conflicts = 0
    capacity_violations = 0
    
    # Maps to track assignments
    time_room_assignments = {}  # time_id -> {room_id -> class_id}
    time_class_assignments = {}  # time_id -> list of class_ids
    class_assignments = {}  # class_id -> (time_id, room_id)
    
    # Process all assignments
    for class_id, (time_id, room_id) in solution.assignments.items():
        # Save for later reference
        class_assignments[class_id] = (time_id, room_id)
        
        # Check time-room conflicts
        if time_id not in time_room_assignments:
            time_room_assignments[time_id] = {}
            time_class_assignments[time_id] = []
        
        if room_id in time_room_assignments[time_id]:
            room_conflicts += 1
        else:
            time_room_assignments[time_id][room_id] = class_id
        
        time_class_assignments[time_id].append(class_id)
        
        # Check room capacity
        class_info = data['classes'].get(class_id, {})
        room_info = data['rooms'].get(room_id, {})
        
        if class_info.get('limit', 0) > room_info.get('capacity', 0):
            capacity_violations += 1
    
    # Check distribution constraints
    for constraint in data['distribution_constraints']:
        # Only evaluate if all classes in the constraint are assigned
        classes_to_check = [c_id for c_id in constraint['classes'] if c_id in class_assignments]
        
        if len(classes_to_check) < 2:
            continue  # Need at least 2 classes to have a constraint between them
        
        # Basic implementation for a few common constraint types
        violated = False
        
        if constraint['type'] == 'SameTime':
            # All classes must be at the same time
            time_ids = set(class_assignments[c_id][0] for c_id in classes_to_check)
            violated = len(time_ids) > 1
        
        elif constraint['type'] == 'DifferentTime':
            # All classes must be at different times
            time_ids = [class_assignments[c_id][0] for c_id in classes_to_check]
            violated = len(time_ids) != len(set(time_ids))
        
        elif constraint['type'] == 'SameRoom':
            # All classes must be in the same room
            room_ids = set(class_assignments[c_id][1] for c_id in classes_to_check)
            violated = len(room_ids) > 1
        
        elif constraint['type'] == 'DifferentRoom':
            # All classes must be in different rooms
            room_ids = [class_assignments[c_id][1] for c_id in classes_to_check]
            violated = len(room_ids) != len(set(room_ids))
        
        if violated:
            # Apply penalty based on whether it's a hard or soft constraint
            if constraint['required']:
                distribution_conflicts += 100  # Hard constraint - severe penalty
            else:
                distribution_conflicts += constraint['penalty']  # Soft constraint - use specified penalty
    
    # Check student conflicts
    for student_id, student_info in data['students'].items():
        # Get classes this student is enrolled in
        student_classes = []
        for course_id in student_info['courses']:
            if course_id in data['course_to_classes']:
                for class_id in data['course_to_classes'][course_id]:
                    if class_id in class_assignments:
                        student_classes.append(class_id)
        
        # Check for time conflicts among this student's classes
        time_slots_taken = {}  # time_id -> class_id
        
        for class_id in student_classes:
            time_id = class_assignments[class_id][0]
            
            if time_id in time_slots_taken:
                # This student has two classes at the same time
                student_conflicts += 1
            else:
                time_slots_taken[time_id] = class_id
    
    # Calculate weighted total based on the dataset weights
    total_violations = (
        data['weights']['ROOM'] * room_conflicts +
        data['weights']['TIME'] * time_conflicts +
        data['weights']['DISTRIBUTION'] * distribution_conflicts +
        data['weights']['STUDENT'] * student_conflicts
    )
    
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
    print(f"Room Conflicts: {solution.fitness[1]}")
    print(f"Time Conflicts: {solution.fitness[2]}")
    print(f"Distribution Conflicts: {solution.fitness[3]}")
    print(f"Student Conflicts: {solution.fitness[4]}")
    print(f"Capacity Violations: {solution.fitness[5]}")
    
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
        solution.fitness = fitness
        fitness_values.append(fitness)
    
    return fitness_values

def nsga2_muni(data, generations=NUM_GENERATIONS):
    """
    Run the NSGA-II algorithm on the munifspsspr17 dataset.
    
    Args:
        data: Dictionary with problem data
        generations: Number of generations to run
    
    Returns:
        Final population of solutions
    """
    # Initialize population
    population = initialize_population(data)
    
    for generation in range(generations):
        start_time = time.time()
        
        # Evaluate the current population
        fitness_values = evaluate_population(population, data)
        
        # Create offspring through selection, crossover, and mutation
        offspring = []
        
        while len(offspring) < POPULATION_SIZE:
            # Select parents using binary tournament selection
            parent_indices = random.sample(range(len(population)), 2)
            parent1 = population[parent_indices[0]]
            parent2 = population[parent_indices[1]]
            
            # Apply crossover with probability CROSSOVER_RATE
            if random.random() < CROSSOVER_RATE:
                child1, child2 = crossover_muni(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Apply mutation with probability MUTATION_RATE
            if random.random() < MUTATION_RATE:
                child1 = mutate_muni(child1, data)
            if random.random() < MUTATION_RATE:
                child2 = mutate_muni(child2, data)
            
            offspring.extend([child1, child2])
        
        # Truncate offspring to match population size
        offspring = offspring[:POPULATION_SIZE]
        
        # Evaluate offspring
        evaluate_population(offspring, data)
        
        # Combine parent and offspring populations
        combined_population = population + offspring
        combined_fitness = [sol.fitness for sol in combined_population]
        
        # Perform non-dominated sorting
        fronts = fast_nondominated_sort(combined_fitness)
        
        # Create next generation population
        new_population = []
        front_index = 0
        
        while len(new_population) + len(fronts[front_index]) <= POPULATION_SIZE:
            # Add all solutions from this front
            for i in fronts[front_index]:
                new_population.append(combined_population[i])
            front_index += 1
            
            if front_index == len(fronts):
                break
        
        # If we need more solutions, select based on crowding distance
        if len(new_population) < POPULATION_SIZE and front_index < len(fronts):
            # Get the current front that needs to be sorted by crowding distance
            current_front = fronts[front_index]
            
            # Calculate crowding distance - passing both the front and the fitness values
            crowding_dist = calculate_crowding_distance(
                current_front, 
                combined_fitness
            )
            
            # Sort by crowding distance (descending)
            sorted_indices = sorted(
                range(len(crowding_dist)), 
                key=lambda i: crowding_dist[i], 
                reverse=True
            )
            
            # Add remaining solutions
            remaining = POPULATION_SIZE - len(new_population)
            for i in range(remaining):
                new_population.append(
                    combined_population[fronts[front_index][sorted_indices[i]]]
                )
        
        # Replace current population
        population = new_population
        
        # Print progress
        if generation % 10 == 0:
            best_fitness = min([sol.fitness[0] for sol in population])
            gen_time = time.time() - start_time
            print(f"Generation {generation}: Best fitness = {best_fitness}, Time = {gen_time:.2f}s")
    
    return population

def crossover_muni(parent1, parent2):
    """
    Perform crossover between two parent solutions.
    
    Args:
        parent1, parent2: Two parent Solution objects
    
    Returns:
        Two child Solution objects
    """
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Get the list of all classes from both parents
    all_classes = list(set(parent1.assignments.keys()) | set(parent2.assignments.keys()))
    
    # Choose a random crossover point
    crossover_point = random.randint(0, len(all_classes))
    
    # Sort the classes for deterministic results
    all_classes.sort()
    
    # Swap assignments after the crossover point
    for i, class_id in enumerate(all_classes):
        if i >= crossover_point:
            # Swap assignments if both parents have this class
            if class_id in parent1.assignments and class_id in parent2.assignments:
                child1.assignments[class_id] = parent2.assignments[class_id]
                child2.assignments[class_id] = parent1.assignments[class_id]
    
    return child1, child2

def mutate_muni(solution, data):
    """
    Mutate a solution by randomly changing some assignments.
    
    Args:
        solution: A Solution object
        data: Dictionary with problem data
    
    Returns:
        Mutated Solution object
    """
    mutated = solution.copy()
    
    # Choose a random class to mutate
    if not solution.assignments:
        return mutated
    
    class_id = random.choice(list(solution.assignments.keys()))
    class_info = data['classes'].get(class_id, {})
    
    # Decide whether to mutate time, room, or both
    mutation_type = random.choice(['time', 'room', 'both'])
    
    if mutation_type in ['time', 'both'] and class_info['times']:
        # Change time assignment
        new_time = random.choice(class_info['times'])
        mutated.assignments[class_id] = (new_time['time_id'], mutated.assignments[class_id][1])
    
    if mutation_type in ['room', 'both'] and class_info['rooms']:
        # Change room assignment
        new_room = random.choice(class_info['rooms'])
        mutated.assignments[class_id] = (mutated.assignments[class_id][0], new_room['room_id'])
    
    return mutated

def find_best_solution(population):
    """
    Find the best solution in the population based on total violations.
    
    Args:
        population: List of Solution objects
    
    Returns:
        Best Solution object
    """
    return min(population, key=lambda sol: sol.fitness[0])

def run_muni_optimization():
    """Main function to run the optimization on the munifspsspr17 dataset."""
    print("Loading munifspsspr17 dataset...")
    data = load_muni_data("munifspsspr17.json")
    
    print(f"Dataset loaded: {len(data['classes'])} classes, {len(data['rooms'])} rooms")
    
    print("Running NSGA-II optimization...")
    start_time = time.time()
    final_population = nsga2_muni(data)
    total_time = time.time() - start_time
    
    print(f"\nOptimization completed in {total_time:.2f} seconds")
    
    # Find and report the best solution
    best_solution = find_best_solution(final_population)
    print_solution_stats(best_solution, data)
    
    # Save results to file
    results = {
        "algorithm": "NSGA-II",
        "dataset": "munifspsspr17",
        "runtime_seconds": total_time,
        "best_fitness": best_solution.fitness[0],
        "assigned_classes": len(best_solution.assignments),
        "total_classes": len(data['classes']),
        "constraint_violations": {
            "room_conflicts": best_solution.fitness[1],
            "time_conflicts": best_solution.fitness[2],
            "distribution_conflicts": best_solution.fitness[3],
            "student_conflicts": best_solution.fitness[4],
            "capacity_violations": best_solution.fitness[5]
        }
    }
    
    with open("nsga2_muni_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return best_solution

if __name__ == "__main__":
    # Run the NSGA-II optimization on munifspsspr17 dataset
    best_solution = run_muni_optimization()
    print("\nNSGA-II optimization on munifspsspr17 dataset completed successfully.")
    print("Results saved to nsga2_muni_results.json")
