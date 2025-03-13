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

# Algorithm parameters
POPULATION_SIZE = 50
ARCHIVE_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
NUM_OBJECTIVES = 5  # Number of objectives in the optimization

def load_muni_dataset(file_path):
    """
    Load and parse the munifspsspr17 dataset from a JSON file.
    """
    print(f"Loading munifspsspr17 dataset...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print("Parsing munifspsspr17.json dataset...")
    
    # Extract key components
    rooms = data['problem']['rooms']['room']
    courses = data['problem']['courses']['course']
    
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
        travels = room.get('travel', [])
        if not isinstance(travels, list):
            travels = [travels]
        
        for travel in travels:
            to_room = travel.get('_room', '')
            travel_time = int(travel.get('_value', 0))
            room_data[room_id]['travel'][to_room] = travel_time
    
    # Process courses, configs, subparts, and classes
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
            'name': course.get('_name', ''),
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
                    
                    # Process room assignments
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
                            'id': room_id,
                            'penalty': penalty
                        })
                    
                    # Process time assignments
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
                            'id': time_id,
                            'penalty': penalty
                        })
                    
                    # Create class object
                    class_data[class_id] = {
                        'course_id': course_id,
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
        'students': students,
        'distribution_constraints': distribution_constraints
    }

def generate_initial_population(data, population_size=POPULATION_SIZE):
    """
    Generate an initial random population of timetables.
    
    Args:
        data: Dictionary containing dataset information
        population_size: Size of the population to generate
    
    Returns:
        List of randomly initialized timetable dictionaries
    """
    population = []
    
    class_ids = list(data['classes'].keys())
    room_ids = list(data['rooms'].keys())
    time_patterns = list(data['time_patterns'].keys())
    
    for _ in range(population_size):
        # Initialize an empty timetable as a list of assignments
        timetable = []
        
        # For each class, assign a random room and time
        for class_id in class_ids:
            if random.random() < 0.8:  # 80% chance to assign a class
                assignment = {
                    'class_id': class_id,
                    'room_id': random.choice(room_ids),
                    'time_id': random.choice(time_patterns),
                    'assigned': True
                }
                timetable.append(assignment)
        
        population.append(timetable)
    
    return population

def evaluate_solution(timetable, data):
    """
    Evaluate a timetable solution against multiple objectives.
    
    Args:
        timetable: A timetable solution (list of assignments)
        data: Dictionary containing dataset information
    
    Returns:
        Tuple of fitness values (total_score, room_conflicts, time_conflicts, distribution_conflicts,
                                student_conflicts, capacity_violations)
    """
    # Initialize counters for constraint violations
    room_conflicts = 0
    time_conflicts = 0
    distribution_conflicts = 0
    student_conflicts = 0
    capacity_violations = 0
    
    # Track room and time usage
    room_time_usage = {}  # (room_id, time_id) -> class_id
    class_to_assignment = {}  # class_id -> (room_id, time_id)
    
    # Assigned classes count
    assigned_classes = 0
    
    # Check assignments
    for assignment in timetable:
        if not assignment.get('assigned', False):
            continue
            
        assigned_classes += 1
        class_id = assignment['class_id']
        room_id = assignment['room_id']
        time_id = assignment['time_id']
        
        # Store the assignment
        class_to_assignment[class_id] = (room_id, time_id)
        
        # Check room capacity
        class_size = data['classes'][class_id]['limit']
        room_capacity = data['rooms'][room_id]['capacity']
        
        if class_size > room_capacity:
            capacity_violations += 1
        
        # Check room-time conflicts
        room_time_key = (room_id, time_id)
        if room_time_key in room_time_usage:
            room_conflicts += 1
        else:
            room_time_usage[room_time_key] = class_id
    
    # Check time conflicts for classes in same course
    for course_id, class_ids in data['course_to_classes'].items():
        assigned_class_times = {}  # time_id -> [class_ids]
        
        for class_id in class_ids:
            if class_id in class_to_assignment:
                _, time_id = class_to_assignment[class_id]
                if time_id not in assigned_class_times:
                    assigned_class_times[time_id] = []
                assigned_class_times[time_id].append(class_id)
        
        # Count conflicts
        for time_id, class_list in assigned_class_times.items():
            if len(class_list) > 1:
                time_conflicts += len(class_list) - 1
    
    # Check distribution constraints
    for constraint in data['distribution_constraints']:
        if not constraint['required']:
            continue  # Skip soft constraints
            
        relevant_classes = [c for c in constraint['classes'] if c in class_to_assignment]
        
        if len(relevant_classes) < 2:
            continue  # Need at least 2 classes for a constraint
            
        # Simple check for some common constraint types
        if constraint['type'] == 'SameTime':
            time_slots = set()
            for class_id in relevant_classes:
                _, time_id = class_to_assignment[class_id]
                time_slots.add(time_id)
            
            if len(time_slots) > 1:
                distribution_conflicts += 1
                
        elif constraint['type'] == 'DifferentTime':
            time_slots = set()
            for class_id in relevant_classes:
                _, time_id = class_to_assignment[class_id]
                time_slots.add(time_id)
            
            if len(time_slots) < len(relevant_classes):
                distribution_conflicts += 1
    
    # Check student conflicts
    for student_id, student in data['students'].items():
        enrolled_classes = []
        
        # Get all classes from enrolled courses
        for course_id in student['courses']:
            if course_id in data['course_to_classes']:
                enrolled_classes.extend(data['course_to_classes'][course_id])
        
        # Check for time conflicts
        assigned_times = {}  # time_id -> class_id
        
        for class_id in enrolled_classes:
            if class_id in class_to_assignment:
                _, time_id = class_to_assignment[class_id]
                
                if time_id in assigned_times:
                    student_conflicts += 1
                else:
                    assigned_times[time_id] = class_id
    
    # Calculate total weighted score
    total_score = (
        room_conflicts * 1000 +
        time_conflicts * 1000 +
        distribution_conflicts * 100 +
        student_conflicts * 10 +
        capacity_violations * 50
    )
    
    # Adjust score based on assigned classes percentage
    total_classes = len(data['classes'])
    assignment_percentage = assigned_classes / total_classes
    assignment_penalty = int((1 - assignment_percentage) * 100000)
    
    total_score += assignment_penalty
    
    return (total_score, room_conflicts, time_conflicts, distribution_conflicts, 
            student_conflicts, capacity_violations)

def evaluate_population(population, data):
    """
    Evaluate the fitness of each timetable in the population.
    
    Args:
        population: List of timetables to evaluate
        data: Dataset information
    
    Returns:
        List of fitness tuples for each timetable
    """
    fitness_values = []
    for timetable in population:
        fitness_values.append(evaluate_solution(timetable, data))
    return fitness_values

def dominates(ind1_fitness, ind2_fitness):
    """
    Check if individual 1 dominates individual 2.
    
    Args:
        ind1_fitness: Fitness of individual 1
        ind2_fitness: Fitness of individual 2
    
    Returns:
        True if individual 1 dominates individual 2, False otherwise
    """
    at_least_one_better = False
    for i in range(len(ind1_fitness)):
        if ind1_fitness[i] > ind2_fitness[i]:
            return False
        elif ind1_fitness[i] < ind2_fitness[i]:
            at_least_one_better = True
    return at_least_one_better

def calculate_strength(population, fitness_values):
    """
    Calculate the strength of each individual in the population.
    
    Args:
        population: List of timetable solutions
        fitness_values: List of fitness values for each solution
    
    Returns:
        List of strength values
    """
    n = len(population)
    dominated_count = np.zeros(n, dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i != j and dominates(fitness_values[i], fitness_values[j]):
                dominated_count[i] += 1
    
    return dominated_count

def calculate_raw_fitness(population, fitness_values, strength):
    """
    Calculate the raw fitness for each individual based on strength.
    
    Args:
        population: List of timetable solutions
        fitness_values: List of fitness values for each solution
        strength: List of strength values
    
    Returns:
        List of raw fitness values
    """
    n = len(population)
    raw_fitness = np.zeros(n)
    
    for i in range(n):
        sum_strengths = 0
        for j in range(n):
            if i != j and dominates(fitness_values[j], fitness_values[i]):
                sum_strengths += strength[j]
        raw_fitness[i] = sum_strengths
    
    return raw_fitness

def calculate_density(fitness_values, k=1):
    """
    Calculate density based on k-nearest neighbor.
    
    Args:
        fitness_values: List of fitness values for each solution
        k: Number of nearest neighbors to consider
    
    Returns:
        List of density values
    """
    n = len(fitness_values)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            # Calculate Euclidean distance between fitness vectors
            dist = 0
            for f1, f2 in zip(fitness_values[i], fitness_values[j]):
                dist += (f1 - f2) ** 2
            dist = np.sqrt(dist)
            
            distances[i, j] = dist
            distances[j, i] = dist
    
    # For each individual, find the kth closest neighbor
    density = np.zeros(n)
    for i in range(n):
        dist_to_i = distances[i]
        sorted_indices = np.argsort(dist_to_i)
        # The first one is the individual itself (distance 0), so take k+1
        kth_neighbor = sorted_indices[min(k+1, len(sorted_indices)-1)]
        density[i] = 1.0 / (distances[i, kth_neighbor] + 2.0)  # +2 to avoid division by zero and ensure density ≤ 0.5
    
    return density

def calculate_fitness(raw_fitness, density):
    """
    Calculate the final fitness by combining raw fitness and density.
    
    Args:
        raw_fitness: List of raw fitness values
        density: List of density values
    
    Returns:
        List of final fitness values
    """
    return raw_fitness + density

def environmental_selection(population, fitness_values, archive_size):
    """
    Perform environmental selection to update the archive.
    
    Args:
        population: Combined population of current population and archive
        fitness_values: Fitness values for the combined population
        archive_size: Maximum size of the archive
    
    Returns:
        Selected individuals and their fitness values
    """
    # Calculate strength
    strength = calculate_strength(population, fitness_values)
    
    # Calculate raw fitness
    raw_fitness = calculate_raw_fitness(population, fitness_values, strength)
    
    # Calculate density
    density = calculate_density(fitness_values)
    
    # Calculate final fitness
    final_fitness = calculate_fitness(raw_fitness, density)
    
    # Select non-dominated individuals (those with raw fitness = 0)
    non_dominated_indices = np.where(raw_fitness == 0)[0]
    
    if len(non_dominated_indices) <= archive_size:
        # If non-dominated set fits, include all of them
        selected_indices = non_dominated_indices
        
        # If more space, fill with dominated individuals
        if len(selected_indices) < archive_size:
            dominated_indices = np.setdiff1d(np.arange(len(population)), non_dominated_indices)
            sorted_dominated = sorted(dominated_indices, key=lambda i: final_fitness[i])
            selected_indices = np.append(selected_indices, 
                                         sorted_dominated[:archive_size - len(selected_indices)])
    else:
        # Need to truncate non-dominated set using density
        # Sort by density and keep those with lowest density (better)
        sorted_non_dom = sorted(non_dominated_indices, key=lambda i: density[i])
        selected_indices = np.array(sorted_non_dom[:archive_size])
    
    # Return selected individuals and their fitness values
    return [population[i] for i in selected_indices], [fitness_values[i] for i in selected_indices]

def crossover(parent1, parent2):
    """
    Perform crossover between two parent timetables.
    
    Args:
        parent1, parent2: Two parent timetable lists
    
    Returns:
        Offspring timetable list
    """
    if random.random() > CROSSOVER_RATE:
        return copy.deepcopy(parent1)
    
    child = []
    
    # Create a map of class_id to assignment for both parents
    parent1_map = {a['class_id']: a for a in parent1 if a.get('assigned', False)}
    parent2_map = {a['class_id']: a for a in parent2 if a.get('assigned', False)}
    
    # Get all unique class IDs
    all_class_ids = set(list(parent1_map.keys()) + list(parent2_map.keys()))
    
    # For each class, choose assignment from one parent
    for class_id in all_class_ids:
        if class_id in parent1_map and class_id in parent2_map:
            # If both parents have this class assigned, choose randomly
            if random.random() < 0.5:
                child.append(copy.deepcopy(parent1_map[class_id]))
            else:
                child.append(copy.deepcopy(parent2_map[class_id]))
        elif class_id in parent1_map:
            # Only in parent1
            child.append(copy.deepcopy(parent1_map[class_id]))
        else:
            # Only in parent2
            child.append(copy.deepcopy(parent2_map[class_id]))
    
    return child

def mutate(individual, data):
    """
    Perform mutation on a timetable.
    
    Args:
        individual: Timetable list to mutate
        data: Dataset information
    
    Returns:
        Mutated timetable list
    """
    if random.random() > MUTATION_RATE:
        return individual
    
    # Deep copy the individual
    mutated = copy.deepcopy(individual)
    
    if len(mutated) == 0:
        return mutated
    
    # Choose a random mutation strategy
    strategy = random.choice(['change_room', 'change_time', 'add_or_remove'])
    
    if strategy == 'change_room':
        # Change room for a random assignment
        idx = random.randrange(len(mutated))
        if mutated[idx].get('assigned', False):
            mutated[idx]['room_id'] = random.choice(list(data['rooms'].keys()))
            
    elif strategy == 'change_time':
        # Change time for a random assignment
        idx = random.randrange(len(mutated))
        if mutated[idx].get('assigned', False):
            mutated[idx]['time_id'] = random.choice(list(data['time_patterns'].keys()))
            
    elif strategy == 'add_or_remove':
        # Either add a new assignment or remove an existing one
        if random.random() < 0.5 and len(mutated) > 0:
            # Remove an assignment
            del mutated[random.randrange(len(mutated))]
        else:
            # Add a new assignment
            unassigned_classes = set(data['classes'].keys()) - {a['class_id'] for a in mutated if a.get('assigned', False)}
            if unassigned_classes:
                class_id = random.choice(list(unassigned_classes))
                assignment = {
                    'class_id': class_id,
                    'room_id': random.choice(list(data['rooms'].keys())),
                    'time_id': random.choice(list(data['time_patterns'].keys())),
                    'assigned': True
                }
                mutated.append(assignment)
    
    return mutated

def select_parents(population, fitness_values):
    """
    Select parents for reproduction using binary tournament selection.
    
    Args:
        population: List of timetable solutions
        fitness_values: List of fitness values for each solution
    
    Returns:
        Two selected parents
    """
    # Convert raw fitness to selection fitness (lower is better)
    selection_fitness = np.array([sum(f) for f in fitness_values])
    
    def binary_tournament():
        # Select two random individuals and return the one with better fitness
        i1, i2 = random.sample(range(len(population)), 2)
        if selection_fitness[i1] <= selection_fitness[i2]:
            return population[i1]
        else:
            return population[i2]
    
    parent1 = binary_tournament()
    parent2 = binary_tournament()
    
    return parent1, parent2

def spea2_muni(data, generations=NUM_GENERATIONS):
    """
    Run the SPEA2 algorithm to optimize university course timetabling.
    
    Args:
        data: Dictionary containing dataset information
        generations: Number of generations to run
    
    Returns:
        Final archive of optimized timetables and their fitness values
    """
    start_time = time.time()
    
    print("Dataset loaded:", len(data['classes']), "classes,", len(data['rooms']), "rooms")
    print("Running SPEA2 optimization...")
    
    # Initialize population
    population = generate_initial_population(data, POPULATION_SIZE)
    population_fitness = evaluate_population(population, data)
    
    # Initialize empty archive
    archive = []
    archive_fitness = []
    
    for generation in range(generations):
        gen_start_time = time.time()
        
        # Combine population and archive
        combined_population = population + archive
        combined_fitness = population_fitness + archive_fitness
        
        # Environmental selection to update archive
        archive, archive_fitness = environmental_selection(combined_population, combined_fitness, ARCHIVE_SIZE)
        
        # Check if maximum generations reached
        if generation == generations - 1:
            break
        
        # Generate offspring population
        offspring = []
        for _ in range(POPULATION_SIZE):
            # Select parents
            parent1, parent2 = select_parents(combined_population, combined_fitness)
            
            # Create offspring through crossover and mutation
            child = crossover(parent1, parent2)
            child = mutate(child, data)
            
            offspring.append(child)
        
        # Replace population with offspring
        population = offspring
        population_fitness = evaluate_population(population, data)
        
        # Print progress every 10 generations
        if generation % 10 == 0 or generation == generations - 1:
            best_idx = min(range(len(archive_fitness)), key=lambda i: archive_fitness[i][0]) if archive_fitness else 0
            best_fitness = archive_fitness[best_idx][0] if archive_fitness else "N/A"
            gen_time = time.time() - gen_start_time
            print(f"Generation {generation}: Archive size = {len(archive)}, Best fitness = {best_fitness}, Time = {gen_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\nOptimization completed in {total_time:.2f} seconds")
    
    return archive, archive_fitness

def find_best_solution(archive, archive_fitness, data):
    """
    Find the best solution in the archive.
    
    Args:
        archive: Archive of timetables
        archive_fitness: Fitness values for each timetable
        data: Dataset information
    
    Returns:
        Best timetable solution and its statistics
    """
    if not archive:
        print("No solutions found in archive!")
        return None, None
    
    # Find solution with minimum total score (first element in the fitness tuple)
    best_idx = min(range(len(archive_fitness)), key=lambda i: archive_fitness[i][0])
    best_timetable = archive[best_idx]
    best_fitness = archive_fitness[best_idx]
    
    # Count assigned classes
    assigned_classes = sum(1 for a in best_timetable if a.get('assigned', False))
    total_classes = len(data['classes'])
    assignment_percentage = (assigned_classes / total_classes) * 100
    
    print("\nSolution Statistics:")
    print(f"Total Classes: {total_classes}")
    print(f"Assigned Classes: {assigned_classes} ({assignment_percentage:.2f}%)")
    print(f"Fitness: {best_fitness}")
    
    print("\nConstraint Violations:")
    print(f"Room Conflicts: {best_fitness[1]}")
    print(f"Time Conflicts: {best_fitness[2]}")
    print(f"Distribution Conflicts: {best_fitness[3]}")
    print(f"Student Conflicts: {best_fitness[4]}")
    print(f"Capacity Violations: {best_fitness[5]}")
    
    print(f"\nTotal Weighted Violation Score: {best_fitness[0]}")
    
    print("\nDataset Statistics:")
    print(f"Courses: {len(data['courses'])}")
    print(f"Classes: {len(data['classes'])}")
    print(f"Fixed Classes: {count_fixed_classes(data)}")
    print(f"Rooms: {len(data['rooms'])}")
    print(f"Students: {len(data['students'])}")
    print(f"Distribution Constraints: {len(data['distribution_constraints'])}")
    
    return best_timetable, best_fitness

def count_fixed_classes(data):
    """Count the number of classes with fixed assignments."""
    fixed_count = 0
    for class_id, class_info in data['classes'].items():
        if len(class_info['rooms']) == 1 and len(class_info['times']) == 1:
            fixed_count += 1
    return fixed_count

def format_solution_for_output(solution, data):
    """
    Format the solution for output in a structured format compatible with frontend.
    
    Args:
        solution: Timetable solution
        data: Dataset information
    
    Returns:
        Dictionary with properly formatted solution
    """
    formatted_solution = []
    
    for assignment in solution:
        if not assignment.get('assigned', False):
            continue
            
        class_id = assignment['class_id']
        room_id = assignment['room_id']
        time_id = assignment['time_id']
        
        class_info = data['classes'][class_id]
        course_id = class_info['course_id']
        course_info = data['courses'][course_id]
        
        time_pattern = data['time_patterns'][time_id]
        room_info = data['rooms'][room_id]
        
        # Format the output to match what the frontend expects
        formatted_activity = {
            "id": class_id,
            "name": f"{course_info['name']} - {class_id}",
            "day": {
                "name": time_pattern['days'],