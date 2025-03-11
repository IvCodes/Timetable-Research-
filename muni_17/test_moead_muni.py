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

# Algorithm parameters
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
NUM_OBJECTIVES = 5  # Number of objectives in the optimization
T = 10  # Neighborhood size
NEIGHBORHOOD_SELECTION_PROB = 0.9  # Probability of selecting from neighborhood

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
        if len(neighborhood) >= 2:
            return random.sample(neighborhood, 2)
        elif len(neighborhood) == 1:
            return [neighborhood[0], random.randrange(POPULATION_SIZE)]
        else:
            return random.sample(range(POPULATION_SIZE), 2)
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

def moead_muni(data, generations=NUM_GENERATIONS):
    """
    Run the MOEA/D algorithm to optimize university course timetabling.
    
    Args:
        data: Dictionary containing dataset information
        generations: Number of generations to run
    
    Returns:
        Final population of optimized timetables and their fitness values
    """
    start_time = time.time()
    
    print("Dataset loaded:", len(data['classes']), "classes,", len(data['rooms']), "rooms")
    print("Running MOEA/D optimization...")
    
    # Initialize weight vectors
    weight_vectors = generate_weight_vectors(POPULATION_SIZE, NUM_OBJECTIVES)
    
    # Compute distances between weight vectors and initialize neighborhoods
    _, sorted_indices = compute_euclidean_distances(weight_vectors)
    neighborhoods = initialize_neighborhoods(sorted_indices, T)
    
    # Initialize population
    population = generate_initial_population(data, POPULATION_SIZE)
    
    # Evaluate initial population
    fitness_values = evaluate_population(population, data)
    
    # Initialize ideal point
    ideal_point = [min(f[i] for f in fitness_values) for i in range(NUM_OBJECTIVES)]
    
    for generation in range(generations):
        gen_start_time = time.time()
        
        # For each subproblem
        for i in range(POPULATION_SIZE):
            # Select parents
            parent_indexes = select_parent_indexes(i, neighborhoods, NEIGHBORHOOD_SELECTION_PROB)
            parent1, parent2 = population[parent_indexes[0]], population[parent_indexes[1]]
            
            # Generate offspring
            offspring = crossover(parent1, parent2)
            offspring = mutate(offspring, data)
            
            # Evaluate offspring
            offspring_fitness = evaluate_solution(offspring, data)
            
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
        
        # Print progress every 10 generations
        if generation % 10 == 0 or generation == generations - 1:
            best_idx = min(range(len(fitness_values)), key=lambda i: fitness_values[i][0])
            best_fitness = fitness_values[best_idx][0]
            gen_time = time.time() - gen_start_time
            print(f"Generation {generation}: Best fitness = {best_fitness}, Time = {gen_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\nOptimization completed in {total_time:.2f} seconds")
    
    return population, fitness_values

def find_best_solution(final_population, fitness_values, data):
    """
    Find the best solution in the final population.
    
    Args:
        final_population: Final population of timetables
        fitness_values: Fitness values for each timetable
        data: Dataset information
    
    Returns:
        Best timetable solution and its statistics
    """
    # Find solution with minimum total score (first element in the fitness tuple)
    best_idx = min(range(len(fitness_values)), key=lambda i: fitness_values[i][0])
    best_timetable = final_population[best_idx]
    best_fitness = fitness_values[best_idx]
    
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
                "code": time_pattern['days']
            },
            "period": {
                "name": f"{time_pattern['start']}-{time_pattern['start'] + time_pattern['length']}",
                "code": f"P{time_pattern['start']}"
            },
            "room": {
                "name": room_id,
                "code": room_id,
                "capacity": room_info['capacity']
            },
            "professor": {
                "name": "Assigned Professor",
                "code": "PROF"
            },
            "student_count": class_info['limit'],
            "algorithm": "MOEAD"  # Ensure algorithm field is set
        }
        
        formatted_solution.append(formatted_activity)
    
    return formatted_solution

def run_muni_optimization():
    """
    Main function to run the MOEA/D optimization for the munifspsspr17 dataset.
    
    Returns:
        Best timetable solution
    """
    # Load data
    data = load_muni_dataset("munifspsspr17.json")
    
    # Run MOEA/D algorithm
    final_population, fitness_values = moead_muni(data)
    
    # Find best solution
    best_timetable, best_fitness = find_best_solution(final_population, fitness_values, data)
    
    # Format solution for output
    formatted_solution = format_solution_for_output(best_timetable, data)
    
    # Save results to file
    result_file = "moead_muni_results.json"
    with open(result_file, 'w') as f:
        json.dump({
            "solution": formatted_solution,
            "fitness": list(best_fitness),
            "assigned_percentage": (sum(1 for a in best_timetable if a.get('assigned', False)) / len(data['classes'])) * 100,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm": "MOEAD"
        }, f, indent=4)
    
    print(f"Results saved to {result_file}")
    
    # Return the best solution
    return best_timetable, best_fitness, formatted_solution

def save_solution_for_frontend(formatted_solution, algorithm="MOEAD"):
    """
    Save the solution in a format compatible with the frontend application.
    
    Args:
        formatted_solution: Solution formatted for output
        algorithm: Algorithm name used for the solution
    """
    # Ensure output directory exists
    output_dir = "frontend_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/moead_muni_solution_{timestamp}.json"
    
    # Prepare data in the format expected by the frontend
    frontend_data = {
        "timetable": formatted_solution,
        "metadata": {
            "algorithm": algorithm,
            "dataset": "munifspsspr17",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "activities_count": len(formatted_solution)
        }
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(frontend_data, f, indent=4)
    
    print(f"Frontend compatible solution saved to {output_file}")
    return output_file

def analyze_solution(solution, data):
    """
    Analyze the solution for quality metrics and detailed statistics.
    
    Args:
        solution: Timetable solution
        data: Dataset information
    
    Returns:
        Dictionary of analysis results
    """
    # Calculate assignment coverage
    total_classes = len(data['classes'])
    assigned_classes = sum(1 for a in solution if a.get('assigned', False))
    assignment_percentage = (assigned_classes / total_classes) * 100
    
    # Calculate room utilization
    room_usage = {}
    for assignment in solution:
        if not assignment.get('assigned', False):
            continue
        
        room_id = assignment['room_id']
        if room_id not in room_usage:
            room_usage[room_id] = 0
        room_usage[room_id] += 1
    
    total_rooms = len(data['rooms'])
    rooms_used = len(room_usage)
    room_utilization = (rooms_used / total_rooms) * 100
    
    # Analyze timeslot usage
    timeslot_usage = {}
    for assignment in solution:
        if not assignment.get('assigned', False):
            continue
        
        time_id = assignment['time_id']
        if time_id not in timeslot_usage:
            timeslot_usage[time_id] = 0
        timeslot_usage[time_id] += 1
    
    # Find most and least used timeslots
    if timeslot_usage:
        most_used_time = max(timeslot_usage.items(), key=lambda x: x[1])
        least_used_time = min(timeslot_usage.items(), key=lambda x: x[1])
    else:
        most_used_time = ("none", 0)
        least_used_time = ("none", 0)
    
    # Course assignment analysis
    course_coverage = {}
    for course_id, class_ids in data['course_to_classes'].items():
        assigned_count = 0
        for class_id in class_ids:
            # Check if this class is assigned in the solution
            if any(a.get('class_id') == class_id and a.get('assigned', False) for a in solution):
                assigned_count += 1
        
        course_coverage[course_id] = {
            "total_classes": len(class_ids),
            "assigned_classes": assigned_count,
            "percentage": (assigned_count / len(class_ids)) * 100 if len(class_ids) > 0 else 0
        }
    
    # Overall metrics
    analysis_results = {
        "assignment_coverage": {
            "total_classes": total_classes,
            "assigned_classes": assigned_classes,
            "percentage": assignment_percentage
        },
        "room_utilization": {
            "total_rooms": total_rooms,
            "rooms_used": rooms_used,
            "percentage": room_utilization,
            "usage_distribution": room_usage
        },
        "timeslot_analysis": {
            "total_timeslots_used": len(timeslot_usage),
            "most_used_timeslot": {
                "id": most_used_time[0],
                "count": most_used_time[1]
            },
            "least_used_timeslot": {
                "id": least_used_time[0],
                "count": least_used_time[1]
            }
        },
        "course_coverage": course_coverage
    }
    
    return analysis_results

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Check if data file exists
    data_file = "munifspsspr17.json"
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found. Please ensure the file exists in the current directory.")
        exit(1)
    
    print("Starting MOEA/D optimization for munifspsspr17 dataset...")
    start_time = time.time()
    
    # Run the optimization
    try:
        best_timetable, best_fitness, formatted_solution = run_muni_optimization()
        
        # Save solution for frontend
        frontend_file = save_solution_for_frontend(formatted_solution)
        
        # Analyze solution
        if best_timetable:
            data = load_muni_dataset("munifspsspr17.json")
            analysis = analyze_solution(best_timetable, data)
            
            # Save analysis to file
            analysis_file = "moead_muni_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=4)
            
            print(f"Analysis saved to {analysis_file}")
            
            # Print summary statistics
            print("\nSummary Statistics:")
            print(f"Total classes: {analysis['assignment_coverage']['total_classes']}")
            print(f"Assigned classes: {analysis['assignment_coverage']['assigned_classes']} ({analysis['assignment_coverage']['percentage']:.2f}%)")
            print(f"Room utilization: {analysis['room_utilization']['percentage']:.2f}%")
            print(f"Execution time: {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("MOEA/D optimization completed.")