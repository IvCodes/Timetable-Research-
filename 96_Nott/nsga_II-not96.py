import os
import json
import random
import numpy as np
import time
from datetime import datetime
from collections import defaultdict

def load_nott_dataset(base_dir):
    """
    Load and parse the Nottingham 96 examination dataset.
    
    Args:
        base_dir: Directory containing the Nott dataset files
    
    Returns:
        Dictionary containing the parsed dataset
    """
    data = {
        'exams': [],
        'rooms': [],
        'timeslots': [],
        'enrollments': defaultdict(list),
        'student_exams': defaultdict(list),
        'coincidences': [],
        'room_assignments': {},
        'earliness_priority': {}
    }
    
    # Load exams
    with open(os.path.join(base_dir, 'exams'), 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) < 2:
                continue
                
            exam_code = parts[0]
            rest = parts[1]
            
            # Extract duration and department
            dept = rest[-2:]
            rest = rest[:-2].strip()
            
            # Parse duration like "3:00" to minutes
            duration_str = rest[-4:].strip()
            rest = rest[:-4].strip()
            
            try:
                if ':' in duration_str:
                    hours, mins = map(int, duration_str.split(':'))
                    duration = hours * 60 + mins
                else:
                    duration = int(duration_str) * 60
            except ValueError:
                duration = 120  # Default to 2 hours if parsing fails
            
            description = rest
            
            data['exams'].append({
                'code': exam_code,
                'description': description,
                'duration': duration,
                'department': dept
            })
    
    # Load data file for rooms and other constraints
    rooms_loaded = False
    coincidences_loaded = False
    room_assignments_loaded = False
    earliness_loaded = False
    
    with open(os.path.join(base_dir, 'data'), 'r') as f:
        current_section = None
        
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if line.upper() == "ROOMS":
                current_section = "rooms"
                rooms_loaded = True
                continue
            elif line.upper() == "COINCIDENCES":
                current_section = "coincidences"
                coincidences_loaded = True
                continue
            elif line.upper() == "ROOM ASSIGNMENTS":
                current_section = "room_assignments"
                room_assignments_loaded = True
                continue
            elif "EARLINESS PRIORITY" in line.upper():
                current_section = "earliness"
                earliness_loaded = True
                continue
            elif "DATES" in line.upper() or "TIMES" in line.upper():
                current_section = None
                continue
                
            # Parse based on current section
            if current_section == "rooms" and rooms_loaded:
                if line[0].isalpha():  # Room line
                    parts = line.split()
                    if len(parts) >= 2:
                        room_name = parts[0]
                        try:
                            capacity = int(parts[1])
                            data['rooms'].append({
                                'id': room_name,
                                'name': room_name,
                                'capacity': capacity,
                                'combined': False
                            })
                        except ValueError:
                            pass
            
            elif current_section == "coincidences" and coincidences_loaded:
                exams = [x.strip() for x in line.split() if x.strip()]
                if exams:
                    data['coincidences'].append(exams)
            
            elif current_section == "room_assignments" and room_assignments_loaded:
                parts = line.split()
                if len(parts) >= 2:
                    exam_code = parts[0]
                    room_id = ' '.join(parts[1:])
                    data['room_assignments'][exam_code] = room_id
            
            elif current_section == "earliness" and earliness_loaded:
                parts = line.split()
                if len(parts) >= 2:
                    exam_code = parts[0]
                    try:
                        priority = int(parts[1])
                        data['earliness_priority'][exam_code] = priority
                    except ValueError:
                        pass
    
    # Create timeslots based on the information from README
    # Mon - Fri 9:00, 13:30, 16:30, Sat 9:00
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    times = ["9:00", "13:30", "16:30"]
    
    for day_idx, day in enumerate(days):
        for time_idx, time in enumerate(times):
            # Saturday only has 9:00 slot
            if day == "Sat" and time != "9:00":
                continue
                
            data['timeslots'].append({
                'id': f"{day}-{time}",
                'day': day,
                'time': time,
                'index': day_idx * 3 + time_idx
            })
    
    # Load enrollments
    with open(os.path.join(base_dir, 'enrolements'), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                student_id, exam_code = parts
                data['enrollments'][exam_code].append(student_id)
                data['student_exams'][student_id].append(exam_code)
    
    return data

def generate_initial_population(data, population_size=100):
    """
    Generate an initial population of exam timetables.
    
    Args:
        data: Dataset information
        population_size: Size of the population to generate
    
    Returns:
        List of timetables
    """
    population = []
    
    for _ in range(population_size):
        # Initialize empty timetable
        timetable = []
        
        # Create list of exams to schedule
        exams_to_schedule = list(range(len(data['exams'])))
        random.shuffle(exams_to_schedule)
        
        # Group coincident exams
        coincidence_groups = []
        for group in data['coincidences']:
            indices = []
            for exam_code in group:
                for i, exam in enumerate(data['exams']):
                    if exam['code'] == exam_code:
                        indices.append(i)
                        break
            if indices:
                coincidence_groups.append(indices)
        
        # Remove coincident exams from main list
        for group in coincidence_groups:
            for idx in group:
                if idx in exams_to_schedule:
                    exams_to_schedule.remove(idx)
        
        # Assign timeslots and rooms to exams
        available_timeslots = list(range(len(data['timeslots'])))
        available_rooms = list(range(len(data['rooms'])))
        
        # First schedule coincident exam groups
        for group in coincidence_groups:
            if not available_timeslots:
                break
                
            timeslot = random.choice(available_timeslots)
            
            # Assign same timeslot to all exams in the group
            for exam_idx in group:
                if not available_rooms:
                    room = -1  # No room available
                else:
                    room = random.choice(available_rooms)
                
                timetable.append({
                    'exam_idx': exam_idx,
                    'timeslot_idx': timeslot,
                    'room_idx': room,
                    'assigned': room != -1
                })
        
        # Then schedule remaining exams
        for exam_idx in exams_to_schedule:
            if not available_timeslots:
                # No more timeslots, assign -1
                timetable.append({
                    'exam_idx': exam_idx,
                    'timeslot_idx': -1,
                    'room_idx': -1,
                    'assigned': False
                })
                continue
            
            timeslot = random.choice(available_timeslots)
            
            if not available_rooms:
                room = -1  # No room available
            else:
                room = random.choice(available_rooms)
            
            timetable.append({
                'exam_idx': exam_idx,
                'timeslot_idx': timeslot,
                'room_idx': room,
                'assigned': room != -1 and timeslot != -1
            })
        
        population.append(timetable)
    
    return population

def evaluate_solution(timetable, data):
    """
    Evaluate a timetable solution against multiple objectives.
    
    Args:
        timetable: The timetable to evaluate
        data: Dataset information
    
    Returns:
        Tuple of fitness values (total_score, unassigned_exams, room_capacity_violations,
                                time_conflicts, coincidence_violations)
    """
    # Initialize counters for constraint violations
    unassigned_exams = 0
    room_capacity_violations = 0
    time_conflicts = 0
    coincidence_violations = 0
    
    # Track assignments by timeslot and room
    timeslot_assignments = defaultdict(list)
    room_timeslot_assignments = defaultdict(list)
    
    # Track which timeslot each exam is assigned to
    exam_timeslots = {}
    
    # Check all assignments
    for assignment in timetable:
        exam_idx = assignment['exam_idx']
        timeslot_idx = assignment['timeslot_idx']
        room_idx = assignment['room_idx']
        
        if not assignment.get('assigned', False):
            unassigned_exams += 1
            continue
        
        # Store exam's timeslot for coincidence checking
        exam_timeslots[exam_idx] = timeslot_idx
        
        # Add to timeslot assignments
        timeslot_assignments[timeslot_idx].append(exam_idx)
        
        # Add to room-timeslot assignments
        if room_idx >= 0:
            room_timeslot_assignments[(room_idx, timeslot_idx)].append(exam_idx)
    
    # Check room capacity violations
    for (room_idx, timeslot_idx), exams in room_timeslot_assignments.items():
        if room_idx < 0 or room_idx >= len(data['rooms']):
            continue
            
        room = data['rooms'][room_idx]
        room_capacity = room['capacity']
        
        for exam_idx in exams:
            exam = data['exams'][exam_idx]
            exam_code = exam['code']
            
            # Get number of students taking this exam
            student_count = len(data['enrollments'].get(exam_code, []))
            
            if student_count > room_capacity:
                room_capacity_violations += 1
    
    # Check for student conflicts (same student having multiple exams in same timeslot)
    for timeslot_idx, exams in timeslot_assignments.items():
        if len(exams) <= 1:
            continue
            
        # Check each pair of exams
        for i in range(len(exams)):
            for j in range(i+1, len(exams)):
                exam1_idx = exams[i]
                exam2_idx = exams[j]
                
                exam1_code = data['exams'][exam1_idx]['code']
                exam2_code = data['exams'][exam2_idx]['code']
                
                # Get students for each exam
                students1 = set(data['enrollments'].get(exam1_code, []))
                students2 = set(data['enrollments'].get(exam2_code, []))
                
                # Count conflicts (students having both exams)
                conflicts = len(students1.intersection(students2))
                time_conflicts += conflicts
    
    # Check coincidence violations
    for group in data['coincidences']:
        exam_indices = []
        for exam_code in group:
            for i, exam in enumerate(data['exams']):
                if exam['code'] == exam_code:
                    exam_indices.append(i)
                    break
        
        # Skip if not all exams in the group are scheduled
        if not all(idx in exam_timeslots for idx in exam_indices):
            continue
            
        # Check if all exams in the group are in the same timeslot
        timeslots = set(exam_timeslots[idx] for idx in exam_indices)
        if len(timeslots) > 1:
            coincidence_violations += 1
    
    # Calculate total score (negative because we're minimizing)
    total_score = -(unassigned_exams * 1000 + room_capacity_violations * 100 + 
                   time_conflicts + coincidence_violations * 500)
    
    return (total_score, unassigned_exams, room_capacity_violations, 
            time_conflicts, coincidence_violations)

def evaluate_population(population, data):
    """
    Evaluate the entire population of timetables.
    
    Args:
        population: List of timetables
        data: Dataset information
    
    Returns:
        List of fitness values for each timetable
    """
    fitness_values = []
    
    for timetable in population:
        fitness = evaluate_solution(timetable, data)
        fitness_values.append(fitness)
    
    return fitness_values

def tournament_selection(population, fitness_values, tournament_size=2):
    """
    Select individuals using binary tournament selection.
    
    Args:
        population: List of timetables
        fitness_values: List of fitness values for each timetable
        tournament_size: Size of the tournament
    
    Returns:
        Selected individual (timetable)
    """
    # Select tournament_size individuals at random
    indices = random.sample(range(len(population)), tournament_size)
    
    # Select the best individual from the tournament
    best_idx = indices[0]
    best_fitness = fitness_values[best_idx]
    
    for idx in indices[1:]:
        if fitness_values[idx][0] > best_fitness[0]:  # Higher is better
            best_idx = idx
            best_fitness = fitness_values[idx]
    
    return population[best_idx]

def crossover(parent1, parent2):
    """
    Perform crossover between two parent timetables.
    
    Args:
        parent1, parent2: Parent timetables
    
    Returns:
        Two offspring timetables
    """
    # Create copies of parents
    offspring1 = [dict(assignment) for assignment in parent1]
    offspring2 = [dict(assignment) for assignment in parent2]
    
    # Ensure same length
    if len(parent1) != len(parent2):
        return offspring1, offspring2
    
    # Perform uniform crossover
    for i in range(len(parent1)):
        if random.random() < 0.5:
            # Swap timeslot and room assignments
            offspring1[i]['timeslot_idx'] = parent2[i]['timeslot_idx']
            offspring1[i]['room_idx'] = parent2[i]['room_idx']
            offspring1[i]['assigned'] = parent2[i]['assigned']
            
            offspring2[i]['timeslot_idx'] = parent1[i]['timeslot_idx']
            offspring2[i]['room_idx'] = parent1[i]['room_idx']
            offspring2[i]['assigned'] = parent1[i]['assigned']
    
    return offspring1, offspring2

def mutate(timetable, data, mutation_rate=0.1):
    """
    Apply mutation to a timetable.
    
    Args:
        timetable: The timetable to mutate
        data: Dataset information
        mutation_rate: Probability of mutation
    
    Returns:
        Mutated timetable
    """
    mutated_timetable = [dict(assignment) for assignment in timetable]
    
    for i in range(len(mutated_timetable)):
        # Skip with probability (1 - mutation_rate)
        if random.random() > mutation_rate:
            continue
        
        # Apply one of several mutation operators
        mutation_type = random.choice(["timeslot", "room", "swap", "assign"])
        
        if mutation_type == "timeslot":
            # Change timeslot
            new_timeslot = random.randint(0, len(data['timeslots']) - 1)
            mutated_timetable[i]['timeslot_idx'] = new_timeslot
            # If it was unassigned but now has both timeslot and room, mark as assigned
            if mutated_timetable[i]['room_idx'] >= 0:
                mutated_timetable[i]['assigned'] = True
                
        elif mutation_type == "room":
            # Change room
            new_room = random.randint(0, len(data['rooms']) - 1)
            mutated_timetable[i]['room_idx'] = new_room
            # If it was unassigned but now has both timeslot and room, mark as assigned
            if mutated_timetable[i]['timeslot_idx'] >= 0:
                mutated_timetable[i]['assigned'] = True
                
        elif mutation_type == "swap":
            # Swap with another assignment
            j = random.randint(0, len(mutated_timetable) - 1)
            if i != j:
                mutated_timetable[i]['timeslot_idx'], mutated_timetable[j]['timeslot_idx'] = \
                    mutated_timetable[j]['timeslot_idx'], mutated_timetable[i]['timeslot_idx']
                mutated_timetable[i]['room_idx'], mutated_timetable[j]['room_idx'] = \
                    mutated_timetable[j]['room_idx'], mutated_timetable[i]['room_idx']
                mutated_timetable[i]['assigned'], mutated_timetable[j]['assigned'] = \
                    mutated_timetable[j]['assigned'], mutated_timetable[i]['assigned']
                
        elif mutation_type == "assign":
            # Toggle assignment status
            if mutated_timetable[i]['assigned']:
                mutated_timetable[i]['assigned'] = False
            else:
                # Only assign if it has valid timeslot and room
                if mutated_timetable[i]['timeslot_idx'] >= 0 and mutated_timetable[i]['room_idx'] >= 0:
                    mutated_timetable[i]['assigned'] = True
    
    return mutated_timetable

def non_dominated_sort(fitness_values):
    """
    Perform non-dominated sorting of the population.
    
    Args:
        fitness_values: List of fitness values for each individual
    
    Returns:
        List of lists, where each inner list contains the indices of individuals in a front
    """
    population_size = len(fitness_values)
    
    # Initialize domination counters and dominated sets
    domination_count = [0] * population_size
    dominated_solutions = [[] for _ in range(population_size)]
    
    # Initialize fronts
    fronts = [[] for _ in range(population_size + 1)]
    
    # Find domination relations
    for p in range(population_size):
        for q in range(population_size):
            if p == q:
                continue
                
            # Check if p dominates q
            p_dominates_q = True
            q_dominates_p = True
            
            for obj_idx in range(1, len(fitness_values[p])):  # Skip first objective (total score)
                if fitness_values[p][obj_idx] > fitness_values[q][obj_idx]:
                    p_dominates_q = False
                if fitness_values[p][obj_idx] < fitness_values[q][obj_idx]:
                    q_dominates_p = False
            
            if p_dominates_q:
                dominated_solutions[p].append(q)
            elif q_dominates_p:
                domination_count[p] += 1
        
        # If p belongs to the first front
        if domination_count[p] == 0:
            fronts[0].append(p)
    
    # Identify remaining fronts
    front_index = 0
    while fronts[front_index]:
        next_front = []
        
        for p in fronts[front_index]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        
        front_index += 1
        fronts[front_index] = next_front
    
    # Remove empty fronts
    return [front for front in fronts[:front_index] if front]

def crowding_distance(fitness_values, front):
    """
    Calculate crowding distance for a front.
    
    Args:
        fitness_values: List of fitness values for each individual
        front: List of indices of individuals in the front
    
    Returns:
        List of crowding distances for individuals in the front
    """
    front_size = len(front)
    if front_size <= 2:
        return [float('inf')] * front_size
    
    # Initialize distances
    distances = [0.0] * front_size
    
    # Calculate crowding distance for each objective
    for obj_idx in range(len(fitness_values[0])):
        # Sort front by objective value
        sorted_front = sorted(range(front_size), key=lambda i: fitness_values[front[i]][obj_idx])
        
        # Set infinity for boundary points
        distances[sorted_front[0]] = float('inf')
        distances[sorted_front[-1]] = float('inf')
        
        # Calculate distances
        f_max = fitness_values[front[sorted_front[-1]]][obj_idx]
        f_min = fitness_values[front[sorted_front[0]]][obj_idx]
        
        # Skip if all values are the same
        if f_max == f_min:
            continue
        
        # Calculate distances for intermediate points
        for i in range(1, front_size - 1):
            distances[sorted_front[i]] += (
                fitness_values[front[sorted_front[i+1]]][obj_idx] - 
                fitness_values[front[sorted_front[i-1]]][obj_idx]
            ) / (f_max - f_min)
    
    return distances

def environmental_selection(population, fitness_values, population_size):
    """
    Select individuals for the next generation using non-dominated sorting
    and crowding distance.
    
    Args:
        population: List of timetables
        fitness_values: List of fitness values for each timetable
        population_size: Desired population size
    
    Returns:
        Selected population and their fitness values
    """
    # Perform non-dominated sorting
    fronts = non_dominated_sort(fitness_values)
    
    # Initialize new population
    new_population = []
    new_fitness = []
    
    # Add fronts to the new population
    for front in fronts:
        # If adding this front exceeds the population size, select based on crowding distance
        if len(new_population) + len(front) > population_size:
            # Calculate crowding distance
            distances = crowding_distance(fitness_values, front)
            
            # Sort front by crowding distance (descending)
            sorted_indices = sorted(range(len(front)), key=lambda i: -distances[i])
            
            # Add individuals until population size is reached
            slots_remaining = population_size - len(new_population)
            for i in range(slots_remaining):
                idx = front[sorted_indices[i]]
                new_population.append(population[idx])
                new_fitness.append(fitness_values[idx])
                
            break
        else:
            # Add the entire front
            for idx in front:
                new_population.append(population[idx])
                new_fitness.append(fitness_values[idx])
    
    return new_population, new_fitness

def nsga2_nott(data, population_size=100, generations=100):
    """
    Run the NSGA-II algorithm for the Nottingham dataset.
    
    Args:
        data: Dataset information
        population_size: Size of the population
        generations: Number of generations
    
    Returns:
        Final population and their fitness values
    """
    # Generate initial population
    population = generate_initial_population(data, population_size)
    
    # Evaluate initial population
    fitness_values = evaluate_population(population, data)
    
    # Run for specified number of generations
    for generation in range(generations):
        # Print progress update every 10 generations
        if generation % 10 == 0:
            best_fitness = max([f[0] for f in fitness_values])
            unassigned = min([f[1] for f in fitness_values])
            print(f"Generation {generation}: Best fitness = {best_fitness}, Unassigned exams = {unassigned}")
        
        # Create offspring population through selection, crossover, and mutation
        offspring = []
        
        while len(offspring) < population_size:
            # Select parents
            parent1 = tournament_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values)
            
            # Perform crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Perform mutation
            child1 = mutate(child1, data)
            child2 = mutate(child2, data)
            
            # Add to offspring
            offspring.append(child1)
            offspring.append(child2)
        
        # Ensure offspring population size
        offspring = offspring[:population_size]
        
        # Evaluate offspring
        offspring_fitness = evaluate_population(offspring, data)
        
        # Combine parent and offspring populations
        combined_population = population + offspring
        combined_fitness = fitness_values + offspring_fitness
        
        # Perform environmental selection
        population, fitness_values = environmental_selection(
            combined_population, combined_fitness, population_size
        )
    
    # Print final results
    best_idx = max(range(len(fitness_values)), key=lambda i: fitness_values[i][0])
    best_fitness = fitness_values[best_idx]
    print(f"Final best fitness: {best_fitness[0]}")
    print(f"Unassigned exams: {best_fitness[1]}")
    print(f"Room capacity violations: {best_fitness[2]}")
    print(f"Time conflicts: {best_fitness[3]}")
    print(f"Coincidence violations: {best_fitness[4]}")
    
    return population, fitness_values

def find_best_solution(population, fitness_values, data):
    """
    Find the best solution in the population.
    
    Args:
        population: List of timetables
        fitness_values: List of fitness values for each timetable
        data: Dataset information
    
    Returns:
        Best timetable and its fitness
    """
    # Find solution with best (highest) total score
    best_idx = max(range(len(fitness_values)), key=lambda i: fitness_values[i][0])
    
    return population[best_idx], fitness_values[best_idx]

def format_solution_for_output(timetable, data):
    """
    Format the timetable solution for output.
    
    Args:
        timetable: The timetable solution
        data: Dataset information
    
    Returns:
        Formatted solution
    """
    formatted_solution = []
    
    for assignment in timetable:
        if not assignment.get('assigned', False):
            continue
            
        exam_idx = assignment['exam_idx']
        timeslot_idx = assignment['timeslot_idx']
        room_idx = assignment['room_idx']
        
        exam = data['exams'][exam_idx]
        timeslot = data['timeslots'][timeslot_idx] if timeslot_idx >= 0 else None
        room = data['rooms'][room_idx] if room_idx >= 0 else None
        
        if not timeslot or not room:
            continue
        
        # Format according to frontend requirements
        formatted_activity = {
            "id": f"exam_{exam['code']}",
            "code": exam['code'],
            "name": exam['description'],
            "day": {
                "name": timeslot['day'],
                "code": timeslot['day']
            },
            "period": {
                "name": timeslot['time'],
                "code": f"P{timeslot['time'].replace(':', '')}"
            },
            "room": {
                "name": room['name'],
                "code": room['id'],
                "capacity": room['capacity']
            },
            "professor": {
                "name": "Exam Supervisor",
                "code": "EXAM_SUP"
            },
            "student_count": len(data['enrollments'].get(exam['code'], [])),
            "algorithm": "NSGA2"  # Ensure algorithm field is set
        }
        
        formatted_solution.append(formatted_activity)
    
    return formatted_solution

def run_nott_optimization(base_dir, population_size=100, generations=100):
    """
    Main function to run the NSGA-II optimization for the Nottingham dataset.
    
    Args:
        base_dir: Directory containing the Nott dataset files
        population_size: Size of the population
        generations: Number of generations
    
    Returns:
        Best timetable solution
    """
    # Load data
    print(f"Loading data from {base_dir}...")
    data = load_nott_dataset(base_dir)
    
    print(f"Dataset loaded: {len(data['exams'])} exams, {len(data['rooms'])} rooms, "
          f"{len(data['timeslots'])} timeslots, {len(data['coincidences'])} coincidence groups")
    
    # Run NSGA-II algorithm
    print("Running NSGA-II optimization...")
    final_population, fitness_values = nsga2_nott(data, population_size, generations)
    
    # Find best solution
    best_timetable, best_fitness = find_best_solution(final_population, fitness_values, data)
    
    # Format solution for output
    formatted_solution = format_solution_for_output(best_timetable, data)
    
    # Save results to file
    result_file = "nsga2_nott_results.json"
    with open(result_file, 'w') as f:
        json.dump({
            "solution": formatted_solution,
            "fitness": list(best_fitness),
            "assigned_percentage": (sum(1 for a in best_timetable if a.get('assigned', False)) / len(data['exams'])) * 100,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm": "NSGA-II"
        })
    
    print(f"Results saved to {result_file}")
    
    return formatted_solution

# Main execution block
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run NSGA-II optimization for Nottingham 96 dataset')
    parser.add_argument('--data_dir', type=str, default='datasets/nott', 
                        help='Directory containing the Nottingham dataset files')
    parser.add_argument('--population', type=int, default=100, 
                        help='Population size for the NSGA-II algorithm')
    parser.add_argument('--generations', type=int, default=100, 
                        help='Number of generations to run')
    
    args = parser.parse_args()
    
    # Run optimization
    start_time = time.time()
    solution = run_nott_optimization(args.data_dir, args.population, args.generations)
    end_time = time.time()
    
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    print(f"Scheduled {len(solution)} exams")