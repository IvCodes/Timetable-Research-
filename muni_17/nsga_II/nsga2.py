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

# Keep track of vacant rooms for analysis
vacant_rooms = []

# This function is kept from original implementation but not used directly in the new NSGA-II implementation
def evaluator(timetable):
    vacant_room = 0
    prof_conflicts = 0
    room_size_conflicts = 0
    sub_group_conflicts = 0
    unasigned_activities = len(activities_dict)
    activities_set = set()

    for slot in timetable:
        prof_set = set()
        sub_group_set = set()
        for room in timetable[slot]:
            activity = timetable[slot][room]

            if not isinstance(activity, Activity):
                vacant_room += 1
                vacant_rooms.append((slot, room))

            else:
                activities_set.add(activity.id)
                if activity.teacher_id in prof_set:
                    prof_conflicts += 1

                sub_group_conflicts += len(
                    set(activity.group_ids).intersection(sub_group_set))

                group_size = 0
                for group_id in activity.group_ids:
                    group_size += groups_dict[group_id].size
                    sub_group_set.add(group_id)

                if group_size > spaces_dict[room].size:
                    room_size_conflicts += 1
                teacher_id = activity.teacher_id
                prof_set.add(teacher_id)
    unasigned_activities -= len(activities_set)
    return vacant_room, prof_conflicts, room_size_conflicts, sub_group_conflicts, unasigned_activities

# Functions below are from the provided implementation

def evaluate_population(population, activities_dict, groups_dict, spaces_dict):
    """Evaluate each individual using the provided evaluator function."""
    fitness_values = []
    for timetable in population:
        fitness_values.append(evaluate_hard_constraints(timetable, activities_dict, groups_dict, spaces_dict))
    return fitness_values

def mutate(individual, slots, spaces_dict):
    """Perform mutation by randomly swapping activities in the timetable."""
    if random.random() > MUTATION_RATE:
        return individual

    # Choose two random slots and spaces
    slot1, slot2 = random.sample(slots, 2)
    space1, space2 = random.choice(list(spaces_dict.keys())), random.choice(list(spaces_dict.keys()))
    
    # Swap activities
    individual[slot1][space1], individual[slot2][space2] = individual[slot2][space2], individual[slot1][space1]
    
    return individual

def crossover(parent1, parent2):
    """Perform crossover by swapping time slots between two parents."""
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()
    
    child1, child2 = parent1.copy(), parent2.copy()
    slots = list(parent1.keys())
    
    # One-point crossover
    split = random.randint(0, len(slots) - 1)
    
    for i in range(split, len(slots)):
        child1[slots[i]], child2[slots[i]] = parent2[slots[i]], parent1[slots[i]]
    
    return child1, child2

def fast_nondominated_sort(fitness_values):
    """Perform non-dominated sorting based on the multi-objective fitness values."""
    fronts = [[]]
    S = [[] for _ in range(len(fitness_values))]
    n = [0] * len(fitness_values)
    rank = [0] * len(fitness_values)

    for p in range(len(fitness_values)):
        for q in range(len(fitness_values)):
            if dominates(fitness_values[p], fitness_values[q]):
                S[p].append(q)
            elif dominates(fitness_values[q], fitness_values[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]

def calculate_crowding_distance(front, fitness_values):
    """Calculate crowding distance for a front."""
    distances = [0] * len(front)
    num_objectives = len(fitness_values[0])

    for m in range(num_objectives):
        front.sort(key=lambda x: fitness_values[x][m])
        distances[0] = distances[-1] = float('inf')

        min_value = fitness_values[front[0]][m]
        max_value = fitness_values[front[-1]][m]
        if max_value == min_value:
            continue

        for i in range(1, len(front) - 1):
            distances[i] += (fitness_values[front[i + 1]][m] -
                           fitness_values[front[i - 1]][m]) / (max_value - min_value)

    return distances

def select_parents(population, fitness_values, fronts=None, crowding_distances=None):
    """Perform tournament selection based on non-dominated sorting and crowding distance."""
    if fronts is None:
        fronts = fast_nondominated_sort(fitness_values)
    
    selected = []

    for front in fronts:
        if len(selected) + len(front) > POPULATION_SIZE:
            crowding_distances = calculate_crowding_distance(front, fitness_values)
            sorted_front = sorted(
                zip(front, crowding_distances), key=lambda x: x[1], reverse=True)
            selected.extend(
                [x[0] for x in sorted_front[:POPULATION_SIZE - len(selected)]])
            break
        else:
            selected.extend(front)

    return [population[i] for i in selected]

def generate_initial_population(slots, spaces_dict, activities_dict, groups_dict):
    """Generate an initial population with random timetables."""
    population = []

    for _ in range(POPULATION_SIZE):
        # Initialize empty timetable
        timetable = {}
        for slot in slots:
            timetable[slot] = {}
            for space in spaces_dict:
                timetable[slot][space] = ""
        
        # Assign activities to slots and rooms - using approach from provided code
        activities_ids = [activity_id for activity_id in activities_dict.keys() 
                         for _ in range(activities_dict[activity_id].duration)]
        random.shuffle(activities_ids)
        
        for activity_id in activities_ids:
            activity = activities_dict[activity_id]
            # Find suitable slot and room
            available_slots = list(slots)
            random.shuffle(available_slots)
            
            for slot in available_slots:
                available_rooms = [room_id for room_id in spaces_dict.keys() 
                                  if timetable[slot][room_id] == "" and 
                                  get_classsize(activity) <= spaces_dict[room_id].size]
                
                if available_rooms:
                    room = random.choice(available_rooms)
                    timetable[slot][room] = activity
                    break
            
        population.append(timetable)
    
    return population

def nsga2(activities_dict, groups_dict, spaces_dict, slots, generations=NUM_GENERATIONS):
    """Main NSGA-II algorithm loop."""
    # Initialize population
    population = generate_initial_population(slots, spaces_dict, activities_dict, groups_dict)
    
    for generation in range(generations):
        # Evaluate the current population
        fitness_values = evaluate_population(population, activities_dict, groups_dict, spaces_dict)
        
        # Create new population through selection, crossover, and mutation
        new_population = []
        
        while len(new_population) < POPULATION_SIZE:
            # Select parents (using random selection as in provided code)
            parent1, parent2 = random.sample(population, 2)
            
            # Apply crossover
            if random.random() < CROSSOVER_RATE:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Apply mutation
            child1 = mutate(child1, slots, spaces_dict)
            child2 = mutate(child2, slots, spaces_dict)
            
            # Add to new population
            new_population.extend([child1, child2])
        
        # Truncate to maintain population size
        new_population = new_population[:POPULATION_SIZE]
        
        # Evaluate new population
        new_fitness_values = evaluate_population(new_population, activities_dict, groups_dict, spaces_dict)
        
        # Select survivors for next generation using non-dominated sorting
        fronts = fast_nondominated_sort(new_fitness_values)
        population = select_parents(new_population, new_fitness_values, fronts)
        
        # Print progress
        if generation % 10 == 0:
            print(f"Generation {generation}: Population size {len(population)}")
    
    return population

def find_best_solution(final_population, activities_dict, groups_dict, spaces_dict):
    """Find the best solution in the final population based on constraint violations."""
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
    """Main function to run the NSGA-II optimization for timetable scheduling."""
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
    
    print("NSGA-II optimization completed successfully.")
