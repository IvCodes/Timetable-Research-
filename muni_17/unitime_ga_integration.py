"""
Integration example for using the UniTime dataset with the Genetic Algorithm.

This module demonstrates how to integrate the UniTime dataset loader with
the existing Genetic Algorithm implementation for timetable scheduling.
"""
import sys
import os
import json
import random
import numpy as np
from deap import base, creator, tools, algorithms
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import time
import traceback

# Import the UniTime dataset loader
from unitime_loader import load_unitime_json, load_unitime_xml

# Path to the UniTime dataset
UNITIME_JSON_PATH = '../Advance-Timetable-Scheduling-Backend/Data/munifspsspr17.json'

# Define the Genetic Algorithm parameters
POP_SIZE = 200
GENERATIONS = 100
CXPB = 0.8  # Crossover probability
MUTPB = 0.2  # Mutation probability

# Set up logging
LOG_DIR = "./logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"unitime_ga_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def evaluate_unitime_constraints(timetable, activities_dict, groups_dict, spaces_dict, lecturers_dict):
    """
    Evaluate constraints specific to the UniTime dataset.
    
    Args:
        timetable: The timetable to evaluate
        activities_dict: Dictionary of activities
        groups_dict: Dictionary of groups
        spaces_dict: Dictionary of spaces
        lecturers_dict: Dictionary of lecturers
        
    Returns:
        Tuple of (room_constraints, time_constraints, distribution_constraints, student_constraints)
    """
    # Initialize constraint violations
    room_constraints = 0
    time_constraints = 0
    distribution_constraints = 0
    student_constraints = 0
    
    # Extract all assignments from the timetable
    assignments = {}  # {activity_id: (slot, room)}
    for activity_id, slot, room in timetable:
        assignments[activity_id] = (slot, room)
    
    # Check room capacity constraints
    for activity_id, (slot, room) in assignments.items():
        activity = activities_dict[activity_id]
        room_obj = spaces_dict[room]
        
        # Calculate total student count for this activity
        student_count = 0
        for group_id in activity.group_ids:
            if group_id in groups_dict:
                student_count += groups_dict[group_id].size
        
        # Check if room is too small
        if student_count > room_obj.size:
            room_constraints += 1
    
    # Check time conflicts (same group scheduled at the same time)
    slot_activities = {}  # {slot: [activity_ids]}
    for activity_id, (slot, room) in assignments.items():
        if slot not in slot_activities:
            slot_activities[slot] = []
        slot_activities[slot].append(activity_id)
    
    for slot, activity_ids in slot_activities.items():
        if len(activity_ids) <= 1:
            continue
            
        # Check for group conflicts
        group_conflicts = set()
        for i in range(len(activity_ids)):
            for j in range(i+1, len(activity_ids)):
                activity1 = activities_dict[activity_ids[i]]
                activity2 = activities_dict[activity_ids[j]]
                
                # Check for shared groups
                shared_groups = set(activity1.group_ids).intersection(set(activity2.group_ids))
                if shared_groups:
                    group_conflicts.add(frozenset([activity_ids[i], activity_ids[j]]))
        
        time_constraints += len(group_conflicts)
    
    # Check lecturer conflicts (same lecturer scheduled at the same time)
    for slot, activity_ids in slot_activities.items():
        if len(activity_ids) <= 1:
            continue
            
        # Check for lecturer conflicts
        lecturer_conflicts = set()
        for i in range(len(activity_ids)):
            for j in range(i+1, len(activity_ids)):
                activity1 = activities_dict[activity_ids[i]]
                activity2 = activities_dict[activity_ids[j]]
                
                # Check if same lecturer
                if activity1.teacher_id == activity2.teacher_id:
                    lecturer_conflicts.add(frozenset([activity_ids[i], activity_ids[j]]))
        
        time_constraints += len(lecturer_conflicts)
    
    # Check room conflicts (same room scheduled at the same time)
    for slot, activity_ids in slot_activities.items():
        if len(activity_ids) <= 1:
            continue
            
        # Check for room conflicts
        room_conflicts = set()
        for i in range(len(activity_ids)):
            for j in range(i+1, len(activity_ids)):
                a1_id, a2_id = activity_ids[i], activity_ids[j]
                room1 = assignments[a1_id][1]
                room2 = assignments[a2_id][1]
                
                # Check if same room
                if room1 == room2:
                    room_conflicts.add(frozenset([a1_id, a2_id]))
        
        room_constraints += len(room_conflicts)
    
    # Return all constraint violations
    return room_constraints, time_constraints, distribution_constraints, student_constraints

def generate_html_report(results, logbook, best_fitness, output_file, runtime):
    """
    Generate an HTML report of the GA results.
    
    Args:
        results: The timetable results
        logbook: DEAP logbook with statistics
        best_fitness: The fitness values of the best individual
        output_file: Output file path for HTML
        runtime: Runtime in seconds
    """
    # Create a stats dictionary from the best fitness
    stats = {
        'total_assignments': len(results),
        'room_constraints': best_fitness[0],
        'time_constraints': best_fitness[1],
        'distribution_constraints': best_fitness[2],
        'student_constraints': best_fitness[3],
        'best_fitness': best_fitness
    }
    
    # Extract GA parameters section
    ga_params = f"""
    <div class="section">
        <h2>Genetic Algorithm Parameters</h2>
        <p><strong>Population Size:</strong> {POP_SIZE}</p>
        <p><strong>Generations:</strong> {GENERATIONS}</p>
        <p><strong>Crossover Probability:</strong> {CXPB}</p>
        <p><strong>Mutation Probability:</strong> {MUTPB}</p>
    </div>
    """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>UniTime GA Integration Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .section {{ 
                background-color: #f8f9fa;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .constraint {{ font-weight: bold; }}
            .good {{ color: green; }}
            .warning {{ color: orange; }}
            .bad {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>UniTime GA Integration Results</h1>
        
        {ga_params}
        
        <div class="section">
            <h2>Fitness Statistics</h2>
            <p><strong>Total Assignments:</strong> {stats['total_assignments']}</p>
            <p><strong>Room Constraint Violations:</strong> <span class="constraint">{stats['room_constraints']}</span></p>
            <p><strong>Time Constraint Violations:</strong> <span class="constraint">{stats['time_constraints']}</span></p>
            <p><strong>Distribution Constraint Violations:</strong> <span class="constraint">{stats['distribution_constraints']}</span></p>
            <p><strong>Student Constraint Violations:</strong> <span class="constraint">{stats['student_constraints']}</span></p>
            <p><strong>Best Fitness Score:</strong> {stats['best_fitness']}</p>
            <p><strong>Total Runtime:</strong> {runtime:.2f} seconds</p>
        </div>
        
        <h2>Convergence Plot</h2>
        <div class="section">
            <img src="convergence_plot.png" alt="Convergence Plot" style="max-width:100%; height:auto;">
        </div>
        
        <div class="section">
            <h2>Timetable Results</h2>
            <table>
                <tr>
                    <th>Activity</th>
                    <th>Day</th>
                    <th>Period</th>
                    <th>Room</th>
                </tr>
    """
    
    # Add timetable rows
    for entry in results:
        activity_name = entry.get('name', f"Activity {entry.get('activity_id', 'Unknown')}")
        day_name = entry.get('day', {}).get('name', 'Unknown')
        period_name = entry.get('period', {}).get('name', 'Unknown')
        room_name = entry.get('room', {}).get('name', 'Unknown')
        
        html_content += f"""
                <tr>
                    <td>{activity_name}</td>
                    <td>{day_name}</td>
                    <td>{period_name}</td>
                    <td>{room_name}</td>
                </tr>
        """
    
    # Close the HTML
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(html_content)
    logger.info(f"Saved HTML report to {output_file}")

def plot_convergence(logbook, filename):
    """
    Plot the convergence of the best fitness over generations.
    
    Args:
        logbook: DEAP logbook with stats
        filename: Filename to save the plot
    """
    # Extract statistics
    generations = list(range(len(logbook)))
    
    # For multi-objective fitness, we need to extract each component separately
    # Extract the first component (room constraints)
    room_fitness = [logbook.select("min")[i][0] for i in range(len(generations))]
    
    # Extract the second component (time constraints)
    time_fitness = [logbook.select("min")[i][1] for i in range(len(generations))]
    
    # Extract the third component (distribution constraints)
    dist_fitness = [logbook.select("min")[i][2] for i in range(len(generations))]
    
    # Extract the fourth component (student constraints)
    student_fitness = [logbook.select("min")[i][3] for i in range(len(generations))]
    
    # Create the figure
    plt.figure(figsize=(10, 6))
    
    # Plot each component
    plt.plot(generations, room_fitness, label='Room Constraints', marker='o')
    plt.plot(generations, time_fitness, label='Time Constraints', marker='s')
    plt.plot(generations, dist_fitness, label='Distribution Constraints', marker='^')
    plt.plot(generations, student_fitness, label='Student Constraints', marker='x')
    
    # Add labels and title
    plt.xlabel('Generation')
    plt.ylabel('Constraint Violations')
    plt.title('Convergence of Constraint Violations Over Generations')
    plt.grid(True)
    plt.legend()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Convergence plot saved to {filename}")

def run_ga_with_unitime(dataset_path, format_type='unitime_json', algorithm_name='ga'):
    """
    Run the Genetic Algorithm using the UniTime dataset.
    
    Args:
        dataset_path (str): Path to the UniTime dataset file
        format_type (str): Format of the dataset ('unitime_json' or 'unitime_xml')
        algorithm_name (str): Name of the algorithm ('ga', 'rl', or 'co')
        
    Returns:
        tuple: Best schedule, statistics, and other relevant information
    """
    logger.info(f"Running GA with UniTime dataset: {dataset_path}")
    
    # Step 1: Load the UniTime dataset
    if format_type == 'unitime_json':
        spaces_dict, groups_dict, lecturers_dict, activities_dict, slots = load_unitime_json(
            dataset_path, algorithm_name
        )
    elif format_type == 'unitime_xml':
        spaces_dict, groups_dict, lecturers_dict, activities_dict, slots = load_unitime_xml(
            dataset_path, algorithm_name
        )
    else:
        raise ValueError(f"Unsupported format type: {format_type}")
    
    # Step 2: Set up the DEAP framework for genetic algorithm
    # Create fitness and individual types if they don't exist
    if 'FitnessMulti' in dir(creator):
        del creator.FitnessMulti
    if 'Individual' in dir(creator):
        del creator.Individual
        
    # Set weights according to UniTime dataset optimization values:
    # - Room Weight: Increased to 10 (was 1) to prioritize room constraint optimization
    # - Time Weight: 25
    # - Distribution Weight: 15
    # - Student Weight: 100
    creator.create("FitnessMulti", base.Fitness, weights=(-10.0, -25.0, -15.0, -100.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    
    # Helper functions for constraint evaluation
    def evaluate_room_constraints(activity, room_obj):
        """Calculate penalty for room capacity violations"""
        student_count = sum(groups_dict[group_id].size for group_id in activity.group_ids if group_id in groups_dict)
        violation = max(0, student_count - room_obj.size)
        return violation
    
    def evaluate_time_constraints(individual):
        """Evaluate time constraints including time conflicts for groups and lecturers"""
        time_violations = 0
        
        # Create a mapping of slots to activities
        slot_activities = {}  # {slot: [activity_ids]}
        for activity_id, slot, room in individual:
            if slot not in slot_activities:
                slot_activities[slot] = []
            slot_activities[slot].append(activity_id)
        
        # Check for group conflicts (same group scheduled at the same time)
        for slot, activity_ids in slot_activities.items():
            if len(activity_ids) <= 1:
                continue
                
            # Check for group conflicts
            group_conflicts = set()
            for i in range(len(activity_ids)):
                for j in range(i+1, len(activity_ids)):
                    activity1 = activities_dict[activity_ids[i]]
                    activity2 = activities_dict[activity_ids[j]]
                    
                    # Check for shared groups
                    shared_groups = set(activity1.group_ids).intersection(set(activity2.group_ids))
                    if shared_groups:
                        group_conflicts.add(frozenset([activity_ids[i], activity_ids[j]]))
            
            time_violations += len(group_conflicts)
        
        # Check for lecturer conflicts
        for slot, activity_ids in slot_activities.items():
            if len(activity_ids) <= 1:
                continue
                
            # Check for lecturer conflicts
            lecturer_conflicts = set()
            for i in range(len(activity_ids)):
                for j in range(i+1, len(activity_ids)):
                    activity1 = activities_dict[activity_ids[i]]
                    activity2 = activities_dict[activity_ids[j]]
                    
                    # Check if same lecturer
                    if activity1.teacher_id == activity2.teacher_id and activity1.teacher_id is not None:
                        lecturer_conflicts.add(frozenset([activity_ids[i], activity_ids[j]]))
            
            time_violations += len(lecturer_conflicts)
            
        return time_violations
    
    def evaluate_distribution_constraints(individual):
        """Evaluate distribution constraints (hard and soft)"""
        # This is a placeholder for actual distribution constraint evaluation
        # In a real implementation, we would need to load and check the distribution
        # constraints from the dataset
        distribution_violations = 0
        
        # Example: Check if consecutive activities for same lecturer aren't too close or far
        # (This is simplified - actual distribution constraints would be more complex)
        lecturer_slots = {}  # {lecturer_id: [(activity_id, slot)]}
        
        for activity_id, slot, room in individual:
            activity = activities_dict[activity_id]
            lecturer_id = activity.teacher_id
            
            if lecturer_id is not None:
                if lecturer_id not in lecturer_slots:
                    lecturer_slots[lecturer_id] = []
                lecturer_slots[lecturer_id].append((activity_id, slot))
        
        # Check for each lecturer's consecutive classes
        for lecturer_id, assignments in lecturer_slots.items():
            if len(assignments) <= 1:
                continue
                
            # Sort by slot
            sorted_assignments = sorted(assignments, key=lambda x: x[1])
            
            # Check consecutive assignments
            for i in range(len(sorted_assignments) - 1):
                current_slot = sorted_assignments[i][1]
                next_slot = sorted_assignments[i + 1][1]
                
                # Check if on same day
                if current_slot[:3] == next_slot[:3]:
                    current_period = int(current_slot[3:])
                    next_period = int(next_slot[3:])
                    
                    # If periods are consecutive, no violation
                    if next_period - current_period == 1:
                        continue
                    
                    # Otherwise, add a soft constraint violation
                    distribution_violations += 1
        
        return distribution_violations
    
    def evaluate_student_constraints(individual):
        """Evaluate student-related constraints"""
        student_violations = 0
        
        # Create a mapping of slots to activities
        slot_activities = {}  # {slot: [activity_ids]}
        for activity_id, slot, room in individual:
            if slot not in slot_activities:
                slot_activities[slot] = []
            slot_activities[slot].append(activity_id)
            
        # Check for student conflicts
        for slot, activity_ids in slot_activities.items():
            if len(activity_ids) <= 1:
                continue
                
            # In a more sophisticated implementation, we would check student enrollments
            # and calculate actual conflicts
            
            # For now, assume student conflicts are already covered by group conflicts in time_constraints
            pass
            
        return student_violations
    
    # Add an improved room selection strategy
    def select_room(activity):
        """
        Select the most suitable room for an activity based on student count and capacity.
        """
        # Get all possible rooms
        possible_rooms = list(spaces_dict.values())
        
        # Calculate total student count
        student_count = sum(groups_dict[group_id].size for group_id in activity.group_ids if group_id in groups_dict)
        
        # Filter rooms by capacity - find rooms that can fit all students
        suitable_rooms = [room for room in possible_rooms if room.size >= student_count]
        
        # If no suitable rooms, use the largest available room
        if not suitable_rooms:
            suitable_rooms = sorted(possible_rooms, key=lambda r: r.size, reverse=True)
            
        # Sort by smallest suitable room first to optimize room usage
        else:
            # Sort by capacity (smallest suitable room first)
            suitable_rooms = sorted(suitable_rooms, key=lambda r: r.size)
        
        # Return a random room from the top 3 suitable rooms to add diversity
        room_sample = suitable_rooms[:3] if len(suitable_rooms) >= 3 else suitable_rooms
        return random.choice(room_sample).code

    # Add improved time slot selection strategy
    def select_time_slot(activity, room):
        """
        Select the most suitable time slot for an activity.
        Takes into account room unavailability data.
        """
        # Get all possible slots
        possible_slots = slots.copy()
        
        # Check if room has unavailability data
        if hasattr(spaces_dict[room], 'unavailable') and spaces_dict[room].unavailable:
            # Get unavailable slots
            unavailable_slots = set()
            
            for unavail in spaces_dict[room].unavailable:
                if 'days' in unavail and unavail['days'] and 'start' in unavail and 'length' in unavail:
                    days_pattern = unavail['days']
                    start_time = int(unavail['start'])
                    length = int(unavail['length'])
                    
                    # Map day pattern to day codes
                    day_map = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
                    for i, day_bit in enumerate(days_pattern):
                        if day_bit == '1' and i < len(day_map):
                            day_code = day_map[i]
                            
                            # Calculate all affected periods
                            # This is a simplified approach - in a real application, 
                            # we would need more complex time calculations
                            for period in range(1, 10):  # Assuming up to 10 periods
                                unavailable_slots.add(f"{day_code}{period}")
            
            # Filter out unavailable slots
            available_slots = [slot for slot in possible_slots if slot not in unavailable_slots]
        else:
            # If no unavailability data, all slots are available
            available_slots = possible_slots
        
        # If no available slots, use any slot
        if not available_slots:
            available_slots = possible_slots
        
        # Return a random available slot
        return random.choice(available_slots)

    # Step 3: Define the individual generator function
    def generate_individual():
        """
        Generate a random individual using improved room and time slot selection.
        """
        individual = []
        
        for activity_id, activity in activities_dict.items():
            # Select a room using the improved room selection strategy
            room = select_room(activity)
            
            # Select a time slot using the improved time slot selection strategy
            slot = select_time_slot(activity, room)
            
            # Add this assignment to the individual
            individual.append((activity_id, slot, room))
        
        return individual
        
    # Register the individual and population generation functions
    toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Step 4: Define the evaluation function
    def evaluate(individual):
        """
        Evaluate a timetable considering all constraints with proper weights.
        Returns a tuple of constraint violations:
        (room_constraints, time_constraints, distribution_constraints, student_constraints)
        """
        # Room constraints (weight 10)
        room_constraints = 0
        slot_room_assignments = {}
        
        for activity_id, slot, room in individual:
            activity = activities_dict[activity_id]
            room_obj = spaces_dict[room]
            
            # Calculate room capacity violations
            room_constraints += evaluate_room_constraints(activity, room_obj)
            
            # Track room assignments for room conflict detection
            if slot not in slot_room_assignments:
                slot_room_assignments[slot] = {}
            if room not in slot_room_assignments[slot]:
                slot_room_assignments[slot][room] = []
            slot_room_assignments[slot][room].append(activity_id)
        
        # Add room conflicts (same room used by multiple activities at same time)
        for slot, rooms in slot_room_assignments.items():
            for room, activities in rooms.items():
                if len(activities) > 1:
                    room_conflicts = len(activities) - 1
                    # Room conflicts are also counted as room constraints
                    room_constraints += room_conflicts
        
        # Time constraints (weight 25)
        time_constraints = evaluate_time_constraints(individual)
        
        # Distribution constraints (weight 15)
        distribution_constraints = evaluate_distribution_constraints(individual)
        
        # Student constraints (weight 100)
        student_constraints = evaluate_student_constraints(individual)
        
        return room_constraints, time_constraints, distribution_constraints, student_constraints
    
    # Register the genetic operators
    toolbox.register("evaluate", evaluate)
    
    # Custom crossover operator that preserves good room assignments
    def custom_crossover(ind1, ind2):
        """
        Custom crossover operator that preserves good room assignments.
        For each activity, choose the assignment from parent with better fitness for that particular activity.
        """
        # Make sure individuals have the same length
        if len(ind1) != len(ind2):
            raise ValueError("Individuals must have the same length")
            
        # Create offspring
        offspring1 = creator.Individual(ind1[:])  # Create a copy as a proper Individual
        offspring2 = creator.Individual(ind2[:])  # Create a copy as a proper Individual
        
        # For each activity position
        for i in range(len(ind1)):
            activity_id1, slot1, room1 = ind1[i]
            activity_id2, slot2, room2 = ind2[i]
            
            # Make sure we're dealing with the same activity
            if activity_id1 != activity_id2:
                # If activities are different, sort both individuals by activity_id
                ind1_sorted = sorted(ind1, key=lambda x: x[0])
                ind2_sorted = sorted(ind2, key=lambda x: x[0])
                offspring1 = creator.Individual(tools.cxTwoPoint(ind1_sorted, ind2_sorted)[0])
                offspring2 = creator.Individual(tools.cxTwoPoint(ind1_sorted, ind2_sorted)[1])
                return offspring1, offspring2
            
            # Calculate fitness contribution for each assignment
            # We'll simplify by just looking at room capacity violations
            activity = activities_dict[activity_id1]
            room_obj1 = spaces_dict[room1]
            room_obj2 = spaces_dict[room2]
            
            violation1 = evaluate_room_constraints(activity, room_obj1)
            violation2 = evaluate_room_constraints(activity, room_obj2)
            
            # Choose better assignment with 70% probability, otherwise random
            if random.random() < 0.7:
                if violation1 <= violation2:
                    offspring1[i] = (activity_id1, slot1, room1)
                    # Introduce some slot variation for the second offspring
                    offspring2[i] = (activity_id2, random.choice(slots), room2)
                else:
                    offspring1[i] = (activity_id1, random.choice(slots), room1)
                    offspring2[i] = (activity_id2, slot2, room2)
            else:
                if random.random() < 0.5:
                    offspring1[i] = (activity_id1, slot1, room1)
                    offspring2[i] = (activity_id2, slot2, room2)
                else:
                    offspring1[i] = (activity_id1, slot2, room2)
                    offspring2[i] = (activity_id2, slot1, room1)
        
        return offspring1, offspring2
    
    # Custom mutation operator that targets room constraint violations
    def custom_mutation(individual, indpb):
        """
        Custom mutation operator that specifically targets assignments with high room constraint violations.
        indpb: Independent probability of each attribute to be mutated
        """
        # Create a copy as a proper Individual to avoid mutating the original
        result = creator.Individual(individual[:])
        
        for i in range(len(result)):
            if random.random() < indpb:
                activity_id, slot, room = result[i]
                activity = activities_dict[activity_id]
                
                # Calculate current violation
                room_obj = spaces_dict[room]
                violation = evaluate_room_constraints(activity, room_obj)
                
                # If violation exists, try to find a better room
                if violation > 0:
                    # Get all rooms that have enough capacity
                    student_count = sum(groups_dict[group_id].size for group_id in activity.group_ids if group_id in groups_dict)
                    suitable_rooms = [r for r in spaces_dict.values() if r.size >= student_count]
                    
                    if suitable_rooms:
                        # Choose a random suitable room
                        new_room = random.choice(suitable_rooms).code
                        
                        # Update assignment with new room and possibly new slot
                        if random.random() < 0.3:  # 30% chance to also change the slot
                            new_slot = select_time_slot(activity, new_room)
                            result[i] = (activity_id, new_slot, new_room)
                        else:
                            result[i] = (activity_id, slot, new_room)
                else:
                    # No room violation, possibly change slot to optimize time constraints
                    if random.random() < 0.2:  # 20% chance
                        new_slot = select_time_slot(activity, room)
                        result[i] = (activity_id, new_slot, room)
        
        return (result,)
    
    # Register custom operators
    toolbox.register("mate", custom_crossover)
    toolbox.register("mutate", custom_mutation, indpb=0.1)  # 10% chance to mutate each assignment
    
    # Use tournament selection with higher pressure
    toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament size 3
    
    # Step 5: Create an initial population
    pop = toolbox.population(n=POP_SIZE)
    
    # Step 6: Set up statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    # Create a hall of fame to store the best individuals
    hof = tools.HallOfFame(1)
    
    # Step 7: Run the algorithm
    start_time = datetime.now()
    pop, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=CXPB, mutpb=MUTPB, 
        ngen=GENERATIONS, stats=stats, halloffame=hof, verbose=True
    )
    end_time = datetime.now()
    total_runtime = (end_time - start_time).total_seconds()
    logger.info(f"Total runtime: {total_runtime:.2f} seconds")

    # Generate convergence plot
    plot_convergence(logbook, "convergence_plot.png")

    # Log final statistics
    logger.info("\n===== FINAL STATISTICS =====")
    gen_stats = logbook.select("min", "max", "std")
    for gen, (min_val, max_val, std_val) in enumerate(zip(logbook.select("min"), logbook.select("max"), logbook.select("std"))):
        logger.info(f"Gen {gen}: min={min_val}, max={max_val}, std={std_val}")

    # Step 8: Process the results
    best_individual = hof[0]
    best_timetable = {}
    
    for assignment in best_individual:
        activity_id, slot, room = assignment
        if slot not in best_timetable:
            best_timetable[slot] = {}
        
        # Get the activity object
        activity = activities_dict[activity_id]
        
        # Parse day and period from slot (e.g., "MON1" -> "MON", 1)
        day_name = slot[:3]
        period_num = int(slot[3:])
        
        # Create properly structured objects for the frontend as per the memory requirements
        day_obj = {"name": day_name, "code": day_name}
        period_obj = {"name": f"Period {period_num}", "code": str(period_num)}
        room_obj = {"name": spaces_dict[room].code, "code": room, "capacity": spaces_dict[room].size}
        
        # Get activity details
        subject = activity.subject
        lecturer = lecturers_dict[activity.teacher_id].name if activity.teacher_id in lecturers_dict else "Unknown"
        group_ids = activity.group_ids
        
        # Create the activity with the structure expected by the frontend
        activity_entry = {
            "id": activity_id,
            "subject": subject,
            "lecturer": lecturer,
            "groups": group_ids,
            "day": day_obj,         # Properly structured day object
            "period": period_obj,   # Properly structured period object
            "room": room_obj,       # Properly structured room object
            "algorithm": algorithm_name  # Ensure algorithm field is set
        }
        
        # Add to timetable
        if slot not in best_timetable:
            best_timetable[slot] = {}
        best_timetable[slot][room] = activity_entry
    
    # Print the results
    logger.info("\n===== GENETIC ALGORITHM RESULTS =====")
    logger.info(f"Best fitness: {best_individual.fitness.values}")
    logger.info(f"Total assignments: {len(best_individual)}")
    
    # Return the results
    return best_timetable, pop, logbook, hof

def run_ga_with_restart(toolbox, pop_size, ngen, cxpb, mutpb, max_restarts=3):
    """
    Run the genetic algorithm with restarts to avoid local optima.
    """
    best_individuals = []
    best_fitnesses = []
    all_logbooks = []
    
    for restart in range(max_restarts):
        logger.info(f"Starting GA run {restart+1}/{max_restarts}")
        
        # Create a new random population
        pop = toolbox.population(n=pop_size)
        
        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
            
        # Keep track of the best individual
        hof = tools.HallOfFame(1)
        
        # Create statistics object
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        stats.register("std", np.std, axis=0)

        # Run the algorithm
        pop, logbook = algorithms.eaSimple(
            pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, 
            stats=stats, halloffame=hof, verbose=True
        )
        
        # Store results
        best_individuals.append(hof[0])
        best_fitnesses.append(hof[0].fitness.values)
        all_logbooks.append(logbook)
        
        # If we found a good solution, stop early
        best_fitness = hof[0].fitness.values
        if best_fitness[0] < 50 and best_fitness[1] < 5 and best_fitness[2] < 10:
            logger.info(f"Found good solution in run {restart+1}, stopping early")
            break
    
    # Find the best run
    best_run_idx = np.argmin([fit[0] + fit[1]*25 + fit[2]*15 + fit[3]*100 for fit in best_fitnesses])
    
    return best_individuals[best_run_idx], pop, all_logbooks[best_run_idx], tools.HallOfFame(1)

def main():
    """
    Main function to run the example.
    """
    logger.info("Starting UniTime GA integration")
    
    # Choose the algorithm name ('ga', 'rl', or 'co')
    algorithm_name = 'ga'
    start_time = time.time()
    
    try:
        # Run the GA with the UniTime dataset
        best_timetable, pop, logbook, hof = run_ga_with_unitime(
            UNITIME_JSON_PATH, 'unitime_json', algorithm_name
        )
        
        # Log the runtime
        runtime = time.time() - start_time
        logger.info(f"Total runtime: {runtime:.2f} seconds")
        
        # Generate a plot of the convergence
        plot_convergence(logbook, "convergence_plot.png")
        
        # Log statistics
        logger.info("\n===== FINAL STATISTICS =====")
        for i, gen in enumerate(logbook.select("gen")):
            min_vals = logbook.select("min")[i]
            max_vals = logbook.select("max")[i]
            std_vals = logbook.select("std")[i]
            logger.info(f"Gen {gen}: min={min_vals}, max={max_vals}, std={std_vals}        ")
            
        # Process the results
        best_fitness = hof[0].fitness.values
        
        # Log the best fitness
        logger.info("\n===== GENETIC ALGORITHM RESULTS =====")
        logger.info(f"Best fitness: {best_fitness}")
        logger.info(f"Total assignments: {len(best_timetable)}")
        
        # Log statistics and results
        logger.info("\n===== GA STATISTICS =====")
        logger.info(f"Total assignments: {len(best_timetable)}")
        logger.info(f"Room constraint violations: {best_fitness[0]}")
        logger.info(f"Time constraint violations: {best_fitness[1]}")
        logger.info(f"Distribution constraint violations: {best_fitness[2]}")
        logger.info(f"Student constraint violations: {best_fitness[3]}")
        logger.info(f"Best fitness score: {best_fitness}")
        
        # Save results to JSON file
        output_file = f"unitime_ga_results_{algorithm_name}.json"
        with open(output_file, 'w') as f:
            json.dump(best_timetable, f, indent=2)
        logger.info(f"Results saved to {output_file}")
        
        # Generate HTML report
        html_file = f"unitime_ga_report_{algorithm_name}.html"
        generate_html_report(best_timetable, logbook, best_fitness, html_file, runtime)
        logger.info(f"HTML report generated: {html_file}")
        logger.info(f"This timetable can be loaded into your frontend application.")
        
    except Exception as e:
        logger.error(f"Error running GA: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
