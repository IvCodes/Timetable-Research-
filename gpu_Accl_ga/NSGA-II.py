#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU-Accelerated Genetic Algorithm for University Course Scheduling
"""

import json
import time
import random
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from deap import base, creator, tools, algorithms
from collections import defaultdict

# Set style for plots
sns.set_style("whitegrid")

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_dataset(filename="sliit_computing_dataset.json"):
    """Load the dataset from a JSON file"""
    try:
        with open(filename, 'r') as f:
            dataset = json.load(f)
            
        # Extract components for easy access
        modules = dataset['modules']
        spaces = dataset['spaces']
        lecturers = [user for user in dataset['users'] if user.get('id', '').startswith('FA')]
        
        # Create student groups based on subgroup IDs in activities
        student_groups = set()
        semester_groups = {'1': [], '2': []}
        
        print("\nAnalyzing student groups and modules:")
        for activity in dataset['activities']:
            module_code = activity.get('subject', '')
            # Updated semester detection logic
            module_semester = 2 if module_code[3] in ['5', '6', '7', '8', '9'] else 1
            print(f"\nProcessing activity: {activity.get('name', 'Unknown')}")
            print(f"Module: {module_code}, Semester: {module_semester}")
            
            subgroup_ids = activity.get('subgroup_ids', [])
            for group_id in subgroup_ids:
                if len(group_id) >= 4 and group_id.startswith('Y') and 'S' in group_id:
                    # Extract year and semester from format "Y1S1.1"
                    year = int(group_id[1])
                    s_index = group_id.find('S')
                    if s_index != -1 and s_index + 1 < len(group_id):
                        semester = int(group_id[s_index + 1])
                        # Get base group (e.g., "Y1S1")
                        base_group = group_id[:s_index + 2]
                        student_groups.add(base_group)
                        semester_groups[str(semester)].append(base_group)
                        print(f"Found group: {base_group} from {group_id}")
                        print(f"  Year: {year}, Semester: {semester}")
                        
                        # Verify semester consistency
                        if semester != module_semester:
                            print(f"Warning: Semester mismatch!")
                            print(f"  Group semester: {semester}")
                            print(f"  Module semester: {module_semester}")
                    else:
                        print(f"Warning: Invalid semester format in group ID: {group_id}")
                else:
                    print(f"Warning: Invalid group ID format: {group_id}")
        
        # Convert to list of dictionaries with metadata
        student_groups = [
            {
                'id': group_id,
                'year': int(group_id[1]),
                'semester': int(group_id[3]),
                'size': 40  # Approximate size per group
            } 
            for group_id in sorted(student_groups)
        ]
        
        print(f"\nLoaded {len(modules)} modules")
        print(f"Loaded {len(spaces)} spaces")
        print(f"Loaded {len(lecturers)} lecturers")
        print(f"Loaded {len(dataset['activities'])} activities")
        print(f"Found {len(student_groups)} student groups:")
        
        print("\nSemester 1 groups:")
        for group in sorted(set(semester_groups['1'])):
            print(f"  - {group}")
            
        print("\nSemester 2 groups:")
        for group in sorted(set(semester_groups['2'])):
            print(f"  - {group}")
        
        return dataset, modules, spaces, lecturers, student_groups
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

def create_tensor_representation(individual):
    """Convert schedule to tensor representation"""
    try:
        # Convert schedule to tensor format
        schedule_data = []
        for event in individual:
            time_slot = event['time_slot']
            room_id = hash(event['space_id']) % 1000
            teacher_id = hash(event['teacher_ids'][0]) % 1000
            group_id = hash(event['subgroup_ids'][0]) % 1000
            
            schedule_data.append([
                float(time_slot),
                float(room_id),
                float(teacher_id),
                float(group_id)
            ])
        
        # Create tensor and move to GPU if available
        tensor = torch.tensor(schedule_data, dtype=torch.float32)
        return tensor.to(device)
        
    except Exception as e:
        print(f"Error creating tensor representation: {str(e)}")
        raise

def compute_conflicts_gpu(schedule_tensor):
    """Compute conflicts using GPU acceleration with batched operations"""
    n_events = schedule_tensor.size(0)
    
    # Pre-compute all masks at once
    times = schedule_tensor[:, 0].view(-1, 1)
    rooms = schedule_tensor[:, 1].view(-1, 1)
    teachers = schedule_tensor[:, 2].view(-1, 1)
    groups = schedule_tensor[:, 3].view(-1, 1)
    
    # Use batched operations for conflict detection
    with torch.no_grad():  # Disable gradient computation for speed
        time_conflicts = (times == times.T)
        conflicts = torch.zeros_like(time_conflicts, dtype=torch.float)
        
        # Where there are time conflicts, check other resources
        mask = time_conflicts
        conflicts += ((rooms == rooms.T) & mask).float()
        conflicts += ((teachers == teachers.T) & mask).float()
        conflicts += ((groups == groups.T) & mask).float()
    
    # Remove self-conflicts and count only unique conflicts
    conflicts.fill_diagonal_(0)
    conflicts = torch.triu(conflicts)
    
    return torch.sum(conflicts, dim=1)

def compute_preference_score_gpu(schedule_tensor):
    """Compute preference score"""
    # Time slot distribution
    slot_counts = torch.bincount(schedule_tensor[:, 0], minlength=40).float()
    avg_slots = torch.mean(slot_counts)
    slot_score = -torch.sum(torch.abs(slot_counts - avg_slots))
    
    # Resource distribution scores
    resource_score = 0
    for i in range(1, 4):
        resource_counts = torch.bincount(schedule_tensor[:, i]).float()
        avg_resource = torch.mean(resource_counts)
        resource_score -= torch.sum(torch.abs(resource_counts - avg_resource))
    
    return (slot_score + resource_score).item()

def create_random_schedule():
    """Create a random schedule"""
    schedule = []
    for activity in dataset['activities']:
        event = activity.copy()  # Copy all activity data
        # Add random assignments
        event['space_id'] = random.choice(spaces)['name']
        event['time_slot'] = random.randint(0, 39)  # 8 slots per day * 5 days
        schedule.append(event)
    return schedule

def custom_mutate(individual, indpb):
    """GPU-optimized mutation operator"""
    schedule_tensor = create_tensor_representation(individual)
    
    num_events = len(individual)
    mutation_mask = torch.rand(num_events, device=device) < indpb
    events_to_mutate = torch.where(mutation_mask)[0]
    
    for idx in events_to_mutate:
        event = individual[idx.item()]
        
        # Time slot mutation
        if random.random() < indpb:
            current_conflicts = compute_conflicts_gpu(schedule_tensor)[idx]
            best_time_slot = event['time_slot']
            min_conflicts = current_conflicts
            
            for _ in range(3):
                test_slot = random.randint(0, 39)
                old_slot = schedule_tensor[idx, 0].item()
                schedule_tensor[idx, 0] = test_slot
                conflicts = compute_conflicts_gpu(schedule_tensor)[idx]
                
                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    best_time_slot = test_slot
                else:
                    schedule_tensor[idx, 0] = old_slot
            
            event['time_slot'] = best_time_slot
        
        # Room mutation
        if random.random() < indpb:
            # Get group size (assuming 40 students per group)
            group_size = len(event['subgroup_ids']) * 40
            
            suitable_rooms = [
                space for space in spaces 
                if int(space.get('capacity', 0)) >= group_size
            ]
            if suitable_rooms:
                new_room = random.choice(suitable_rooms)
                event['space_id'] = new_room['name']
                schedule_tensor[idx, 1] = hash(new_room['name']) % 1000
    
    return individual,

def evaluate_batch_gpu(individuals):
    """Evaluate a batch of individuals using GPU acceleration"""
    try:
        batch_size = len(individuals)
        tensors = [create_tensor_representation(ind) for ind in individuals]
        batch_tensor = torch.stack(tensors).to(device)
        
        with torch.no_grad():
            fitnesses = []
            for i in range(batch_size):
                schedule_tensor = batch_tensor[i]
                conflicts = analyze_detailed_conflicts(schedule_tensor)
                
                # Return all four conflict types as separate fitness values
                fitness = (
                    float(conflicts['teacher_conflicts']),
                    float(conflicts['room_conflicts']),
                    float(conflicts['interval_conflicts']),
                    float(conflicts['period_conflicts'])
                )
                fitnesses.append(fitness)
            
            return fitnesses
            
    except Exception as e:
        print(f"Error in batch evaluation: {str(e)}")
        raise

def analyze_detailed_conflicts(schedule_tensor):
    """Analyze different types of conflicts in detail"""
    n_events = schedule_tensor.size(0)
    
    # Convert to long/int64 for comparison operations
    times = schedule_tensor[:, 0].long().view(-1, 1)
    rooms = schedule_tensor[:, 1].long().view(-1, 1)
    teachers = schedule_tensor[:, 2].long().view(-1, 1)
    groups = schedule_tensor[:, 3].long().view(-1, 1)
    
    with torch.no_grad():
        # Time conflicts matrix (same time slot)
        time_conflicts = (times == times.T).to(dtype=torch.float32)
        
        # Teacher conflicts (same teacher, same time)
        teacher_mask = (teachers == teachers.T).to(dtype=torch.float32)
        teacher_conflicts = torch.logical_and(teacher_mask, time_conflicts).to(dtype=torch.float32)
        teacher_conflicts.fill_diagonal_(0)
        teacher_conflicts = torch.triu(teacher_conflicts)
        
        # Room conflicts (same room, same time)
        room_mask = (rooms == rooms.T).to(dtype=torch.float32)
        room_conflicts = torch.logical_and(room_mask, time_conflicts).to(dtype=torch.float32)
        room_conflicts.fill_diagonal_(0)
        room_conflicts = torch.triu(room_conflicts)
        
        # Period conflicts (consecutive periods for same group)
        period_mask = (torch.abs(times - times.T) == 1).to(dtype=torch.float32)
        group_mask = (groups == groups.T).to(dtype=torch.float32)
        period_conflicts = torch.logical_and(group_mask, period_mask).to(dtype=torch.float32)
        period_conflicts.fill_diagonal_(0)
        period_conflicts = torch.triu(period_conflicts)
        
        # Interval conflicts (more than 2 consecutive periods)
        interval_mask = (torch.abs(times - times.T) <= 2).to(dtype=torch.float32)
        interval_conflicts = torch.logical_and(group_mask, interval_mask).to(dtype=torch.float32)
        interval_conflicts.fill_diagonal_(0)
        interval_conflicts = torch.triu(interval_conflicts)
        
        # Count conflicts
        return {
            'teacher_conflicts': int(teacher_conflicts.sum().item()),
            'room_conflicts': int(room_conflicts.sum().item()),
            'period_conflicts': int(period_conflicts.sum().item()),
            'interval_conflicts': int(interval_conflicts.sum().item())
        }

def run_parallel_ga_gpu():
    """Run the genetic algorithm with GPU acceleration and batch processing"""
    try:
        print("\nInitializing GPU-accelerated GA...")
        print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        
        POPULATION_SIZE = 100
        GENERATIONS = 50
        CXPB, MUTPB = 0.7, 0.3
        TOURNAMENT_SIZE = 3
        BATCH_SIZE = 32
        
        population = toolbox.population(n=POPULATION_SIZE)
        
        # Print header for evolution stats
        header = f"\n{'Generation':>5} {'Evaluations':>12} {'Teacher':>10} {'Room':>10} {'Interval':>10} {'Period':>10} {'Total':>10}"
        print(header)
        print("-" * len(header))
        
        # Evaluate initial population
        for i in range(0, POPULATION_SIZE, BATCH_SIZE):
            batch = population[i:i + BATCH_SIZE]
            fitnesses = evaluate_batch_gpu(batch)
            for ind, fit in zip(batch, fitnesses):
                ind.fitness.values = fit
        
        # Main evolution loop
        for gen in range(GENERATIONS):
            try:
                offspring = tools.selTournament(population, len(population), TOURNAMENT_SIZE)
                offspring = [toolbox.clone(ind) for ind in offspring]
                
                # Apply crossover and mutation
                for i in range(1, len(offspring), 2):
                    if random.random() < CXPB:
                        offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
                        del offspring[i-1].fitness.values, offspring[i].fitness.values
                
                for i in range(len(offspring)):
                    if random.random() < MUTPB:
                        offspring[i] = toolbox.mutate(offspring[i])[0]
                        del offspring[i].fitness.values
                
                # Evaluate invalid individuals
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                nevals = len(invalid_ind)
                
                for i in range(0, len(invalid_ind), BATCH_SIZE):
                    batch = invalid_ind[i:i + BATCH_SIZE]
                    fitnesses = evaluate_batch_gpu(batch)
                    for ind, fit in zip(batch, fitnesses):
                        ind.fitness.values = fit
                
                # Replace population
                population[:] = offspring
                
                # Get best individual's conflicts
                best_ind = tools.selBest(population, 1)[0]
                conflicts = {
                    'teacher': int(best_ind.fitness.values[0]),
                    'room': int(best_ind.fitness.values[1]),
                    'interval': int(best_ind.fitness.values[2]),
                    'period': int(best_ind.fitness.values[3])
                }
                
                total_conflicts = sum(conflicts.values())
                
                # Print current generation stats
                print(f"{gen:5d} {nevals:12d} {conflicts['teacher']:10d} {conflicts['room']:10d} "
                      f"{conflicts['interval']:10d} {conflicts['period']:10d} {total_conflicts:10d}")
                
                # Early stopping if perfect solution found
                if total_conflicts == 0:
                    print("\nPerfect solution found!")
                    break
                    
            except Exception as e:
                print(f"\nError in generation {gen}: {str(e)}")
                continue
        
        return population
        
    except Exception as e:
        print(f"Fatal error in GA: {str(e)}")
        raise

def cleanup_gpu():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def visualize_results(population, hof):
    """Visualize the results of the genetic algorithm"""
    plt.figure(figsize=(15, 5))
    
    # Plot constraint violations
    plt.subplot(1, 2, 1)
    violations = [ind.fitness.values[0] for ind in population]
    plt.hist(violations, bins=30, color='red', alpha=0.7)
    plt.title('Distribution of Constraint Violations')
    plt.xlabel('Number of Violations')
    plt.ylabel('Frequency')
    
    # Plot preference scores
    plt.subplot(1, 2, 2)
    scores = [ind.fitness.values[1] for ind in population]
    plt.hist(scores, bins=30, color='blue', alpha=0.7)
    plt.title('Distribution of Preference Scores')
    plt.xlabel('Preference Score')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def save_best_solution(best_solution):
    """Save the best solution to a JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'best_schedule_gpu_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump([{
            'activity': event['activity'],
            'module': event['module'],
            'lecturer': event['lecturer']['first_name'],
            'space': event['space']['name'],
            'group': event['group']['id'],
            'time_slot': event['time_slot']
        } for event in best_solution], f, indent=2)
    
    print(f"\nBest schedule saved to: {output_file}")

def time_slot_to_readable(slot_id):
    """Convert time slot ID to readable time format"""
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    times = ['8:30-10:30', '10:30-12:30', '13:30-15:30', '15:30-17:30']
                
    day = days[slot_id // 8]
    time = times[(slot_id % 8) // 2]
    return f"{day} {time}"

def analyze_constraints(solution):
    """Analyze the constraints and generate statistics for a solution"""
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    time_slots = ['08:30-09:30', '09:30-10:30', '10:30-11:30', '11:30-12:30',
                 '13:30-14:30', '14:30-15:30', '15:30-16:30', '16:30-17:30']
    
    # Initialize statistics
    stats = {
        'events_per_day': {day: 0 for day in days},
        'events_per_time': {slot: 0 for slot in time_slots},
        'room_utilization': defaultdict(int),
        'teacher_load': defaultdict(int),
        'group_schedule': defaultdict(int),
        'total_events': len(solution)
    }
    
    try:
        # Analyze each scheduled event
        for event in solution:
            # Convert float32 to int for indexing
            slot = int(event['time_slot'])
            day = days[slot // 8]
            time = time_slots[slot % 8]
            
            # Update statistics
            stats['events_per_day'][day] += 1
            stats['events_per_time'][time] += 1
            stats['room_utilization'][event['space_id']] += 1
            stats['teacher_load'][event['teacher_ids'][0]] += 1
            stats['group_schedule'][event['subgroup_ids'][0]] += 1
            
        # Calculate percentages and averages
        total_slots = len(days) * len(time_slots)
        stats['time_slot_utilization'] = (len(solution) / total_slots) * 100
        stats['avg_events_per_day'] = len(solution) / len(days)
        stats['avg_events_per_time'] = len(solution) / len(time_slots)
        
        return stats
        
    except Exception as e:
        print(f"Error in constraint analysis: {str(e)}")
        raise

def generate_timetable(solution):
    """Generate a formatted timetable from the solution"""
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    time_slots = ['08:30-09:30', '09:30-10:30', '10:30-11:30', '11:30-12:30',
                 '13:30-14:30', '14:30-15:30', '15:30-16:30', '16:30-17:30']
    
    # Initialize empty timetable
    timetable = {day: {time: [] for time in time_slots} for day in days}
    
    try:
        # Place events in timetable
        for event in solution:
            # Convert float32 to int for indexing
            slot = int(event['time_slot'])
            day = days[slot // 8]
            time = time_slots[slot % 8]
            
            # Get module details
            module_code = event.get('subject', '')
            module_name = next((m['name'] for m in modules if m['code'] == module_code), module_code)
            
            # Get teacher details
            teacher_ids = event.get('teacher_ids', [])
            teacher_names = []
            for teacher_id in teacher_ids:
                teacher = next((t for t in lecturers if t['id'] == teacher_id), None)
                if teacher:
                    teacher_names.append(f"{teacher['first_name']} {teacher['last_name']}")
            
            # Get room details
            room_id = event.get('space_id', '')
            room = next((r for r in spaces if r['name'] == room_id), {'name': room_id})
            
            # Get group details
            subgroup_ids = event.get('subgroup_ids', [])
            
            # Format event details
            event_details = {
                'module': f"{module_name} ({module_code})",
                'lecturer': ', '.join(teacher_names) if teacher_names else 'TBA',
                'room': room['name'],
                'group': ', '.join(subgroup_ids),
                'activity': event.get('name', ''),
                'duration': event.get('duration', 1)
            }
            
            timetable[day][time].append(event_details)
            
        return timetable
        
    except Exception as e:
        print(f"Error generating timetable: {str(e)}")
        print(f"Event causing error: {event}")
        raise

def save_timetable(timetable):
    """Save the timetable to an HTML file with improved formatting"""
    try:
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>University Timetable</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2196F3; text-align: center; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .event { margin-bottom: 5px; padding: 5px; background-color: #e7f3fe; border-radius: 3px; }
                .module { font-weight: bold; color: #2196F3; }
                .lecturer { color: #4CAF50; }
                .room { color: #f44336; }
                .group { color: #9C27B0; }
            </style>
        </head>
        <body>
            <h1>University Timetable</h1>
            <table>
                <tr>
                    <th>Time</th>
        """
        
        # Add days to header
        for day in timetable.keys():
            html_content += f"<th>{day}</th>"
        html_content += "</tr>"
        
        # Add time slots and events
        for time in list(timetable['Monday'].keys()):
            html_content += f"<tr><td>{time}</td>"
            
            for day in timetable.keys():
                html_content += "<td>"
                if timetable[day][time]:
                    for event in timetable[day][time]:
                        html_content += f"""
                        <div class='event'>
                            <div class='module'>{event['module']}</div>
                            <div class='lecturer'>{event['lecturer']}</div>
                            <div class='room'>Room: {event['room']}</div>
                            <div class='group'>Group: {event['group']}</div>
                            <div class='activity'>Activity: {event['activity']}</div>
                        </div>
                        """
                html_content += "</td>"
            html_content += "</tr>"
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open('timetable.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print("\nTimetable saved to timetable.html")
        
    except Exception as e:
        print(f"Error saving timetable: {str(e)}")
        raise

def generate_year_semester_timetables(solution):
    """Generate separate timetables for each specific student group"""
    # Group events by specific student groups
    group_timetables = {}
    
    print("\nGrouping events by specific student groups:")
    for event in solution:
        try:
            # Get subgroup IDs from the event
            subgroup_ids = event.get('subgroup_ids', [])
            if not subgroup_ids:
                print(f"Warning: No subgroup IDs for event: {event.get('name', 'Unknown')}")
                continue
                
            # Process each subgroup ID
            for group_id in subgroup_ids:
                # Handle group IDs in format "Y1S1.1"
                if group_id.startswith('Y') and 'S' in group_id:
                    # Extract year and semester
                    s_index = group_id.find('S')
                    if s_index != -1 and s_index + 1 < len(group_id):
                        year = group_id[1:s_index]  # Get the year number
                        semester = group_id[s_index+1]  # Get the semester number
                        
                        # Verify semester consistency
                        module_code = event.get('subject', '')
                        module_semester = 2 if module_code[3] in ['5', '6', '7', '8', '9'] else 1
                        
                        if int(semester) != module_semester:
                            print(f"Warning: Semester mismatch for event {event.get('name', 'Unknown')}")
                            print(f"Group semester: {semester}, Module semester: {module_semester}")
                            continue
                        
                        # Use the full group ID
                        if group_id not in group_timetables:
                            group_timetables[group_id] = []
                        
                        # Only add unique events to the group
                        if event not in group_timetables[group_id]:
                            group_timetables[group_id].append(event)
                            print(f"Added event {event.get('name', 'Unknown')} to group {group_id}")
                    else:
                        print(f"Warning: Invalid semester format in group ID: {group_id}")
                else:
                    print(f"Warning: Invalid group ID format: {group_id}")
                    
        except Exception as e:
            print(f"Warning: Error processing event: {str(e)}")
            continue
    
    if not group_timetables:
        print("Warning: No valid groups found. Creating single timetable.")
        return {"combined": generate_timetable(solution)}
    
    # Generate timetable for each specific group
    timetables = {}
    print("\nGenerating timetables for specific groups:")
    for group_id, events in group_timetables.items():
        print(f"Generating timetable for {group_id} with {len(events)} events")
        timetables[group_id] = generate_timetable(events)
    
    return timetables

def save_year_semester_timetables(timetables):
    """Save all timetables in a single HTML file, grouped by year and semester"""
    try:
        # Import os to handle directory creation
        import os
        
        # Create timetables directory if it doesn't exist
        timetables_dir = 'timetables'
        os.makedirs(timetables_dir, exist_ok=True)
        
        # Organize timetables by year and semester
        organized_timetables = {}
        for group_id in timetables.keys():
            if group_id == 'combined':
                continue
                
            # Extract year and semester from group ID (format: Y1S1.1)
            s_index = group_id.find('S')
            if s_index != -1:
                year_sem = group_id[:s_index+2]  # Get "Y1S1" part
                if year_sem not in organized_timetables:
                    organized_timetables[year_sem] = []
                organized_timetables[year_sem].append(group_id)
        
        # Create single HTML file
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>University Timetables</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 20px; 
                    max-width: 1400px; 
                    margin: 0 auto; 
                    padding: 20px;
                }
                h1 { color: #2196F3; text-align: center; }
                h2 { 
                    color: #4CAF50; 
                    border-bottom: 2px solid #4CAF50; 
                    padding-bottom: 10px; 
                    margin-top: 40px; 
                }
                h3 { 
                    color: #2196F3; 
                    margin-top: 20px; 
                    background-color: #f0f0f0; 
                    padding: 10px; 
                    border-radius: 5px; 
                }
                .nav-section {
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .nav-section h2 {
                    margin-top: 0;
                    color: #2196F3;
                    border-bottom: none;
                }
                .nav-links {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin-bottom: 15px;
                }
                .nav-links a {
                    display: inline-block;
                    padding: 8px 16px;
                    text-decoration: none;
                    color: white;
                    background: #4CAF50;
                    border-radius: 4px;
                    transition: background 0.3s;
                }
                .nav-links a:hover {
                    background: #45a049;
                }
                table { 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin-top: 20px; 
                    margin-bottom: 30px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                }
                th, td { 
                    border: 1px solid #ddd; 
                    padding: 8px; 
                    text-align: left; 
                }
                th { 
                    background-color: #4CAF50; 
                    color: white; 
                    position: sticky; 
                    top: 0; 
                    z-index: 10; 
                }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .event { 
                    margin-bottom: 5px; 
                    padding: 5px; 
                    background-color: #e7f3fe; 
                    border-radius: 3px; 
                }
                .module { font-weight: bold; color: #2196F3; }
                .lecturer { color: #4CAF50; }
                .room { color: #f44336; }
                .group { color: #9C27B0; }
                @media print {
                    body { max-width: none; }
                    table { page-break-inside: avoid; }
                }
            </style>
        </head>
        <body>
            <h1>University Timetables</h1>
            
            <!-- Navigation Section -->
            <div class="nav-section">
                <h2>Quick Navigation</h2>
        """
        
        # Sort year-semester groups to ensure consistent order
        sorted_year_sems = sorted(organized_timetables.keys())
        
        # Add navigation links grouped by year-semester
        for year_sem in sorted_year_sems:
            html_content += f'<h3>{year_sem}</h3><div class="nav-links">'
            groups = sorted(organized_timetables[year_sem])
            for group_id in groups:
                html_content += f'<a href="#{group_id}">{group_id}</a>'
            html_content += '</div>'
        
        html_content += '</div>'  # Close navigation section
        
        # Add timetables
        for year_sem in sorted_year_sems:
            # Add year-semester section
            html_content += f"<h2>{year_sem}</h2>"
            
            # Sort groups within this year-semester
            groups = sorted(organized_timetables[year_sem])
            
            for group_id in groups:
                # Add group timetable with ID for navigation
                html_content += f'<h3 id="{group_id}">{group_id}</h3>'
                html_content += """
                <table>
                    <tr>
                        <th>Time</th>
                """
                
                # Add days to header
                for day in timetables[group_id].keys():
                    html_content += f"<th>{day}</th>"
                html_content += "</tr>"
                
                # Add time slots and events
                for time in list(timetables[group_id]['Monday'].keys()):
                    html_content += f"<tr><td>{time}</td>"
                    
                    for day in timetables[group_id].keys():
                        html_content += "<td>"
                        if timetables[group_id][day][time]:
                            for event in timetables[group_id][day][time]:
                                html_content += f"""
                                <div class='event'>
                                    <div class='module'>{event['module']}</div>
                                    <div class='lecturer'>{event['lecturer']}</div>
                                    <div class='room'>Room: {event['room']}</div>
                                    <div class='group'>Group: {event['group']}</div>
                                    <div class='activity'>Activity: {event['activity']}</div>
                                </div>
                                """
                        html_content += "</td>"
                    html_content += "</tr>"
                
                html_content += "</table>"
        
        # Close HTML
        html_content += """
        </body>
        </html>
        """
        
        # Save to file in timetables directory
        output_file = os.path.join(timetables_dir, 'timetables.html')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"\nAll timetables saved to {output_file}")
            
    except Exception as e:
        print(f"Error saving timetables: {str(e)}")
        raise

def print_evolution_stats(gen, nevals, conflicts):
    """Print evolution statistics in a clean tabular format"""
    print(f"\nGeneration {gen}:")
    print("-" * 60)
    print(f"{'Conflict Type':<20} {'Count':>10}")
    print("-" * 60)
    for conflict_type, count in conflicts.items():
        print(f"{conflict_type.replace('_', ' ').title():<20} {count:>10}")
    print("-" * 60)
    print(f"Total Evaluations: {nevals}")
    print(f"Total Conflicts: {sum(conflicts.values())}")

def print_final_stats(solution, execution_time):
    """Print final statistics in a clean format"""
    print("\n" + "="*60)
    print("FINAL SOLUTION STATISTICS")
    print("="*60)
    
    # Calculate statistics
    stats = analyze_constraints(solution)
    
    # Print event distribution
    print("\nEvents per Day:")
    print("-" * 40)
    for day, count in stats['events_per_day'].items():
        print(f"{day:<15} {count:>5}")
    
    print("\nEvents per Time Slot:")
    print("-" * 40)
    for slot, count in stats['events_per_time'].items():
        print(f"{slot:<15} {count:>5}")
    
    print("\nRoom Utilization:")
    print("-" * 40)
    for room, count in stats['room_utilization'].items():
        print(f"{room:<20} {count:>5}")
    
    print("\nTeacher Load:")
    print("-" * 40)
    for teacher, count in stats['teacher_load'].items():
        print(f"{teacher:<20} {count:>5}")
    
    print("\nOverall Statistics:")
    print("-" * 40)
    print(f"Total Events Scheduled: {stats['total_events']}")
    print(f"Average Events per Day: {stats['avg_events_per_day']:.2f}")
    print(f"Time Slot Utilization: {stats['time_slot_utilization']:.2f}%")
    print(f"Execution Time: {execution_time}")
    print("="*60)

def main():
    """Main execution function"""
    try:
        # Initialize DEAP
        if 'FitnessMulti' in creator.__dict__:
            del creator.FitnessMulti
        if 'Individual' in creator.__dict__:
            del creator.Individual
            
        # Create fitness and individual with four objectives (all minimizing)
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        global toolbox
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, create_random_schedule)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate_batch_gpu)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", custom_mutate, indpb=0.2)
        toolbox.register("select", tools.selNSGA2)
        
        # Load dataset
        global dataset, modules, spaces, lecturers, student_groups
        dataset, modules, spaces, lecturers, student_groups = load_dataset()
        
        # Run genetic algorithm
        population = run_parallel_ga_gpu()
        
        # Generate timetables
        best_solution = tools.selBest(population, 1)[0]
        timetables = generate_year_semester_timetables(best_solution)
        
        # Save timetables
        save_year_semester_timetables(timetables)
        
        print("\nTimetable generation complete!")
        print("Check the output HTML files for the generated timetables.")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise
    finally:
        cleanup_gpu()

if __name__ == "__main__":
    main()
