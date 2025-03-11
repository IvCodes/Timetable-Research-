"""
Utility functions and data structures for timetable scheduling algorithms.
"""
import json
import numpy as np
import random

class Space:
    """Class representing a room or space in the university."""
    def __init__(self, code, size):
        self.code = code
        self.size = size
    
    def __repr__(self):
        return f"Space(code={self.code}, size={self.size})"

class Group:
    """Class representing a student group."""
    def __init__(self, id, size):
        self.id = id
        self.size = size
    
    def __repr__(self):
        return f"Group(id={self.id}, size={self.size})"

class Activity:
    """Class representing an academic activity (course, lab, etc.)."""
    def __init__(self, id, subject, teacher_id, group_ids, duration):
        self.id = id
        self.subject = subject
        self.teacher_id = teacher_id
        self.group_ids = group_ids
        self.duration = duration
    
    def __repr__(self):
        return f"Activity(id={self.id}, subject={self.subject}, teacher_id={self.teacher_id}, group_ids={self.group_ids}, duration={self.duration})"

class Lecturer:
    """Class representing a lecturer."""
    def __init__(self, id, name, department):
        self.id = id
        self.name = name
        self.department = department
    
    def __repr__(self):
        return f"Lecturer(id={self.id}, name={self.name}, department={self.department})"

def load_data(file_path):
    """Load data from JSON file and initialize objects."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Initialize spaces (rooms)
    spaces_dict = {}
    for space in data['spaces']:
        spaces_dict[space['code']] = Space(space['code'], space['size'])
    
    # Initialize student groups
    groups_dict = {}
    for group in data['groups']:
        groups_dict[group['id']] = Group(group['id'], group['size'])
    
    # Initialize lecturers
    lecturers_dict = {}
    for lecturer in data['lecturers']:
        lecturers_dict[lecturer['id']] = Lecturer(lecturer['id'], lecturer['name'], lecturer['department'])
    
    # Initialize activities
    activities_dict = {}
    for activity in data['activities']:
        activities_dict[activity['id']] = Activity(
            activity['id'], activity['subject'], activity['teacher_id'], 
            activity['group_ids'], activity['duration']
        )
    
    # Define time slots
    slots = []
    days = ["MON", "TUE", "WED", "THU", "FRI"]
    for day in days:
        for i in range(1, 9):  # 8 time slots per day
            slots.append(f"{day}{i}")
    
    return spaces_dict, groups_dict, lecturers_dict, activities_dict, slots

def evaluate_hard_constraints(timetable, activities_dict, groups_dict, spaces_dict):
    """
    Evaluate hard constraints in a timetable.
    
    Returns:
        Tuple of (vacant_room, prof_conflicts, room_size_conflicts, sub_group_conflicts, unasigned_activities)
    """
    vacant_rooms = []
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

            if not isinstance(activity, Activity):  # Ensure it's an Activity object
                vacant_room += 1
                vacant_rooms.append((slot, room))
            else:
                activities_set.add(activity.id)

                # Lecturer Conflict Check
                if activity.teacher_id in prof_set:
                    prof_conflicts += 1
                prof_set.add(activity.teacher_id)

                # Student Group Conflict Check
                sub_group_conflicts += len(
                    set(activity.group_ids).intersection(sub_group_set))

                group_size = 0
                for group_id in activity.group_ids:
                    group_size += groups_dict[group_id].size
                    sub_group_set.add(group_id)

                # Room Capacity Constraint Check
                if group_size > spaces_dict[room].size:
                    room_size_conflicts += 1

    # Unassigned Activity Count
    unasigned_activities -= len(activities_set)

    return vacant_room, prof_conflicts, room_size_conflicts, sub_group_conflicts, unasigned_activities

def evaluate_soft_constraints(schedule, groups_dict, lecturers_dict, slots):
    """
    Evaluates the soft constraints of a given schedule, handling missing (None) activities.
    This function measures:
    - Student group metrics: fatigue, idle time, lecture spread.
    - Lecturer metrics: fatigue, idle time, lecture spread, and workload balance.

    Parameters:
    - schedule (dict): The scheduled activities mapped by time slots and locations.
    - groups_dict (dict): Dictionary of student groups with group IDs as keys.
    - lecturers_dict (dict): Dictionary of lecturers with lecturer IDs as keys.
    - slots (list): Ordered list of available time slots.

    Returns:
    - final_score (float): Computed soft constraint score representing 
      schedule quality based on fatigue, idle time, spread, and workload balance.
    """

    # Initialize student group metrics
    group_fatigue = {g: 0 for g in groups_dict.keys()}
    group_idle_time = {g: 0 for g in groups_dict.keys()}
    group_lecture_spread = {g: 0 for g in groups_dict.keys()}

    # Initialize lecturer metrics
    lecturer_fatigue = {l: 0 for l in lecturers_dict.keys()}
    lecturer_idle_time = {l: 0 for l in lecturers_dict.keys()}
    lecturer_lecture_spread = {l: 0 for l in lecturers_dict.keys()}
    lecturer_workload = {l: 0 for l in lecturers_dict.keys()}

    # Track when each student group has classes
    group_slot_map = {g: [] for g in groups_dict.keys()}
    
    # Track when each lecturer teaches
    lecturer_slot_map = {l: [] for l in lecturers_dict.keys()}

    # Process each slot and room to extract relevant data
    for slot_idx, slot in enumerate(slots):
        if slot not in schedule:
            continue
            
        for room, activity in schedule[slot].items():
            if not isinstance(activity, Activity):
                continue
                
            # Update group metrics
            for group_id in activity.group_ids:
                group_fatigue[group_id] += 1
                group_slot_map[group_id].append(slot_idx)
                
            # Update lecturer metrics
            lecturer_id = activity.teacher_id
            lecturer_fatigue[lecturer_id] += 1
            lecturer_slot_map[lecturer_id].append(slot_idx)
            lecturer_workload[lecturer_id] += 1  # Simple workload measure

    # Normalize fatigue
    def normalize(metric_dict):
        if not metric_dict:
            return {}
        values = list(metric_dict.values())
        min_val = min(values) if values else 0
        max_val = max(values) if values else 0
        
        if max_val == min_val:
            return {k: 1.0 for k in metric_dict}
        
        return {k: 1 - ((v - min_val) / (max_val - min_val)) for k, v in metric_dict.items()}

    # Calculate lecture spread and idle time
    for group_id, slots_used in group_slot_map.items():
        if slots_used:
            slots_used.sort()
            # Spread is measured as how scattered the lectures are
            group_lecture_spread[group_id] = max(slots_used) - min(slots_used) + 1 - len(slots_used)
            
            # Idle time is measured as gaps between lectures
            idle_time = 0
            for i in range(len(slots_used) - 1):
                gap = slots_used[i + 1] - slots_used[i] - 1
                idle_time += gap
            group_idle_time[group_id] = idle_time

    for lecturer_id, slots_used in lecturer_slot_map.items():
        if slots_used:
            slots_used.sort()
            # Similar metrics for lecturers
            lecturer_lecture_spread[lecturer_id] = max(slots_used) - min(slots_used) + 1 - len(slots_used)
            
            idle_time = 0
            for i in range(len(slots_used) - 1):
                gap = slots_used[i + 1] - slots_used[i] - 1
                idle_time += gap
            lecturer_idle_time[lecturer_id] = idle_time
    
    # Normalize all metrics
    group_fatigue = normalize(group_fatigue)
    group_idle_time = normalize(group_idle_time)
    group_lecture_spread = normalize(group_lecture_spread)
    
    lecturer_fatigue = normalize(lecturer_fatigue)
    lecturer_idle_time = normalize(lecturer_idle_time)
    lecturer_lecture_spread = normalize(lecturer_lecture_spread)

    # Compute lecturer workload balance
    workload_values = np.array(list(lecturer_workload.values()))
    if len(workload_values) > 0 and workload_values.std() > 0:
        workload_balance = 1 - (workload_values.std() / workload_values.mean())
    else:
        workload_balance = 1.0  # Perfect balance if no variance
    
    # Compute average metrics
    avg_group_fatigue = sum(group_fatigue.values()) / len(group_fatigue) if group_fatigue else 0
    avg_group_idle = sum(group_idle_time.values()) / len(group_idle_time) if group_idle_time else 0
    avg_group_spread = sum(group_lecture_spread.values()) / len(group_lecture_spread) if group_lecture_spread else 0
    
    avg_lecturer_fatigue = sum(lecturer_fatigue.values()) / len(lecturer_fatigue) if lecturer_fatigue else 0
    avg_lecturer_idle = sum(lecturer_idle_time.values()) / len(lecturer_idle_time) if lecturer_idle_time else 0
    avg_lecturer_spread = sum(lecturer_lecture_spread.values()) / len(lecturer_lecture_spread) if lecturer_lecture_spread else 0
    
    # Combine into a weighted final score
    # Higher is better
    weights = {
        'student_fatigue': 0.15,
        'student_idle': 0.15,
        'student_spread': 0.15,
        'lecturer_fatigue': 0.15,
        'lecturer_idle': 0.15,
        'lecturer_spread': 0.15,
        'workload_balance': 0.1
    }
    
    final_score = (
        weights['student_fatigue'] * avg_group_fatigue +
        weights['student_idle'] * avg_group_idle +
        weights['student_spread'] * avg_group_spread +
        weights['lecturer_fatigue'] * avg_lecturer_fatigue +
        weights['lecturer_idle'] * avg_lecturer_idle + 
        weights['lecturer_spread'] * avg_lecturer_spread +
        weights['workload_balance'] * workload_balance
    )
    
    result = {
        'student_fatigue': avg_group_fatigue,
        'student_idle': avg_group_idle,
        'student_spread': avg_group_spread,
        'lecturer_fatigue': avg_lecturer_fatigue,
        'lecturer_idle': avg_lecturer_idle,
        'lecturer_spread': avg_lecturer_spread,
        'workload_balance': workload_balance,
        'final_score': final_score
    }
    
    return result

def evaluate(schedule, groups_dict, lecturers_dict, activities_dict, spaces_dict, slots):
    """Evaluate both hard and soft constraints and print a summary."""
    # Evaluate hard constraints
    hard_results = evaluate_hard_constraints(schedule, activities_dict, groups_dict, spaces_dict)
    
    print("\n--- Hard Constraint Evaluation Results ---")
    print(f"Vacant Rooms Count: {hard_results[0]}")
    print(f"Lecturer Conflict Violations: {hard_results[1]}")
    print(f"Student Group Conflict Violations: {hard_results[2]}")
    print(f"Room Capacity Violations: {hard_results[3]}")
    print(f"Unassigned Activity Violations: {hard_results[4]}")
    
    total_violations = sum(hard_results[1:])  # Exclude vacant rooms which isn't necessarily a violation
    print(f"\nTotal Hard Constraint Violations: {total_violations}")
    
    # Evaluate soft constraints
    soft_results = evaluate_soft_constraints(schedule, groups_dict, lecturers_dict, slots)
    
    print("\n--- Soft Constraint Evaluation Results ---")
    print(f"Student Fatigue Factor: {soft_results['student_fatigue']:.2f}")
    print(f"Student Idle Time Factor: {soft_results['student_idle']:.2f}")
    print(f"Student Lecture Spread Factor: {soft_results['student_spread']:.2f}")
    print(f"Lecturer Fatigue Factor: {soft_results['lecturer_fatigue']:.2f}")
    print(f"Lecturer Idle Time Factor: {soft_results['lecturer_idle']:.2f}")
    print(f"Lecturer Lecture Spread Factor: {soft_results['lecturer_spread']:.2f}")
    print(f"Lecturer Workload Balance Factor: {soft_results['workload_balance']:.2f}")
    print(f"\nFinal Soft Constraint Score: {soft_results['final_score']:.2f}")
    
    return hard_results, soft_results

def get_classsize(activity, groups_dict):
    """Compute the total class size for an activity."""
    classsize = 0
    for id in activity.group_ids:
        classsize += groups_dict[id].size
    return classsize

def dominates(fitness1, fitness2):
    """
    Determines if fitness1 dominates fitness2 in a multi-objective minimization context.
    
    Returns True if fitness1 dominates fitness2 (i.e., is better in at least one objective
    and not worse in any objective).
    """
    return all(f1 <= f2 for f1, f2 in zip(fitness1, fitness2)) and any(f1 < f2 for f1, f2 in zip(fitness1, fitness2))
