from metrics import track_constraint_violations

def check_room_conflicts(solution):
    """Check for room conflicts (same room, same time)."""
    room_conflicts = 0
    room_time_assignments = {}
    
    for class_id, (time_id, room_id) in solution.assignments.items():
        if room_id and time_id:
            key = (room_id, time_id)
            if key in room_time_assignments:
                room_conflicts += 1
            else:
                room_time_assignments[key] = class_id
    
    return room_conflicts

def check_time_conflicts(solution, class_data):
    """Check for time conflicts (same course, same time)."""
    time_conflicts = 0
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
    
    return time_conflicts

def check_distribution_constraints(solution, distribution_constraints):
    """Check distribution constraints."""
    distribution_conflicts = 0
    
    for constraint in distribution_constraints:
        if constraint.get('required', False):  # Only check hard constraints
            classes = constraint.get('classes', [])
            constraint_type = constraint.get('type')
            
            # Check if all classes in the constraint are assigned
            assigned_classes = [c for c in classes if c in solution.assignments]
            
            if len(assigned_classes) >= 2:  # Need at least 2 classes to check constraints
                distribution_conflicts += check_specific_constraint(solution, constraint_type, assigned_classes)
    
    return distribution_conflicts

def check_specific_constraint(solution, constraint_type, assigned_classes):
    """Check a specific type of constraint."""
    if constraint_type == 'SameTime':
        # All classes should be at the same time
        times = [solution.assignments[c][0] for c in assigned_classes if solution.assignments[c][0]]
        if len(set(times)) > 1:  # More than one unique time
            return 1
    
    elif constraint_type == 'DifferentTime':
        # All classes should be at different times
        times = [solution.assignments[c][0] for c in assigned_classes if solution.assignments[c][0]]
        if len(set(times)) < len(times):  # Duplicate times exist
            return 1
    
    return 0

def check_student_conflicts(solution, course_to_classes):
    """Check student conflicts (same student, overlapping times)."""
    student_conflicts = 0
    
    for course_id, class_list in course_to_classes.items():
        assigned_classes = [c for c in class_list if c in solution.assignments]
        student_conflicts += count_time_overlaps(solution, assigned_classes)
    
    return student_conflicts

def count_time_overlaps(solution, assigned_classes):
    """Count time overlaps between classes."""
    conflicts = 0
    for i in range(len(assigned_classes)):
        for j in range(i+1, len(assigned_classes)):
            class1 = assigned_classes[i]
            class2 = assigned_classes[j]
            time1 = solution.assignments[class1][0] if class1 in solution.assignments else None
            time2 = solution.assignments[class2][0] if class2 in solution.assignments else None
            
            if time1 and time2 and time1 == time2:  # Same time slot
                conflicts += 1
    return conflicts

def check_capacity_violations(solution, class_data, room_data):
    """Check room capacity violations."""
    capacity_violations = 0
    
    for class_id, (_, room_id) in solution.assignments.items():
        if room_id and class_id in class_data and room_id in room_data:
            class_limit = class_data[class_id].get('limit', 0)
            room_capacity = room_data[room_id].get('capacity', 0)
            
            if class_limit > room_capacity:
                capacity_violations += 1
    
    return capacity_violations

def calculate_total_violations(weights, violations):
    """Calculate total weighted violations."""
    return sum(weights.get(key, 1.0) * value for key, value in violations.items())

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
    
    # Check different types of conflicts
    room_conflicts = check_room_conflicts(solution)
    time_conflicts = check_time_conflicts(solution, class_data)
    distribution_conflicts = check_distribution_constraints(solution, distribution_constraints)
    student_conflicts = check_student_conflicts(solution, course_to_classes)
    capacity_violations = check_capacity_violations(solution, class_data, room_data)
    
    # Calculate total weighted violations
    violation_dict = {
        'ROOM': room_conflicts,
        'TIME': time_conflicts,
        'DISTRIBUTION': distribution_conflicts,
        'STUDENT': student_conflicts,
        'CAPACITY': capacity_violations
    }
    
    total_violations = calculate_total_violations(weights, violation_dict)
    
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
        'total_counts': {
            'room_conflicts': room_conflicts,
            'time_conflicts': time_conflicts,
            'distribution_conflicts': distribution_conflicts,
            'student_conflicts': student_conflicts,
            'capacity_violations': capacity_violations,
            'total_weighted_score': total_violations
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