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