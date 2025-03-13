import numpy as np

# Global reference point for hypervolume calculation
REFERENCE_POINT = [1.0, 1.0]  # Default reference point, should be defined based on your specific problem

def _calculate_simple_hypervolume(points, reference_point):
    """Helper function to calculate hypervolume for cases with limited dimension variation."""
    hypervolume = 1.0
    for i in range(points.shape[1]):
        dimension_min = np.min(points[:, i])
        hypervolume *= (reference_point[i] - dimension_min)
    return hypervolume

def _calculate_2d_hypervolume(points, reference_point):
    """Helper function to calculate hypervolume for 2D fronts."""
    # Sort points by first objective
    sorted_points = points[points[:, 0].argsort()]
    
    # Calculate hypervolume as sum of rectangles
    hypervolume = 0.0
    for i in range(len(sorted_points)):
        if i == 0:
            width = reference_point[0] - sorted_points[i, 0]
            height = reference_point[1] - sorted_points[i, 1]
        else:
            width = sorted_points[i-1, 0] - sorted_points[i, 0]
            height = reference_point[1] - sorted_points[i, 1]
        
        if width > 0 and height > 0:
            hypervolume += width * height
    
    return hypervolume

def _is_point_dominated(sample, points):
    """Check if a sample point is dominated by any point in the front."""
    for point in points:
        if np.all(point <= sample) and np.any(point < sample):
            return True
    return False

def _calculate_monte_carlo_hypervolume(filtered_points, filtered_reference):
    """Calculate hypervolume using Monte Carlo sampling."""
    n_samples = 10000
    
    # Define bounds for sampling
    lower_bounds = np.min(filtered_points, axis=0)
    upper_bounds = filtered_reference
    
    # Generate random points within the reference volume using the newer Generator API
    rng = np.random.default_rng(seed=42)  # Create a Generator instance with a fixed seed
    samples = rng.uniform(
        low=lower_bounds,
        high=upper_bounds,
        size=(n_samples, len(filtered_reference))
    )
    
    # Count points dominated by the Pareto front
    count_dominated = sum(1 for sample in samples if _is_point_dominated(sample, filtered_points))
    
    # Calculate hypervolume
    dominated_ratio = count_dominated / n_samples
    reference_volume = np.prod(upper_bounds - lower_bounds)
    return dominated_ratio * reference_volume

def calculate_hypervolume(front, reference_point=None):
    """
    Calculate the hypervolume indicator for a Pareto front.
    
    Args:
        front: List of fitness values for solutions in the Pareto front
        reference_point: Reference point for hypervolume calculation (worst possible values)
        
    Returns:
        Hypervolume value
    """
    if not front:
        return 0.0
    
    # Use provided reference point or update the global one
    if reference_point is None:
        reference_point = REFERENCE_POINT
    
    # Convert to numpy array for easier manipulation
    points = np.array(front)
    
    # Check which dimensions have variation
    dimension_ranges = np.ptp(points, axis=0)
    active_dimensions = dimension_ranges > 1e-10
    
    # If we have less than 2 active dimensions, use a simple approach
    if np.sum(active_dimensions) < 2:
        return _calculate_simple_hypervolume(points, reference_point)
    
    # For 2D fronts, use a simpler calculation
    if points.shape[1] == 2:
        return _calculate_2d_hypervolume(points, reference_point)
    
    # For higher dimensions, use a custom calculation approach
    try:
        # Only keep active dimensions to avoid qhull errors
        active_dim_indices = np.nonzero(active_dimensions)[0]
        
        if len(active_dim_indices) < 2:  # Need at least 2 dimensions
            return _calculate_simple_hypervolume(points, reference_point)
        
        # Filter points to only include active dimensions
        filtered_points = points[:, active_dim_indices]
        filtered_reference = np.array(reference_point)[active_dim_indices]
        
        return _calculate_monte_carlo_hypervolume(filtered_points, filtered_reference)
        
    except Exception as e:
        print(f"Error calculating hypervolume: {e}")
        return _calculate_simple_hypervolume(points, reference_point)

def _check_room_conflicts(solution, violations):
    """Check for room conflicts (same room, same time)"""
    room_time_assignments = {}  # (room_id, time_id) -> class_id
    
    for class_id, assignment in solution.assignments.items():
        if not assignment or len(assignment) < 2:
            continue
            
        try:
            room_id, time_id = assignment
        except (ValueError, TypeError):
            continue
        
        if not room_id or not time_id:
            continue
            
        key = (room_id, time_id)
        if key in room_time_assignments:
            conflicting_class = room_time_assignments[key]
            violations['room_conflicts'][(class_id, conflicting_class)] = 1
            violations['total_by_type']['room_conflicts'] += 1
        else:
            room_time_assignments[key] = class_id
    
    return violations

def _check_time_conflicts(solution, data, violations):
    """Check for time conflicts (same course, same time)"""
    class_data = data.get('classes', {})
    course_time_assignments = {}
    
    for class_id, assignment in solution.assignments.items():
        if not assignment or len(assignment) < 2:
            continue
            
        try:
            _, time_id = assignment
        except (ValueError, TypeError):
            continue
            
        if not time_id or class_id not in class_data:
            continue
            
        course_id = class_data[class_id].get('course_id')
        if course_id:
            key = (course_id, time_id)
            if key in course_time_assignments:
                conflicting_class = course_time_assignments[key]
                violations['time_conflicts'][(class_id, conflicting_class)] = 1
                violations['total_by_type']['time_conflicts'] += 1
            else:
                course_time_assignments[key] = class_id
    
    return violations

def _check_distribution_constraints(solution, data, violations):
    """Check distribution constraints"""
    distribution_constraints = data.get('distribution_constraints', [])
    
    for constraint in distribution_constraints:
        constraint_type = constraint.get('type')
        classes = constraint.get('classes', [])
        
        if not constraint_type or not classes:
            continue
            
        class_ids = [str(c_id) for c in classes for c_id in ([c.get('class_id')] if isinstance(c, dict) else [c])]
        assigned_classes = [c_id for c_id in class_ids if c_id in solution.assignments]
        
        if constraint_type == 'SameTime' and len(assigned_classes) > 1:
            violations = _check_same_time_constraint(solution, assigned_classes, violations)
        elif constraint_type == 'DifferentTime' and len(assigned_classes) > 1:
            violations = _check_different_time_constraint(solution, assigned_classes, violations)
    
    return violations

def _check_class_pair_conflict(class1, class2, solution, violations):
    """Helper function to check if two classes have a time conflict for students"""
    assignment1 = solution.assignments.get(class1)
    assignment2 = solution.assignments.get(class2)
    
    if not assignment1 or not assignment2 or len(assignment1) < 2 or len(assignment2) < 2:
        return
        
    time1, time2 = assignment1[1], assignment2[1]
    
    if time1 and time2 and time1 == time2:
        violations['student_conflicts'][(class1, class2)] = 1
        violations['total_by_type']['student_conflicts'] += 1

def _check_student_conflicts(solution, data, violations):
    """Check for student conflicts"""
    student_enrollments = data.get('student_enrollments', {})
    
    for student_id, enrolled_classes in student_enrollments.items():
        assigned_classes = [c_id for c_id in enrolled_classes if c_id in solution.assignments]
        
        for i in range(len(assigned_classes)):
            for j in range(i+1, len(assigned_classes)):
                _check_class_pair_conflict(
                    assigned_classes[i], 
                    assigned_classes[j], 
                    solution,
                    violations
                )
    
    return violations

def _check_capacity_violations(solution, data, violations):
    """Check for room capacity violations"""
    rooms = data.get('rooms', {})
    class_data = data.get('classes', {})
    
    for class_id, assignment in solution.assignments.items():
        if not assignment or len(assignment) < 2:
            continue
            
        room_id = assignment[0]
        if room_id in rooms and class_id in class_data:
            room_capacity = rooms[room_id].get('capacity', 0)
            enrolled_students = len(class_data[class_id].get('students', []))
            
            if enrolled_students > room_capacity:
                violations['capacity_violations'][class_id] = enrolled_students - room_capacity
                violations['total_by_type']['capacity_violations'] += 1
    
    return violations

def _check_same_time_constraint(solution, assigned_classes, violations):
    """Check 'SameTime' distribution constraint"""
    times = set()
    for c_id in assigned_classes:
        assignment = solution.assignments.get(c_id)
        if assignment and len(assignment) >= 2:
            times.add(assignment[1])
    
    if len(times) > 1:  # Classes are not at the same time
        for i in range(len(assigned_classes)):
            for j in range(i+1, len(assigned_classes)):
                violations['distribution_conflicts'][(assigned_classes[i], assigned_classes[j])] = 1
                violations['total_by_type']['distribution_conflicts'] += 1
    
    return violations

def _check_different_time_constraint(solution, assigned_classes, violations):
    """Check 'DifferentTime' distribution constraint"""
    time_classes = {}
    
    # Group classes by time slot
    for c_id in assigned_classes:
        assignment = solution.assignments.get(c_id)
        if not assignment or len(assignment) < 2:
            continue
            
        time_id = assignment[1]
        time_classes.setdefault(time_id, []).append(c_id)
    
    # Record conflicts for classes that should be at different times but aren't
    for classes in time_classes.values():
        if len(classes) <= 1:
            continue  # No conflicts if only one class or none at this time
            
        # Record all pairwise conflicts
        for i, class1 in enumerate(classes):
            for class2 in classes[i+1:]:
                violations['distribution_conflicts'][(class1, class2)] = 1
                violations['total_by_type']['distribution_conflicts'] += 1
    
    return violations

def _calculate_weighted_score(violations):
    """Calculate total weighted violation score"""
    return (
        violations['total_by_type']['room_conflicts'] * 10 +
        violations['total_by_type']['time_conflicts'] * 20 +
        violations['total_by_type']['distribution_conflicts'] * 10 +
        violations['total_by_type']['student_conflicts'] * 5 +
        violations['total_by_type']['capacity_violations'] * 2
    )

def track_constraint_violations(solution, data):
    """
    Track detailed constraint violations in a solution.
    
    Args:
        solution: A Solution object
        data: Dictionary with problem data
    
    Returns:
        Dictionary with constraint violation details
    """
    violations = {
        'room_conflicts': {},
        'time_conflicts': {},
        'distribution_conflicts': {},
        'student_conflicts': {},
        'capacity_violations': {},
        'total_by_type': {
            'room_conflicts': 0,
            'time_conflicts': 0,
            'distribution_conflicts': 0,
            'student_conflicts': 0,
            'capacity_violations': 0
        }
    }
    
    # Check all types of violations
    violations = _check_room_conflicts(solution, violations)
    violations = _check_time_conflicts(solution, data, violations)
    violations = _check_distribution_constraints(solution, data, violations)
    violations = _check_student_conflicts(solution, data, violations)
    violations = _check_capacity_violations(solution, data, violations)
    
    # Calculate total weighted violation score
    total_weighted_score = _calculate_weighted_score(violations)
    
    # Add the total counts for easy access
    violations['total_counts'] = {
        'room_conflicts': violations['total_by_type']['room_conflicts'],
        'time_conflicts': violations['total_by_type']['time_conflicts'],
        'distribution_conflicts': violations['total_by_type']['distribution_conflicts'],
        'student_conflicts': violations['total_by_type']['student_conflicts'],
        'capacity_violations': violations['total_by_type']['capacity_violations'],
        'total_weighted_score': total_weighted_score
    }
    
    return violations

def calculate_spacing(front):
    """
    Calculate the spacing metric for a Pareto front.
    
    The spacing metric measures how evenly the solutions are distributed along the front.
    Lower values indicate more uniform spacing.
    
    Args:
        front: List of fitness values for solutions in the Pareto front
        
    Returns:
        Spacing metric value
    """
    if not front or len(front) < 2:
        return 0.0
    
    # Convert to numpy array
    points = np.array(front)
    n = len(points)
    
    # Calculate distances between consecutive points
    distances = []
    
    for i in range(n):
        # Find minimum distance to any other point
        min_dist = float('inf')
        for j in range(n):
            if i != j:
                # Euclidean distance
                dist = np.sqrt(np.sum((points[i] - points[j])**2))
                min_dist = min(min_dist, dist)
        distances.append(min_dist)
    
    # Calculate mean distance
    mean_dist = np.mean(distances)
    
    # Calculate standard deviation of distances
    spacing = np.sqrt(np.sum((distances - mean_dist)**2) / (n - 1))
    
    return spacing

def calculate_igd(front, reference_front):
    """
    Calculate the Inverted Generational Distance (IGD) metric.
    
    IGD measures how far the approximated Pareto front is from the true Pareto front.
    Lower values indicate better convergence to the true front.
    
    Args:
        front: List of fitness values for solutions in the approximated Pareto front
        reference_front: List of fitness values for solutions in the true/reference Pareto front
        
    Returns:
        IGD value
    """
    if not front or not reference_front:
        return float('inf')
    
    # Convert to numpy arrays and normalize the values
    points = np.array(front, dtype=float)
    ref_points = np.array(reference_front, dtype=float)
    
    # Find max values for normalization
    max_values = np.maximum(np.max(points, axis=0), np.max(ref_points, axis=0))
    max_values[max_values == 0] = 1  # Avoid division by zero
    
    # Normalize both sets of points
    points = points / max_values
    ref_points = ref_points / max_values
    
    # Calculate minimum distance from each reference point to any point in the front
    total_dist = 0.0
    for ref_point in ref_points:
        # Calculate distances to all points
        distances = np.sqrt(np.sum((points - ref_point)**2, axis=1))
        # Take minimum distance
        min_dist = np.min(distances)
        total_dist += min_dist
    
    # Average distance
    igd = total_dist / len(ref_points)
    
    return igd

def analyze_constraint_violations(population, data):
    """
    Analyze constraint violations across the population.
    
    Args:
        population: List of Solution objects
        data: Dictionary with problem data
        
    Returns:
        Dictionary with constraint violation statistics
    """
    # Initialize statistics
    violation_stats = {
        'room_conflicts': {
            'min': float('inf'),
            'max': 0,
            'avg': 0,
            'total': 0
        },
        'time_conflicts': {
            'min': float('inf'),
            'max': 0,
            'avg': 0,
            'total': 0
        },
        'distribution_conflicts': {
            'min': float('inf'),
            'max': 0,
            'avg': 0,
            'total': 0
        },
        'student_conflicts': {
            'min': float('inf'),
            'max': 0,
            'avg': 0,
            'total': 0
        },
        'capacity_violations': {
            'min': float('inf'),
            'max': 0,
            'avg': 0,
            'total': 0
        },
        'total': {
            'min': float('inf'),
            'max': 0,
            'avg': 0,
            'total': 0
        }
    }
    
    # If population is empty, return default values
    if not population:
        for key in violation_stats:
            violation_stats[key]['min'] = 0
        return violation_stats
    
    # Track violations for each solution
    for solution in population:
        # Track constraint violations for this solution
        violations = track_constraint_violations(solution, data)
        
        # Extract totals for each violation type
        room_conflicts = violations['total_by_type']['room_conflicts']
        time_conflicts = violations['total_by_type']['time_conflicts']
        distribution_conflicts = violations['total_by_type']['distribution_conflicts']
        student_conflicts = violations['total_by_type']['student_conflicts']
        capacity_violations = violations['total_by_type']['capacity_violations']
        
        # Calculate total weighted violations
        total_weighted = violations.get('total_counts', {}).get('total_weighted_score', 
            room_conflicts * 10 + 
            time_conflicts * 20 + 
            distribution_conflicts * 10 + 
            student_conflicts * 5 + 
            capacity_violations * 2)
        
        # Update room conflicts stats
        violation_stats['room_conflicts']['min'] = min(violation_stats['room_conflicts']['min'], room_conflicts)
        violation_stats['room_conflicts']['max'] = max(violation_stats['room_conflicts']['max'], room_conflicts)
        violation_stats['room_conflicts']['total'] += room_conflicts
        
        # Update time conflicts stats
        violation_stats['time_conflicts']['min'] = min(violation_stats['time_conflicts']['min'], time_conflicts)
        violation_stats['time_conflicts']['max'] = max(violation_stats['time_conflicts']['max'], time_conflicts)
        violation_stats['time_conflicts']['total'] += time_conflicts
        
        # Update distribution conflicts stats
        violation_stats['distribution_conflicts']['min'] = min(violation_stats['distribution_conflicts']['min'], distribution_conflicts)
        violation_stats['distribution_conflicts']['max'] = max(violation_stats['distribution_conflicts']['max'], distribution_conflicts)
        violation_stats['distribution_conflicts']['total'] += distribution_conflicts
        
        # Update student conflicts stats
        violation_stats['student_conflicts']['min'] = min(violation_stats['student_conflicts']['min'], student_conflicts)
        violation_stats['student_conflicts']['max'] = max(violation_stats['student_conflicts']['max'], student_conflicts)
        violation_stats['student_conflicts']['total'] += student_conflicts
        
        # Update capacity violations stats
        violation_stats['capacity_violations']['min'] = min(violation_stats['capacity_violations']['min'], capacity_violations)
        violation_stats['capacity_violations']['max'] = max(violation_stats['capacity_violations']['max'], capacity_violations)
        violation_stats['capacity_violations']['total'] += capacity_violations
        
        # Update total stats
        violation_stats['total']['min'] = min(violation_stats['total']['min'], total_weighted)
        violation_stats['total']['max'] = max(violation_stats['total']['max'], total_weighted)
        violation_stats['total']['total'] += total_weighted
    
    # Calculate averages
    n = len(population)
    violation_stats['room_conflicts']['avg'] = violation_stats['room_conflicts']['total'] / n
    violation_stats['time_conflicts']['avg'] = violation_stats['time_conflicts']['total'] / n
    violation_stats['distribution_conflicts']['avg'] = violation_stats['distribution_conflicts']['total'] / n
    violation_stats['student_conflicts']['avg'] = violation_stats['student_conflicts']['total'] / n
    violation_stats['capacity_violations']['avg'] = violation_stats['capacity_violations']['total'] / n
    violation_stats['total']['avg'] = violation_stats['total']['total'] / n
    
    # Handle case where there were no violations (min was not updated)
    for key in violation_stats:
        if violation_stats[key]['min'] == float('inf'):
            violation_stats[key]['min'] = 0
    
    return violation_stats
