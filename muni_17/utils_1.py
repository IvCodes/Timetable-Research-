import json
import numpy as np

class TupleKeyEncoder(json.JSONEncoder):

    """
    Custom JSON encoder that handles dictionaries with tuple keys by converting them to strings.
    Also handles any other non-serializable types by converting them to strings.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)
            
    def encode(self, obj):
        def _preprocess_for_json(o):
            if isinstance(o, dict):
                # Process dictionaries
                result = {}
                for k, v in o.items():
                    # Convert tuple keys to strings
                    if isinstance(k, tuple):
                        new_key = str(k)
                    else:
                        new_key = k
                    
                    # Recursively process values
                    result[new_key] = _preprocess_for_json(v)
                return result
            elif isinstance(o, (list, tuple)):
                # Process lists and tuples
                return [_preprocess_for_json(item) for item in o]
            elif hasattr(o, 'keys') and callable(getattr(o, 'keys')):
                # Handle dict-like objects that aren't directly instance of dict
                result = {}
                for k in o.keys():
                    if isinstance(k, tuple):
                        result[str(k)] = _preprocess_for_json(o[k])
                    else:
                        result[k] = _preprocess_for_json(o[k])
                return result
            else:
                # Return other objects as is
                return o
        
        return super().encode(_preprocess_for_json(obj))

def process_constraint_violations(violations):
    """Helper function to process constraint violations dictionaries."""
    violations_dict = {}
    for key, value in violations.items():
        # Handle nested dictionaries with tuple keys
        if isinstance(value, dict):
            processed_dict = {}
            for sub_key, sub_value in value.items():
                # Convert tuple keys to strings
                if isinstance(sub_key, tuple):
                    processed_dict[str(sub_key)] = sub_value
                else:
                    processed_dict[sub_key] = sub_value
            violations_dict[key] = processed_dict
        else:
            violations_dict[key] = value
    return violations_dict

def process_best_solution(solution):
    """Process the best solution into a serializable dictionary."""
    # Process assignments to handle tuple keys and values
    processed_assignments = {}
    for k, v in solution.assignments.items():
        key = str(k) if isinstance(k, tuple) else k
        value = str(v) if isinstance(v, tuple) else v
        processed_assignments[key] = value
    
    solution_dict = {
        'fitness': solution.fitness,
        'assignments': processed_assignments,  # Use processed assignments
        'num_assigned': len(solution.assignments),
        'crowding_distance': solution.crowding_distance
    }
    
    # Process constraint violations if they exist
    if hasattr(solution, 'constraint_violations') and solution.constraint_violations:
        solution_dict['constraint_violations'] = process_constraint_violations(solution.constraint_violations)
    
    return solution_dict

def process_pareto_fronts(fronts):
    """Process Pareto fronts into serializable format."""
    pareto_fronts = []
    for front in fronts:
        front_dict = []
        for solution in front:
            solution_dict = {
                'fitness': solution.fitness,
                'num_assigned': len(solution.assignments),
                'crowding_distance': solution.crowding_distance
            }
            front_dict.append(solution_dict)
        pareto_fronts.append(front_dict)
    return pareto_fronts

def process_metric_history(metrics):
    """Process metrics history into serializable format."""
    metric_history = {}
    for metric, values in metrics.items():
        if metric == 'constraint_violations':
            processed_violations = process_constraint_violations_in_metrics(values)
            metric_history[metric] = processed_violations
        elif isinstance(values, list):
            metric_history[metric] = values
    return metric_history

def process_constraint_violations_in_metrics(violations_list):
    """Process constraint violations in metrics history."""
    processed_violations = []
    for violation_data in violations_list:
        processed_violation = process_constraint_violations(violation_data)
        processed_violations.append(processed_violation)
    return processed_violations

def create_dataset_summary(data):
    """Create a summary of the dataset."""
    return {
        'num_courses': len(data.get('courses', {})),
        'num_classes': len(data.get('classes', {})),
        'num_rooms': len(data.get('rooms', {})),
        'num_students': data.get('num_students', 0),
        'num_distribution_constraints': len(data.get('distribution_constraints', []))
    }

def save_results(best_solution, fronts, metrics, data, output_file='nsga2_muni_results.json'):
    """
    Save the optimization results to a JSON file.
    
    Args:
        best_solution: The best Solution object
        fronts: List of fronts (lists of Solution objects)
        metrics: Dictionary with performance metrics
        data: Dictionary with problem data
        output_file: Path to output JSON file
    """
    # Process components using helper functions
    best_solution_dict = process_best_solution(best_solution)
    pareto_fronts = process_pareto_fronts(fronts)
    metric_history = process_metric_history(metrics)
    dataset_summary = create_dataset_summary(data)
    
    # Create final results dictionary
    results = {
        'best_solution': best_solution_dict,
        'pareto_fronts': pareto_fronts,
        'metrics': metric_history,
        'dataset': dataset_summary
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, cls=TupleKeyEncoder)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")


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
    
    print("\nSolution Statistics:")
    print(f"Total Classes: {total_classes}")
    print(f"Assigned Classes: {assigned_classes} ({assignment_rate:.2f}%)")
    print(f"Fitness: {solution.fitness}")
    
    # Count conflicts by type
    print("\nConstraint Violations:")
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
    
    print("\nDataset Statistics:")
    print(f"Courses: {len(data['courses'])}")
    print(f"Classes: {len(data['classes'])}")
    print(f"Fixed Classes: {fixed_classes}")
    print(f"Rooms: {len(data['rooms'])}")
    print(f"Students: {len(data['students'])}")
    print(f"Distribution Constraints: {len(data['distribution_constraints'])}")