import json

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
    # Convert complex datastructures to JSON-serializable format
    
    # Convert the best solution
    best_solution_dict = {
        'fitness': best_solution.fitness,
        'assignments': best_solution.assignments,
        'num_assigned': len(best_solution.assignments),
        'crowding_distance': best_solution.crowding_distance
    }
    
    # Process constraint violations safely
    if hasattr(best_solution, 'constraint_violations') and best_solution.constraint_violations:
        violations_dict = {}
        for key, value in best_solution.constraint_violations.items():
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
        best_solution_dict['constraint_violations'] = violations_dict
    
    # Convert the Pareto fronts
    pareto_fronts = []
    for i, front in enumerate(fronts):
        front_dict = []
        for solution in front:
            solution_dict = {
                'fitness': solution.fitness,
                'num_assigned': len(solution.assignments),
                'crowding_distance': solution.crowding_distance
            }
            front_dict.append(solution_dict)
        pareto_fronts.append(front_dict)
    
    # Convert metric history to lists
    metric_history = {}
    for metric, values in metrics.items():
        if isinstance(values, list):
            metric_history[metric] = values
    
    # Create final results dictionary
    results = {
        'best_solution': best_solution_dict,
        'pareto_fronts': pareto_fronts,
        'metrics': metric_history,
        'dataset': {
            'num_courses': len(data.get('courses', {})),
            'num_classes': len(data.get('classes', {})),
            'num_rooms': len(data.get('rooms', {})),
            'num_students': data.get('num_students', 0),
            'num_distribution_constraints': len(data.get('distribution_constraints', []))
        }
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, cls=TupleKeyEncoder)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
