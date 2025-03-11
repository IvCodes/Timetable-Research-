"""
Main script to run and compare all three multi-objective evolutionary algorithms
for timetable scheduling: NSGA-II, MOEA/D, and SPEA2.

This script provides a unified interface to:
1. Run any or all of the algorithms
2. Compare their performance on the same dataset
3. Output results for analysis
"""
import json
import time
from utils import evaluate, load_data
from nsga2 import run_nsga2_optimization
from moead import run_moead_optimization
from spea2 import run_spea2_optimization

# Algorithm choices
ALGORITHMS = {
    'nsga2': run_nsga2_optimization,
    'moead': run_moead_optimization,
    'spea2': run_spea2_optimization
}

def save_timetable(timetable, filename):
    """
    Save a timetable solution to a JSON file.
    
    Args:
        timetable: Timetable dictionary to save
        filename: Target filename
    """
    with open(filename, 'w') as f:
        json.dump(timetable, f, indent=2)
    print(f"Saved timetable to {filename}")

def format_evaluation_result(result):
    """
    Format the evaluation result dictionary for display.
    
    Args:
        result: Evaluation result dictionary
    
    Returns:
        Formatted string representation
    """
    output = []
    # Hard constraints
    output.append("HARD CONSTRAINTS:")
    output.append(f"  Room Vacant: {result['hard_constraints']['vacant']}")
    output.append(f"  Room Overbooking: {result['hard_constraints']['overbooking']}")
    output.append(f"  Slot Conflicts: {result['hard_constraints']['slot_conflicts']}")
    output.append(f"  Professor Conflicts: {result['hard_constraints']['professor_conflicts']}")
    output.append(f"  Group Conflicts: {result['hard_constraints']['group_conflicts']}")
    output.append(f"  Unassigned Activities: {result['hard_constraints']['unassigned_activities']}")
    
    # Soft constraints
    output.append("\nSOFT CONSTRAINTS:")
    output.append(f"  Student Fatigue: {result['soft_constraints']['student_fatigue']}")
    output.append(f"  Student Idle Time: {result['soft_constraints']['student_idle_time']}")
    output.append(f"  Student Lecture Spread: {result['soft_constraints']['student_lecture_spread']}")
    output.append(f"  Lecturer Fatigue: {result['soft_constraints']['lecturer_fatigue']}")
    output.append(f"  Lecturer Idle Time: {result['soft_constraints']['lecturer_idle_time']}")
    output.append(f"  Lecturer Lecture Spread: {result['soft_constraints']['lecturer_lecture_spread']}")
    output.append(f"  Lecturer Workload Balance: {result['soft_constraints']['lecturer_workload_balance']}")
    
    return "\n".join(output)

def compare_algorithms(data_file, algorithms=None):
    """
    Run and compare multiple algorithms on the same dataset.
    
    Args:
        data_file: Path to the JSON data file
        algorithms: List of algorithm names to run (default: all)
    
    Returns:
        Dictionary mapping algorithm names to their best solutions and evaluation results
    """
    # Load data once for all algorithms
    spaces_dict, groups_dict, lecturers_dict, activities_dict, slots = load_data(data_file)
    
    # If no algorithms specified, run all
    if algorithms is None:
        algorithms = list(ALGORITHMS.keys())
    
    results = {}
    
    for algo_name in algorithms:
        if algo_name not in ALGORITHMS:
            print(f"Warning: Algorithm '{algo_name}' not found. Skipping.")
            continue
        
        print(f"\n{'=' * 50}")
        print(f"Running {algo_name.upper()}...")
        print(f"{'=' * 50}")
        
        # Measure execution time
        start_time = time.time()
        
        # Run the algorithm
        best_timetable = ALGORITHMS[algo_name](data_file)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Evaluate the solution
        evaluation_result = evaluate(best_timetable, groups_dict, lecturers_dict, 
                              activities_dict, spaces_dict, slots, as_dict=True)
        
        # Save results
        results[algo_name] = {
            'timetable': best_timetable,
            'evaluation': evaluation_result,
            'execution_time': execution_time
        }
        
        # Save timetable to file
        save_timetable(best_timetable, f"{algo_name}_solution.json")
        
        # Print results
        print(f"\nResults for {algo_name.upper()}:")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(format_evaluation_result(evaluation_result))
    
    return results

def print_comparison_table(results):
    """
    Print a comparison table of algorithm results.
    
    Args:
        results: Dictionary of algorithm results
    """
    if not results:
        print("No results to compare.")
        return
    
    print("\n" + "=" * 80)
    print("ALGORITHM COMPARISON")
    print("=" * 80)
    
    # Table header
    header = ["Algorithm", "Exec Time", "Unassigned", "Overbooking", "Conflicts", "Student Fatigue", "Lecturer Fatigue"]
    print(f"{header[0]:<10} {header[1]:<12} {header[2]:<10} {header[3]:<12} {header[4]:<10} {header[5]:<15} {header[6]:<15}")
    print("-" * 80)
    
    # Table rows
    for algo_name, result in results.items():
        eval_result = result['evaluation']
        row = [
            algo_name.upper(),
            f"{result['execution_time']:.2f}s",
            eval_result['hard_constraints']['unassigned_activities'],
            eval_result['hard_constraints']['overbooking'],
            eval_result['hard_constraints']['slot_conflicts'] + 
            eval_result['hard_constraints']['professor_conflicts'] + 
            eval_result['hard_constraints']['group_conflicts'],
            eval_result['soft_constraints']['student_fatigue'],
            eval_result['soft_constraints']['lecturer_fatigue']
        ]
        print(f"{row[0]:<10} {row[1]:<12} {row[2]:<10} {row[3]:<12} {row[4]:<10} {row[5]:<15} {row[6]:<15}")
    
    print("=" * 80)

def main():
    """Main function to run the comparison."""
    data_file = "sliit_computing_dataset.json"
    
    print("Timetable Scheduling - Multi-Objective Optimization Comparison")
    print(f"Dataset: {data_file}")
    
    # Run all algorithms and get results
    results = compare_algorithms(data_file)
    
    # Print comparison table
    print_comparison_table(results)
    
    print("\nOptimization complete. Individual solutions saved to JSON files.")

if __name__ == "__main__":
    main()
