"""
Example script to demonstrate the use of the standalone TimetableEvaluator.

This script shows how to:
1. Evaluate solutions from existing files
2. Run and evaluate algorithms directly
3. Generate comparison reports and visualizations
"""

from evaluator import TimetableEvaluator
from nsga2 import run_nsga2_optimization
from moead import run_moead_optimization

# Try to import SPEA2 if available
try:
    from spea2 import run_spea2_optimization
    has_spea2 = True
except ImportError:
    has_spea2 = False
    print("SPEA2 algorithm not found. Continuing with NSGA-II and MOEA/D only.")

def evaluate_existing_solutions():
    """Evaluate algorithms from existing solution files."""
    print("\n" + "=" * 60)
    print("EVALUATION OF EXISTING SOLUTION FILES")
    print("=" * 60)
    
    # Initialize the evaluator
    data_file = "sliit_computing_dataset.json"
    evaluator = TimetableEvaluator(data_file)
    
    # Evaluate each solution file
    evaluator.evaluate_solution_file("nsga2_solution.json", "NSGA-II")
    evaluator.evaluate_solution_file("moead_solution.json", "MOEA/D")
    
    if has_spea2:
        evaluator.evaluate_solution_file("spea2_solution.json", "SPEA2")
    
    # Print comparison table
    evaluator.print_comparison_table()
    
    # Generate visualization
    evaluator.visualize_comparison(save_path="existing_solutions_comparison.png")
    print(f"Visualization saved to 'existing_solutions_comparison.png'")
    
    # Save detailed results
    evaluator.save_results_to_file("existing_solutions_comparison.json")
    
def run_and_evaluate_algorithms():
    """Run each algorithm and evaluate results directly."""
    print("\n" + "=" * 60)
    print("RUNNING AND EVALUATING ALGORITHMS")
    print("=" * 60)
    
    # Initialize the evaluator
    data_file = "sliit_computing_dataset.json"
    evaluator = TimetableEvaluator(data_file)
    
    # Define algorithms to run and evaluate
    algorithms = {
        'NSGA-II': run_nsga2_optimization,
        'MOEA/D': run_moead_optimization
    }
    
    if has_spea2:
        algorithms['SPEA2'] = run_spea2_optimization
    
    # Run and evaluate each algorithm
    for name, func in algorithms.items():
        print(f"\nRunning and evaluating {name}...")
        result = evaluator.run_and_evaluate_algorithm(func, name)
        
        # Print detailed results for this algorithm
        evaluator.print_evaluation(result)
    
    # Print comparison table
    evaluator.print_comparison_table()
    
    # Generate visualization
    evaluator.visualize_comparison(save_path="algorithm_comparison.png")
    print(f"Visualization saved to 'algorithm_comparison.png'")
    
    # Save detailed results
    evaluator.save_results_to_file("algorithm_comparison.json")

def custom_algorithm_evaluation():
    """
    Example of how to evaluate a custom algorithm or a modification
    of an existing algorithm with different parameters.
    """
    print("\n" + "=" * 60)
    print("CUSTOM ALGORITHM EVALUATION")
    print("=" * 60)
    
    # Initialize the evaluator
    data_file = "sliit_computing_dataset.json"
    evaluator = TimetableEvaluator(data_file)
    
    # Example: Run NSGA-II with different parameters by using a lambda
    # This could be a separate algorithm implementation in practice
    custom_nsga2 = lambda data_file: run_nsga2_optimization(
        data_file, 
        # Here you could pass custom parameters to the algorithm
        # population_size=100,
        # generations=200,
        # etc.
    )
    
    # Evaluate the custom algorithm
    result = evaluator.run_and_evaluate_algorithm(custom_nsga2, "Custom NSGA-II")
    
    # Print detailed results
    evaluator.print_evaluation(result)

def main():
    """Main function to run all examples."""
    print("TIMETABLE SCHEDULING ALGORITHM EVALUATION EXAMPLES")
    
    # Uncomment the example you want to run
    
    # Example 1: Evaluate existing solution files
    evaluate_existing_solutions()
    
    # Example 2: Run and evaluate algorithms directly
    # This will take longer as it runs the algorithms
    # run_and_evaluate_algorithms()
    
    # Example 3: Custom algorithm evaluation
    # custom_algorithm_evaluation()
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
