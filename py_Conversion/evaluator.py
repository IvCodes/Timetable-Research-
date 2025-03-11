"""
Standalone Evaluation Module for Timetable Scheduling Algorithms

This module provides tools for evaluating timetable scheduling solutions
from different algorithms. It allows for standardized evaluation metrics
and comparison between algorithms.
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from utils import (
    load_data, evaluate, evaluate_hard_constraints, evaluate_soft_constraints
)

class TimetableEvaluator:
    """
    A class for evaluating and comparing timetable scheduling algorithms.
    """
    
    def __init__(self, data_file):
        """
        Initialize the evaluator with a dataset.
        
        Args:
            data_file: Path to the JSON data file containing timetable scheduling data
        """
        self.data_file = data_file
        self.spaces_dict, self.groups_dict, self.lecturers_dict, self.activities_dict, self.slots = load_data(data_file)
        self.algorithm_results = {}
        
    def evaluate_solution(self, timetable, algorithm_name=None, execution_time=None, save_results=True):
        """
        Evaluate a single timetable solution.
        
        Args:
            timetable: The timetable solution to evaluate
            algorithm_name: Optional name of the algorithm that produced the solution
            execution_time: Optional execution time of the algorithm in seconds
            save_results: Whether to save the results in the internal storage
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Evaluate hard constraints
        hard_results = evaluate_hard_constraints(timetable, self.activities_dict, self.groups_dict, self.spaces_dict)
        vacant_rooms, prof_conflicts, group_conflicts, room_size_conflicts, unassigned_activities = hard_results
        total_hard_violations = sum(hard_results[1:])  # Exclude vacant rooms
        
        # Evaluate soft constraints
        soft_results = evaluate_soft_constraints(timetable, self.groups_dict, self.lecturers_dict, self.slots)
        
        # Combine results
        results = {
            'algorithm_name': algorithm_name,
            'execution_time': execution_time,
            'timetable': timetable,
            'hard_constraints': {
                'vacant': vacant_rooms,
                'overbooking': room_size_conflicts,
                'slot_conflicts': 0,  # Not directly returned by evaluate_hard_constraints
                'professor_conflicts': prof_conflicts,
                'group_conflicts': group_conflicts,
                'unassigned_activities': unassigned_activities,
                'total_violations': total_hard_violations
            },
            'soft_constraints': {
                'student_fatigue': soft_results['student_fatigue'],
                'student_idle_time': soft_results['student_idle'],
                'student_lecture_spread': soft_results['student_spread'],
                'lecturer_fatigue': soft_results['lecturer_fatigue'],
                'lecturer_idle_time': soft_results['lecturer_idle'],
                'lecturer_lecture_spread': soft_results['lecturer_spread'],
                'lecturer_workload_balance': soft_results['workload_balance'],
                'final_score': soft_results['final_score']
            },
            'overall_score': total_hard_violations * 1000 + soft_results['final_score']  # Hard constraints weighted more heavily
        }
        
        # Save results if requested
        if save_results and algorithm_name:
            self.algorithm_results[algorithm_name] = results
            
        return results
        
    def evaluate_solution_file(self, solution_file, algorithm_name=None):
        """
        Evaluate a timetable solution from a JSON file.
        
        Args:
            solution_file: Path to the JSON file containing the timetable solution
            algorithm_name: Optional name of the algorithm that produced the solution
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Load timetable from file
        with open(solution_file, 'r') as f:
            timetable = json.load(f)
            
        # Use the filename as algorithm name if not provided
        if not algorithm_name:
            import os
            algorithm_name = os.path.splitext(os.path.basename(solution_file))[0]
            
        return self.evaluate_solution(timetable, algorithm_name)
        
    def run_and_evaluate_algorithm(self, algorithm_func, algorithm_name, **kwargs):
        """
        Run an algorithm function and evaluate its solution.
        
        Args:
            algorithm_func: Function that runs the algorithm and returns a timetable
            algorithm_name: Name of the algorithm
            **kwargs: Additional keyword arguments to pass to the algorithm function
            
        Returns:
            Dictionary containing evaluation metrics
        """
        start_time = time.time()
        timetable = algorithm_func(self.data_file, **kwargs)
        execution_time = time.time() - start_time
        
        return self.evaluate_solution(timetable, algorithm_name, execution_time)
        
    def compare_algorithms(self, algorithms_dict=None):
        """
        Compare multiple algorithms based on their evaluation results.
        
        Args:
            algorithms_dict: Dictionary mapping algorithm names to their functions
                             If None, uses previously evaluated results
                             
        Returns:
            Dictionary mapping algorithm names to their evaluation results
        """
        if algorithms_dict:
            # Run and evaluate each algorithm
            for algo_name, algo_func in algorithms_dict.items():
                print(f"\n{'=' * 50}")
                print(f"Running and evaluating {algo_name.upper()}...")
                print(f"{'=' * 50}")
                self.run_and_evaluate_algorithm(algo_func, algo_name)
                
        # Return the results for all evaluated algorithms
        return self.algorithm_results
        
    def print_evaluation(self, result):
        """
        Print the evaluation result in a formatted way.
        
        Args:
            result: Evaluation result dictionary
        """
        algo_name = result.get('algorithm_name', 'Unknown')
        print(f"\n{'=' * 50}")
        print(f"Evaluation Results for {algo_name.upper()}")
        print(f"{'=' * 50}")
        
        if result.get('execution_time'):
            print(f"Execution Time: {result['execution_time']:.2f} seconds")
        
        # Hard constraints
        print("\n--- Hard Constraint Evaluation Results ---")
        print(f"Vacant Rooms Count: {result['hard_constraints']['vacant']}")
        print(f"Room Overbooking Violations: {result['hard_constraints']['overbooking']}")
        print(f"Professor Conflicts: {result['hard_constraints']['professor_conflicts']}")
        print(f"Group Conflicts: {result['hard_constraints']['group_conflicts']}")
        print(f"Unassigned Activities: {result['hard_constraints']['unassigned_activities']}")
        print(f"\nTotal Hard Constraint Violations: {result['hard_constraints']['total_violations']}")
        
        # Soft constraints
        print("\n--- Soft Constraint Evaluation Results ---")
        print(f"Student Fatigue Factor: {result['soft_constraints']['student_fatigue']:.2f}")
        print(f"Student Idle Time Factor: {result['soft_constraints']['student_idle_time']:.2f}")
        print(f"Student Lecture Spread Factor: {result['soft_constraints']['student_lecture_spread']:.2f}")
        print(f"Lecturer Fatigue Factor: {result['soft_constraints']['lecturer_fatigue']:.2f}")
        print(f"Lecturer Idle Time Factor: {result['soft_constraints']['lecturer_idle_time']:.2f}")
        print(f"Lecturer Lecture Spread Factor: {result['soft_constraints']['lecturer_lecture_spread']:.2f}")
        print(f"Lecturer Workload Balance Factor: {result['soft_constraints']['lecturer_workload_balance']:.2f}")
        print(f"\nFinal Soft Constraint Score: {result['soft_constraints']['final_score']:.2f}")
        
        # Overall score
        print(f"\nOverall Algorithm Score: {result['overall_score']:.2f}")
        
    def print_comparison_table(self):
        """
        Print a comparison table of all evaluated algorithms.
        """
        if not self.algorithm_results:
            print("No algorithms have been evaluated yet.")
            return
            
        print("\n" + "=" * 80)
        print("ALGORITHM COMPARISON")
        print("=" * 80)
        
        # Prepare table data
        headers = ["Algorithm", "Exec Time", "Unassigned", "Overbooking", "Conflicts", "Hard Total", "Soft Score", "Overall"]
        rows = []
        
        for algo_name, result in self.algorithm_results.items():
            row = [
                algo_name.upper(),
                f"{result.get('execution_time', 'N/A')}s" if result.get('execution_time') else "N/A",
                result['hard_constraints']['unassigned_activities'],
                result['hard_constraints']['overbooking'],
                result['hard_constraints']['professor_conflicts'] + result['hard_constraints']['group_conflicts'],
                result['hard_constraints']['total_violations'],
                f"{result['soft_constraints']['final_score']:.2f}",
                f"{result['overall_score']:.2f}"
            ]
            rows.append(row)
        
        # Sort by overall score (ascending is better)
        rows.sort(key=lambda x: float(x[-1].replace(',', '')))
        
        # Print table
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        print("=" * 80)
        
    def visualize_comparison(self, figsize=(12, 8), save_path=None):
        """
        Create a visual comparison of algorithm performance.
        
        Args:
            figsize: Figure size tuple (width, height)
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if not self.algorithm_results:
            print("No algorithms have been evaluated yet.")
            return None
            
        # Sort algorithms by overall score
        sorted_algos = sorted(
            self.algorithm_results.items(), 
            key=lambda x: x[1]['overall_score']
        )
        algo_names = [a[0].upper() for a in sorted_algos]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Algorithm Comparison', fontsize=16)
        
        # Hard constraints violations
        ax1 = axes[0, 0]
        hard_data = {
            'Unassigned': [a[1]['hard_constraints']['unassigned_activities'] for a in sorted_algos],
            'Overbooking': [a[1]['hard_constraints']['overbooking'] for a in sorted_algos],
            'Prof Conflicts': [a[1]['hard_constraints']['professor_conflicts'] for a in sorted_algos],
            'Group Conflicts': [a[1]['hard_constraints']['group_conflicts'] for a in sorted_algos]
        }
        
        bottom = np.zeros(len(algo_names))
        for label, values in hard_data.items():
            ax1.bar(algo_names, values, bottom=bottom, label=label)
            bottom += np.array(values)
            
        ax1.set_title('Hard Constraint Violations')
        ax1.legend()
        ax1.set_ylabel('Number of Violations')
        ax1.tick_params(axis='x', rotation=45)
        
        # Soft constraints scores
        ax2 = axes[0, 1]
        soft_metrics = [
            'student_fatigue', 'student_idle_time', 'student_lecture_spread',
            'lecturer_fatigue', 'lecturer_idle_time', 'lecturer_lecture_spread',
            'lecturer_workload_balance'
        ]
        
        soft_labels = [m.replace('_', ' ').title() for m in soft_metrics]
        
        # Create grouped bar chart for soft constraints
        x = np.arange(len(algo_names))
        width = 0.7 / len(soft_metrics)
        
        for i, metric in enumerate(soft_metrics):
            values = [a[1]['soft_constraints'][metric] for a in sorted_algos]
            ax2.bar(x + i*width - 0.35, values, width, label=soft_labels[i])
            
        ax2.set_title('Soft Constraint Scores')
        ax2.set_xticks(x)
        ax2.set_xticklabels(algo_names)
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylabel('Score (Lower is Better)')
        ax2.legend(fontsize='x-small')
        
        # Execution time
        ax3 = axes[1, 0]
        exec_times = [a[1].get('execution_time', 0) for a in sorted_algos]
        ax3.bar(algo_names, exec_times)
        ax3.set_title('Execution Time')
        ax3.set_ylabel('Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Overall score
        ax4 = axes[1, 1]
        overall_scores = [a[1]['overall_score'] for a in sorted_algos]
        ax4.bar(algo_names, overall_scores)
        ax4.set_title('Overall Score (Lower is Better)')
        ax4.set_ylabel('Score')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def save_results_to_file(self, filename='algorithm_comparison.json'):
        """
        Save all algorithm results to a JSON file.
        
        Args:
            filename: Target filename
        """
        if not self.algorithm_results:
            print("No algorithms have been evaluated yet.")
            return
            
        # Create a serializable version of the results
        serializable_results = {}
        for algo_name, result in self.algorithm_results.items():
            # Create a copy without the timetable which may have non-serializable objects
            result_copy = result.copy()
            result_copy.pop('timetable', None)
            serializable_results[algo_name] = result_copy
            
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"Saved algorithm comparison results to {filename}")


# Example usage
def main():
    """Example usage of the TimetableEvaluator class."""
    from nsga2 import run_nsga2_optimization
    from moead import run_moead_optimization
    
    try:
        from spea2 import run_spea2_optimization
        has_spea2 = True
    except ImportError:
        has_spea2 = False
    
    # Initialize evaluator
    data_file = "sliit_computing_dataset.json"
    evaluator = TimetableEvaluator(data_file)
    
    # Define algorithms to compare
    algorithms = {
        'nsga2': run_nsga2_optimization,
        'moead': run_moead_optimization
    }
    
    if has_spea2:
        algorithms['spea2'] = run_spea2_optimization
    
    # Option 1: Compare algorithms (runs each algorithm)
    # results = evaluator.compare_algorithms(algorithms)
    
    # Option 2: Evaluate existing solution files
    evaluator.evaluate_solution_file("nsga2_solution.json")
    evaluator.evaluate_solution_file("moead_solution.json")
    if has_spea2:
        evaluator.evaluate_solution_file("spea2_solution.json")
    
    # Print comparison table
    evaluator.print_comparison_table()
    
    # Visualize comparison
    evaluator.visualize_comparison(save_path="algorithm_comparison.png")
    
    # Save results
    evaluator.save_results_to_file()


if __name__ == "__main__":
    main()
