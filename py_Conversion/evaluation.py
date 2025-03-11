"""
Evaluation module for timetable scheduling algorithms.

This module provides comprehensive tools to:
1. Evaluate individual algorithm performance
2. Compare multiple algorithm solutions
3. Visualize results using various metrics
4. Calculate statistical measures of algorithm performance
5. Export evaluation results for further analysis
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import evaluate, load_data
from nsga2 import run_nsga2_optimization
from moead import run_moead_optimization
from spea2 import run_spea2_optimization

class TimetableEvaluator:
    """Class to evaluate and compare timetable scheduling algorithms."""
    
    def __init__(self, data_file):
        """
        Initialize the evaluator with a dataset.
        
        Args:
            data_file: Path to the JSON data file
        """
        self.data_file = data_file
        self.spaces_dict, self.groups_dict, self.lecturers_dict, self.activities_dict, self.slots = load_data(data_file)
        self.algorithm_results = {}
        
    def run_algorithm(self, algo_name, generations=100):
        """
        Run a specified algorithm and store its results.
        
        Args:
            algo_name: Name of the algorithm to run (nsga2, moead, or spea2)
            generations: Number of generations to run
        
        Returns:
            Dictionary with algorithm results
        """
        algorithm_functions = {
            'nsga2': run_nsga2_optimization,
            'moead': run_moead_optimization,
            'spea2': run_spea2_optimization
        }
        
        if algo_name not in algorithm_functions:
            raise ValueError(f"Algorithm '{algo_name}' not supported. Choose from: {list(algorithm_functions.keys())}")
        
        print(f"\n{'=' * 50}")
        print(f"Running {algo_name.upper()}...")
        print(f"{'=' * 50}")
        
        # Measure execution time
        start_time = time.time()
        
        # Run the algorithm
        best_timetable = algorithm_functions[algo_name](self.data_file)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Evaluate the solution
        evaluation_result = evaluate(best_timetable, self.groups_dict, self.lecturers_dict, 
                           self.activities_dict, self.spaces_dict, self.slots, as_dict=True)
        
        # Store results
        result = {
            'timetable': best_timetable,
            'evaluation': evaluation_result,
            'execution_time': execution_time
        }
        
        self.algorithm_results[algo_name] = result
        return result
    
    def run_all_algorithms(self, generations=100):
        """
        Run all supported algorithms and store their results.
        
        Args:
            generations: Number of generations to run for each algorithm
        
        Returns:
            Dictionary with all algorithm results
        """
        for algo_name in ['nsga2', 'moead', 'spea2']:
            self.run_algorithm(algo_name, generations)
        
        return self.algorithm_results
    
    def evaluate_hard_constraints(self, algo_name=None):
        """
        Evaluate hard constraint performance for a specific algorithm or all algorithms.
        
        Args:
            algo_name: Name of the algorithm to evaluate (or None for all)
        
        Returns:
            Dictionary with hard constraint violation counts
        """
        results = {}
        
        if algo_name:
            if algo_name not in self.algorithm_results:
                raise ValueError(f"Algorithm '{algo_name}' has not been run yet.")
            
            algorithms = {algo_name: self.algorithm_results[algo_name]}
        else:
            algorithms = self.algorithm_results
        
        for name, result in algorithms.items():
            eval_result = result['evaluation']
            hard_constraints = eval_result['hard_constraints']
            
            results[name] = {
                'overbooking': hard_constraints['overbooking'],
                'slot_conflicts': hard_constraints['slot_conflicts'],
                'professor_conflicts': hard_constraints['professor_conflicts'],
                'group_conflicts': hard_constraints['group_conflicts'],
                'unassigned_activities': hard_constraints['unassigned_activities'],
                'total_violations': sum([
                    hard_constraints['overbooking'],
                    hard_constraints['slot_conflicts'],
                    hard_constraints['professor_conflicts'],
                    hard_constraints['group_conflicts'],
                    hard_constraints['unassigned_activities']
                ])
            }
        
        return results
    
    def evaluate_soft_constraints(self, algo_name=None):
        """
        Evaluate soft constraint performance for a specific algorithm or all algorithms.
        
        Args:
            algo_name: Name of the algorithm to evaluate (or None for all)
        
        Returns:
            Dictionary with soft constraint scores
        """
        results = {}
        
        if algo_name:
            if algo_name not in self.algorithm_results:
                raise ValueError(f"Algorithm '{algo_name}' has not been run yet.")
            
            algorithms = {algo_name: self.algorithm_results[algo_name]}
        else:
            algorithms = self.algorithm_results
        
        for name, result in algorithms.items():
            eval_result = result['evaluation']
            soft_constraints = eval_result['soft_constraints']
            
            results[name] = {
                'student_fatigue': soft_constraints['student_fatigue'],
                'student_idle_time': soft_constraints['student_idle_time'],
                'student_lecture_spread': soft_constraints['student_lecture_spread'],
                'lecturer_fatigue': soft_constraints['lecturer_fatigue'],
                'lecturer_idle_time': soft_constraints['lecturer_idle_time'],
                'lecturer_lecture_spread': soft_constraints['lecturer_lecture_spread'],
                'lecturer_workload_balance': soft_constraints['lecturer_workload_balance'],
                'total_soft_score': sum([
                    soft_constraints['student_fatigue'],
                    soft_constraints['student_idle_time'],
                    soft_constraints['student_lecture_spread'],
                    soft_constraints['lecturer_fatigue'],
                    soft_constraints['lecturer_idle_time'],
                    soft_constraints['lecturer_lecture_spread'],
                    soft_constraints['lecturer_workload_balance']
                ])
            }
        
        return results
    
    def get_combined_metrics(self):
        """
        Get combined metrics for all algorithms.
        
        Returns:
            Dictionary with combined metrics
        """
        if not self.algorithm_results:
            raise ValueError("No algorithms have been run yet.")
        
        hard_constraints = self.evaluate_hard_constraints()
        soft_constraints = self.evaluate_soft_constraints()
        
        combined = {}
        for algo_name in self.algorithm_results:
            combined[algo_name] = {
                'execution_time': self.algorithm_results[algo_name]['execution_time'],
                'hard_constraints': hard_constraints[algo_name],
                'soft_constraints': soft_constraints[algo_name]
            }
        
        return combined
    
    def plot_hard_constraints(self, save_path=None):
        """
        Plot hard constraint violations for all algorithms.
        
        Args:
            save_path: Path to save the plot (or None to display)
        """
        if not self.algorithm_results:
            raise ValueError("No algorithms have been run yet.")
        
        hard_constraints = self.evaluate_hard_constraints()
        
        # Extract constraint types and algorithm names
        constraint_types = ['overbooking', 'slot_conflicts', 'professor_conflicts', 
                           'group_conflicts', 'unassigned_activities']
        algo_names = list(hard_constraints.keys())
        
        # Create a figure with appropriate size
        plt.figure(figsize=(12, 8))
        
        # Bar width and positions
        bar_width = 0.15
        r = np.arange(len(constraint_types))
        
        # Plot bars for each algorithm
        for i, algo_name in enumerate(algo_names):
            values = [hard_constraints[algo_name][constraint] for constraint in constraint_types]
            plt.bar(r + i * bar_width, values, width=bar_width, label=algo_name.upper())
        
        # Add labels and title
        plt.xlabel('Constraint Type')
        plt.ylabel('Number of Violations')
        plt.title('Hard Constraint Violations by Algorithm')
        plt.xticks(r + bar_width * (len(algo_names) - 1) / 2, constraint_types)
        plt.legend()
        
        # Save or display the plot
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_soft_constraints(self, save_path=None):
        """
        Plot soft constraint scores for all algorithms.
        
        Args:
            save_path: Path to save the plot (or None to display)
        """
        if not self.algorithm_results:
            raise ValueError("No algorithms have been run yet.")
        
        soft_constraints = self.evaluate_soft_constraints()
        
        # Extract constraint types and algorithm names
        constraint_types = ['student_fatigue', 'student_idle_time', 'student_lecture_spread',
                           'lecturer_fatigue', 'lecturer_idle_time', 'lecturer_lecture_spread',
                           'lecturer_workload_balance']
        algo_names = list(soft_constraints.keys())
        
        # Create a figure with appropriate size
        plt.figure(figsize=(14, 8))
        
        # Bar width and positions
        bar_width = 0.15
        r = np.arange(len(constraint_types))
        
        # Plot bars for each algorithm
        for i, algo_name in enumerate(algo_names):
            values = [soft_constraints[algo_name][constraint] for constraint in constraint_types]
            plt.bar(r + i * bar_width, values, width=bar_width, label=algo_name.upper())
        
        # Add labels and title
        plt.xlabel('Constraint Type')
        plt.ylabel('Score')
        plt.title('Soft Constraint Scores by Algorithm')
        plt.xticks(r + bar_width * (len(algo_names) - 1) / 2, constraint_types, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save or display the plot
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_execution_times(self, save_path=None):
        """
        Plot execution times for all algorithms.
        
        Args:
            save_path: Path to save the plot (or None to display)
        """
        if not self.algorithm_results:
            raise ValueError("No algorithms have been run yet.")
        
        # Extract algorithm names and execution times
        algo_names = list(self.algorithm_results.keys())
        execution_times = [self.algorithm_results[name]['execution_time'] for name in algo_names]
        
        # Create a figure
        plt.figure(figsize=(10, 6))
        
        # Plot bars
        plt.bar(algo_names, execution_times)
        
        # Add labels and title
        plt.xlabel('Algorithm')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Algorithm Execution Times')
        
        # Save or display the plot
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def export_results(self, export_file):
        """
        Export evaluation results to a JSON file.
        
        Args:
            export_file: Path to the export file
        """
        if not self.algorithm_results:
            raise ValueError("No algorithms have been run yet.")
        
        # Create exportable results (excluding timetables which are too large)
        export_data = {}
        for algo_name, result in self.algorithm_results.items():
            export_data[algo_name] = {
                'evaluation': result['evaluation'],
                'execution_time': result['execution_time']
            }
        
        # Export to file
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Results exported to {export_file}")
    
    def print_comparison_table(self):
        """Print a comparison table of algorithm results."""
        if not self.algorithm_results:
            print("No results to compare.")
            return
        
        print("\n" + "=" * 80)
        print("ALGORITHM COMPARISON")
        print("=" * 80)
        
        # Table header
        header = ["Algorithm", "Exec Time", "Unassigned", "Conflicts", "Student Score", "Lecturer Score"]
        print(f"{header[0]:<10} {header[1]:<12} {header[2]:<10} {header[3]:<12} {header[4]:<15} {header[5]:<15}")
        print("-" * 80)
        
        # Table rows
        for algo_name, result in self.algorithm_results.items():
            eval_result = result['evaluation']
            
            # Calculate total conflicts
            total_conflicts = (
                eval_result['hard_constraints']['overbooking'] +
                eval_result['hard_constraints']['slot_conflicts'] +
                eval_result['hard_constraints']['professor_conflicts'] +
                eval_result['hard_constraints']['group_conflicts']
            )
            
            # Calculate student and lecturer scores
            student_score = (
                eval_result['soft_constraints']['student_fatigue'] +
                eval_result['soft_constraints']['student_idle_time'] +
                eval_result['soft_constraints']['student_lecture_spread']
            )
            
            lecturer_score = (
                eval_result['soft_constraints']['lecturer_fatigue'] +
                eval_result['soft_constraints']['lecturer_idle_time'] +
                eval_result['soft_constraints']['lecturer_lecture_spread'] +
                eval_result['soft_constraints']['lecturer_workload_balance']
            )
            
            row = [
                algo_name.upper(),
                f"{result['execution_time']:.2f}s",
                eval_result['hard_constraints']['unassigned_activities'],
                total_conflicts,
                student_score,
                lecturer_score
            ]
            print(f"{row[0]:<10} {row[1]:<12} {row[2]:<10} {row[3]:<12} {row[4]:<15} {row[5]:<15}")
        
        print("=" * 80)
    
    def analyze_timetable_quality(self, algo_name):
        """
        Analyze timetable quality metrics for a specific algorithm.
        
        Args:
            algo_name: Name of the algorithm to analyze
        
        Returns:
            Dictionary with quality metrics
        """
        if algo_name not in self.algorithm_results:
            raise ValueError(f"Algorithm '{algo_name}' has not been run yet.")
        
        timetable = self.algorithm_results[algo_name]['timetable']
        
        # Number of slots and spaces used
        slots_used = set()
        spaces_used = set()
        activities_scheduled = 0
        
        for slot in timetable:
            for space in timetable[slot]:
                if timetable[slot][space]:
                    slots_used.add(slot)
                    spaces_used.add(space)
                    activities_scheduled += 1
        
        # Calculate resource utilization
        total_slots = len(self.slots)
        total_spaces = len(self.spaces_dict)
        total_activities = len(self.activities_dict)
        
        slot_utilization = len(slots_used) / total_slots * 100
        space_utilization = len(spaces_used) / total_spaces * 100
        activities_scheduled_pct = activities_scheduled / total_activities * 100
        
        return {
            'total_slots': total_slots,
            'slots_used': len(slots_used),
            'slot_utilization_pct': slot_utilization,
            'total_spaces': total_spaces, 
            'spaces_used': len(spaces_used),
            'space_utilization_pct': space_utilization,
            'total_activities': total_activities,
            'activities_scheduled': activities_scheduled,
            'activities_scheduled_pct': activities_scheduled_pct
        }

def run_evaluation(data_file="sliit_computing_dataset.json", generations=100, export_results=True):
    """
    Run a full evaluation of all timetable scheduling algorithms.
    
    Args:
        data_file: Path to the JSON data file
        generations: Number of generations to run for each algorithm
        export_results: Whether to export results to a file
    
    Returns:
        TimetableEvaluator instance with results
    """
    print(f"Starting evaluation with dataset: {data_file}")
    print(f"Running each algorithm for {generations} generations")
    
    # Create evaluator
    evaluator = TimetableEvaluator(data_file)
    
    # Run all algorithms
    evaluator.run_all_algorithms(generations)
    
    # Print comparison table
    evaluator.print_comparison_table()
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    evaluator.plot_hard_constraints("hard_constraints_comparison.png")
    evaluator.plot_soft_constraints("soft_constraints_comparison.png")
    evaluator.plot_execution_times("execution_times_comparison.png")
    
    # Export results if requested
    if export_results:
        evaluator.export_results("algorithm_evaluation_results.json")
    
    print("\nEvaluation complete.")
    return evaluator

if __name__ == "__main__":
    # Run a full evaluation
    evaluator = run_evaluation()
    
    # Analyze quality for each algorithm
    for algo in ["nsga2", "moead", "spea2"]:
        print(f"\nTimetable Quality Analysis for {algo.upper()}:")
        quality = evaluator.analyze_timetable_quality(algo)
        print(f"Slot Utilization: {quality['slot_utilization_pct']:.2f}%")
        print(f"Space Utilization: {quality['space_utilization_pct']:.2f}%")
        print(f"Activities Scheduled: {quality['activities_scheduled']} / {quality['total_activities']} ({quality['activities_scheduled_pct']:.2f}%)")
