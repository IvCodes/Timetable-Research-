"""
Benchmark module for timetable scheduling algorithms.

This module provides tools to:
1. Run performance benchmarks on algorithms with different parameters
2. Test scalability with different dataset sizes
3. Generate benchmark reports
4. Compare algorithm convergence rates
"""

import time
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import evaluate, load_data, dominates
from nsga2 import nsga2, evaluate_population
from moead import moead
from spea2 import spea2

class AlgorithmBenchmark:
    """Class to benchmark timetable scheduling algorithms."""
    
    def __init__(self, data_file):
        """
        Initialize the benchmark with a dataset.
        
        Args:
            data_file: Path to the JSON data file
        """
        self.data_file = data_file
        self.spaces_dict, self.groups_dict, self.lecturers_dict, self.activities_dict, self.slots = load_data(data_file)
        self.benchmark_results = {}
    
    def benchmark_algorithm(self, algo_name, param_sets, iterations=3):
        """
        Benchmark an algorithm with different parameter sets.
        
        Args:
            algo_name: Name of the algorithm to benchmark (nsga2, moead, or spea2)
            param_sets: List of parameter dictionaries
            iterations: Number of iterations to run for each parameter set
        
        Returns:
            Dictionary with benchmark results
        """
        algorithm_functions = {
            'nsga2': nsga2,
            'moead': moead,
            'spea2': spea2
        }
        
        if algo_name not in algorithm_functions:
            raise ValueError(f"Algorithm '{algo_name}' not supported. Choose from: {list(algorithm_functions.keys())}")
        
        algorithm_func = algorithm_functions[algo_name]
        results = {}
        
        print(f"\n{'=' * 50}")
        print(f"Benchmarking {algo_name.upper()} with {len(param_sets)} parameter sets")
        print(f"{'=' * 50}")
        
        for i, params in enumerate(param_sets):
            print(f"\nParameter Set {i+1}: {params}")
            param_results = []
            
            for iteration in range(iterations):
                print(f"  Iteration {iteration+1}/{iterations}...")
                
                # Measure execution time
                start_time = time.time()
                
                # Run the algorithm with the given parameters
                if algo_name == 'nsga2':
                    generations = params.get('generations', 100)
                    population = algorithm_func(
                        self.activities_dict, 
                        self.groups_dict, 
                        self.spaces_dict, 
                        self.slots,
                        generations=generations
                    )
                    fitness_values = evaluate_population(population, self.activities_dict, self.groups_dict, self.spaces_dict)
                    
                elif algo_name == 'moead':
                    generations = params.get('generations', 100)
                    population, fitness_values = algorithm_func(
                        self.activities_dict, 
                        self.groups_dict, 
                        self.spaces_dict, 
                        self.slots,
                        generations=generations
                    )
                    
                elif algo_name == 'spea2':
                    generations = params.get('generations', 100)
                    population = algorithm_func(
                        self.activities_dict, 
                        self.groups_dict, 
                        self.spaces_dict, 
                        self.slots,
                        generations=generations
                    )
                    fitness_values = evaluate_population(population, self.activities_dict, self.groups_dict, self.spaces_dict)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Find best solution
                best_solution = None
                min_violations = float('inf')
                best_fitness = None
                
                for j, individual in enumerate(population):
                    total_violations = sum(fitness_values[j][1:])  # Exclude vacant rooms count
                    
                    if total_violations < min_violations:
                        min_violations = total_violations
                        best_solution = individual
                        best_fitness = fitness_values[j]
                
                # Evaluate best solution
                evaluation_result = evaluate(
                    best_solution, 
                    self.groups_dict, 
                    self.lecturers_dict, 
                    self.activities_dict, 
                    self.spaces_dict, 
                    self.slots, 
                    as_dict=True
                )
                
                param_results.append({
                    'execution_time': execution_time,
                    'best_fitness': best_fitness,
                    'evaluation': evaluation_result
                })
            
            # Calculate average metrics across iterations
            avg_execution_time = np.mean([result['execution_time'] for result in param_results])
            avg_hard_violations = np.mean([
                sum([
                    result['evaluation']['hard_constraints']['overbooking'],
                    result['evaluation']['hard_constraints']['slot_conflicts'],
                    result['evaluation']['hard_constraints']['professor_conflicts'],
                    result['evaluation']['hard_constraints']['group_conflicts'],
                    result['evaluation']['hard_constraints']['unassigned_activities']
                ]) 
                for result in param_results
            ])
            
            results[f"param_set_{i+1}"] = {
                'parameters': params,
                'avg_execution_time': avg_execution_time,
                'avg_hard_violations': avg_hard_violations,
                'detailed_results': param_results
            }
            
            print(f"  Average Execution Time: {avg_execution_time:.2f} seconds")
            print(f"  Average Hard Violations: {avg_hard_violations:.2f}")
        
        self.benchmark_results[algo_name] = results
        return results
    
    def benchmark_all_algorithms(self, param_sets_dict, iterations=3):
        """
        Benchmark all algorithms with their respective parameter sets.
        
        Args:
            param_sets_dict: Dictionary mapping algorithm names to lists of parameter dictionaries
            iterations: Number of iterations to run for each parameter set
        
        Returns:
            Dictionary with all benchmark results
        """
        for algo_name, param_sets in param_sets_dict.items():
            self.benchmark_algorithm(algo_name, param_sets, iterations)
        
        return self.benchmark_results
    
    def plot_execution_times(self, save_path=None):
        """
        Plot execution times for all benchmarked algorithms.
        
        Args:
            save_path: Path to save the plot (or None to display)
        """
        if not self.benchmark_results:
            raise ValueError("No benchmarks have been run yet.")
        
        plt.figure(figsize=(12, 8))
        
        bar_width = 0.2
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (algo_name, results) in enumerate(self.benchmark_results.items()):
            param_sets = list(results.keys())
            exec_times = [results[param_set]['avg_execution_time'] for param_set in param_sets]
            
            x = np.arange(len(param_sets))
            plt.bar(x + i * bar_width, exec_times, width=bar_width, label=algo_name.upper(), color=colors[i % len(colors)])
        
        plt.xlabel('Parameter Set')
        plt.ylabel('Average Execution Time (seconds)')
        plt.title('Algorithm Execution Times by Parameter Set')
        plt.legend()
        
        # Set x-ticks at the center of each group
        num_algos = len(self.benchmark_results)
        plt.xticks(np.arange(len(param_sets)) + bar_width * (num_algos - 1) / 2, [f"Set {i+1}" for i in range(len(param_sets))])
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_constraint_violations(self, save_path=None):
        """
        Plot constraint violations for all benchmarked algorithms.
        
        Args:
            save_path: Path to save the plot (or None to display)
        """
        if not self.benchmark_results:
            raise ValueError("No benchmarks have been run yet.")
        
        plt.figure(figsize=(12, 8))
        
        bar_width = 0.2
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (algo_name, results) in enumerate(self.benchmark_results.items()):
            param_sets = list(results.keys())
            violations = [results[param_set]['avg_hard_violations'] for param_set in param_sets]
            
            x = np.arange(len(param_sets))
            plt.bar(x + i * bar_width, violations, width=bar_width, label=algo_name.upper(), color=colors[i % len(colors)])
        
        plt.xlabel('Parameter Set')
        plt.ylabel('Average Hard Constraint Violations')
        plt.title('Algorithm Constraint Violations by Parameter Set')
        plt.legend()
        
        # Set x-ticks at the center of each group
        num_algos = len(self.benchmark_results)
        plt.xticks(np.arange(len(param_sets)) + bar_width * (num_algos - 1) / 2, [f"Set {i+1}" for i in range(len(param_sets))])
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def analyze_convergence(self, algo_name, generations_list=[10, 25, 50, 100, 200], iterations=3):
        """
        Analyze algorithm convergence over different numbers of generations.
        
        Args:
            algo_name: Name of the algorithm to analyze
            generations_list: List of generation counts to test
            iterations: Number of iterations to run for each generation count
        
        Returns:
            Dictionary with convergence analysis results
        """
        param_sets = [{'generations': gen} for gen in generations_list]
        benchmark_results = self.benchmark_algorithm(algo_name, param_sets, iterations)
        
        # Extract relevant data for plotting
        generations = generations_list
        avg_violations = [benchmark_results[f"param_set_{i+1}"]["avg_hard_violations"] for i in range(len(generations_list))]
        
        # Plot convergence
        plt.figure(figsize=(10, 6))
        plt.plot(generations, avg_violations, 'o-', linewidth=2)
        plt.xlabel('Number of Generations')
        plt.ylabel('Average Hard Constraint Violations')
        plt.title(f'Convergence Analysis for {algo_name.upper()}')
        plt.grid(True)
        plt.savefig(f"{algo_name}_convergence.png")
        
        return {
            'generations': generations,
            'avg_violations': avg_violations
        }
    
    def export_benchmark_report(self, export_file):
        """
        Export benchmark results to a JSON file.
        
        Args:
            export_file: Path to the export file
        """
        if not self.benchmark_results:
            raise ValueError("No benchmarks have been run yet.")
        
        # Create a simplified version of results for export
        export_data = {}
        for algo_name, results in self.benchmark_results.items():
            export_data[algo_name] = {}
            for param_set, data in results.items():
                export_data[algo_name][param_set] = {
                    'parameters': data['parameters'],
                    'avg_execution_time': data['avg_execution_time'],
                    'avg_hard_violations': data['avg_hard_violations']
                }
        
        # Export to file
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Benchmark report exported to {export_file}")

def run_standard_benchmark(data_file="sliit_computing_dataset.json"):
    """
    Run a standard benchmark for all algorithms with predefined parameter sets.
    
    Args:
        data_file: Path to the JSON data file
    
    Returns:
        AlgorithmBenchmark instance with results
    """
    print(f"Starting standard benchmark with dataset: {data_file}")
    
    # Create benchmarker
    benchmarker = AlgorithmBenchmark(data_file)
    
    # Define parameter sets for each algorithm
    param_sets = {
        'nsga2': [
            {'generations': 50, 'population_size': 50, 'mutation_rate': 0.1, 'crossover_rate': 0.8},
            {'generations': 50, 'population_size': 100, 'mutation_rate': 0.1, 'crossover_rate': 0.8},
            {'generations': 50, 'population_size': 50, 'mutation_rate': 0.2, 'crossover_rate': 0.8}
        ],
        'moead': [
            {'generations': 50, 'population_size': 50, 'neighborhood_size': 10, 'neighborhood_selection_prob': 0.9},
            {'generations': 50, 'population_size': 100, 'neighborhood_size': 10, 'neighborhood_selection_prob': 0.9},
            {'generations': 50, 'population_size': 50, 'neighborhood_size': 20, 'neighborhood_selection_prob': 0.9}
        ],
        'spea2': [
            {'generations': 50, 'population_size': 50, 'archive_size': 50},
            {'generations': 50, 'population_size': 100, 'archive_size': 50},
            {'generations': 50, 'population_size': 50, 'archive_size': 100}
        ]
    }
    
    # Run benchmarks with fewer iterations for quicker results
    benchmarker.benchmark_all_algorithms(param_sets, iterations=2)
    
    # Generate plots
    print("\nGenerating benchmark plots...")
    benchmarker.plot_execution_times("benchmark_execution_times.png")
    benchmarker.plot_constraint_violations("benchmark_constraint_violations.png")
    
    # Export benchmark report
    benchmarker.export_benchmark_report("algorithm_benchmark_report.json")
    
    print("\nBenchmark complete.")
    return benchmarker

def analyze_algorithm_convergence(data_file="sliit_computing_dataset.json"):
    """
    Analyze convergence for all algorithms.
    
    Args:
        data_file: Path to the JSON data file
    
    Returns:
        Dictionary with convergence analysis results
    """
    print(f"Starting convergence analysis with dataset: {data_file}")
    
    # Create benchmarker
    benchmarker = AlgorithmBenchmark(data_file)
    
    # Analyze convergence for each algorithm
    results = {}
    for algo_name in ['nsga2', 'moead', 'spea2']:
        print(f"\nAnalyzing convergence for {algo_name.upper()}...")
        results[algo_name] = benchmarker.analyze_convergence(
            algo_name, 
            generations_list=[10, 25, 50, 75, 100], 
            iterations=2  # Reduced iterations for quicker analysis
        )
    
    # Plot combined convergence
    plt.figure(figsize=(12, 8))
    for algo_name, data in results.items():
        plt.plot(data['generations'], data['avg_violations'], 'o-', linewidth=2, label=algo_name.upper())
    
    plt.xlabel('Number of Generations')
    plt.ylabel('Average Hard Constraint Violations')
    plt.title('Convergence Analysis for All Algorithms')
    plt.legend()
    plt.grid(True)
    plt.savefig("combined_convergence.png")
    
    print("\nConvergence analysis complete.")
    return results

if __name__ == "__main__":
    # Run standard benchmark
    benchmarker = run_standard_benchmark()
    
    # Analyze algorithm convergence
    convergence_results = analyze_algorithm_convergence()
