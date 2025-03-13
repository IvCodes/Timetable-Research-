import matplotlib.pyplot as plt


def plot_metrics(metrics):
    """Plot the metrics from the optimization."""
    plt.figure(figsize=(15, 10))
    
    # Plot hypervolume
    plt.subplot(2, 2, 1)
    plt.plot(metrics['hypervolume'])
    plt.title('Hypervolume')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    
    # Plot spacing
    plt.subplot(2, 2, 2)
    plt.plot(metrics['spacing'])
    plt.title('Spacing')
    plt.xlabel('Generation')
    plt.ylabel('Spacing')
    
    # Plot IGD
    plt.subplot(2, 2, 3)
    plt.plot(metrics['igd'])
    plt.title('Inverted Generational Distance')
    plt.xlabel('Generation')
    plt.ylabel('IGD')
    
    # Plot fitness
    plt.subplot(2, 2, 4)
    plt.plot(metrics['best_fitness'], label='Best')
    plt.plot(metrics['average_fitness'], label='Average')
    plt.title('Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (lower is better)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('nsga2_metrics.png')
    plt.close()
    
    # Plot constraint violations
    plt.figure(figsize=(15, 8))
    violations = [v['total_counts'] for v in metrics['constraint_violations']]
    
    for violation_type in ['room_conflicts', 'time_conflicts', 'distribution_conflicts', 
                          'student_conflicts', 'capacity_violations']:
        values = [v[violation_type] for v in violations]
        plt.plot(values, label=violation_type)
    
    plt.title('Constraint Violations')
    plt.xlabel('Generation')
    plt.ylabel('Number of Violations')
    plt.legend()
    plt.tight_layout()
    plt.savefig('nsga2_violations.png')
    plt.close()
    
    # Plot Pareto front size
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['pareto_front_size'])
    plt.title('Pareto Front Size')
    plt.xlabel('Generation')
    plt.ylabel('Number of Solutions')
    plt.tight_layout()
    plt.savefig('nsga2_pareto_size.png')
    plt.close()

def plot_constraint_violations(violations):
    """
    Plot the constraint violations over generations.
    
    Args:
        violations: List of dictionaries containing violation counts per generation
    """
    plt.figure(figsize=(15, 8))
    
    for violation_type in ['room_conflicts', 'time_conflicts', 'distribution_conflicts', 
                          'student_conflicts', 'capacity_violations']:
        values = [v[violation_type] for v in violations]
        plt.plot(values, label=violation_type)
    
    plt.title('Constraint Violations')
    plt.xlabel('Generation')
    plt.ylabel('Number of Violations')
    plt.legend()
    plt.tight_layout()
    plt.savefig('nsga2_violations.png')
    plt.close()

def plot_pareto_size(pareto_sizes):
    """
    Plot the size of the Pareto front over generations.
    
    Args:
        pareto_sizes: List of integers representing the size of the Pareto front per generation
    """
    plt.figure(figsize=(10, 6))
    plt.plot(pareto_sizes)
    plt.title('Pareto Front Size')
    plt.xlabel('Generation')
    plt.ylabel('Number of Solutions')
    plt.tight_layout()
    plt.savefig('nsga2_pareto_size.png')
    plt.close()