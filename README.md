# Timetable Scheduling Multi-Objective Optimization

This project implements advanced multi-objective evolutionary algorithms to solve complex university timetable scheduling challenges. It provides efficient, flexible solutions to optimize class schedules while balancing multiple constraints.

## Implemented Algorithms

1. **NSGA-II (Non-dominated Sorting Genetic Algorithm II)**
   - Uses non-dominated sorting for ranking solutions
   - Employs crowding distance for diversity preservation
   - Tournament selection based on rank and crowding distance

2. **MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)**
   - Decomposes multi-objective problem into single-objective subproblems
   - Uses weight vectors to define subproblems
   - Applies neighborhood-based selection and replacement
   - Implements Tchebycheff scalarizing function

3. **SPEA2 (Strength Pareto Evolutionary Algorithm 2)**
   - Fine-grained fitness assignment based on dominated and dominating solutions
   - Nearest neighbor density estimation for breaking ties
   - Environmental selection preserving boundary solutions
   - Archive of non-dominated solutions

## Optimization Objectives

### Hard Constraints
- Room overbooking
- Slot conflicts
- Professor conflicts
- Student group conflicts
- Unassigned activities

### Soft Constraints
- Student fatigue
- Student idle time
- Student lecture spread
- Lecturer fatigue
- Lecturer idle time
- Lecturer lecture spread
- Lecturer workload balance

## Project Structure

- **utils.py**: Shared utility functions and data structures
- **nsga2.py**: Implementation of the NSGA-II algorithm
- **moead.py**: Implementation of the MOEA/D algorithm
- **spea2.py**: Implementation of the SPEA2 algorithm
- **main.py**: Script to run and compare all algorithms
- **Notebooks**: Original Jupyter notebooks (`Genetic_Algorithm_Scheduling.ipynb` & `RL_Scheduling.ipynb`)

## Usage

To run all algorithms and compare their performance:

```bash
python main.py
```

To run a specific algorithm:

```python
from nsga2 import run_nsga2_optimization

# Run NSGA-II
best_timetable = run_nsga2_optimization("sliit_computing_dataset.json")
```

## Algorithm Parameters

Common parameters across all algorithms:
- `POPULATION_SIZE = 50`
- `NUM_GENERATIONS = 100`
- `MUTATION_RATE = 0.1`
- `CROSSOVER_RATE = 0.8`

## Data Format

The algorithms expect a JSON data file with the following structure:
- Spaces (rooms)
- Groups (student groups)
- Activities (lectures, labs, etc.)
- Lecturers

## Future Improvements

- More sophisticated mutation and crossover strategies
- Advanced constraint handling
- Hyperparameter tuning
- Performance optimization
- Interactive visualization tools

## Original Notes

Consider `Genetic_Algorithm_Scheduling.ipynb` & `RL_Scheduling.ipynb` as the final Evaluation Outputs
