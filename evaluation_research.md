# Evaluation Methodology for Timetable Scheduling Optimization Algorithms

## Introduction

This document outlines the comprehensive evaluation methodology used to assess and compare various multi-objective optimization algorithms for timetable scheduling. Our research utilizes two distinct algorithmic approaches: evolutionary algorithms (EAs) and reinforcement learning (RL), each with their own evaluation metrics and performance measures.

## 1. Evolutionary Algorithm Evaluation Framework

Our implementation includes three major evolutionary algorithms: NSGA-II, MOEA/D, and SPEA2. These algorithms are evaluated using a systematic multi-objective approach:

### 1.1 Hard Constraint Evaluation

Hard constraints represent inviolable rules that must be satisfied for a timetable to be considered valid. Our framework evaluates five key hard constraints:

| Constraint Type | Description | Evaluation Method |
|-----------------|-------------|------------------|
| Room Overbooking | Multiple activities assigned to the same room/slot | Count of instances where a room hosts multiple activities in the same slot |
| Professor Conflicts | Professors assigned to multiple activities simultaneously | Count of instances where a lecturer is assigned to teach multiple classes in the same timeslot |
| Student Group Conflicts | Student groups assigned to multiple activities at once | Count of instances where a student group is assigned to attend multiple activities simultaneously |
| Room Capacity Violations | Room capacity insufficient for assigned activity | Count of instances where the total student count exceeds room capacity |
| Unassigned Activities | Activities not scheduled in any slot | Count of activities that could not be placed in the timetable |

The total hard constraint violation score is calculated as:
```
total_violations = overbooking + slot_conflicts + professor_conflicts + group_conflicts + unassigned_activities
```

Lower scores indicate better performance, with zero representing a perfectly feasible timetable.

### 1.2 Soft Constraint Evaluation

Soft constraints represent optimization objectives that should be satisfied when possible but can be compromised if necessary. Our framework evaluates seven soft constraints:

| Constraint Type | Description | Evaluation Method |
|-----------------|-------------|------------------|
| Student Fatigue | Overall workload on students | Normalized score based on total lecture attendance per student group |
| Student Idle Time | Time gaps between classes | Calculated based on non-consecutive slot assignments per day |
| Student Lecture Spread | Distribution of lectures across available slots | Based on variance of lecture distribution across the week |
| Lecturer Fatigue | Workload on lecturers | Normalized score based on number of teaching sessions |
| Lecturer Idle Time | Time gaps between teaching sessions | Calculated based on non-consecutive teaching slots per day |
| Lecturer Lecture Spread | Distribution of teaching across available slots | Based on variance of teaching slots across the week |
| Lecturer Workload Balance | Equity in teaching load distribution | Calculated as inverse of the workload variance across lecturers |

Each soft constraint is normalized to a 0-1 scale, where higher values indicate better satisfaction of constraints. The final soft constraint score is calculated as a weighted average:
```
final_score = (w1 * student_fatigue + w2 * student_idle + w3 * student_spread + 
              w4 * lecturer_fatigue + w5 * lecturer_idle + w6 * lecturer_spread + 
              w7 * workload_balance) / (w1 + w2 + w3 + w4 + w5 + w6 + w7)
```

Higher scores indicate better overall satisfaction of soft constraints, with 1.0 representing ideal optimization.

### 1.3 Resource Utilization Metrics

Our framework also evaluates resource utilization aspects:

| Metric | Calculation |
|--------|-------------|
| Slot Utilization | Percentage of available timeslots used |
| Space Utilization | Percentage of available rooms used |
| Activity Scheduling Rate | Percentage of activities successfully scheduled |

### 1.4 Execution Performance Metrics

Implementation efficiency is measured through:

| Metric | Description |
|--------|-------------|
| Execution Time | Total time taken to generate a solution |
| Convergence Rate | Number of generations needed to reach stability |
| Memory Usage | Peak memory consumption during execution |

### 1.5 Comparative Analysis Methodology

For comparing multiple algorithms, we employ:

1. **Pareto Frontier Analysis**: Identifying non-dominated solutions
2. **Statistical Significance Testing**: Using t-tests or ANOVA for performance comparison
3. **Visual Representation**: Using radar charts and parallel coordinate plots for multi-objective visualization

## 2. Reinforcement Learning Evaluation Framework

Our Deep Q-Learning approach to timetable scheduling is evaluated using a distinct methodology appropriate for RL-based optimization:

### 2.1 Reward Function Analysis

The RL agent is guided by a comprehensive reward function that incorporates multiple objectives:

| Reward Component | Value | Description |
|------------------|-------|-------------|
| Valid Placement | +10 | Reward for each successfully placed activity |
| Teacher Conflict | -20 | Penalty for double-booking a teacher |
| Group Conflict | -15 | Penalty for assigning a student group to multiple activities simultaneously |
| Student Overlap | -25 | Penalty for student groups overlapping in the same slot |
| Room Capacity Violation | -30 | Penalty when room capacity is exceeded |

The cumulative reward serves as a primary indicator of solution quality, with higher values indicating better timetables.

### 2.2 Learning Performance Metrics

The RL approach is also evaluated on its learning capabilities:

| Metric | Description |
|--------|-------------|
| Training Convergence | Number of episodes needed to reach stable rewards |
| Exploration Efficiency | How quickly the ε-greedy strategy transitions to exploitation |
| Action Space Coverage | Percentage of possible actions explored during training |
| Q-Value Stability | Variance in Q-values as training progresses |

### 2.3 Model Architecture Impact

We assess how neural network architecture affects performance:

| Component | Analysis |
|-----------|----------|
| Network Depth | Impact of adding/removing hidden layers |
| Layer Width | Effect of changing neuron counts in hidden layers |
| Activation Functions | Performance comparison between ReLU, tanh, and sigmoid |
| Learning Rate | Sensitivity to different learning rate values |

### 2.4 Solution Quality Over Time

Unlike EAs that produce generations of solutions, RL provides a single solution that improves over time. We track:

| Metric | Description |
|--------|-------------|
| Reward Trajectory | Plot of cumulative reward across training episodes |
| Constraint Violation Reduction | How quickly hard constraints are satisfied during training |
| Final Solution Quality | Comparison of final solution to evolutionary approaches |

## 3. Comparative Methodology: EA vs. RL

To facilitate fair comparison between fundamentally different approaches, we employ:

### 3.1 Shared Performance Metrics

| Metric | Description |
|--------|-------------|
| Solution Quality | Measured by hard and soft constraint satisfaction |
| Computation Efficiency | Time and resources required to reach solutions of comparable quality |
| Scalability | Performance change as problem size increases |
| Adaptability | Ability to handle changes in problem constraints |

### 3.2 Specialized Advantages

We also document the unique strengths of each approach:

| Approach | Distinctive Advantages |
|----------|------------------------|
| Evolutionary | Population-based exploration, explicit Pareto frontier generation |
| Reinforcement Learning | Sequential decision making, potential for transfer learning |

## 4. Implementation and Visualization

Our evaluation framework includes robust implementation tools:

1. **Automated Execution**: Batch processing of algorithm runs with different parameters
2. **Statistical Analysis**: Python-based analysis using scipy and numpy
3. **Visualization Suite**: Comprehensive plots for constraint violations, solution quality, and algorithm comparison

## Conclusion

This multi-faceted evaluation methodology provides a holistic assessment of timetable scheduling algorithms. By combining hard constraint violation counts, soft constraint satisfaction scores, execution metrics, and learning performance measures, we gain comprehensive insight into the strengths and weaknesses of different optimization approaches.

The framework is designed to be extensible, supporting the addition of new algorithms and metrics as research progresses, while maintaining consistent comparison standards across fundamentally different optimization paradigms.
