from copy import deepcopy

class Solution:
    """Class to represent a timetable solution."""
    def __init__(self):
        self.assignments = {}  # Maps class_id to (time_id, room_id)
        self.fitness = None
        self.constraint_violations = None  # Will store detailed constraint violation info
        self.crowding_distance = None  # Crowding distance for diversity preservation
        self.generation = None  # Generation number for dynamic mutation rate
    
    def copy(self):
        """Create a deep copy of the solution."""
        new_solution = Solution()
        new_solution.assignments = deepcopy(self.assignments)
        new_solution.fitness = deepcopy(self.fitness) if self.fitness else None
        new_solution.constraint_violations = deepcopy(self.constraint_violations) if self.constraint_violations else None
        new_solution.crowding_distance = self.crowding_distance
        new_solution.generation = self.generation
        return new_solution
