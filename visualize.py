"""
Visualization module for timetable scheduling solutions.

This module provides tools to:
1. Generate visual representations of timetables
2. Create heatmaps for room/slot utilization
3. Visualize constraint violations spatially
4. Compare multiple timetable solutions visually
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from utils import load_data

class TimetableVisualizer:
    """Class to visualize timetable scheduling solutions."""
    
    def __init__(self, data_file):
        """
        Initialize the visualizer with a dataset.
        
        Args:
            data_file: Path to the JSON data file
        """
        self.data_file = data_file
        self.spaces_dict, self.groups_dict, self.lecturers_dict, self.activities_dict, self.slots = load_data(data_file)
    
    def load_timetable(self, timetable_file):
        """
        Load a timetable from a JSON file.
        
        Args:
            timetable_file: Path to the timetable JSON file
        
        Returns:
            Loaded timetable dictionary
        """
        with open(timetable_file, 'r') as f:
            timetable = json.load(f)
        return timetable
    
    def create_timetable_dataframe(self, timetable):
        """
        Create a DataFrame representation of a timetable.
        
        Args:
            timetable: Timetable dictionary
        
        Returns:
            DataFrame representation of the timetable
        """
        # Create a matrix representation
        spaces = list(self.spaces_dict.keys())
        slots = self.slots
        
        # Initialize DataFrame with empty strings
        df = pd.DataFrame(index=slots, columns=spaces, data='')
        
        # Fill in the activities
        for slot in timetable:
            for space in timetable[slot]:
                if space in spaces and timetable[slot][space]:
                    activity = timetable[slot][space]
                    if isinstance(activity, dict) and 'code' in activity:
                        df.loc[slot, space] = activity['code']
                    else:
                        df.loc[slot, space] = str(activity)
        
        return df
    
    def create_utilization_heatmap(self, timetable, save_path=None):
        """
        Create a heatmap showing room and slot utilization.
        
        Args:
            timetable: Timetable dictionary
            save_path: Path to save the plot (or None to display)
        """
        df = self.create_timetable_dataframe(timetable)
        
        # Create a binary utilization matrix (1 if occupied, 0 if empty)
        utilization = df.applymap(lambda x: 1 if x else 0)
        
        plt.figure(figsize=(12, 8))
        cmap = colors.ListedColormap(['white', 'green'])
        plt.pcolormesh(utilization.T, cmap=cmap, edgecolors='lightgray', linewidth=0.5)
        
        # Add labels
        plt.yticks(np.arange(0.5, len(utilization.columns)), utilization.columns)
        plt.xticks(np.arange(0.5, len(utilization.index)), utilization.index, rotation=90)
        plt.title('Room Utilization by Time Slot')
        plt.xlabel('Time Slot')
        plt.ylabel('Room')
        
        # Add a colorbar
        cbar = plt.colorbar(ticks=[0.25, 0.75])
        cbar.set_ticklabels(['Vacant', 'Occupied'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def create_group_schedule(self, timetable, group_id, save_path=None):
        """
        Create a visual representation of a specific group's schedule.
        
        Args:
            timetable: Timetable dictionary
            group_id: ID of the group to visualize
            save_path: Path to save the plot (or None to display)
        """
        if group_id not in self.groups_dict:
            raise ValueError(f"Group '{group_id}' not found in the dataset.")
        
        # Find all activities for this group
        group_activities = {}
        for activity_id, activity in self.activities_dict.items():
            if isinstance(activity, dict) and 'groups' in activity and group_id in activity['groups']:
                group_activities[activity_id] = activity
        
        # Find where these activities are scheduled in the timetable
        schedule = {}
        for slot in timetable:
            for space in timetable[slot]:
                activity = timetable[slot][space]
                if not activity:
                    continue
                
                activity_id = None
                if isinstance(activity, dict) and 'id' in activity:
                    activity_id = activity['id']
                elif isinstance(activity, str) and activity in self.activities_dict:
                    activity_id = activity
                
                if activity_id in group_activities:
                    if slot not in schedule:
                        schedule[slot] = []
                    schedule[slot].append({
                        'space': space,
                        'activity': activity_id,
                        'name': group_activities[activity_id].get('name', 'Unknown')
                    })
        
        # Create a visual schedule
        slots = sorted(self.slots)
        
        plt.figure(figsize=(14, 8))
        y_positions = np.arange(len(slots))
        
        for i, slot in enumerate(slots):
            activities = schedule.get(slot, [])
            
            for j, activity in enumerate(activities):
                plt.text(
                    j * 0.3, i, 
                    f"{activity['activity']}\n{activity['space']}", 
                    fontsize=8, 
                    bbox=dict(facecolor='lightblue', alpha=0.5)
                )
        
        plt.yticks(y_positions, slots)
        plt.title(f"Schedule for Group: {group_id} - {self.groups_dict[group_id].get('name', 'Unknown')}")
        plt.xlabel('Activities')
        plt.ylabel('Time Slot')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def create_lecturer_schedule(self, timetable, lecturer_id, save_path=None):
        """
        Create a visual representation of a specific lecturer's schedule.
        
        Args:
            timetable: Timetable dictionary
            lecturer_id: ID of the lecturer to visualize
            save_path: Path to save the plot (or None to display)
        """
        if lecturer_id not in self.lecturers_dict:
            raise ValueError(f"Lecturer '{lecturer_id}' not found in the dataset.")
        
        # Find all activities for this lecturer
        lecturer_activities = {}
        for activity_id, activity in self.activities_dict.items():
            if isinstance(activity, dict) and 'lecturer' in activity and activity['lecturer'] == lecturer_id:
                lecturer_activities[activity_id] = activity
        
        # Find where these activities are scheduled in the timetable
        schedule = {}
        for slot in timetable:
            for space in timetable[slot]:
                activity = timetable[slot][space]
                if not activity:
                    continue
                
                activity_id = None
                if isinstance(activity, dict) and 'id' in activity:
                    activity_id = activity['id']
                elif isinstance(activity, str) and activity in self.activities_dict:
                    activity_id = activity
                
                if activity_id in lecturer_activities:
                    if slot not in schedule:
                        schedule[slot] = []
                    schedule[slot].append({
                        'space': space,
                        'activity': activity_id,
                        'name': lecturer_activities[activity_id].get('name', 'Unknown')
                    })
        
        # Create a visual schedule
        slots = sorted(self.slots)
        
        plt.figure(figsize=(14, 8))
        y_positions = np.arange(len(slots))
        
        for i, slot in enumerate(slots):
            activities = schedule.get(slot, [])
            
            for j, activity in enumerate(activities):
                plt.text(
                    j * 0.3, i, 
                    f"{activity['activity']}\n{activity['space']}", 
                    fontsize=8, 
                    bbox=dict(facecolor='lightgreen', alpha=0.5)
                )
        
        plt.yticks(y_positions, slots)
        plt.title(f"Schedule for Lecturer: {lecturer_id} - {self.lecturers_dict[lecturer_id].get('name', 'Unknown')}")
        plt.xlabel('Activities')
        plt.ylabel('Time Slot')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def create_conflict_heatmap(self, timetable, save_path=None):
        """
        Create a heatmap showing conflicts in the timetable.
        
        Args:
            timetable: Timetable dictionary
            save_path: Path to save the plot (or None to display)
        """
        df = self.create_timetable_dataframe(timetable)
        spaces = list(self.spaces_dict.keys())
        slots = self.slots
        
        # Initialize conflict matrix
        conflict_matrix = np.zeros((len(slots), len(spaces)))
        
        # Check for room size conflicts
        for i, slot in enumerate(slots):
            for j, space in enumerate(spaces):
                if slot in timetable and space in timetable[slot] and timetable[slot][space]:
                    activity = timetable[slot][space]
                    activity_id = None
                    
                    if isinstance(activity, dict) and 'id' in activity:
                        activity_id = activity['id']
                    elif isinstance(activity, str) and activity in self.activities_dict:
                        activity_id = activity
                    
                    if activity_id:
                        # Check room capacity
                        if isinstance(activity, dict) and 'groups' in activity:
                            total_students = 0
                            for group_id in activity['groups']:
                                if group_id in self.groups_dict:
                                    total_students += self.groups_dict[group_id].get('size', 0)
                            
                            room_capacity = self.spaces_dict[space].get('capacity', 0)
                            if total_students > room_capacity:
                                conflict_matrix[i, j] = 1  # Room capacity conflict
        
        # Create a conflict heatmap
        plt.figure(figsize=(12, 8))
        cmap = colors.ListedColormap(['white', 'red'])
        plt.pcolormesh(conflict_matrix.T, cmap=cmap, edgecolors='lightgray', linewidth=0.5)
        
        # Add labels
        plt.yticks(np.arange(0.5, len(spaces)), spaces)
        plt.xticks(np.arange(0.5, len(slots)), slots, rotation=90)
        plt.title('Room Capacity Conflicts')
        plt.xlabel('Time Slot')
        plt.ylabel('Room')
        
        # Add a colorbar
        cbar = plt.colorbar(ticks=[0.25, 0.75])
        cbar.set_ticklabels(['No Conflict', 'Conflict'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def create_timetable_summary(self, timetable, save_path=None):
        """
        Create a visual summary of a timetable.
        
        Args:
            timetable: Timetable dictionary
            save_path: Path to save the plot (or None to display)
        """
        df = self.create_timetable_dataframe(timetable)
        
        # Calculate utilization by day and time
        # Assuming slot format like "Monday_9:00"
        days = sorted(set([slot.split('_')[0] for slot in df.index]))
        times = sorted(set([slot.split('_')[1] for slot in df.index]))
        
        utilization_by_day_time = np.zeros((len(days), len(times)))
        
        for i, day in enumerate(days):
            for j, time in enumerate(times):
                slot = f"{day}_{time}"
                if slot in df.index:
                    utilization = sum(1 for x in df.loc[slot] if x)
                    utilization_by_day_time[i, j] = utilization / len(df.columns)
        
        # Create the heatmap
        plt.figure(figsize=(12, 8))
        plt.pcolormesh(utilization_by_day_time, cmap='Blues', edgecolors='lightgray', linewidth=0.5)
        
        # Add labels
        plt.yticks(np.arange(0.5, len(days)), days)
        plt.xticks(np.arange(0.5, len(times)), times, rotation=90)
        plt.title('Room Utilization by Day and Time')
        plt.xlabel('Time')
        plt.ylabel('Day')
        
        # Add a colorbar
        cbar = plt.colorbar()
        cbar.set_label('Utilization Ratio')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def visualize_timetable(self, timetable_file, output_dir='visualizations'):
        """
        Create a complete set of visualizations for a timetable.
        
        Args:
            timetable_file: Path to the timetable JSON file
            output_dir: Directory to save visualizations
        """
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load timetable
        timetable = self.load_timetable(timetable_file)
        filename = os.path.basename(timetable_file).split('.')[0]
        
        # Generate visualizations
        print(f"Creating visualizations for {timetable_file}...")
        
        # Utilization heatmap
        self.create_utilization_heatmap(timetable, f"{output_dir}/{filename}_utilization.png")
        
        # Conflict heatmap
        self.create_conflict_heatmap(timetable, f"{output_dir}/{filename}_conflicts.png")
        
        # Timetable summary
        self.create_timetable_summary(timetable, f"{output_dir}/{filename}_summary.png")
        
        # Create schedules for some groups and lecturers
        groups = list(self.groups_dict.keys())[:3]  # First 3 groups
        for group_id in groups:
            self.create_group_schedule(timetable, group_id, f"{output_dir}/{filename}_group_{group_id}.png")
        
        lecturers = list(self.lecturers_dict.keys())[:3]  # First 3 lecturers
        for lecturer_id in lecturers:
            self.create_lecturer_schedule(timetable, lecturer_id, f"{output_dir}/{filename}_lecturer_{lecturer_id}.png")
        
        print(f"Visualizations saved to {output_dir}/ directory.")

def visualize_all_solutions(data_file="sliit_computing_dataset.json"):
    """
    Visualize solutions from all algorithms.
    
    Args:
        data_file: Path to the JSON data file
    """
    # Create visualizer
    visualizer = TimetableVisualizer(data_file)
    
    # Visualize solutions
    algorithm_files = [
        "nsga2_solution.json",
        "moead_solution.json",
        "spea2_solution.json"
    ]
    
    for file in algorithm_files:
        if os.path.exists(file):
            visualizer.visualize_timetable(file)
        else:
            print(f"Warning: Solution file {file} not found.")

if __name__ == "__main__":
    import os
    
    # Visualize all solutions if they exist
    visualize_all_solutions()
    
    # Or visualize a specific solution
    # visualizer = TimetableVisualizer("sliit_computing_dataset.json")
    # visualizer.visualize_timetable("nsga2_solution.json")
