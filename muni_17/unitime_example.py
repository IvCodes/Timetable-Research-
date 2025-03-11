"""
Example usage of the UniTime dataset loader with existing algorithms.

This script demonstrates how to load the minifspsspr17.json dataset
and use it with the existing timetable scheduling algorithms.
"""
import sys
import os
import json
from unitime_loader import load_unitime_json, load_unitime_xml
from utils import evaluate_hard_constraints, evaluate_soft_constraints, evaluate

# Path to the UniTime dataset
UNITIME_JSON_PATH = '../Advance-Timetable-Scheduling-Backend/Data/minifspsspr17.json'
UNITIME_XML_PATH = '../Advance-Timetable-Scheduling-Backend/Data/muni-fsps-spr17.xml'

def main():
    """
    Main function to demonstrate the usage of UniTime dataset loader.
    """
    # Choose which format to load (JSON or XML)
    use_json = True
    algorithm_name = 'ga'  # Change to 'rl' or 'co' for other algorithms
    
    # Load the dataset
    if use_json:
        print(f"Loading UniTime JSON dataset from {UNITIME_JSON_PATH}...")
        spaces_dict, groups_dict, lecturers_dict, activities_dict, slots = load_unitime_json(
            UNITIME_JSON_PATH, algorithm_name
        )
    else:
        print(f"Loading UniTime XML dataset from {UNITIME_XML_PATH}...")
        spaces_dict, groups_dict, lecturers_dict, activities_dict, slots = load_unitime_xml(
            UNITIME_XML_PATH, algorithm_name
        )
    
    # Print summary of loaded data
    print("\n===== LOADED DATA SUMMARY =====")
    print(f"Spaces: {len(spaces_dict)}")
    print(f"Groups: {len(groups_dict)}")
    print(f"Lecturers: {len(lecturers_dict)}")
    print(f"Activities: {len(activities_dict)}")
    print(f"Time Slots: {len(slots)}")
    
    # Example: Print first 5 activities with their attributes
    print("\n===== SAMPLE ACTIVITIES =====")
    for i, (activity_id, activity) in enumerate(activities_dict.items()):
        if i >= 5:
            break
        print(f"Activity ID: {activity_id}")
        print(f"  Subject: {activity.subject}")
        print(f"  Teacher: {activity.teacher_id}")
        print(f"  Groups: {activity.group_ids}")
        print(f"  Duration: {activity.duration}")
        print(f"  Algorithm: {getattr(activity, 'algorithm', 'not set')}")
        print(f"  Room Preferences: {getattr(activity, 'room_preferences', [])[:2]}...")
        print(f"  Time Preferences: {getattr(activity, 'time_preferences', [])[:2]}...")
        print()
    
    # Example: Print some room information with travel times
    print("\n===== SAMPLE ROOMS =====")
    for i, (space_id, space) in enumerate(spaces_dict.items()):
        if i >= 3:
            break
        print(f"Room ID: {space_id}")
        print(f"  Capacity: {space.size}")
        print(f"  Sample Travel Times: {list(getattr(space, 'travel_times', {}).items())[:3]}...")
        print()
    
    # At this point, you would typically call your scheduling algorithm functions
    # For example:
    # schedule = run_genetic_algorithm(spaces_dict, groups_dict, lecturers_dict, activities_dict, slots)
    
    print("\nData loaded successfully. Ready to use with scheduling algorithms.")
    print("To use this data with your algorithms, modify the algorithm entry points to accept")
    print("the output of the unitime_loader functions instead of the standard load_data function.")

if __name__ == "__main__":
    main()
