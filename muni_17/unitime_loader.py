"""
Data loading utilities for UniTime datasets (muni-fsps-spr17) in both XML and JSON formats.

This module provides functions to load and convert UniTime dataset formats into
objects compatible with the existing timetable scheduling algorithms.
"""
import os
import json
import xml.etree.ElementTree as ET
from utils import Space, Group, Lecturer, Activity

def load_unitime_json(file_path, algorithm_name='ga'):
    """
    Load data from UniTime JSON format (minifspsspr17.json) and convert to objects
    compatible with existing algorithms.
    
    Parameters:
        file_path (str): Path to the UniTime JSON file
        algorithm_name (str): Name of the algorithm to assign to activities (ga, rl, co)
        
    Returns:
        Tuple of (spaces_dict, groups_dict, lecturers_dict, activities_dict, slots)
    """
    print(f"Loading UniTime JSON data from {file_path}...")
    
    # Initialize data structures
    spaces_dict = {}
    groups_dict = {}
    lecturers_dict = {}
    activities_dict = {}
    slots = []  # Will store time slots like MON1, MON2, etc.
    
    try:
        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Process rooms
        rooms = data.get('problem', {}).get('rooms', {}).get('room', [])
        if not isinstance(rooms, list):
            rooms = [rooms]
            
        for room in rooms:
            if not room:  # Skip if room data is None or empty
                continue
                
            room_id = str(room.get('_id', ''))
            if not room_id:  # Skip if no valid room ID
                continue
                
            capacity = int(room.get('_capacity', 0))
            
            # Create space object
            spaces_dict[room_id] = Space(room_id, capacity)
            
            # Store travel times
            travel_times = {}
            travels = room.get('travel', [])
            if not isinstance(travels, list):
                travels = [travels] if travels else []
                
            for travel in travels:
                if isinstance(travel, dict):
                    target_room = str(travel.get('_room', ''))
                    if target_room:
                        value = int(travel.get('_value', 0))
                        travel_times[target_room] = value
            
            # Add travel times as attribute
            spaces_dict[room_id].travel_times = travel_times
            
            # Store unavailability periods
            unavailable = []
            unavails = room.get('unavailable', [])
            if not isinstance(unavails, list):
                unavails = [unavails] if unavails else []
                
            for unavail in unavails:
                if isinstance(unavail, dict):
                    days_pattern = unavail.get('_days', '')
                    start_slot = int(unavail.get('_start', 0))
                    length = int(unavail.get('_length', 0))
                    weeks_pattern = unavail.get('_weeks', '')
                    unavailable.append({
                        'days': days_pattern,
                        'start': start_slot,
                        'length': length,
                        'weeks': weeks_pattern
                    })
            spaces_dict[room_id].unavailable = unavailable
        
        # Process courses and their configurations
        courses = data.get('problem', {}).get('courses', {}).get('course', [])
        if not isinstance(courses, list):
            courses = [courses] if courses else []
            
        activity_id = 1  # Initialize activity counter
        
        for course in courses:
            if not course:  # Skip if course data is None or empty
                continue
                
            course_id = str(course.get('_id', ''))
            if not course_id:  # Skip if no valid course ID
                continue
                
            configs = course.get('config', [])
            if not isinstance(configs, list):
                configs = [configs] if configs else []
            
            for config in configs:
                if not config:  # Skip if config is None or empty
                    continue
                    
                subparts = config.get('subpart', [])
                if not isinstance(subparts, list):
                    subparts = [subparts] if subparts else []
                
                for subpart in subparts:
                    if not subpart:  # Skip if subpart is None or empty
                        continue
                        
                    classes = subpart.get('class', [])
                    if not isinstance(classes, list):
                        classes = [classes] if classes else []
                    
                    for class_ in classes:
                        if not class_:  # Skip if class data is None or empty
                            continue
                            
                        class_id = str(class_.get('_id', ''))
                        if not class_id:  # Skip if no valid class ID
                            continue
                            
                        limit = int(class_.get('_limit', 0))
                        
                        # Create a group for this class
                        group_id = f"group_{class_id}"
                        groups_dict[group_id] = Group(group_id, limit)
                        
                        # Create a placeholder lecturer
                        lecturer_id = f"lecturer_{class_id}"
                        lecturers_dict[lecturer_id] = Lecturer(lecturer_id, f"Lecturer {class_id}", "Department")
                        
                        # Get room preferences
                        room_prefs = []
                        rooms = class_.get('room', [])
                        if not isinstance(rooms, list):
                            rooms = [rooms] if rooms else []
                        
                        for room in rooms:
                            if isinstance(room, dict):
                                room_id = str(room.get('_id', ''))
                                if room_id:
                                    penalty = int(room.get('_penalty', 0))
                                    room_prefs.append({
                                        'id': room_id,
                                        'penalty': penalty
                                    })
                        
                        # Get time preferences
                        time_prefs = []
                        times = class_.get('time', [])
                        if not isinstance(times, list):
                            times = [times] if times else []
                        
                        # Default length for activities
                        default_length = 1
                        
                        for time in times:
                            if isinstance(time, dict):
                                days = time.get('_days', '')
                                start = int(time.get('_start', 0))
                                length = int(time.get('_length', 0))
                                weeks = time.get('_weeks', '')
                                time_prefs.append({
                                    'days': days,
                                    'start': start,
                                    'length': length,
                                    'weeks': weeks
                                })
                                # Update default_length from the first valid time preference
                                default_length = length
                        
                        # Create an activity for this class
                        activity = Activity(
                            str(activity_id),
                            f"Course {course_id} Class {class_id}",
                            lecturer_id,
                            [group_id],
                            default_length  # Use the length from time preferences
                        )
                        
                        # Add additional attributes
                        activity.algorithm = algorithm_name
                        activity.room_preferences = room_prefs
                        activity.time_preferences = time_prefs
                        
                        activities_dict[str(activity_id)] = activity
                        activity_id += 1
        
        # Generate time slots (e.g., MON1, MON2, etc.)
        days = ['MON', 'TUE', 'WED', 'THU', 'FRI']
        periods_per_day = 12  # Adjust based on your needs
        for day in days:
            for period in range(1, periods_per_day + 1):
                slots.append(f"{day}{period}")
        
        print(f"Loaded {len(spaces_dict)} rooms, {len(groups_dict)} groups, "
              f"{len(lecturers_dict)} lecturers, {len(activities_dict)} activities")
        
        return spaces_dict, groups_dict, lecturers_dict, activities_dict, slots
        
    except Exception as e:
        print(f"Error loading UniTime JSON data: {str(e)}")
        raise

def load_unitime_xml(file_path, algorithm_name='ga'):
    """
    Load data from UniTime XML format (muni-fsps-spr17.xml) and convert to objects
    compatible with existing algorithms.
    
    Parameters:
        file_path (str): Path to the UniTime XML file
        algorithm_name (str): Name of the algorithm to assign to activities (ga, rl, co)
        
    Returns:
        Tuple of (spaces_dict, groups_dict, lecturers_dict, activities_dict, slots)
    """
    print(f"Loading UniTime XML data from {file_path}...")
    
    # Parse XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Extract optimization weights for later use in algorithms
    optimization = root.find('optimization')
    weights = {
        'time': int(optimization.get('time', 25)),
        'room': int(optimization.get('room', 1)),
        'distribution': int(optimization.get('distribution', 15)),
        'student': int(optimization.get('student', 100))
    }
    
    # Get problem parameters
    nr_days = int(root.get('nrDays', 7))
    slots_per_day = int(root.get('slotsPerDay', 288))
    nr_weeks = int(root.get('nrWeeks', 19))
    
    print(f"Problem settings: {nr_days} days, {slots_per_day} slots per day, {nr_weeks} weeks")
    print(f"Optimization weights: {weights}")
    
    # =============== STEP 1: Parse rooms ===============
    spaces_dict = {}
    rooms_xml = root.find('rooms').findall('room')
    
    for room_xml in rooms_xml:
        room_id = room_xml.get('id')
        capacity = int(room_xml.get('capacity', 0))
        
        # Create space object
        spaces_dict[room_id] = Space(room_id, capacity)
        
        # Store travel times for later use (can be added as an attribute)
        travel_times = {}
        for travel in room_xml.findall('travel'):
            target_room = travel.get('room')
            value = int(travel.get('value', 0))
            travel_times[target_room] = value
        
        # Add travel times as attribute
        spaces_dict[room_id].travel_times = travel_times
        
        # Store unavailability periods for later use
        unavailable = []
        for unavail in room_xml.findall('unavailable'):
            days_pattern = unavail.get('days')
            start_slot = int(unavail.get('start', 0))
            length = int(unavail.get('length', 0))
            weeks_pattern = unavail.get('weeks')
            unavailable.append({
                'days': days_pattern,
                'start': start_slot,
                'length': length,
                'weeks': weeks_pattern
            })
        
        # Add unavailability as attribute
        spaces_dict[room_id].unavailable = unavailable
    
    print(f"Loaded {len(spaces_dict)} rooms/spaces")
    
    # =============== STEP 2: Generate student groups ===============
    # Since UniTime doesn't have explicit student groups like our system,
    # we'll create them based on classes and their limits
    
    groups_dict = {}
    courses_xml = root.find('courses').findall('course')
    
    # First pass: identify unique classes and create placeholder groups
    for course_xml in courses_xml:
        course_id = course_xml.get('id')
        
        for config_xml in course_xml.findall('config'):
            for subpart_xml in config_xml.findall('subpart'):
                # Handle both single class and multiple classes
                classes_xml = subpart_xml.findall('class')
                if not classes_xml:
                    classes_xml = [subpart_xml.find('class')]
                
                for class_xml in classes_xml:
                    if class_xml is None:
                        continue
                    
                    class_id = class_xml.get('id')
                    limit = int(class_xml.get('limit', 0))
                    
                    # Create a group ID based on course and class
                    group_id = f"C{course_id}_{class_id}"
                    groups_dict[group_id] = Group(group_id, limit)
    
    print(f"Created {len(groups_dict)} student groups")
    
    # =============== STEP 3: Create placeholder lecturers ===============
    # Since UniTime doesn't have explicit lecturers, we'll create placeholders
    lecturers_dict = {}
    for i in range(1, 21):  # Create 20 placeholder lecturers
        lecturer_id = f"L{i}"
        lecturers_dict[lecturer_id] = Lecturer(lecturer_id, f"Lecturer {i}", "Department")
    
    print(f"Created {len(lecturers_dict)} placeholder lecturers")
    
    # =============== STEP 4: Create activities from classes ===============
    activities_dict = {}
    activity_counter = 1
    
    for course_xml in courses_xml:
        course_id = course_xml.get('id')
        
        for config_xml in course_xml.findall('config'):
            for subpart_xml in config_xml.findall('subpart'):
                subpart_id = subpart_xml.get('id')
                
                # Handle both single class and multiple classes
                classes_xml = subpart_xml.findall('class')
                if not classes_xml:
                    classes_xml = [subpart_xml.find('class')]
                
                for class_xml in classes_xml:
                    if class_xml is None:
                        continue
                    
                    class_id = class_xml.get('id')
                    limit = int(class_xml.get('limit', 0))
                    
                    # Generate activity ID
                    activity_id = f"AC-{activity_counter:03d}"
                    activity_counter += 1
                    
                    # Create group ID for this class
                    group_id = f"C{course_id}_{class_id}"
                    
                    # Determine activity duration based on the first time slot's length
                    # (converting from UniTime's minute-based units to our time slot units)
                    time_xml = class_xml.find('time')
                    duration = 1  # Default
                    if time_xml is not None:
                        length_minutes = int(time_xml.get('length', 60))
                        duration = max(1, length_minutes // 30)  # Assuming 30 min per slot in our system
                    
                    # Assign random lecturer (in a real system, this would be based on actual assignment)
                    lecturer_id = f"L{(int(class_id) % 20) + 1}"
                    
                    # Create activity
                    activity = Activity(
                        activity_id, 
                        f"C{course_id}",  # Use course ID as subject 
                        lecturer_id,
                        [group_id],  # List of group IDs
                        duration
                    )
                    
                    # Important: Set the algorithm field as mentioned in the memory
                    activity.algorithm = algorithm_name
                    
                    # Add constraints as attributes
                    room_preferences = []
                    for room_xml in class_xml.findall('room'):
                        room_id = room_xml.get('id')
                        penalty = int(room_xml.get('penalty', 0))
                        room_preferences.append({'id': room_id, 'penalty': penalty})
                    
                    time_preferences = []
                    for time_xml in class_xml.findall('time'):
                        days = time_xml.get('days')
                        start = int(time_xml.get('start', 0))
                        length = int(time_xml.get('length', 0))
                        weeks = time_xml.get('weeks')
                        penalty = int(time_xml.get('penalty', 0))
                        time_preferences.append({
                            'days': days,
                            'start': start,
                            'length': length,
                            'weeks': weeks,
                            'penalty': penalty
                        })
                    
                    # Store these constraints as attributes
                    activity.room_preferences = room_preferences
                    activity.time_preferences = time_preferences
                    
                    activities_dict[activity_id] = activity
    
    print(f"Created {len(activities_dict)} activities")
    
    # =============== STEP 5: Create time slots ===============
    # Converting from UniTime's fine-grained slots to our system
    days = ["MON", "TUE", "WED", "THU", "FRI"]
    slots = []
    for day in days:
        for i in range(1, 9):  # 8 time slots per day as in your existing system
            slots.append(f"{day}{i}")
    
    print(f"Created {len(slots)} time slots")
    
    return spaces_dict, groups_dict, lecturers_dict, activities_dict, slots
