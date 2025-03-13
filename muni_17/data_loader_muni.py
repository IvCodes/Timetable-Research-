"""
Data loading utilities for the MUni timetabling problem.

This module contains functions for loading and processing the munifspsspr17 dataset.
"""
import json

def load_muni_data(file_path):
    """
    Load data from the munifspsspr17.json file.
    
    Returns:
        Dictionary with rooms, courses, classes, time patterns, etc.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print("Parsing munifspsspr17.json dataset...")
    
    # Extract key components
    rooms = data['problem']['rooms']['room']
    courses = data['problem']['courses']['course']
    
    # Process optimization weights
    optimization = data['problem'].get('optimization', {})
    weights = {
        'TIME': int(optimization.get('_time', 25)),
        'ROOM': int(optimization.get('_room', 1)),
        'DISTRIBUTION': int(optimization.get('_distribution', 15)),
        'STUDENT': int(optimization.get('_student', 100))
    }
    
    # Process rooms
    room_data = {}
    for room in rooms:
        room_id = room.get('_id', '')
        room_capacity = int(room.get('_capacity', 0))
        room_data[room_id] = {
            'capacity': room_capacity,
            'travel': {},
            'features': []
        }
        
        # Add travel times
        if 'travel' in room:
            travels = room['travel']
            if not isinstance(travels, list):
                travels = [travels]
            for travel in travels:
                dest_room = travel.get('_room', '')
                value = int(travel.get('_value', 0))
                room_data[room_id]['travel'][dest_room] = value
    
    # Process courses and classes
    class_data = {}
    course_data = {}
    time_patterns = {}
    class_count = 0
    
    # Track relationships between courses and classes
    course_to_classes = {}
    
    # Process all courses and their classes
    for course in courses:
        course_id = course.get('_id', '')
        course_data[course_id] = {
            'configs': []
        }
        
        course_to_classes[course_id] = []
        
        configs = course.get('config', [])
        if not isinstance(configs, list):
            configs = [configs]
        
        for config in configs:
            config_id = config.get('_id', '')
            course_data[course_id]['configs'].append(config_id)
            
            subparts = config.get('subpart', [])
            if not isinstance(subparts, list):
                subparts = [subparts]
            
            for subpart in subparts:
                subpart_id = subpart.get('_id', '')
                
                classes = subpart.get('class', [])
                if not isinstance(classes, list):
                    classes = [classes]
                
                for cls in classes:
                    class_id = cls.get('_id', '')
                    class_limit = int(cls.get('_limit', 0))
                    class_count += 1
                    
                    # Add to course-class relationship
                    course_to_classes[course_id].append(class_id)
                    
                    # Process rooms for this class
                    rooms_for_class = []
                    class_rooms = cls.get('room', [])
                    if not isinstance(class_rooms, list):
                        class_rooms = [class_rooms]
                    
                    for room in class_rooms:
                        if not room:  # Skip if room is None or empty
                            continue
                        room_id = room.get('_id', '')
                        penalty = int(room.get('_penalty', 0))
                        rooms_for_class.append({
                            'room_id': room_id,
                            'penalty': penalty
                        })
                    
                    # Process time patterns for this class
                    times_for_class = []
                    class_times = cls.get('time', [])
                    if not isinstance(class_times, list):
                        class_times = [class_times]
                    
                    for time in class_times:
                        if not time:  # Skip if time is None or empty
                            continue
                        days = time.get('_days', '')
                        start = int(time.get('_start', 0))
                        length = int(time.get('_length', 0))
                        weeks = time.get('_weeks', '')
                        penalty = int(time.get('_penalty', 0)) if '_penalty' in time else 0
                        
                        time_id = f"{days}_{start}_{length}_{weeks}"
                        time_patterns[time_id] = {
                            'days': days,
                            'start': start,
                            'length': length,
                            'weeks': weeks
                        }
                        
                        times_for_class.append({
                            'time_id': time_id,
                            'penalty': penalty
                        })
                    
                    # Store class data
                    class_data[class_id] = {
                        'course_id': course_id,
                        'config_id': config_id,
                        'subpart_id': subpart_id,
                        'limit': class_limit,
                        'rooms': rooms_for_class,
                        'times': times_for_class
                    }
    
    # Track distribution constraints
    distribution_constraints = []
    if 'distributions' in data['problem']:
        distributions = data['problem']['distributions'].get('distribution', [])
        if not isinstance(distributions, list):
            distributions = [distributions]
        
        for dist in distributions:
            constraint = {
                'id': dist.get('_id', ''),
                'type': dist.get('_type', ''),
                'required': dist.get('_required', 'true') == 'true',
                'penalty': int(dist.get('_penalty', 0)),
                'classes': []
            }
            
            # Extract classes involved in this constraint
            classes = dist.get('class', [])
            if not isinstance(classes, list):
                classes = [classes]
                
            for cls in classes:
                if cls and '_id' in cls:
                    constraint['classes'].append(cls.get('_id', ''))
            
            distribution_constraints.append(constraint)
    
    # Process students and enrollments
    students = {}
    if 'students' in data['problem']:
        student_data = data['problem']['students'].get('student', [])
        if not isinstance(student_data, list):
            student_data = [student_data]
            
        for student in student_data:
            student_id = student.get('_id', '')
            students[student_id] = {
                'courses': []
            }
            
            # Get courses this student is enrolled in
            courses = student.get('course', [])
            if not isinstance(courses, list):
                courses = [courses]
                
            for course in courses:
                if course and '_id' in course:
                    students[student_id]['courses'].append(course.get('_id', ''))
    
    # Print summary of loaded data
    print(f"Loaded {len(course_data)} courses, {len(class_data)} classes, {len(room_data)} rooms, {len(students)} students, {len(distribution_constraints)} distributions")
    
    return {
        'rooms': room_data,
        'courses': course_data,
        'classes': class_data,
        'time_patterns': time_patterns,
        'course_to_classes': course_to_classes,
        'distribution_constraints': distribution_constraints,
        'students': students,
        'weights': weights
    }