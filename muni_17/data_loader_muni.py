"""
Data loading utilities for the MUni timetabling problem.

This module contains functions for loading and processing the munifspsspr17 dataset.
"""
import json

def _ensure_list(item):
    """Convert single items to a list if they aren't already."""
    if not isinstance(item, list):
        return [item]
    return item

def _process_optimization_weights(problem_data):
    """Extract optimization weights from problem data."""
    optimization = problem_data.get('optimization', {})
    return {
        'TIME': int(optimization.get('_time', 25)),
        'ROOM': int(optimization.get('_room', 1)),
        'DISTRIBUTION': int(optimization.get('_distribution', 15)),
        'STUDENT': int(optimization.get('_student', 100))
    }

def _process_rooms(rooms):
    """Process room data and travel times."""
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
            travels = _ensure_list(room['travel'])
            for travel in travels:
                dest_room = travel.get('_room', '')
                value = int(travel.get('_value', 0))
                room_data[room_id]['travel'][dest_room] = value
    
    return room_data

def _process_class_rooms(cls):
    """Process room assignments for a class."""
    rooms_for_class = []
    class_rooms = _ensure_list(cls.get('room', []))
    
    for room in class_rooms:
        if not room:  # Skip if room is None or empty
            continue
        room_id = room.get('_id', '')
        penalty = int(room.get('_penalty', 0))
        rooms_for_class.append({
            'room_id': room_id,
            'penalty': penalty
        })
    
    return rooms_for_class

def _process_class_times(cls, time_patterns):
    """Process time patterns for a class."""
    times_for_class = []
    class_times = _ensure_list(cls.get('time', []))
    
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
    
    return times_for_class

def _process_courses_and_classes(courses):
    """Process course and class data."""
    class_data = {}
    course_data = {}
    time_patterns = {}
    course_to_classes = {}
    class_count = 0
    
    for course in courses:
        course_id = course.get('_id', '')
        course_data[course_id] = {'configs': []}
        course_to_classes[course_id] = []
        
        configs = _ensure_list(course.get('config', []))
        for config in configs:
            config_id = config.get('_id', '')
            course_data[course_id]['configs'].append(config_id)
            
            subparts = _ensure_list(config.get('subpart', []))
            for subpart in subparts:
                subpart_id = subpart.get('_id', '')
                classes = _ensure_list(subpart.get('class', []))
                
                for cls in classes:
                    class_id = cls.get('_id', '')
                    class_limit = int(cls.get('_limit', 0))
                    class_count += 1
                    
                    # Add to course-class relationship
                    course_to_classes[course_id].append(class_id)
                    
                    # Process rooms and times for this class
                    rooms_for_class = _process_class_rooms(cls)
                    times_for_class = _process_class_times(cls, time_patterns)
                    
                    # Store class data
                    class_data[class_id] = {
                        'course_id': course_id,
                        'config_id': config_id,
                        'subpart_id': subpart_id,
                        'limit': class_limit,
                        'rooms': rooms_for_class,
                        'times': times_for_class
                    }
    
    return class_data, course_data, time_patterns, course_to_classes, class_count

def _process_distribution_constraints(problem_data):
    """Process distribution constraints."""
    distribution_constraints = []
    if 'distributions' not in problem_data:
        return distribution_constraints
    
    distributions = _ensure_list(problem_data['distributions'].get('distribution', []))
    for dist in distributions:
        constraint = {
            'id': dist.get('_id', ''),
            'type': dist.get('_type', ''),
            'required': dist.get('_required', 'true') == 'true',
            'penalty': int(dist.get('_penalty', 0)),
            'classes': []
        }
        
        # Extract classes involved in this constraint
        classes = _ensure_list(dist.get('class', []))
        for cls in classes:
            if cls and '_id' in cls:
                constraint['classes'].append(cls.get('_id', ''))
        
        distribution_constraints.append(constraint)
    
    return distribution_constraints

def _process_students(problem_data):
    """Process student enrollments."""
    students = {}
    if 'students' not in problem_data:
        return students
    
    student_data = _ensure_list(problem_data['students'].get('student', []))
    for student in student_data:
        student_id = student.get('_id', '')
        students[student_id] = {'courses': []}
        
        # Get courses this student is enrolled in
        courses = _ensure_list(student.get('course', []))
        for course in courses:
            if course and '_id' in course:
                students[student_id]['courses'].append(course.get('_id', ''))
    
    return students

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
    problem_data = data['problem']
    rooms = problem_data['rooms']['room']
    courses = problem_data['courses']['course']
    
    # Process data using helper functions
    weights = _process_optimization_weights(problem_data)
    room_data = _process_rooms(rooms)
    class_data, course_data, time_patterns, course_to_classes, class_count = _process_courses_and_classes(courses)
    distribution_constraints = _process_distribution_constraints(problem_data)
    students = _process_students(problem_data)
    
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