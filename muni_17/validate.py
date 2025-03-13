def get_valid_rooms(data, class_id):
    """
    Get valid rooms for a class.
    
    Args:
        data: Dictionary with problem data
        class_id: ID of the class
    
    Returns:
        List of valid room IDs
    """
    class_info = data['classes'].get(class_id, {})
    valid_rooms = []
    
    for room in class_info.get('rooms', []):
        room_id = room.get('room_id')
        if room_id and room_id in data['rooms']:
            valid_rooms.append(room_id)
    
    return valid_rooms

def get_valid_times(data, class_id, room_id, solution):
    """
    Get valid times for a class in a specific room.
    
    Args:
        data: Dictionary with problem data
        class_id: ID of the class
        room_id: ID of the room
        solution: Current solution (to check for conflicts)
    
    Returns:
        List of valid time IDs
    """
    class_info = data['classes'].get(class_id, {})
    valid_times = []
    
    for time in class_info.get('times', []):
        time_id = time.get('time_id')
        if not time_id:
            continue
            
        # Check for conflicts with other classes in the same room
        conflict = False
        for other_class, assignment in solution.assignments.items():
            if other_class != class_id and assignment and len(assignment) >= 2:
                if assignment[0] == room_id and assignment[1] == time_id:
                    conflict = True
                    break
        
        if not conflict:
            valid_times.append(time_id)
    
    return valid_times