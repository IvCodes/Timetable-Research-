import os
import json
from collections import defaultdict
import argparse

def load_nott_dataset(base_dir):
    """
    Load and parse the Nottingham 96 dataset.
    
    Args:
        base_dir: Directory containing the Nott dataset files
        
    Returns:
        Dictionary containing the parsed dataset
    """
    # Check if directory exists
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Dataset directory '{base_dir}' not found. Please check the path.")
    
    # Initialize data structures
    exams = []
    rooms = []
    timeslots = []
    enrollments = defaultdict(list)  # exam_id -> list of student_ids
    student_exams = defaultdict(list)  # student_id -> list of exam_ids
    coincidences = []
    room_assignments = []
    earliness_priority = []
    combined_rooms = []  # Track which rooms can be combined
    
    # Load exams
    exam_path = os.path.join(base_dir, 'exams')
    if not os.path.exists(exam_path):
        raise FileNotFoundError(f"Exam file not found at '{exam_path}'. Please check if the dataset structure is correct.")
    
    with open(exam_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) < 2:
                continue
                
            exam_code = parts[0]
            rest = parts[1]
            
            # Extract duration and department
            dept = rest[-2:]
            rest = rest[:-2].strip()
            
            # Parse duration like "3:00" to minutes
            duration_str = rest[-4:].strip()
            rest = rest[:-4].strip()
            
            try:
                if ':' in duration_str:
                    hours, mins = map(int, duration_str.split(':'))
                    duration = hours * 60 + mins
                else:
                    duration = int(duration_str) * 60
            except ValueError:
                duration = 120  # Default to 2 hours if parsing fails
            
            description = rest
            
            exams.append({
                'code': exam_code,
                'description': description,
                'duration': duration,
                'department': dept
            })
    
    # Load data file for rooms and other constraints
    data_path = os.path.join(base_dir, 'data')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at '{data_path}'. Please check if the dataset structure is correct.")
    
    rooms_loaded = False
    coincidences_loaded = False
    room_assignments_loaded = False
    earliness_loaded = False
    
    with open(data_path, 'r') as f:
        current_section = None
        # Track rooms that can be combined
        current_combined_group = []
        
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if line.upper() == "ROOMS":
                current_section = "rooms"
                rooms_loaded = True
                continue
            elif line.upper() == "COINCIDENCES":
                current_section = "coincidences"
                coincidences_loaded = True
                continue
            elif line.upper() == "ROOM ASSIGNMENTS":
                current_section = "room_assignments"
                room_assignments_loaded = True
                continue
            elif "EARLINESS PRIORITY" in line.upper():
                current_section = "earliness"
                earliness_loaded = True
                continue
            elif "DATES" in line.upper() or "TIMES" in line.upper():
                current_section = None
                continue
                
            # Parse based on current section
            if current_section == "rooms" and rooms_loaded:
                # If this is a room line
                if not line.startswith('#') and len(line.strip()) > 0:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        room_id = parts[0]
                        capacity = int(parts[1])
                        
                        # Check if there are additional markers for combined rooms
                        combined_marker = None
                        if len(parts) >= 3 and parts[2] in ['\\', '/']:
                            combined_marker = parts[2]
                            
                        # Track rooms that can be combined
                        if combined_marker == '\\':  # Start of a new combined group
                            if current_combined_group:  # Save any previous group
                                if len(current_combined_group) > 1:
                                    combined_rooms.append(current_combined_group.copy())
                            current_combined_group = [room_id]
                        elif combined_marker == '/':  # End of a combined group
                            current_combined_group.append(room_id)
                            if len(current_combined_group) > 1:
                                combined_rooms.append(current_combined_group.copy())
                            current_combined_group = []
                        elif current_combined_group:  # Continue an existing group
                            current_combined_group.append(room_id)
                        
                        rooms.append({
                            'id': room_id,
                            'capacity': capacity
                        })
            
            elif current_section == "coincidences" and coincidences_loaded:
                exams = [x.strip() for x in line.split() if x.strip()]
                if exams:
                    coincidences.append(exams)
            
            elif current_section == "room_assignments" and room_assignments_loaded:
                parts = line.split()
                if len(parts) >= 2:
                    exam_code = parts[0]
                    room_id = ' '.join(parts[1:])
                    room_assignments.append({
                        'exam_code': exam_code,
                        'room_id': room_id
                    })
            
            elif current_section == "earliness" and earliness_loaded:
                parts = line.split()
                if len(parts) >= 2:
                    exam_code = parts[0]
                    try:
                        priority = int(parts[1])
                        earliness_priority.append({
                            'exam_code': exam_code,
                            'priority': priority
                        })
                    except ValueError:
                        pass
    
    # If there's a remaining combined group, add it
    if current_combined_group and len(current_combined_group) > 1:
        combined_rooms.append(current_combined_group)
    
    # Create timeslots based on the information from README
    # Mon - Fri 9:00, 13:30, 16:30, Sat 9:00
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    times = ["9:00", "13:30", "16:30"]
    
    for day_idx, day in enumerate(days):
        for time_idx, time in enumerate(times):
            # Saturday only has 9:00 slot
            if day == "Sat" and time != "9:00":
                continue
                
            timeslots.append({
                'id': f"{day}-{time}",
                'day': day,
                'time': time,
                'index': day_idx * 3 + time_idx
            })
    
    # Load enrollments
    enrollment_path = os.path.join(base_dir, 'enrolements')
    if not os.path.exists(enrollment_path):
        raise FileNotFoundError(f"Enrollment file not found at '{enrollment_path}'. Please check if the dataset structure is correct.")
    
    with open(enrollment_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                student_id, exam_code = parts
                enrollments[exam_code].append(student_id)
                student_exams[student_id].append(exam_code)
    
    return {
        'exams': exams,
        'rooms': rooms,
        'timeslots': timeslots,
        'enrollments': enrollments,
        'student_exams': student_exams,
        'coincidences': coincidences,
        'room_assignments': room_assignments,
        'earliness_priority': earliness_priority,
        'combined_rooms': combined_rooms  # Add the combined rooms information
    }

def convert_nott_dataset_to_json(base_dir, output_file='nott_dataset.json'):
    """
    Convert the Nottingham 96 dataset to JSON format and save it to a file.
    
    Args:
        base_dir: Directory containing the Nott dataset files
        output_file: Path to save the JSON output
    """
    print(f"Loading data from {base_dir}...")
    try:
        # Load data using the load function
        data = load_nott_dataset(base_dir)
        
        print(f"Dataset loaded: {len(data['exams'])} exams, {len(data['rooms'])} rooms, "
              f"{len(data['timeslots'])} timeslots, {len(data['coincidences'])} coincidence groups")
        
        # Convert defaultdicts to regular dicts for JSON serialization
        json_data = {
            'exams': data['exams'],
            'rooms': data['rooms'],
            'timeslots': data['timeslots'],
            'enrollments': {k: v for k, v in data['enrollments'].items()},
            'student_exams': {k: v for k, v in data['student_exams'].items()},
            'coincidences': data['coincidences'],
            'room_assignments': data['room_assignments'],
            'earliness_priority': data['earliness_priority'],
            'combined_rooms': data['combined_rooms']  # Include combined rooms in JSON
        }
        
        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Dataset converted to JSON and saved to {output_file}")
        
        return json_data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure the directory '{base_dir}' contains the Nottingham dataset files (exams, enrolements, data)")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Nottingham 96 dataset to JSON format')
    parser.add_argument('--data_dir', type=str, default='Nott', 
                        help='Directory containing the Nottingham dataset files')
    parser.add_argument('--output', type=str, default='nott_dataset.json',
                        help='Path to save the JSON dataset')
    
    args = parser.parse_args()
    
    # Convert dataset to JSON
    convert_nott_dataset_to_json(args.data_dir, args.output)