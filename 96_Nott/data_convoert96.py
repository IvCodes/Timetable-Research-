import os
import json
import argparse
from collections import defaultdict
import time

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
    
    print(f"Loading exams from {exam_path}...")
    with open(exam_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Split by whitespace, but limit to 4 parts to keep description intact
            parts = line.split(None, 4)
            if len(parts) < 3:  # Need at least exam code, duration, department
                continue
                
            exam_code = parts[0]
            
            # We know from the dataset documentation the second field is duration
            # Just set a default duration of 180 minutes (3 hours) for now
            duration = 180
            
            # If there's a department code, extract it (usually third or fourth field)
            department = parts[2] if len(parts) > 2 else ""
            
            # Get the description (rest of the line)
            description = parts[-1] if len(parts) > 3 else ""
            
            exams.append({
                'id': exam_code,
                'description': description,
                'duration': duration,
                'department': department
            })
    
    print(f"Loaded {len(exams)} exams")
    
    # Load data file for rooms and other constraints
    data_path = os.path.join(base_dir, 'data')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at '{data_path}'. Please check if the dataset structure is correct.")
    
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        current_section = None
        # Track rooms that can be combined
        current_combined_group = []
        
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Check for section headers - they're uppercase followed by dashes in the data file
            if line.isupper() and all(c == '-' for c in line if not c.isupper() and not c.isspace()):
                current_section = None
                continue
                
            # After a header line, the next line contains dashes, so skip it
            if all(c == '-' for c in line if c.strip()):
                continue
                
            # Check for specific section headers in the content
            if "ROOMS" in line and line.strip() == "ROOMS":
                current_section = "rooms"
                continue
            elif "ROOM ASSIGNMENTS" in line:
                current_section = "room"
                continue
            elif "COINCIDENCES" in line:
                current_section = "coincidence"
                continue
            elif "EARLINESS PRIORITY" in line:
                current_section = "earliness"
                continue
            elif "TIMES" in line and line.strip() == "TIMES":
                current_section = "times"
                continue
            elif "DATES" in line and line.strip() == "DATES":
                current_section = "dates"
                continue
                
            # Process data based on current section
            if current_section == "rooms":
                parts = line.split()
                if len(parts) >= 2:
                    room_id = parts[0]
                    try:
                        capacity = int(parts[1])
                    except ValueError:
                        print(f"Warning: Invalid capacity for room {room_id}: {parts[1]}")
                        continue
                    
                    # Check for room combination markers
                    combined_marker = None
                    for part in parts[2:]:
                        if part in ['\\', '/']:
                            combined_marker = part
                            break
                    
                    # Track rooms that can be combined
                    if combined_marker == '\\':  # Start of a new combined group
                        if current_combined_group and len(current_combined_group) > 1:
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
            
            elif current_section == "times":
                # Parse the timeslots from the times section
                # Format: Mon - Fri  9:00 (3hrs), 13:30 (2hrs), 16:30 (2hrs)
                if "Mon - Fri" in line:
                    time_parts = line.replace("Mon - Fri", "").strip().split(",")
                    for i, part in enumerate(time_parts):
                        part = part.strip()
                        if "(" in part and ")" in part:
                            time_str = part.split("(")[0].strip()
                            # Create timeslots for weekdays (1-5)
                            for day in range(1, 6):
                                timeslot_id = f"D{day}P{i+1}"
                                timeslots.append({
                                    'id': timeslot_id,
                                    'day': day,
                                    'period': i+1,
                                    'time': time_str
                                })
                elif "Sat" in line:
                    time_parts = line.replace("Sat", "").strip().split(",")
                    for i, part in enumerate(time_parts):
                        part = part.strip()
                        if "(" in part and ")" in part:
                            time_str = part.split("(")[0].strip()
                            # Create timeslot for Saturday (day 6)
                            timeslot_id = f"D6P{i+1}"
                            timeslots.append({
                                'id': timeslot_id,
                                'day': 6,
                                'period': i+1,
                                'time': time_str
                            })
            
            elif current_section == "coincidence":
                # Parse coincidences - exams that need to be scheduled together
                exams_in_coincidence = [x.strip() for x in line.split() if x.strip()]
                # Filter out special characters that might appear in the line
                exams_in_coincidence = [x for x in exams_in_coincidence if not all(c in '{}&()' for c in x)]
                if len(exams_in_coincidence) >= 2:
                    coincidences.append(exams_in_coincidence)
            
            elif current_section == "room":
                parts = line.split()
                if len(parts) >= 2:
                    exam_id = parts[0]
                    room_ids = [r for r in parts[1:] if r not in ['&', 'and']]
                    room_assignments.append({
                        'exam_id': exam_id,
                        'room_ids': room_ids
                    })
            
            elif current_section == "earliness":
                parts = line.split()
                if len(parts) >= 1:
                    exam_id = parts[0]
                    priority = 1
                    if len(parts) >= 2:
                        try:
                            priority = int(parts[1])
                        except ValueError:
                            pass
                    earliness_priority.append({
                        'exam_id': exam_id,
                        'priority': priority
                    })
    
    # If there's a remaining combined group, add it
    if current_combined_group and len(current_combined_group) > 1:
        combined_rooms.append(current_combined_group)
    
    # Load enrollments
    enrollment_path = os.path.join(base_dir, 'enrolements')
    if not os.path.exists(enrollment_path):
        raise FileNotFoundError(f"Enrollment file not found at '{enrollment_path}'. Please check if the dataset structure is correct.")
    
    print(f"Loading enrollments from {enrollment_path}...")
    enrollment_count = 0
    with open(enrollment_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if len(parts) >= 2:
                student_id, exam_id = parts
                enrollments[exam_id].append(student_id)
                student_exams[student_id].append(exam_id)
                enrollment_count += 1
    
    print(f"Loaded {enrollment_count} enrollments")
    
    # Generate numeric timeslots if none are parsed from the data file
    if not timeslots:
        # From the data file we know there are 23 timeslots across 13 days
        days = 13  # 2 weeks excluding weekends
        periods_per_day = 3  # 3 periods most days (9:00, 13:30, 16:30)
        
        for day in range(1, days + 1):
            for period in range(1, periods_per_day + 1):
                if day == 6 or day == 13:  # Saturdays only have 1 slot
                    if period == 1:
                        timeslot_id = f"D{day}P{period}"
                        timeslots.append({
                            'id': timeslot_id,
                            'day': day,
                            'period': period
                        })
                else:
                    timeslot_id = f"D{day}P{period}"
                    timeslots.append({
                        'id': timeslot_id,
                        'day': day,
                        'period': period
                    })
    
    return {
        'exams': exams,
        'rooms': rooms,
        'timeslots': timeslots,
        'enrollments': enrollments,
        'student_exams': student_exams,
        'coincidences': coincidences,
        'room_assignments': room_assignments,
        'earliness_priority': earliness_priority,
        'combined_rooms': combined_rooms
    }

def convert_nott_dataset_to_json(base_dir, output_file='nott_dataset.json'):
    """
    Convert the Nottingham 96 dataset to JSON format and save it to a file.
    
    Args:
        base_dir: Directory containing the Nott dataset files
        output_file: Path to save the JSON output
    """
    # Load the dataset
    dataset = load_nott_dataset(base_dir)
    
    # Print dataset stats
    exams_count = len(dataset['exams'])
    rooms_count = len(dataset['rooms'])
    timeslots_count = len(dataset['timeslots'])
    coincidence_count = len(dataset['coincidences'])
    
    print(f"Dataset loaded: {exams_count} exams, {rooms_count} rooms, {timeslots_count} timeslots, {coincidence_count} coincidence groups")
    
    # Convert to JSON format
    json_dataset = {
        'exams': dataset['exams'],
        'rooms': dataset['rooms'],
        'timeslots': dataset['timeslots'],
        'enrollments': {k: v for k, v in dataset['enrollments'].items()},
        'coincidences': dataset['coincidences'],
        'room_assignments': dataset['room_assignments'],
        'earliness_priority': dataset['earliness_priority'],
        'combined_rooms': dataset['combined_rooms'],
        'metadata': {
            'exams_count': exams_count,
            'rooms_count': rooms_count,
            'timeslots_count': timeslots_count,
            'coincidence_count': coincidence_count,
            'conversion_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(json_dataset, f, indent=2)
        
    print(f"Dataset converted to JSON and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Nottingham 96 dataset to JSON format')
    parser.add_argument('--data_dir', type=str, default='Nott', 
                        help='Directory containing the Nottingham dataset files')
    parser.add_argument('--output', type=str, default='nott_dataset.json',
                        help='Path to save the output JSON file')
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_dir}...")
    convert_nott_dataset_to_json(args.data_dir, args.output)