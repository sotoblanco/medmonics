import sys

filename = "/home/pastor/projects_linux/medmonics/app.py"
start_line = 696
end_line = 904

try:
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # Python list is 0-indexed, so line 696 is index 695
    # We want to remove from index 695 up to index 904 (inclusive of line 904?)
    # Lines are 1-indexed. 
    # To remove 696 to 904:
    # keep lines[:695] + lines[904:]
    
    # Check bounds
    if len(lines) < start_line:
        print(f"File has only {len(lines)} lines, cannot start at {start_line}")
        sys.exit(1)
        
    print(f"Original length: {len(lines)}")
    print(f"Removing lines {start_line} to {end_line}")
    
    # Slice
    new_lines = lines[:start_line-1] + lines[end_line:]
    
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
        
    print(f"New length: {len(new_lines)}")
    print("Successfully removed lines.")
    
except Exception as e:
    print(f"Error: {e}")
