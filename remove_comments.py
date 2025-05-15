import re

def remove_comments(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace full-line comments
    content = re.sub(r'^\s*#.*$', '', content, flags=re.MULTILINE)
    
    # Process line by line to handle inline comments without affecting color codes
    lines = content.split('\n')
    processed_lines = []
    
    for line in lines:
        # Skip if line is empty after previous processing
        if not line.strip():
            processed_lines.append(line)
            continue
        
        # Find positions of all # characters
        hash_positions = [m.start() for m in re.finditer(r'#', line)]
        
        if not hash_positions:
            # No # in the line
            processed_lines.append(line)
            continue
        
        # Find all string literals to avoid removing # inside strings
        string_ranges = []
        # Find single-quoted strings
        for m in re.finditer(r"'[^']*'", line):
            string_ranges.append((m.start(), m.end()))
        # Find double-quoted strings
        for m in re.finditer(r'"[^"]*"', line):
            string_ranges.append((m.start(), m.end()))
        
        # Check each # position
        comment_pos = None
        for pos in hash_positions:
            # Check if # is inside any string
            in_string = False
            for start, end in string_ranges:
                if start <= pos < end:
                    in_string = True
                    break
            
            # If # is not inside a string, it's the start of a comment
            if not in_string:
                comment_pos = pos
                break
        
        if comment_pos is not None:
            # Keep only the part before the comment
            processed_lines.append(line[:comment_pos])
        else:
            # No comment found, keep the line as is
            processed_lines.append(line)
    
    # Join lines back together
    result = '\n'.join(processed_lines)
    
    # Remove consecutive blank lines
    result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)

if __name__ == "__main__":
    remove_comments("train_models.py", "train_models_no_comments.py")
    print("Comments removed. Output saved to train_models_no_comments.py") 