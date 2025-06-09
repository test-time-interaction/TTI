import json
import os
from pathlib import Path

# Directory containing JSON task files
tasks_dir = '.'  # Update this to your actual directory path

# Output file path
output_file = './webarena_test_data.jsonl'

def process_tasks():
    # Array to store all processed tasks
    summarized_tasks = []
    
    # Get all JSON files in the directory
    task_files = [f for f in os.listdir(tasks_dir) if f.endswith('.json')]
    
    # Process each file
    for file in task_files:
        file_path = os.path.join(tasks_dir, file)
        if 'test' in str(file) or 'task_list' in str(file):
            continue
        
        try:
            # Read and parse the JSON content
            with open(file_path, 'r', encoding='utf-8') as f:
                task = json.load(f)
            
            # Create a summarized version with the additional keys
            summarized_task = {
                # Include all original task properties
                **task,
                
                # Add the four required keys
                'web_name': task['sites'][0],
                'id': f"{task['sites'][0]}--{task['task_id']}",
                'ques': task['intent'],
                'web': task['start_url']
            }
            
            # Add to the array of summarized tasks
            summarized_tasks.append(summarized_task)
            
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    # Write the summarized tasks to a JSONL file (one JSON object per line)
    with open(output_file, 'w', encoding='utf-8') as f:
        for task in summarized_tasks:
            f.write(json.dumps(task) + '\n')
    
    print(f"Processed {len(summarized_tasks)} tasks. Output saved to {output_file} in JSONL format")

if __name__ == "__main__":
    process_tasks()