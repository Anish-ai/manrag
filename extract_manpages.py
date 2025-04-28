#!/usr/bin/env python3
import os
import subprocess
import re
from pathlib import Path
import json
import time
import multiprocessing
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/extraction.log")
    ]
)
logger = logging.getLogger("ManPageExtractor")

def get_all_man_commands():
    """Get a list of all available man commands."""
    try:
        # Find all man directories
        man_dirs_output = subprocess.check_output("man --path", shell=True).decode('utf-8').strip()
        man_dirs = man_dirs_output.split(':')
        
        commands = set()
        
        for man_dir in man_dirs:
            # Skip if directory doesn't exist
            if not os.path.isdir(man_dir):
                continue
                
            # Look for man pages in each section (1-9)
            for section in range(1, 10):
                section_dir = os.path.join(man_dir, f"man{section}")
                if not os.path.isdir(section_dir):
                    continue
                    
                for filename in os.listdir(section_dir):
                    # Extract command name from filename (remove extensions like .gz)
                    command = re.sub(r'\.[^.]*$', '', filename)
                    if command:
                        commands.add((command, str(section)))
        
        return sorted(list(commands))
    except Exception as e:
        logger.error(f"Error getting man commands: {e}")
        return []

def extract_man_page(args):
    """
    Extract a single man page content.
    
    Args:
        args: A tuple of (command, section)
        
    Returns:
        A dictionary with the extracted man page data or None if extraction failed
    """
    command, section = args
    
    try:
        # Using --ascii option to avoid font warnings
        content = subprocess.check_output(f"man --ascii {section} {command} 2>/dev/null", 
                                         shell=True).decode('utf-8')
        
        # Clean up the content (remove formatting codes)
        content = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', content)
        
        return {
            "command": command,
            "section": section,
            "content": content,
            "id": f"{command}_{section}",
            "added_timestamp": int(time.time())
        }
    except:
        try:
            # Try without section
            content = subprocess.check_output(f"man --ascii {command} 2>/dev/null", 
                                            shell=True).decode('utf-8')
            
            # Clean up the content (remove formatting codes)
            content = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', content)
            
            return {
                "command": command,
                "section": section,
                "content": content,
                "id": f"{command}_{section}",
                "added_timestamp": int(time.time())
            }
        except:
            return None

def load_existing_manpages(file_path):
    """Load existing man pages from JSON file."""
    if not os.path.exists(file_path):
        return []
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading existing man pages: {e}")
        return []

def main():
    start_time = time.time()
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    json_path = data_dir / "manpages.json"
    
    # Load existing man pages
    existing_manpages = load_existing_manpages(json_path)
    logger.info(f"Loaded {len(existing_manpages)} existing man pages.")
    
    # Create a set of existing command/section pairs for quick lookups
    existing_pairs = {(page["command"], page["section"]) for page in existing_manpages}
    
    # Get all commands with man pages
    logger.info("Finding all available man pages...")
    all_commands = get_all_man_commands()
    logger.info(f"Found {len(all_commands)} total man pages.")
    
    # Filter to only new commands
    new_commands = [cmd for cmd in all_commands if (cmd[0], cmd[1]) not in existing_pairs]
    logger.info(f"Detected {len(new_commands)} new man pages to extract.")
    
    if not new_commands:
        logger.info("No new man pages found. Exiting.")
        return
    
    # Use all available CPU cores
    num_cores = multiprocessing.cpu_count()
    logger.info(f"Using {num_cores} CPU cores for parallel extraction")
    
    # Extract man pages in parallel using Pool
    new_results = []
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Use imap to process results as they come in - allows for progress bar
        for result in tqdm(
            pool.imap(extract_man_page, new_commands), 
            total=len(new_commands),
            desc=f"Extracting man pages (using {num_cores} cores)"
        ):
            if result:
                new_results.append(result)
    
    # Combine with existing results
    combined_results = existing_manpages + new_results
    
    # Save to JSON file
    with open(json_path, "w") as f:
        json.dump(combined_results, f)
    
    # If any new pages were added, create a timestamp file to indicate update
    if new_results:
        with open(data_dir / "last_update.txt", "w") as f:
            f.write(f"{int(time.time())}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"Successfully extracted {len(new_results)} new man pages")
    logger.info(f"Total man pages in database: {len(combined_results)}")
    logger.info(f"Total extraction time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Set multiprocessing start method based on OS
    if os.name == 'posix':
        # For Linux/macOS
        multiprocessing.set_start_method('fork')
    else:
        # For Windows
        multiprocessing.set_start_method('spawn')
    
    main()