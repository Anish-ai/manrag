#!/usr/bin/env python3
import json
import os
from pathlib import Path
import time

# Create data directory
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Sample man pages for common commands
sample_data = [
    {
        "command": "find",
        "section": "1",
        "content": """
FIND(1)                  User Commands                  FIND(1)

NAME
       find - search for files in a directory hierarchy

SYNOPSIS
       find [-H] [-L] [-P] [-D debugopts] [-Olevel] [starting-point...] [expression]

DESCRIPTION
       This  manual  page documents the GNU version of find.  GNU find searches the directory tree rooted at each
       given starting-point by evaluating the given expression from left to right, according to  the  rules  of
       precedence (see section OPERATORS), until the outcome is known (the left hand side is false for AND operations,
       true for OR), at which point find moves on to the next file name.

EXAMPLES
       find /tmp -name "*.txt" -type f
              Find all text files in /tmp directory
       
       find . -type f -name "*.jpg" -size +1M
              Find all JPEG files larger than 1 MB in current directory
       
       find /home -user john -mtime -7
              Find all files owned by john modified in the last 7 days

       find . -name "*.c" -o -name "*.h"
              Find all C source and header files
        """,
        "id": "find_1",
        "added_timestamp": int(time.time())
    },
    {
        "command": "grep",
        "section": "1",
        "content": """
GREP(1)                  User Commands                  GREP(1)

NAME
       grep, egrep, fgrep, rgrep - print lines that match patterns

SYNOPSIS
       grep [OPTIONS] PATTERN [FILE...]
       grep [OPTIONS] -e PATTERN ... [FILE...]
       grep [OPTIONS] -f FILE ... [FILE...]

DESCRIPTION
       grep  searches  for  PATTERN  in  each  FILE.  A FILE of "-" stands for standard input.  If no FILE is given,
       recursive searches examine the working directory, and nonrecursive searches read standard input.

EXAMPLES
       grep "hello" file.txt
              Search for "hello" in file.txt
       
       grep -i "error" *.log
              Search for "error" (case insensitive) in all .log files
       
       grep -r "TODO" --include="*.py" .
              Recursively search for "TODO" in all Python files in current directory
        """,
        "id": "grep_1",
        "added_timestamp": int(time.time())
    },
    {
        "command": "ls",
        "section": "1",
        "content": """
LS(1)                    User Commands                    LS(1)

NAME
       ls - list directory contents

SYNOPSIS
       ls [OPTION]... [FILE]...

DESCRIPTION
       List  information  about  the FILEs (the current directory by default).  Sort entries alphabetically if none of
       -cftuvSUX nor --sort is specified.

EXAMPLES
       ls
              List files in the current directory
       
       ls -l
              List files in long format
       
       ls -la
              List all files including hidden ones in long format
       
       ls -lh
              List files in long format with human readable sizes
        """,
        "id": "ls_1",
        "added_timestamp": int(time.time())
    },
    {
        "command": "cp",
        "section": "1",
        "content": """
CP(1)                    User Commands                    CP(1)

NAME
       cp - copy files and directories

SYNOPSIS
       cp [OPTION]... [-T] SOURCE DEST
       cp [OPTION]... SOURCE... DIRECTORY
       cp [OPTION]... -t DIRECTORY SOURCE...

DESCRIPTION
       Copy SOURCE to DEST, or multiple SOURCE(s) to DIRECTORY.

EXAMPLES
       cp file1 file2
              Copy file1 to file2
       
       cp -r dir1 dir2
              Copy directory dir1 to dir2 recursively
       
       cp -p file1 file2
              Copy file1 to file2 and preserve mode, ownership, timestamps
        """,
        "id": "cp_1",
        "added_timestamp": int(time.time())
    },
    {
        "command": "chmod",
        "section": "1",
        "content": """
CHMOD(1)                 User Commands                 CHMOD(1)

NAME
       chmod - change file mode bits

SYNOPSIS
       chmod [OPTION]... MODE[,MODE]... FILE...
       chmod [OPTION]... OCTAL-MODE FILE...
       chmod [OPTION]... --reference=RFILE FILE...

DESCRIPTION
       Change the file mode bits of each FILE to MODE.

EXAMPLES
       chmod 755 script.sh
              Make script.sh executable by owner and readable by others
       
       chmod u+x file
              Add execute permission for the owner
       
       chmod -R go-w directory
              Remove write permission for group and others recursively
        """,
        "id": "chmod_1",
        "added_timestamp": int(time.time())
    }
]

# Save to JSON file
json_path = data_dir / "manpages.json"
with open(json_path, "w") as f:
    json.dump(sample_data, f, indent=2)

print(f"Created sample dataset with {len(sample_data)} man pages in {json_path}")
print("Next, run 'python process_and_load.py' to process and load into the vector database") 