"""
File Operations Tools
=====================
Tools for file system operations.
"""

import logging
import os
import json
import csv
import aiofiles
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..shared import format_success, format_error, validate_path

logger = logging.getLogger(__name__)


async def read_file(
    file_path: str,
    encoding: str = "utf-8",
    mode: str = "text"
) -> Dict[str, Any]:
    """
    Read contents of a file.
    
    Args:
        file_path: Path to the file
        encoding: File encoding for text mode
        mode: Read mode (text or binary)
    
    Returns:
        File contents and metadata
    """
    try:
        # Validate path
        if not validate_path(file_path):
            return format_error("Invalid file path")
        
        path = Path(file_path)
        
        if not path.exists():
            return format_error(f"File not found: {file_path}")
        
        if not path.is_file():
            return format_error(f"Not a file: {file_path}")
        
        # Get file stats
        stats = path.stat()
        
        # Read file
        if mode == "text":
            async with aiofiles.open(path, mode='r', encoding=encoding) as f:
                content = await f.read()
        else:
            async with aiofiles.open(path, mode='rb') as f:
                content = await f.read()
        
        return format_success({
            "path": str(path.absolute()),
            "name": path.name,
            "size": stats.st_size,
            "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "encoding": encoding if mode == "text" else None,
            "content": content
        })
        
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading file: {e}")
        return format_error(f"Encoding error: {str(e)}")
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return format_error(str(e))


async def write_file(
    file_path: str,
    content: Any,
    encoding: str = "utf-8",
    mode: str = "text",
    create_dirs: bool = True
) -> Dict[str, Any]:
    """
    Write content to a file.
    
    Args:
        file_path: Path to the file
        content: Content to write
        encoding: File encoding for text mode
        mode: Write mode (text or binary)
        create_dirs: Create parent directories if they don't exist
    
    Returns:
        Write operation result
    """
    try:
        # Validate path
        if not validate_path(file_path):
            return format_error("Invalid file path")
        
        path = Path(file_path)
        
        # Create parent directories if needed
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        if mode == "text":
            async with aiofiles.open(path, mode='w', encoding=encoding) as f:
                if isinstance(content, (dict, list)):
                    await f.write(json.dumps(content, indent=2))
                else:
                    await f.write(str(content))
        else:
            async with aiofiles.open(path, mode='wb') as f:
                await f.write(content if isinstance(content, bytes) else str(content).encode())
        
        # Get file stats
        stats = path.stat()
        
        return format_success({
            "path": str(path.absolute()),
            "size": stats.st_size,
            "created": datetime.now().isoformat(),
            "mode": mode
        })
        
    except Exception as e:
        logger.error(f"Error writing file: {e}")
        return format_error(str(e))


async def list_files(
    directory: str,
    pattern: str = "*",
    recursive: bool = False,
    include_hidden: bool = False
) -> Dict[str, Any]:
    """
    List files in a directory.
    
    Args:
        directory: Directory path
        pattern: File pattern to match
        recursive: Search recursively
        include_hidden: Include hidden files
    
    Returns:
        List of files with metadata
    """
    try:
        # Validate path
        if not validate_path(directory):
            return format_error("Invalid directory path")
        
        path = Path(directory)
        
        if not path.exists():
            return format_error(f"Directory not found: {directory}")
        
        if not path.is_dir():
            return format_error(f"Not a directory: {directory}")
        
        files = []
        
        # Get files based on pattern
        if recursive:
            file_paths = path.rglob(pattern)
        else:
            file_paths = path.glob(pattern)
        
        for file_path in file_paths:
            # Skip hidden files if not included
            if not include_hidden and file_path.name.startswith('.'):
                continue
            
            if file_path.is_file():
                stats = file_path.stat()
                files.append({
                    "name": file_path.name,
                    "path": str(file_path.relative_to(path)),
                    "size": stats.st_size,
                    "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                    "is_hidden": file_path.name.startswith('.')
                })
        
        # Sort by name
        files.sort(key=lambda x: x["name"])
        
        return format_success({
            "directory": str(path.absolute()),
            "pattern": pattern,
            "recursive": recursive,
            "count": len(files),
            "files": files
        })
        
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return format_error(str(e))


async def process_csv(
    file_path: str,
    operation: str = "read",
    data: Optional[List[Dict[str, Any]]] = None,
    delimiter: str = ",",
    has_header: bool = True
) -> Dict[str, Any]:
    """
    Process CSV files.
    
    Args:
        file_path: Path to CSV file
        operation: Operation to perform (read, write, append)
        data: Data to write (for write/append operations)
        delimiter: CSV delimiter
        has_header: Whether CSV has header row
    
    Returns:
        CSV processing result
    """
    try:
        path = Path(file_path)
        
        if operation == "read":
            if not path.exists():
                return format_error(f"CSV file not found: {file_path}")
            
            rows = []
            async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
                content = await f.read()
                
            # Parse CSV
            reader = csv.DictReader(
                content.splitlines(),
                delimiter=delimiter
            ) if has_header else csv.reader(
                content.splitlines(),
                delimiter=delimiter
            )
            
            for row in reader:
                rows.append(row)
            
            return format_success({
                "file": str(path.absolute()),
                "rows": len(rows),
                "columns": list(rows[0].keys()) if rows and has_header else None,
                "data": rows
            })
            
        elif operation in ["write", "append"]:
            if not data:
                return format_error("No data provided for write operation")
            
            # Prepare CSV content
            if has_header and data:
                fieldnames = list(data[0].keys())
                output = []
                
                writer = csv.DictWriter(
                    output,
                    fieldnames=fieldnames,
                    delimiter=delimiter
                )
                
                if operation == "write" or not path.exists():
                    writer.writeheader()
                
                writer.writerows(data)
                content = '\n'.join(output)
            else:
                output = []
                writer = csv.writer(output, delimiter=delimiter)
                writer.writerows(data)
                content = '\n'.join(output)
            
            # Write to file
            mode = 'a' if operation == "append" and path.exists() else 'w'
            async with aiofiles.open(path, mode=mode, encoding='utf-8') as f:
                await f.write(content)
                if mode == 'a':
                    await f.write('\n')
            
            return format_success({
                "file": str(path.absolute()),
                "operation": operation,
                "rows_written": len(data)
            })
            
        else:
            return format_error(f"Invalid operation: {operation}")
            
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        return format_error(str(e))