#!/usr/bin/env python3
"""
Main application entry point
"""

import json
import os
from pathlib import Path

def load_config():
    """Load configuration from config.json"""
    config_path = Path('config.json')
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def main():
    """Main application function"""
    print("Starting application...")
    
    # Load configuration
    config = load_config()
    print(f"Loaded configuration: {config}")
    
    # Your application logic here
    print("Application running...")
    
if __name__ == "__main__":
    main()