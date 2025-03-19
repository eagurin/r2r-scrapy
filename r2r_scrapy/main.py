import sys
import os
from r2r_scrapy.cli.main import cli
import asyncio

def main():
    """Main entry point for R2R Scrapy"""
    # Add the project directory to the Python path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Run the CLI
    cli()

if __name__ == "__main__":
    # Run the main function
    main()
