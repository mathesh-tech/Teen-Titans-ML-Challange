import os
import sys

# Wrapper script allowing execution from C:\ml model> directly
project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'project')
os.chdir(project_dir)
sys.path.insert(0, project_dir)

from train import run_pipeline

if __name__ == "__main__":
    run_pipeline()
