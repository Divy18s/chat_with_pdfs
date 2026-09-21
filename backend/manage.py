#!/usr/bin/env python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
from django.core.management import execute_from_command_line
if __name__ == '__main__':
    execute_from_command_line(sys.argv)
