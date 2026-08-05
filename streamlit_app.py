import os
import sys
import runpy

# Ensure root and app directories are in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(root_dir, 'app')

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Execute main Streamlit app script in app/app.py
target_script = os.path.join(app_dir, 'app.py')
runpy.run_path(target_script, run_name='__main__')
