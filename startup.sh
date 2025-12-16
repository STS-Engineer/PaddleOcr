
#!/bin/bash

# Install required system libraries for OpenCV
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0

# Start Gunicorn
gunicorn --bind=0.0.0.0 --timeout 600 ocr_api:app