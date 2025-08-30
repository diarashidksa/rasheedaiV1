# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Create the data and flask_session directories and set permissions
RUN mkdir -p data flask_session && chmod -R 777 data flask_session

# Copy the requirements file and install the dependencies
# This is done in two steps for a cleaner build
COPY requirements.txt ./

# Install PyTorch CPU-only
RUN pip install --no-cache-dir torch==2.8.0 --extra-index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download the sentence-transformers model to prevent startup failures
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the entire application code into the container
COPY . .

# Expose the port on which the app will run
EXPOSE 8000

# Define the command to start the application using Gunicorn
# Use the shell form to correctly evaluate the $PORT variable
CMD gunicorn --bind 0.0.0.0:$PORT app:app