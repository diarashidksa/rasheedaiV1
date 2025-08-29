# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file and install the dependencies
# This step is done separately to leverage Docker's caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Expose the port on which the app will run
EXPOSE 8000

# Define the command to start the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]