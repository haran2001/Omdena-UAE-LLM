# Use the official Python image
# FROM python:3.10
FROM python:3.8.18

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file first to leverage Docker cache
COPY requirements.txt .

# Install required Python packages
RUN pip install -r requirements.txt --default-timeout=100 future

# Copy the rest of the application files to the container's working directory
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run your Streamlit application
CMD ["streamlit", "run", "app/chatbot_app.py"]



# for docker compose
# FROM python:3.8.18

# WORKDIR /usr/src/app

# # dont write pyc files
# # dont buffer to stdout/stderr
# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1

# COPY ./requirements.txt /usr/src/app/requirements.txt

# # dependencies
# RUN pip install --upgrade pip setuptools wheel \
#     && pip install -r requirements.txt --default-timeout=100 future\
#     && rm -rf /root/.cache/pip

# COPY ./ /usr/src/app
