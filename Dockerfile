FROM python:3.9

# Maintainer info
LABEL maintainer="zhzchen327@zohomail.com"

# Make working directories
RUN  mkdir -p  /p4ds_mst_chatbot
WORKDIR  /p4ds_mst_chatbot

# Upgrade pip with no cache
RUN pip install --no-cache-dir -U pip

# Copy application requirements file to the created working directory
COPY requirements.txt .

# Install application dependencies from the requirements file
RUN pip install -r requirements.txt

# Copy every file in the source folder to the created working directory
COPY  . .

# Run the python application
CMD ["python", "main.py"]