# This file tells a computer how to package the web service so it can run the same way anywhere.

# Start from a small Python 3.11 setup.
FROM python:3.11-slim
# Work inside a folder called /app.
WORKDIR /app
# Copy the list of libraries first.
COPY requirements.txt .
# Install those libraries and do not keep the download cache.
RUN pip install --no-cache-dir -r requirements.txt
# Copy the rest of the project.
COPY . .
# The web service listens on port 8000.
EXPOSE 8000
# Start the web service so other computers can reach it.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
