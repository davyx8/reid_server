# Multi-Video Person Tracking API

This is a FastAPI application that provides an API for tracking people across multiple videos.

## Features

*   Upload videos for processing.
*   Detects people and objects in the videos.
*   Tracks people within and across videos.
*   Identifies anomalies such as changes in clothing or objects being carried.
*   Provides an API to retrieve tracking data and annotated videos.

## API Endpoints

*   `POST /videos`: Upload a video for processing.
*   `GET /persons`: List all tracked persons and their appearances in videos.
*   `GET /persons/{pid}/anomalies`: Get anomalies for a specific person.
*   `GET /videos/{vid}/annotated`: Get the annotated video with tracking information.

## Setup

1.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the FastAPI application:
    ```bash
    uvicorn api:app --host 0.0.0.0 --port 8000
    ```

## Usage

Use a tool like `curl` or an API client to interact with the API.

### Upload a video

```bash
curl -X POST -F "file=@/path/to/your/video.mp4" http://localhost:8000/videos
```

### List persons

```bash
curl http://localhost:8000/persons
```
