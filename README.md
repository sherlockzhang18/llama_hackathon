# AI Video Scanner with Llama 3.2 Vision


A FastAPI web application that analyzes video frames using Meta's Llama 3.2 Vision model to detect and describe specified objects in videos. Built for the Meta Llama Stack Challenge.

## Features

- **Object Detection**: Find specific objects in video frames using natural language descriptions
- **Frame-by-Frame Analysis**: Processes one frame per second for efficient analysis
- **Detailed Results**: Provides confidence scores and descriptive analysis for each frame
- **Streaming Response**: Real-time results as the video is being processed
- **Image Enhancement**: CLAHE preprocessing for better model performance
- **Responsive UI**: Clean, modern interface with animated results display

## Technology Stack

- **Backend**: FastAPI (Python)
- **AI Model**: Llama 3.2 Vision 11b (via Ollama)
- **Frontend**: HTML5, CSS3, JavaScript (Bootstrap 5)
- **Computer Vision**: OpenCV
- **Server**: Uvicorn

## Installation


1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure you have Ollama installed and the Llama 3.2 Vision model downloaded:
   ```bash
   ollama pull llama3.2-vision
   ```

3. Create the required directories:
   ```bash
   mkdir -p uploads frames
   ```

## Usage

1. Start the FastAPI server:
   ```bash
   python main.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

3. Upload a video file and specify the object you want to find (examples below):

4. View the real-time analysis results showing:
   - Which frames contain the object
   - Detailed descriptions of the object's appearance and location
   - Confidence scores for each detection (1-10 scale)
   - Processed frame images with visual indicators