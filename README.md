# Vanderbilt-SPC-Anycam
A small Python project that makes it possible to integrate almost any RTSP camera with a Vanderbilt SPC alarm


# RTSP to SPC Camera Stream

This Python script captures video streams from RTSP cameras and serves them as JPEG images over HTTP. It supports multiple streams, hardware acceleration, and optional URL authentication.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Features](#features)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Requirements
- Python 3.6 or higher
- Required Python packages:
  - `opencv-python`
  - `numpy`
  - `requests`
  - `pillow`

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/rtsp-to-spc.git
    cd rtsp-to-spc
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Configuration
### Stream Configuration
The script supports multiple RTSP streams. You can configure the streams by editing the `streams_config.json` file or using the console menu.

Example `streams_config.json`:
```json
{
    "num_streams": 2,
    "auto_start": true,
    "streams": {
        "0": {
            "rtsp_url": "rtsp://192.168.1.100:554/stream",
            "web_port": 8080,
            "username": "user",
            "password": "pass",
            "web_username": "admin",
            "web_password": "admin",
            "config": {
                "brightness": 50,
                "contrast": 50,
                "saturation": 50,
                "exposure": 50,
                "frame_width": 640,
                "frame_height": 480,
                "fps": 30
            }
        },
        "1": {
            "rtsp_url": "rtsp://192.168.1.101:554/stream",
            "web_port": 8081,
            "username": "user",
            "password": "pass",
            "web_username": "admin",
            "web_password": "admin",
            "config": {
                "brightness": 50,
                "contrast": 50,
                "saturation": 50,
                "exposure": 50,
                "frame_width": 640,
                "frame_height": 480,
                "fps": 30
            }
        }
    }
}
```

### Console Menu
The script includes a console menu for configuring and managing streams interactively.

## Usage
1. Run the script:
    ```sh
    python Vanderbilt_Spc_Anycam.py
    ```

2. Use the console menu to configure streams, start/stop streams, and save/load configurations.

### Console Menu Options
- **Set number of streams**: Configure the number of RTSP streams.
- **Configure stream**: Set up the RTSP URL, credentials, and web server port for a stream.
- **List streams**: Display the current stream configurations.
- **Start stream**: Start a specific stream or all streams.
- **Stop stream**: Stop a specific stream or all streams.
- **Show stream settings**: Display the settings of a specific stream.
- **Save configuration**: Save the current configuration to `streams_config.json`.
- **Load configuration**: Load the configuration from `streams_config.json`.
- **Save and enable auto-start**: Save the configuration and enable auto-start.
- **Save and disable auto-start**: Save the configuration and disable auto-start.
- **Exit**: Exit the program.

## Features
- **Multiple Streams**: Support for multiple RTSP streams.
- **Hardware Acceleration**: Detect and use available hardware acceleration (NVIDIA, Intel, AMD).
- **URL Authentication**: Optional URL authentication for accessing the JPEG images.
- **Adaptive Frame Processing**: Adaptive frame processing to maintain target FPS.
- **Error Handling**: Robust error handling and automatic reconnection.

## Troubleshooting
- **Failed to connect to camera**: Ensure the RTSP URL and credentials are correct.
- **No image available**: Check the camera connection and stream configuration.
- **Web server not starting**: Verify the web server port is not in use by another application.
