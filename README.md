# Vanderbilt-SPC-Anycam
A small Python project that makes it possible to integrate almost any RTSP camera with a Vanderbilt SPC alarm


Vanderbilt SPC - Anycam User Manual
Table of Contents
Introduction
System Requirements
Installation
Configuration
Setting Number of Streams
Configuring a Stream
Authentication Settings
Running the Application
Managing Streams
Starting Streams
Stopping Streams
Listing Streams
Viewing Stream Settings
Saving and Loading Configuration
Saving Configuration
Loading Configuration
Auto-Start Configuration
Camera Settings
Troubleshooting
Exiting the Application
Introduction
Vanderbilt SPC - Anycam is a Python-based application designed to manage multiple RTSP camera streams, providing web access to live feeds. It supports features such as authentication, hardware acceleration, and configurable camera settings. This manual provides comprehensive instructions on installing, configuring, and using the Anycam application.

System Requirements
Operating System: Windows
Python Version: Python 3.6 or higher
Hardware: Compatible camera devices with RTSP support
Dependencies:
OpenCV (opencv-python)
NumPy (numpy)
Pillow (PIL)
Requests
Other standard Python libraries as listed in the requirements.txt
Installation
Clone the Repository:
Navigate to the Project Directory:
Install Required Python Packages: Ensure you have pip installed. Run the following command to install dependencies:
Note: If requirements.txt is not provided, install dependencies individually:
Configuration
Setting Number of Streams
Start the Application:
Access the Menu: Upon running, the console menu will appear.
Set Number of Streams:
Select option 1 from the main menu.
Enter the desired number of camera streams (1-10).
Configuring a Stream
Access Configuration:
From the main menu, select option 2 to configure a stream.
Input Stream Details:
RTSP URL: Enter the RTSP URL of your camera (e.g., rtsp://192.168.1.100:554/stream).
Username and Password: If your camera requires authentication, provide the RTSP username and password.
Web Server Port: Specify the port for the web server. If left blank, a default port between 80-90 will be assigned.
Configure Authentication:
Choose whether to enable URL authentication by entering y (yes) or n (no).
If enabled, provide a username and password for accessing the web stream.
Authentication Settings
Enable URL Authentication: Protects access to the camera streams by requiring valid credentials.
Provide Credentials: When enabled, users must supply the correct username and password encoded in Base64 to access the stream.
Running the Application
Start the Application:
Interact with the Console Menu: Use the numbered options to configure and manage your camera streams.
Managing Streams
Starting Streams
Start All Streams:
Select option 4 from the main menu.
Choose A to start all configured streams.
Start Individual Stream:
Select option 4 and enter the stream number (e.g., 0).
Stopping Streams
Stop All Streams:
Select option 5 from the main menu to stop all active streams.
Stop Individual Stream:
Select option 5 and follow prompts to stop specific streams.
Listing Streams
View Current Streams:
Select option 3 to list all configured streams along with their status and access URLs.
Viewing Stream Settings
View Settings of a Specific Stream:
Select option 6 and choose the desired stream to view its configuration details.
Saving and Loading Configuration
Saving Configuration
Save Current Settings:
Select option 7 from the main menu to save all current stream configurations to streams_config.json.
Loading Configuration
Load Saved Settings:
Select option 8 to load configurations from streams_config.json.
Existing streams will be stopped and reconfigured based on the loaded settings.
Auto-Start Configuration
Enable Auto-Start:
After configuring streams, select option 9 to save settings with auto-start enabled.
On application launch, streams will automatically start based on the saved configuration.
Disable Auto-Start:
Select option 10 to save settings with auto-start disabled.
Camera Settings
Adjust Camera Parameters:
The application allows configuring settings such as brightness, contrast, saturation, exposure, frame width, frame height, and FPS.
These settings can be adjusted within the stream configuration section.
Troubleshooting
Failed to Connect to Camera:
Ensure the RTSP URL is correct and the camera is accessible over the network.
Verify network connectivity and firewall settings.
Web Server Port Conflicts:
If the default port is in use, specify an alternative port during stream configuration.
Authentication Issues:
Ensure that the correct credentials are provided and properly encoded in Base64 when accessing authenticated streams.
Hardware Acceleration Problems:
If hardware acceleration fails, the application will fallback to CPU decoding. Ensure that your system supports the desired hardware acceleration method.
Exiting the Application
Graceful Shutdown:
Select option 0 from the main menu to exit the application.
The application will stop all active streams and perform necessary cleanup.
Force Exit:
Press Ctrl + C to forcibly terminate the application. This will trigger cleanup procedures to stop streams.
For further assistance or to report issues, please contact the support team or refer to the project's repository documentation.
