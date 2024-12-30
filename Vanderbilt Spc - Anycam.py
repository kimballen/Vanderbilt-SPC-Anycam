import cv2
import numpy as np  # Fixed numpy import
import socket
import time
import logging
import json
from threading import Thread, Lock
import sys
import traceback
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
from collections import deque
import base64  # Add at top with other imports
import signal  # Add at top with other imports

def check_imports():
    required_modules = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'PIL': 'pillow'
    }
    
    for module, package in required_modules.items():
        try:
            __import__(module)
            print(f"Successfully imported {module}")
        except ImportError:
            print(f"Error: {module} is not installed. Install using:")
            print(f"pip install {package}")
            return False
    return True

if not check_imports():
    input("Press Enter to exit...")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageHandler(BaseHTTPRequestHandler):
    streams = {}  # Dictionary to store frames for each stream
    frame_lock = Lock()

    def validate_auth(self):
        try:
            # Get stream index from path
            stream_idx = int(self.path.split('/')[3].split('?')[0])
            stream = self.server.rtsp_streams.get(stream_idx)
            
            if not stream or not stream.web_auth_enabled:
                return True  # No auth needed
                
            # Parse URL parameters
            if '?' not in self.path:
                return True  # No auth parameters, allow access
                
            query = self.path.split('?', 1)[1]
            params = dict(param.split('=') for param in query.split('&'))
            
            username = params.get('username', '')
            password = params.get('pwd', '')
            
            # If credentials are present, verify them
            if username and password:
                try:
                    decoded_user = base64.b64decode(username).decode()
                    decoded_pwd = base64.b64decode(password).decode()
                    return (decoded_user == stream.web_auth_username and 
                           decoded_pwd == stream.web_auth_password)
                except:
                    pass
            
            return True  # Allow access if no credentials provided
            
        except:
            return True  # Allow access on any error

    def do_GET(self):
        if self.path.startswith('/cgi-bin/stilljpeg'):
            try:
                if not self.validate_auth():
                    self.send_response(403)  # Changed from 401 to 403
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b"Invalid credentials")
                    return

                # Extract stream index from path
                stream_idx = None
                if self.path.count('/') >= 3:  # Path format: /cgi-bin/stilljpeg/INDEX
                    try:
                        stream_idx = int(self.path.split('/')[-1].split('?')[0])
                    except ValueError:
                        pass

                with ImageHandler.frame_lock:
                    frame_data = None
                    if stream_idx is not None:
                        # Get frame for specific stream
                        frame_data = ImageHandler.streams.get(stream_idx)
                    else:
                        # Backwards compatibility - get first available frame
                        for frame in ImageHandler.streams.values():
                            if frame is not None:
                                frame_data = frame
                                break

                    if frame_data is not None:
                        self.send_response(200)
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
                        self.send_header('Pragma', 'no-cache')
                        self.send_header('Expires', '0')
                        self.end_headers()
                        self.wfile.write(frame_data)
                    else:
                        self.send_response(503)
                        self.send_header('Content-Type', 'text/plain')
                        self.end_headers()
                        self.wfile.write(b"No image available")
            except Exception as e:
                logger.error(f"Error serving image: {e}")
                self.send_error(500)
        else:
            self.send_error(404, "Path must be /cgi-bin/stilljpeg[/stream_index]")

    def log_message(self, format, *args):
        logger.debug(f"{self.address_string()} - {format%args}")

class RTSPtoSPC:
    def __init__(self, rtsp_url, web_port=80, username=None, password=None, stream_index=0, web_username=None, web_password=None):
        self.rtsp_url = rtsp_url
        # Remove any existing credentials from URL
        if '@' in self.rtsp_url:
            self.rtsp_url = 'rtsp://' + self.rtsp_url.split('@')[-1]
        self.web_port = web_port
        self.username = username
        self.password = password
        self.running = False
        self.cap = None
        self.server = None
        self.server_thread = None
        self.last_good_frame = None
        self.consecutive_errors = 0
        self.max_errors = 5
        self.config = {}  # Initialize config attribute
        self.stream_index = stream_index  # Add stream index
        self.web_username = web_username or 'admin'  # Default web credentials
        self.web_password = web_password or 'admin'
        self.web_auth_enabled = False  # New flag to control URL authentication
        self.web_auth_username = None  # URL auth credentials
        self.web_auth_password = None
        self.hw_acceleration = None  # Will store detected hardware acceleration type
        self.detect_hw_acceleration()
        
        # Simplified FFMPEG options for better compatibility
        self.rtsp_options = {
            'rtsp_transport': 'tcp',
            'stimeout': '2000000',
            'bufsize': '1024000',
            'fflags': 'nobuffer',
            'flags': 'low_delay',
            'max_delay': '500000',
            'reorder_queue_size': '0',
            'max_error_rate': '0.0'
        }
        
        self.frame_queue = deque(maxlen=5)
        self.max_retries = 3
        self.retry_delay = 2
        self.last_keyframe = None
        self.keyframe_interval = 30

    def detect_hw_acceleration(self):
        """Detect available hardware acceleration methods"""
        try:
            # Try NVIDIA NVENC
            test_cap = cv2.VideoCapture()
            if test_cap.open('test', cv2.CAP_FFMPEG):
                if 'CUDA' in cv2.getBuildInformation():
                    logger.info("NVIDIA GPU acceleration available")
                    self.hw_acceleration = 'nvidia'
                    return

            # Try Intel QuickSync
            if 'QSV' in cv2.getBuildInformation():
                logger.info("Intel QuickSync acceleration available")
                self.hw_acceleration = 'intel'
                return

            # Try AMD AMF
            if 'AMF' in cv2.getBuildInformation():
                logger.info("AMD AMF acceleration available")
                self.hw_acceleration = 'amd'
                return

            logger.info("No hardware acceleration detected, using CPU")
            self.hw_acceleration = None
        except Exception as e:
            logger.error(f"Error detecting hardware acceleration: {e}")
            self.hw_acceleration = None
        finally:
            if 'test_cap' in locals():
                test_cap.release()

    def start_web_server(self):
        try:
            # Allow server reuse
            class ReuseHTTPServer(HTTPServer):
                allow_reuse_address = True
            
            self.server = ReuseHTTPServer(('', self.web_port), ImageHandler)
            self.server_thread = Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            local_ip = socket.gethostbyname(socket.gethostname())
            logger.info(f"Web server started at http://{local_ip}:{self.web_port}/cgi-bin/stilljpeg")
            return True
        except Exception as e:
            logger.error(f"Failed to start web server: {e}")
            return False

    def stop_web_server(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server_thread.join()
            logger.info("Web server stopped")

    def connect_to_camera(self):
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                # Format base URL for camera
                base_url = self.rtsp_url.replace('rtsp://', '')
                if '@' in base_url:
                    base_url = base_url.split('@')[-1]

                # Build URL with credentials
                if self.username and self.password:
                    url = f"rtsp://{self.username}:{self.password}@{base_url}"
                else:
                    url = f"rtsp://{base_url}"

                # Clean URL
                url = url.replace('///', '//').rstrip('/')
                
                # Configure hardware acceleration
                if self.hw_acceleration:
                    if self.hw_acceleration == 'nvidia':
                        self.rtsp_options.update({
                            'hwaccel': 'cuda',
                            'hwaccel_output_format': 'cuda'
                        })
                    elif self.hw_acceleration == 'intel':
                        self.rtsp_options.update({
                            'hwaccel': 'qsv',
                            'hwaccel_output_format': 'qsv'
                        })
                    elif self.hw_acceleration == 'amd':
                        self.rtsp_options.update({
                            'hwaccel': 'amf',
                            'hwaccel_output_format': 'nv12'
                        })

                # First try with hardware acceleration if available
                if self.hw_acceleration:
                    options_str = '&'.join(f"{k}={v}" for k, v in self.rtsp_options.items())
                    final_url = f"{url}?{options_str}"
                    logger.info(f"Trying with hardware acceleration ({self.hw_acceleration})")
                    self.cap = cv2.VideoCapture(final_url, cv2.CAP_FFMPEG)
                    
                    if not self.cap.isOpened():
                        logger.warning("Hardware acceleration failed, falling back to CPU")
                        self.hw_acceleration = None
                        self.cap.release()

                # If no hardware acceleration or it failed, try without
                if not self.hw_acceleration or not self.cap.isOpened():
                    logger.info("Using CPU decoding")
                    self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

                # Verify connection with multiple attempts
                for attempt in range(3):
                    ret = self.cap.grab()
                    if ret:
                        _, frame = self.cap.retrieve()
                        if frame is not None and frame.size > 0:
                            self.last_keyframe = frame.copy()
                            logger.info(f"Successfully connected to stream on attempt {attempt + 1}")
                            return True
                    time.sleep(0.5)
                
                raise Exception("Failed to get valid frame after connection")

            except Exception as e:
                logger.error(f"Connection attempt {retry_count + 1} failed: {str(e)}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                retry_count += 1
                if retry_count < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error("Max retries reached. Failed to connect to camera.")
                    return False
        return False

    def run(self):
        print(f"Starting camera feed for stream {self.stream_index}...")
        if not self.connect_to_camera() or not self.start_web_server():
            return

        print(f"Starting video stream processing for stream {self.stream_index}...")
        self.running = True
        frames_processed = 0
        start_time = time.time()
        last_reconnect = 0
        reconnect_interval = 5  # seconds

        while self.running:
            try:
                current_time = time.time()
                
                # Check if we need to reconnect
                if self.consecutive_errors >= self.max_errors:
                    if current_time - last_reconnect > reconnect_interval:
                        logger.info(f"Stream {self.stream_index}: Too many errors, attempting to reconnect...")
                        self.cap.release()
                        self.frame_buffer.clear()  # Clear buffer on reconnect
                        if self.connect_to_camera():
                            self.consecutive_errors = 0
                            last_reconnect = current_time
                        continue

                # Skip frames if we're falling behind
                if self.consecutive_errors > 0:
                    self.cap.grab()  # Skip frame
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    raise ValueError(f"Failed to read frame from stream {self.stream_index}")

                processed_frame = self.process_frame(frame)
                with ImageHandler.frame_lock:
                    ImageHandler.streams[self.stream_index] = processed_frame  # Store frame with index
                
                frames_processed += 1
                if frames_processed % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frames_processed / elapsed
                    print(f"Stream {self.stream_index}: {fps:.2f} FPS, Frames: {frames_processed}")
                
                # Adaptive sleep based on processing time
                process_time = time.time() - current_time
                sleep_time = max(0.01, 0.04 - process_time)  # Target ~25 FPS
                time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in processing loop for stream {self.stream_index}: {e}")
                self.consecutive_errors += 1
                time.sleep(0.1)

        print(f"Stream {self.stream_index} processing stopped")
        self.cleanup()

    def cleanup(self):
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
            if self.server:
                self.server.shutdown()
                self.server.server_close()
                if self.server_thread and self.server_thread.is_alive():
                    self.server_thread.join(timeout=1)
                self.server = None
                self.server_thread = None
            logger.info(f"Cleanup completed for stream {self.stream_index}")
        except Exception as e:
            logger.error(f"Error in cleanup for stream {self.stream_index}: {e}")

    def process_frame(self, frame):
        try:
            if frame is None or frame.size == 0:
                return self._get_fallback_frame()

            # Store frame in queue
            self.frame_queue.append(frame)

            # Try to use the most recent valid frame
            valid_frame = None
            for f in reversed(list(self.frame_queue)):
                if f is not None and f.size > 0:
                    valid_frame = f
                    break

            if valid_frame is None:
                return self._get_fallback_frame()

            # Update keyframe periodically
            if len(self.frame_queue) % self.keyframe_interval == 0:
                self.last_keyframe = valid_frame.copy()

            # Process frame
            if self.hw_acceleration:
                processed = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
                
                # Optimize encoding parameters for hardware
                if self.hw_acceleration == 'nvidia':
                    encode_param = [
                        int(cv2.IMWRITE_JPEG_QUALITY), 80,
                        int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,
                        int(cv2.IMWRITE_JPEG_PROGRESSIVE), 0,
                        int(cv2.IMWRITE_JPEG_LUMA_QUALITY), 85,
                        int(cv2.IMWRITE_JPEG_CHROMA_QUALITY), 85
                    ]
                else:
                    encode_param = [
                        int(cv2.IMWRITE_JPEG_QUALITY), 80,
                        int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
                    ]
            else:
                processed = cv2.resize(frame, (640, 480))
                encode_param = [
                    int(cv2.IMWRITE_JPEG_QUALITY), 80,
                    int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,
                    int(cv2.IMWRITE_JPEG_PROGRESSIVE), 0
                ]

            success, encoded = cv2.imencode('.jpg', processed, encode_param)
            if not success:
                return self._get_fallback_frame()
            
            self.consecutive_errors = 0
            return encoded.tobytes()
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return self._get_fallback_frame()

    def _get_fallback_frame(self):
        """Return last keyframe or error frame if no keyframe available"""
        if self.last_keyframe is not None:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            success, encoded = cv2.imencode('.jpg', self.last_keyframe, encode_param)
            if success:
                return encoded.tobytes()
        return self._create_error_frame()

    def _create_error_frame(self):
        # Create error frame with timestamp
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, "No Signal", (200, 220), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, timestamp, (200, 260), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        _, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return encoded.tobytes()

    def apply_camera_settings(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.config.get('brightness', 50) / 100.0)
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.config.get('contrast', 50) / 100.0)
            self.cap.set(cv2.CAP_PROP_SATURATION, self.config.get('saturation', 50) / 100.0)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config.get('exposure', 50) / 100.0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get('frame_width', 640))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get('frame_height', 480))
            self.cap.set(cv2.CAP_PROP_FPS, self.config.get('fps', 30))

    def save_config(self, filename='camera_config.json'):
        with open(filename, 'w') as f:
            json.dump(self.config, f)
            
    def load_config(self, filename='camera_config.json'):
        try:
            with open(filename, 'r') as f:
                self.config.update(json.load(f))
        except FileNotFoundError:
            logger.warning("No config file found, using defaults")

class ConsoleMenu:
    def __init__(self):
        try:
            print("Initializing console menu...")
            self.rtsp_streams = {}  # Dictionary to hold multiple streams
            self.stream_threads = {}  # Dictionary to hold stream threads
            self.num_streams = 1  # Default number of streams
            self.running = True
            # Setup signal handlers
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            # Check for auto-start configuration
            self.load_and_autostart()
        except Exception as e:
            print(f"Error initializing console menu: {str(e)}")
            raise

    def signal_handler(self, signum, frame):
        print("\nReceived shutdown signal. Cleaning up...")
        self.running = False
        self.cleanup_streams()

    def cleanup_streams(self):
        try:
            for idx in list(self.rtsp_streams.keys()):
                try:
                    if idx in self.stream_threads:
                        print(f"Stopping stream {idx}...")
                        self.rtsp_streams[idx].running = False
                        self.stream_threads[idx].join(timeout=3)  # 3 second timeout
                        if self.stream_threads[idx].is_alive():
                            print(f"Stream {idx} didn't stop gracefully")
                        del self.stream_threads[idx]
                except Exception as e:
                    print(f"Error stopping stream {idx}: {e}")
        except Exception as e:
            print(f"Error in cleanup: {e}")

    def stop_stream(self, idx):
        if idx in self.rtsp_streams:
            try:
                print(f"Stopping stream {idx}...")
                # Set running flag to False
                self.rtsp_streams[idx].running = False
                
                # Wait for thread to finish with timeout
                if idx in self.stream_threads:
                    self.stream_threads[idx].join(timeout=3)
                    
                    # Force cleanup if thread is still alive
                    if self.stream_threads[idx].is_alive():
                        print(f"Force stopping stream {idx}...")
                        self.rtsp_streams[idx].cleanup()
                    
                    # Remove thread reference
                    del self.stream_threads[idx]
                
                print(f"Stream {idx} stopped")
            except Exception as e:
                print(f"Error stopping stream {idx}: {e}")
                # Force cleanup on error
                try:
                    self.rtsp_streams[idx].cleanup()
                except:
                    pass

    def stop_all_streams(self):
        if not self.rtsp_streams:
            print("\nNo streams configured")
            return
            
        print("\nStopping all streams...")
        # First set all streams to stop
        for stream in self.rtsp_streams.values():
            stream.running = False
        
        # Then wait for all to stop
        for idx in list(self.rtsp_streams.keys()):
            try:
                if idx in self.stream_threads:
                    self.stream_threads[idx].join(timeout=3)
                    if self.stream_threads[idx].is_alive():
                        print(f"Force stopping stream {idx}...")
                        self.rtsp_streams[idx].cleanup()
                    del self.stream_threads[idx]
                print(f"Stream {idx} stopped")
            except Exception as e:
                print(f"Error stopping stream {idx}: {e}")
                # Force cleanup on error
                try:
                    self.rtsp_streams[idx].cleanup()
                except:
                    pass
        
        print("All streams stopped")
        self.stream_threads.clear()

    def load_and_autostart(self):
        try:
            with open('streams_config.json', 'r') as f:
                config = json.load(f)
                if config.get('auto_start', False):
                    print("\nAuto-start configuration found, loading streams...")
                    self.num_streams = config['num_streams']
                    for idx, stream_config in config['streams'].items():
                        idx = int(idx)
                        self.rtsp_streams[idx] = RTSPtoSPC(
                            stream_config['rtsp_url'],
                            stream_config['web_port'],
                            stream_config['username'],
                            stream_config['password'],
                            stream_index=idx,
                            web_username=stream_config.get('web_username', 'admin'),
                            web_password=stream_config.get('web_password', 'admin')
                        )
                        self.rtsp_streams[idx].config = stream_config['config']
                    print("Starting all streams...")
                    self.start_all_streams()
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"Error in auto-start: {e}")

    def save_with_autostart(self):
        try:
            config = {
                'num_streams': self.num_streams,
                'auto_start': True,
                'streams': {
                    idx: {
                        'rtsp_url': stream.rtsp_url,
                        'web_port': stream.web_port,
                        'username': stream.username,
                        'password': stream.password,
                        'web_username': stream.web_username,
                        'web_password': stream.web_password,
                        'config': stream.config
                    }
                    for idx, stream in self.rtsp_streams.items()
                }
            }
            with open('streams_config.json', 'w') as f:
                json.dump(config, f)
            print("\nConfiguration saved with auto-start enabled")
        except Exception as e:
            print(f"Error saving auto-start configuration: {e}")

    def configure_num_streams(self):
        try:
            print("\n=== Stream Count Configuration ===")
            num = input("Enter number of camera streams [1-10]: ").strip()
            try:
                num = int(num)
                if 1 <= num <= 10:
                    # Clean up existing streams if reducing number
                    if num < len(self.rtsp_streams):
                        for i in range(num, len(self.rtsp_streams)):
                            self.stop_stream(i)
                            del self.rtsp_streams[i]
                            if i in self.stream_threads:
                                del self.stream_threads[i]
                    self.num_streams = num
                    print(f"\nNumber of streams set to {num}")
                else:
                    print("Invalid number. Using 1 stream")
                    self.num_streams = 1
            except ValueError:
                print("Invalid input. Using 1 stream")
                self.num_streams = 1
        except Exception as e:
            print(f"Error configuring streams: {str(e)}")

    def list_streams(self):
        print("\n=== Current Streams ===")
        if not self.rtsp_streams:
            print("No streams configured")
            return
        
        try:
            local_ip = socket.gethostbyname(socket.gethostname())
        except Exception as e:
            local_ip = "localhost"
            logger.error(f"Could not get local IP: {e}")
        
        for idx, stream in self.rtsp_streams.items():
            status = "Running" if idx in self.stream_threads and self.stream_threads[idx].is_alive() else "Stopped"
            
            # Generate URL based on auth settings
            auth_params = ""
            if stream.web_auth_enabled and stream.web_auth_username and stream.web_auth_password:
                username_b64 = base64.b64encode(stream.web_auth_username.encode()).decode()
                password_b64 = base64.b64encode(stream.web_auth_password.encode()).decode()
                auth_params = f"?username={username_b64}&pwd={password_b64}"
            
            print(f"Stream {idx}:")
            print(f"  URL: {stream.rtsp_url}")
            print(f"  Port: {stream.web_port}")
            print(f"  Status: {status}")
            print(f"  Authentication: {'Enabled' if stream.web_auth_enabled else 'Disabled'}")
            print(f"  SPC Camera URL: http://{local_ip}:{stream.web_port}/cgi-bin/stilljpeg/{idx}{auth_params}")
            print()

    def start_all_streams(self):
        if not self.rtsp_streams:
            print("\nNo streams configured")
            return
        
        started_count = 0
        for idx in self.rtsp_streams.keys():
            if idx not in self.stream_threads:
                self.stream_threads[idx] = Thread(target=self.rtsp_streams[idx].run)
                self.stream_threads[idx].start()
                started_count += 1
                print(f"Stream {idx} started")
        
        if started_count > 0:
            print(f"\nStarted {started_count} streams successfully")
        else:
            print("\nAll streams were already running")

    def get_stream_selection(self):
        self.list_streams()
        while True:
            try:
                print("\nAvailable stream slots:")
                for i in range(self.num_streams):
                    status = "Configured" if i in self.rtsp_streams else "Empty"
                    print(f"{i}: {status}")
                
                print("A: Start/Stop all streams")  # Modified text
                choice = input(f"\nSelect stream (0-{self.num_streams-1} or A for all): ").strip()
                
                if choice.upper() == 'A':
                    return 'ALL'
                
                idx = int(choice)
                if 0 <= idx < self.num_streams:
                    return idx
                print("Invalid selection")
            except ValueError:
                if choice.upper() == 'A':  # Handle 'A' input here too
                    return 'ALL'
                print("Invalid input")

    def get_config(self):
        try:
            stream_idx = self.get_stream_selection()
            print(f"\n=== Configuring Stream {stream_idx} ===")
            
            rtsp_url = input("Enter RTSP URL [rtsp://192.168.1.100:554/stream]: ").strip()
            if not rtsp_url:
                rtsp_url = "rtsp://192.168.1.100:554/stream"
            
            username = input("Enter RTSP username: ").strip()
            password = input("Enter RTSP password: ").strip()
            
            # Calculate next available port starting from 80
            used_ports = {s.web_port for s in self.rtsp_streams.values()}
            default_port = next(port for port in range(80, 90) if port not in used_ports)
            
            web_port = input(f"Enter web server port [{default_port}]: ").strip()
            if not web_port:
                web_port = default_port
            else:
                try:
                    web_port = int(web_port)
                except ValueError:
                    print(f"Invalid port number, using {default_port}")
                    web_port = default_port

            print("\nWeb Access Security:")
            use_auth = input("Enable URL authentication (y/N): ").strip().lower() == 'y'  # Fixed toLowerCase to lower()
            web_username = None
            web_password = None
            
            if use_auth:
                web_username = input("Enter URL username: ").strip()
                web_password = input("Enter URL password: ").strip()
                if not web_username or not web_password:
                    print("Username and password required when authentication is enabled")
                    use_auth = False
            
            # Configure stream instance
            stream = RTSPtoSPC(rtsp_url, web_port, username, password, stream_index=stream_idx)
            stream.web_auth_enabled = use_auth
            if use_auth:
                stream.web_auth_username = web_username
                stream.web_auth_password = web_password
            
            return stream_idx, stream

        except Exception as e:
            print(f"Error in configuration: {str(e)}")
            return None, None

    def start_stream(self, idx):
        if idx == 'ALL':
            self.start_all_streams()
            return
            
        if idx in self.rtsp_streams and idx not in self.stream_threads:
            self.stream_threads[idx] = Thread(target=self.rtsp_streams[idx].run)
            self.stream_threads[idx].start()
            print(f"\nStream {idx} started")
        else:
            print(f"\nStream {idx} not configured or already running")

    def stop_stream(self, idx):
        if idx in self.rtsp_streams:
            try:
                self.rtsp_streams[idx].running = False
                if idx in self.stream_threads:
                    self.stream_threads[idx].join(timeout=3)
                    if self.stream_threads[idx].is_alive():
                        print(f"Warning: Stream {idx} is taking long to stop")
                    del self.stream_threads[idx]
                print(f"\nStream {idx} stopped")
            except Exception as e:
                print(f"Error stopping stream {idx}: {e}")

    def stop_all_streams(self):
        if not self.rtsp_streams:
            print("\nNo streams configured")
            return
            
        print("\nStopping all streams...")
        for idx in list(self.rtsp_streams.keys()):
            self.stop_stream(idx)
        print("All streams stopped")

    def save_without_autostart(self):
        try:
            config = {
                'num_streams': self.num_streams,
                'auto_start': False,  # Explicitly disable auto-start
                'streams': {
                    idx: {
                        'rtsp_url': stream.rtsp_url,
                        'web_port': stream.web_port,
                        'username': stream.username,
                        'password': stream.password,
                        'web_username': stream.web_username,
                        'web_password': stream.web_password,
                        'config': stream.config
                    }
                    for idx, stream in self.rtsp_streams.items()
                }
            }
            with open('streams_config.json', 'w') as f:
                json.dump(config, f)
            print("\nConfiguration saved with auto-start disabled")
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def show_menu(self):
        try:
            print("\nStarting RTSP to SPC Camera Stream Console")
            while self.running:
                try:
                    print("\n=== RTSP to SPC Camera Stream ===")
                    print("1. Set number of streams")
                    print("2. Configure stream")
                    print("3. List streams")
                    print("4. Start stream")
                    print("5. Stop stream")
                    print("6. Show stream settings")
                    print("7. Save configuration")
                    print("8. Load configuration")
                    print("9. Save and enable auto-start")
                    print("10. Save and disable auto-start")  # New option
                    print("0. Exit")
                    
                    choice = input("\nEnter choice (0-10): ").strip()
                    print(f"Selected option: {choice}")
                    
                    if choice == "1":
                        self.configure_num_streams()
                    
                    elif choice == "2":
                        result = self.get_config()
                        if all(x is not None for x in result):
                            stream_idx, stream = result
                            self.stop_stream(stream_idx)
                            self.rtsp_streams[stream_idx] = stream
                            print(f"\nStream {stream_idx} configured successfully")
                    
                    elif choice == "3":
                        self.list_streams()
                    
                    elif choice == "4":
                        idx = self.get_stream_selection()
                        self.start_stream(idx)
                    
                    elif choice == "5":
                        print("\nStopping all streams...")
                        self.stop_all_streams()
                        print("\nAll streams have been stopped")
                    
                    elif choice == "6":
                        idx = self.get_stream_selection()
                        if idx in self.rtsp_streams:
                            stream = self.rtsp_streams[idx]
                            print(f"\nStream {idx} settings:")
                            print(f"RTSP URL: {stream.rtsp_url}")
                            print(f"Web Server Port: {stream.web_port}")
                            print("\nCamera settings:")
                            for key, value in stream.config.items():
                                print(f"{key}: {value}")
                        else:
                            print(f"\nStream {idx} not configured")
                    
                    elif choice == "7":
                        # Save all stream configurations
                        config = {
                            'num_streams': self.num_streams,
                            'streams': {
                                idx: {
                                    'rtsp_url': stream.rtsp_url,
                                    'web_port': stream.web_port,
                                    'username': stream.username,
                                    'password': stream.password,
                                    'web_username': stream.web_username,
                                    'web_password': stream.web_password,
                                    'config': stream.config
                                }
                                for idx, stream in self.rtsp_streams.items()
                            }
                        }
                        with open('streams_config.json', 'w') as f:
                            json.dump(config, f)
                        print("\nConfiguration saved")
                    
                    elif choice == "8":
                        try:
                            with open('streams_config.json', 'r') as f:
                                config = json.load(f)
                            # Stop all streams before loading
                            for idx in list(self.rtsp_streams.keys()):
                                self.stop_stream(idx)
                            self.rtsp_streams.clear()
                            self.num_streams = config['num_streams']
                            for idx, stream_config in config['streams'].items():
                                idx = int(idx)  # Convert string key to int
                                self.rtsp_streams[idx] = RTSPtoSPC(
                                    stream_config['rtsp_url'],
                                    stream_config['web_port'],
                                    stream_config['username'],
                                    stream_config['password'],
                                    stream_index=idx,
                                    web_username=stream_config.get('web_username', 'admin'),
                                    web_password=stream_config.get('web_password', 'admin')
                                )
                                self.rtsp_streams[idx].config = stream_config['config']
                            print("\nConfiguration loaded")
                        except FileNotFoundError:
                            print("\nNo configuration file found")
                    
                    elif choice == "9":
                        self.save_with_autostart()
                    
                    elif choice == "10":
                        self.save_without_autostart()
                    
                    elif choice == "0":
                        print("Exiting program...")
                        self.running = False
                        self.cleanup_streams()
                        break
                    
                except KeyboardInterrupt:
                    print("\nReceived interrupt signal...")
                    self.running = False
                    self.cleanup_streams()
                    break
                except Exception as e:
                    print(f"Error in menu operation: {str(e)}")
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"Fatal error in menu: {str(e)}")
            traceback.print_exc()
        finally:
            self.cleanup_streams()

if __name__ == "__main__":
    menu = None
    try:
        print("Starting application...")
        print(f"Python version: {sys.version}")
        print("Creating console menu...")
        menu = ConsoleMenu()
        print("Starting menu...")
        menu.show_menu()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal...")
    except Exception as e:
        print(f"Critical error: {str(e)}")
        traceback.print_exc()
    finally:
        if menu:
            menu.cleanup_streams()
        print("\nProgram finished")
        try:
            input("Press Enter to exit...")
        except KeyboardInterrupt:
            pass


