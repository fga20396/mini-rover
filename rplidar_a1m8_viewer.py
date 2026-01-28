"""
Uses the Adafruit CircuitPython RPLidar library to read from a Slamtec RPLidar A1M8 continuously
Visualizes scans in real time using pygame
Runs the sensor reading in a separate thread to keep the UI responsive
Adds a simple range grid, adjustable zoom, and quality‑based coloring

# Install libraries
pip install adafruit-circuitpython-rplidar pygame

Keys:
    + / -: Zoom in/out
    R: Reset zoom
    ESC or Q: Quit
"""

import sys
import math
import threading
import time
import argparse
from collections import deque

import pygame

# Adafruit CircuitPython RPLidar
from adafruit_rplidar import RPLidar


def polar_to_cartesian(angle_deg, distance_mm, scale_px_per_mm, origin):
    """Convert polar (angle in degrees, distance in mm) to pygame cartesian coordinates."""
    theta = math.radians(angle_deg)
    x = origin[0] + distance_mm * math.cos(theta) * scale_px_per_mm
    y = origin[1] - distance_mm * math.sin(theta) * scale_px_per_mm  # Pygame y grows downward
    return int(x), int(y)


def quality_to_color(q):
    """Map RPLidar quality (0-255) to a color. Low quality = red, medium = yellow, high = green."""
    q = max(0, min(255, int(q)))
    if q < 64:
        return (255, 80, 80)         # red-ish
    elif q < 128:
        return (255, 200, 80)        # orange/yellow
    elif q < 192:
        return (120, 220, 120)       # light green
    else:
        return (80, 255, 80)         # green


class LidarReader(threading.Thread):
    """
    Background thread that continuously reads scans from the RPLidar.

    It stores the latest scan as a list of tuples: (angle_deg, distance_mm, quality).
    """

    def __init__(self, port, baudrate=115200, timeout=3.0, max_buf_meas=500, min_len=5):
        super().__init__(daemon=True)
        self.port = port
        self.timeout = timeout
        self.baudrate = baudrate
        self.max_buf_meas = max_buf_meas
        self.min_len = min_len

        self._lidar = None
        self.latest_scan = []  # list[(angle, distance, quality)]
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._started = threading.Event()
        self._error = None

    def run(self):
        try:
            # Adafruit RPLidar constructor: RPLidar(None, port, baudrate=115200, timeout=1)
            self._lidar = RPLidar(None, self.port, baudrate=self.baudrate, timeout=self.timeout)

            # Sometimes motor will be off — start if supported (ignore if not)
            try:
                self._lidar.start_motor()
            except AttributeError:
                pass

            self._started.set()

            # iter_scans yields lists of measurements: (quality, angle, distance)
            for scan in self._lidar.iter_scans(max_buf_meas=self.max_buf_meas, min_len=self.min_len):
                if self._stop_event.is_set():
                    break
                # Convert to (angle, distance, quality) and store
                formatted = [(m[1], m[2], m[0]) for m in scan if m[2] > 0]  # distance > 0 means valid
                with self._lock:
                    self.latest_scan = formatted

        except Exception as e:
            self._error = e
        finally:
            self.stop_lidar()

    def stop_lidar(self):
        # Gracefully stop the lidar device
        try:
            if self._lidar is not None:
                try:
                    self._lidar.stop()
                except Exception:
                    pass
                # Some libs support stop_motor separately
                try:
                    self._lidar.stop_motor()
                except Exception:
                    pass
                try:
                    self._lidar.disconnect()
                except Exception:
                    pass
        finally:
            self._lidar = None

    def stop(self):
        self._stop_event.set()

    def wait_started(self, timeout=5.0):
        return self._started.wait(timeout=timeout)

    def get_latest_scan(self):
        with self._lock:
            return list(self.latest_scan)

    def get_error(self):
        return self._error


def draw_grid(surface, origin, scale_px_per_mm, max_radius_mm):
    """Draw concentric range rings every 1 m and axes."""
    w, h = surface.get_size()
    bg = (18, 18, 22)
    grid = (45, 45, 55)
    axis = (70, 70, 90)

    surface.fill(bg)

    # Concentric circles every 1000 mm (1 m)
    meters = int(max_radius_mm // 1000)
    for i in range(1, meters + 1):
        radius_px = int(i * 1000 * scale_px_per_mm)
        if radius_px < 2:
            continue
        pygame.draw.circle(surface, grid, origin, radius_px, 1)

    # Crosshair axes
    pygame.draw.line(surface, axis, (origin[0], 0), (origin[0], h), 1)
    pygame.draw.line(surface, axis, (0, origin[1]), (w, origin[1]), 1)


def main():
    parser = argparse.ArgumentParser(description="RPLidar A1M8 viewer (Adafruit lib + pygame)")
    parser.add_argument("--port", default="/dev/ttyUSB0",
                        help="Serial port (e.g., /dev/ttyUSB0, COM3, /dev/cu.usbserial-*)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default 115200)")
    parser.add_argument("--timeout", type=float, default=3.0, help="Serial timeout seconds")
    parser.add_argument("--width", type=int, default=900, help="Window width")
    parser.add_argument("--height", type=int, default=900, help="Window height")
    parser.add_argument("--max-mm", type=int, default=6000, help="Max distance to draw (mm)")
    parser.add_argument("--scale", type=float, default=0.10,
                        help="Pixels per mm (e.g., 0.10 px/mm => 6000 mm ~ 600 px radius)")
    parser.add_argument("--point-size", type=int, default=2, help="Point radius in pixels")
    args = parser.parse_args()

    pygame.init()
    pygame.display.set_caption("RPLidar A1M8 Viewer (Adafruit + pygame)")
    screen = pygame.display.set_mode((args.width, args.height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    origin = (args.width // 2, args.height // 2)
    scale_px_per_mm = args.scale
    max_radius_mm = args.max_mm
    point_radius = args.point_size

    # Start lidar reader thread
    reader = LidarReader(args.port, baudrate=args.baud, timeout=args.timeout)
    reader.start()
    if not reader.wait_started(timeout=6.0):
        err = reader.get_error()
        reader.stop()
        pygame.quit()
        if err:
            print(f"Failed to start LIDAR: {err}", file=sys.stderr)
        else:
            print("Failed to start LIDAR (timeout).", file=sys.stderr)
        sys.exit(1)

    running = True
    fps_smooth = deque(maxlen=30)

    # Basic UI help
    help_lines = [
        "Keys:",
        "  ESC / Q    Quit",
        "  + / -      Zoom in / out",
        "  R          Reset zoom",
        "",
        f"Port: {args.port}  Baud: {args.baud}",
    ]

    try:
        while running:
            t0 = time.time()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        scale_px_per_mm *= 1.1
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        scale_px_per_mm /= 1.1
                        scale_px_per_mm = max(0.01, scale_px_per_mm)
                    elif event.key == pygame.K_r:
                        scale_px_per_mm = args.scale

            # Draw background/grid
            draw_grid(screen, origin, scale_px_per_mm, max_radius_mm)

            # Draw scan
            scan = reader.get_latest_scan()
            points_drawn = 0
            for angle, distance, quality in scan:
                if distance <= 0 or distance > max_radius_mm:
                    continue
                x, y = polar_to_cartesian(angle, distance, scale_px_per_mm, origin)
                color = quality_to_color(quality)
                pygame.draw.circle(screen, color, (x, y), point_radius)
                points_drawn += 1

            # Text overlay
            fps = clock.get_fps()
            if fps > 0:
                fps_smooth.append(fps)
            fps_avg = sum(fps_smooth) / len(fps_smooth) if fps_smooth else 0.0

            overlay_lines = [
                f"Points: {points_drawn}",
                f"FPS: {fps_avg:5.1f}",
                f"Scale: {scale_px_per_mm:.3f} px/mm",
                f"Max range drawn: {max_radius_mm/1000:.1f} m",
            ]
            # Render help + overlay
            y = 8
            for line in help_lines + overlay_lines:
                surf = font.render(line, True, (200, 200, 210))
                screen.blit(surf, (8, y))
                y += 18

            pygame.display.flip()
            clock.tick(60)  # limit to ~60 FPS

            # Check for reader errors
            err = reader.get_error()
            if err:
                raise err

    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        reader.join(timeout=3.0)
        pygame.quit()


if __name__ == "__main__":
    main()
