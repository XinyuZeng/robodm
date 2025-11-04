#!/usr/bin/env python3
"""
Script to convert MP4 to MKV format and profile random frame access performance.
Records IO operations and their sizes during random frame reading.
"""

import os
import sys
import subprocess
import random
import time
import psutil
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import tempfile


class IOProfiler:
    """Profiles IO operations during file access."""
    
    def __init__(self):
        self.io_stats = []
        self.start_time = None
        self.end_time = None
        
    def start_profiling(self):
        """Start IO profiling."""
        self.start_time = time.time()
        self.io_stats = []
        
    def stop_profiling(self):
        """Stop IO profiling."""
        self.end_time = time.time()
        
    def get_io_stats(self) -> Dict:
        """Get current IO statistics."""
        try:
            # Get process IO stats
            process = psutil.Process()
            io_counters = process.io_counters()
            
            stats = {
                'read_count': io_counters.read_count,
                'write_count': io_counters.write_count,
                'read_bytes': io_counters.read_bytes,
                'write_bytes': io_counters.write_bytes,
                'timestamp': time.time()
            }
            
            self.io_stats.append(stats)
            return stats
            
        except Exception as e:
            print(f"Warning: Could not get IO stats: {e}")
            return {}
    
    def get_summary(self) -> Dict:
        """Get profiling summary."""
        if not self.io_stats:
            return {}
            
        # Calculate differences between first and last measurement
        first_stats = self.io_stats[0]
        last_stats = self.io_stats[-1]
        
        return {
            'total_time': self.end_time - self.start_time if self.end_time and self.start_time else 0,
            'total_read_count': last_stats['read_count'] - first_stats['read_count'],
            'total_write_count': last_stats['write_count'] - first_stats['write_count'],
            'total_read_bytes': last_stats['read_bytes'] - first_stats['read_bytes'],
            'total_write_bytes': last_stats['write_bytes'] - first_stats['write_bytes'],
            'measurements_taken': len(self.io_stats)
        }


def convert_mp4_to_mkv(input_path: str, output_path: str) -> bool:
    """
    Convert MP4 file to MKV format using ffmpeg.
    
    Args:
        input_path: Path to input MP4 file
        output_path: Path to output MKV file
        
    Returns:
        True if conversion successful, False otherwise
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist")
        return False
        
    print(f"Converting {input_path} to {output_path}...")
    
    try:
        # Use ffmpeg to convert MP4 to MKV
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c', 'copy',  # Copy streams without re-encoding
            '-f', 'matroska',  # Force MKV format
            '-y',  # Overwrite output file
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully converted to {output_path}")
            return True
        else:
            print(f"FFmpeg error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg.")
        return False
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False


def get_video_info(video_path: str) -> Dict:
    """
    Get video information including frame count and duration.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return {}
    
    info = {
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info


def extract_frame_with_ffmpeg(video_path: str, frame_number: int) -> Tuple[bool, np.ndarray | None]:
    """
    Extract a specific frame using ffmpeg to handle AV1 and other codecs.
    
    Args:
        video_path: Path to video file
        frame_number: Frame number to extract (0-based)
        
    Returns:
        Tuple of (success, frame_array)
    """
    try:
        # Create temporary file for the extracted frame
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Use ffmpeg to extract specific frame
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'select=eq(n\\,{frame_number})',
            '-vframes', '1',
            '-f', 'image2',
            '-y',  # Overwrite output
            temp_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(temp_path):
            # Read the extracted frame
            frame = cv2.imread(temp_path)
            os.unlink(temp_path)  # Clean up temp file
            
            if frame is not None:
                return True, frame
            else:
                return False, None
        else:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return False, None
            
    except Exception as e:
        print(f"Error extracting frame {frame_number}: {e}")
        return False, None


def get_video_info_ffmpeg(video_path: str) -> Dict:
    """
    Get video information using ffprobe to handle all codecs.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in data['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
            
            if video_stream:
                info = {
                    'frame_count': int(video_stream.get('nb_frames', 0)),
                    'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                    'width': int(video_stream.get('width', 0)),
                    'height': int(video_stream.get('height', 0)),
                    'codec': video_stream.get('codec_name', 'unknown'),
                    'duration': float(video_stream.get('duration', 0))
                }
                return info
        
        return {}
        
    except Exception as e:
        print(f"Error getting video info: {e}")
        return {}


def random_frame_access(video_path: str, num_accesses: int = 10) -> List[Dict]:
    """
    Perform random frame access and record IO operations.
    Uses ffmpeg for reliable frame extraction with all codecs.
    
    Args:
        video_path: Path to video file
        num_accesses: Number of random frame accesses to perform
        
    Returns:
        List of access results with IO statistics
    """
    # Get video information using ffprobe
    video_info = get_video_info_ffmpeg(video_path)
    
    if not video_info:
        print(f"Error: Could not get video information for {video_path}")
        return []
    
    frame_count = video_info['frame_count']
    fps = video_info['fps']
    codec = video_info['codec']
    
    print(f"Video info: {frame_count} frames, {fps:.2f} FPS, codec: {codec}")
    
    if frame_count == 0:
        print("Error: Video has no frames")
        return []
    
    results = []
    
    for i in range(num_accesses):
        # Generate random frame number
        frame_number = random.randint(0, frame_count - 1)
        
        print(f"Access {i+1}/{num_accesses}: Reading frame {frame_number}")
        
        # Start profiling
        profiler = IOProfiler()
        profiler.start_profiling()
        
        # Get initial IO stats
        initial_stats = profiler.get_io_stats()
        
        # Extract frame using ffmpeg
        success, frame = extract_frame_with_ffmpeg(video_path, frame_number)
        
        # Get final IO stats
        final_stats = profiler.get_io_stats()
        profiler.stop_profiling()
        
        # Calculate IO differences
        io_diff = {}
        if initial_stats and final_stats:
            io_diff = {
                'read_count_diff': final_stats['read_count'] - initial_stats['read_count'],
                'write_count_diff': final_stats['write_count'] - initial_stats['write_count'],
                'read_bytes_diff': final_stats['read_bytes'] - initial_stats['read_bytes'],
                'write_bytes_diff': final_stats['write_bytes'] - initial_stats['write_bytes'],
            }
        
        # Get summary
        summary = profiler.get_summary()
        
        result = {
            'frame_number': frame_number,
            'success': success,
            'frame_shape': frame.shape if success and frame is not None else None,
            'io_stats': io_diff,
            'profiling_summary': summary,
            'timestamp': time.time()
        }
        
        results.append(result)
        
        if success and frame is not None:
            print(f"  Successfully read frame {frame_number}, shape: {frame.shape}")
        else:
            print(f"  Failed to read frame {frame_number}")
    
    return results


def print_results_summary(results: List[Dict]):
    """Print a summary of the profiling results."""
    if not results:
        print("No results to summarize")
        return
    
    print("\n" + "="*60)
    print("PROFILING RESULTS SUMMARY")
    print("="*60)
    
    successful_accesses = [r for r in results if r['success']]
    failed_accesses = [r for r in results if not r['success']]
    
    print(f"Total accesses: {len(results)}")
    print(f"Successful: {len(successful_accesses)}")
    print(f"Failed: {len(failed_accesses)}")
    
    if successful_accesses:
        print("\nIO Statistics (successful accesses only):")
        
        # Calculate totals
        total_read_count = sum(r['io_stats'].get('read_count_diff', 0) for r in successful_accesses)
        total_write_count = sum(r['io_stats'].get('write_count_diff', 0) for r in successful_accesses)
        total_read_bytes = sum(r['io_stats'].get('read_bytes_diff', 0) for r in successful_accesses)
        total_write_bytes = sum(r['io_stats'].get('write_bytes_diff', 0) for r in successful_accesses)
        
        print(f"  Total read operations: {total_read_count}")
        print(f"  Total write operations: {total_write_count}")
        print(f"  Total bytes read: {total_read_bytes:,}")
        print(f"  Total bytes written: {total_write_bytes:,}")
        
        # Calculate averages
        avg_read_count = total_read_count / len(successful_accesses)
        avg_write_count = total_write_count / len(successful_accesses)
        avg_read_bytes = total_read_bytes / len(successful_accesses)
        avg_write_bytes = total_write_bytes / len(successful_accesses)
        
        print(f"\nPer-access averages:")
        print(f"  Average read operations: {avg_read_count:.2f}")
        print(f"  Average write operations: {avg_write_count:.2f}")
        print(f"  Average bytes read: {avg_read_bytes:,.0f}")
        print(f"  Average bytes written: {avg_write_bytes:,.0f}")
        
        # Show individual access details
        print(f"\nIndividual access details:")
        for i, result in enumerate(successful_accesses):
            io_stats = result['io_stats']
            print(f"  Access {i+1}: Frame {result['frame_number']}")
            print(f"    Read ops: {io_stats.get('read_count_diff', 0)}, "
                  f"Write ops: {io_stats.get('write_count_diff', 0)}")
            print(f"    Read bytes: {io_stats.get('read_bytes_diff', 0):,}, "
                  f"Write bytes: {io_stats.get('write_bytes_diff', 0):,}")


def main():
    """Main function to run the profiling script."""
    # Input and output file paths
    input_file = "file-000.mp4"
    output_file = "file-000.mkv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found")
        print("Please ensure file-000.mp4 exists in the current directory")
        return 1
    
    # Convert MP4 to MKV
    print("Step 1: Converting MP4 to MKV format")
    if not convert_mp4_to_mkv(input_file, output_file):
        print("Conversion failed. Exiting.")
        return 1
    
    # Get video information
    print("\nStep 2: Getting video information")
    video_info = get_video_info_ffmpeg(output_file)
    if not video_info:
        print("Failed to get video information. Exiting.")
        return 1
    
    print(f"Video information:")
    for key, value in video_info.items():
        print(f"  {key}: {value}")
    
    # Perform random frame access profiling
    print(f"\nStep 3: Performing random frame access profiling")
    num_accesses = 10  # Number of random frame accesses
    results = random_frame_access(output_file, num_accesses)
    
    if not results:
        print("No profiling results obtained. Exiting.")
        return 1
    
    # Print results summary
    print_results_summary(results)
    
    print(f"\nProfiling complete! Results saved for {len(results)} accesses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
