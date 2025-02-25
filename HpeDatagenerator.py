import os
import random
import threading
import time
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

class StorageSimulator:
    def __init__(self, mount_points_config):
        self.mount_points_config = mount_points_config
        self.running = False
        self.file_locks = {}
        self.lock_dict_lock = threading.Lock()
        self.size_check_interval = 300  # Check directory sizes every 5 minutes
        self.protected_paths = set(config['path'] for config in mount_points_config)
        
        # File type configurations
        self.file_types = {
            'text': {
                'extensions': ['.txt', '.log', '.csv', '.json', '.xml'],
                'weight': 0.3
            },
            'image': {
                'extensions': ['.jpg', '.png', '.gif', '.bmp'],
                'weight': 0.3
            },
            'document': {
                'extensions': ['.pdf', '.doc', '.docx', '.xlsx'],
                'weight': 0.2
            },
            'video': {
                'extensions': ['.mp4', '.avi', '.mkv'],
                'weight': 0.1
            },
            'binary': {
                'extensions': ['.bin', '.dat'],
                'weight': 0.1
            }
        }
        
        # Directory size tracking
        self.dir_sizes = {}
        self.dir_sizes_lock = threading.Lock()
        
        # Ensure mount directories exist
        for mount in mount_points_config:
            Path(mount['path']).mkdir(parents=True, exist_ok=True)
            self.update_directory_size(mount['path'])

    def get_directory_size_rclone(self, path):
        """Get directory size using rclone size command."""
        try:
            cmd = ['rclone', 'size', f"gdrive:{path}", '--json']
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                size_data = json.loads(result.stdout)
                return size_data.get('bytes', 0)
            return 0
        except Exception as e:
            print(f"Error getting directory size: {e}")
            return 0

    def update_directory_size(self, path):
        """Update the stored directory size."""
        with self.dir_sizes_lock:
            self.dir_sizes[path] = self.get_directory_size_rclone(path)

    def is_directory_full(self, path, mount_config):
        """Check if directory has reached its size cap."""
        with self.dir_sizes_lock:
            current_size = self.dir_sizes.get(path, 0)
            return current_size >= mount_config['size_cap']

    def size_monitor(self):
        """Periodically monitor directory sizes."""
        while self.running:
            for mount_config in self.mount_points_config:
                self.update_directory_size(mount_config['path'])
            time.sleep(self.size_check_interval)

    def get_random_size_with_delay(self, mount_config):
        """Generate random file size with weighted probability and corresponding delay."""
        sizes = [
            (1024 * 10, 1024 * 500, 0.4, (30, 60)),       # 10KB - 500KB: 30sec - 1min
            (1024 * 500, 1024 * 1024 * 5, 0.3, (120, 180)),  # 500KB - 5MB: 2-3 minutes
            (1024 * 1024 * 5, 1024 * 1024 * 100, 0.2, (300, 300)),  # 5MB - 100MB: 5 minutes
            (1024 * 1024 * 100, 1024 * 1024 * 1024, 0.1, (600, 900))  # 100MB - 1GB: 10-15 minutes
        ]
        
        adjusted_sizes = []
        for min_size, max_size, base_prob, delay in sizes:
            adjusted_prob = base_prob * mount_config['size_weights'].get(
                f"{min_size}-{max_size}", 1.0)
            adjusted_sizes.append((min_size, max_size, adjusted_prob, delay))
        
        total_prob = sum(size[2] for size in adjusted_sizes)
        adjusted_sizes = [(s[0], s[1], s[2]/total_prob, s[3]) for s in adjusted_sizes]
        
        rand = random.random()
        cumulative = 0
        for min_size, max_size, prob, delay in adjusted_sizes:
            cumulative += prob
            if rand <= cumulative:
                return random.randint(min_size, max_size), delay
        
        return random.randint(1024 * 100, 1024 * 1024), (300, 300)

    def get_file_lock(self, filepath):
        """Get or create a lock for a specific file with thread safety."""
        with self.lock_dict_lock:
            if filepath not in self.file_locks:
                self.file_locks[filepath] = threading.Lock()
            return self.file_locks[filepath]

    def remove_file_lock(self, filepath):
        """Remove lock for deleted files."""
        with self.lock_dict_lock:
            self.file_locks.pop(filepath, None)

    def get_random_file_type(self):
        """Get random file type and extension based on weights."""
        file_type = random.choices(
            list(self.file_types.keys()),
            weights=[t['weight'] for t in self.file_types.values()]
        )[0]
        extension = random.choice(self.file_types[file_type]['extensions'])
        return extension

    def is_protected_path(self, path):
        """Check if the given path is a protected directory."""
        return path in self.protected_paths

    def create_file(self, mount_point, mount_config):
        """Create a new file with random size and type."""
        try:
            # Check if directory is full before creating new file
            if self.is_directory_full(mount_point, mount_config):
                print(f"Directory {mount_point} is full. Skipping file creation.")
                return None, (60, 120)

            size, delay = self.get_random_size_with_delay(mount_config)
            extension = self.get_random_file_type()
            filename = f"file_{int(time.time())}_{random.randint(1000, 9999)}{extension}"
            filepath = os.path.join(mount_point, filename)
            
            with open(filepath, 'wb') as f:
                chunk_size = min(1024 * 1024, size)
                remaining = size
                
                while remaining > 0:
                    write_size = min(chunk_size, remaining)
                    f.write(os.urandom(write_size))
                    remaining -= write_size
            
            # Update directory size after creation
            self.update_directory_size(mount_point)
            return filepath, delay
            
        except Exception as e:
            print(f"Error creating file: {e}")
            return None, (60, 120)

    def update_file(self, filepath):
        """Update existing file with random data."""
        # Skip if path is protected
        if self.is_protected_path(filepath):
            return False

        file_lock = self.get_file_lock(filepath)
        if not file_lock.acquire(timeout=1):
            return False
        
        try:
            if not os.path.exists(filepath):
                return False
                
            current_size = os.path.getsize(filepath)
            modification = random.choice(['append', 'modify', 'truncate'])
            
            with open(filepath, 'rb+') as f:
                if modification == 'append':
                    f.seek(0, 2)  # Seek to end
                    append_size = random.randint(1024, 1024 * 1024)  # 1KB to 1MB
                    f.write(os.urandom(append_size))
                
                elif modification == 'modify' and current_size > 0:
                    position = random.randint(0, current_size - 1)
                    modify_size = min(1024 * 1024, current_size - position)
                    f.seek(position)
                    f.write(os.urandom(modify_size))
                
                elif modification == 'truncate':
                    new_size = random.randint(1024, current_size)
                    f.truncate(new_size)
            
            # Update directory size after modification
            mount_point = os.path.dirname(filepath)
            self.update_directory_size(mount_point)
            return True
            
        except Exception as e:
            print(f"Error updating file: {e}")
            return False
            
        finally:
            file_lock.release()

    def delete_file(self, filepath):
        """Delete a file with proper lock handling."""
        # Skip if path is protected
        if self.is_protected_path(filepath):
            return False

        file_lock = self.get_file_lock(filepath)
        if not file_lock.acquire(timeout=1):
            return False
        
        try:
            if os.path.exists(filepath) and not self.is_protected_path(filepath):
                os.remove(filepath)
                self.remove_file_lock(filepath)
                # Update directory size after deletion
                mount_point = os.path.dirname(filepath)
                self.update_directory_size(mount_point)
                return True
            return False
            
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False
            
        finally:
            file_lock.release()

    def get_existing_files(self, mount_point):
        """Get list of existing files in mount point."""
        try:
            return [os.path.join(mount_point, f) for f in os.listdir(mount_point)
                   if os.path.isfile(os.path.join(mount_point, f)) and 
                   not self.is_protected_path(os.path.join(mount_point, f))]
        except Exception:
            return []

    def simulate_mount_point(self, mount_config):
        """Simulate file operations for a specific mount point with multiple threads."""
        mount_point = mount_config['path']
        
        def worker():
            while self.running:
                try:
                    existing_files = self.get_existing_files(mount_point)
                    num_files = len(existing_files)
                    
                    # Check if directory is full
                    is_full = self.is_directory_full(mount_point, mount_config)
                    
                    # Adjust operations based on directory fullness
                    if is_full:
                        # Only allow update and delete operations
                        operation = random.choices(
                            ['update', 'delete'],
                            weights=[0.6, 0.4]
                        )[0]
                    else:
                        operation = random.choices(
                            ['create', 'update', 'delete'],
                            weights=[
                                mount_config['operation_weights']['create'],
                                mount_config['operation_weights']['update'],
                                mount_config['operation_weights']['delete']
                            ]
                        )[0]
                    
                    delay_range = (60, 120)  # default delay
                    
                    if operation == 'create' and not is_full:
                        filepath, delay_range = self.create_file(mount_point, mount_config)
                    
                    elif operation == 'update' and existing_files:
                        filepath = random.choice(existing_files)
                        self.update_file(filepath)
                        delay_range = (mount_config['update_delay_min'], 
                                     mount_config['update_delay_max'])
                    
                    elif operation == 'delete' and existing_files:
                        filepath = random.choice(existing_files)
                        self.delete_file(filepath)
                        delay_range = (mount_config['delete_delay_min'], 
                                     mount_config['delete_delay_max'])
                    
                    time.sleep(random.uniform(*delay_range))
                    
                except Exception as e:
                    print(f"Error in worker: {e}")
                    time.sleep(5)
        
        num_workers = mount_config['num_workers']
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for _ in range(num_workers):
                executor.submit(worker)

    def start(self):
        """Start storage simulation across all mount points."""
        self.running = True
        
        # Start size monitoring thread
        threading.Thread(target=self.size_monitor, daemon=True).start()
        
        with ThreadPoolExecutor(max_workers=len(self.mount_points_config)) as executor:
            for mount_config in self.mount_points_config:
                executor.submit(self.simulate_mount_point, mount_config)

    def stop(self):
        """Stop storage simulation."""
        self.running = False

if __name__ == "__main__":
    mount_points_config = [
        {
            'path': '/customers',
            'num_workers': 2,
            'size_cap': 2.5 * 1024 * 1024 * 1024 * 1024,  # 2.5TB
            'operation_weights': {'create': 0.4, 'update': 0.4, 'delete': 0.2},  # Balanced operations
            'size_weights': {
                '10240-512000': 1.2,  # Favor smaller files
                '512000-5242880': 1.0,
                '5242880-104857600': 0.8,
                '104857600-1073741824': 0.5
            },
            'update_delay_min': 30,  # 30 seconds
            'update_delay_max': 60,   # 1 minute
            'delete_delay_min': 30,
            'delete_delay_max': 60
        },
        
       
        {
            'path': '/scratch',
            'num_workers': 3,
            'size_cap': 2.5 * 1024 * 1024 * 1024 * 1024,  # 2.5TB
            'operation_weights': {'create': 0.5, 'update': 0.3, 'delete': 0.2},
            'size_weights': {
                '10240-512000': 0.8,
                '512000-5242880': 1.2,  # Favor medium files
                '5242880-104857600': 1.0,
                '104857600-1073741824': 0.6
            },
            'update_delay_min': 120,  # 2 minutes
            'update_delay_max': 180,  # 3 minutes
            'delete_delay_min': 120,
            'delete_delay_max': 180
        },
        {
            'path': '/projects',
            'num_workers': 2,
            'size_cap': 2.5 * 1024 * 1024 * 1024 * 1024,  # 2.5TB
            'operation_weights': {'create': 0.35, 'update': 0.45, 'delete': 0.2},
            'size_weights': {
                '10240-512000': 0.6,
                '512000-5242880': 0.8,
                '5242880-104857600': 1.2,  # Favor larger files
                '104857600-1073741824': 1.0
            },
            'update_delay_min': 240,  # 4 minutes
            'update_delay_max': 240,  # 4 minutes
            'delete_delay_min': 240,
            'delete_delay_max': 240
        },
        {
            'path': '/info',
            'num_workers': 2,  # Increased workers for higher frequency
            'size_cap': 2.5 * 1024 * 1024 * 1024 * 1024,  # 2.5TB
            'operation_weights': {'create': 0.4, 'update': 0.4, 'delete': 0.2},
            'size_weights': {
                '10240-512000': 0.5,
                '512000-5242880': 0.7,
                '5242880-104857600': 1.0,
                '104857600-1073741824': 1.2  # Favor very large files
            },
            'update_delay_min': 60,  # 1 minute
            'update_delay_max': 60,  # 1 minute
            'delete_delay_min': 60,
            'delete_delay_max': 60
        }
    ]
    
    simulator = StorageSimulator(mount_points_config)
    
    try:
        print("Starting storage simulator...")
        print("\nDirectory size caps:")
        for config in mount_points_config:
            print(f"{config['path']}: {config['size_cap'] / (1024**4):.1f}TB")
        
        print("\nFile generation frequencies:")
        print("/customers: 30-60 seconds")
        print("/scratch: 2-3 minutes")
        print("/projects: 4 minutes")
        print("/info: 1 minute")
        
        simulator.start()
        
        # Main loop with size monitoring display
        while True:
            time.sleep(60)  # Update display every minute
            print("\nCurrent directory sizes:")
            for mount_config in mount_points_config:
                path = mount_config['path']
                current_size = simulator.dir_sizes.get(path, 0)
                size_cap = mount_config['size_cap']
                usage_percent = (current_size / size_cap) * 100
                print(f"{path}: {current_size / (1024**4):.2f}TB / {size_cap / (1024**4):.1f}TB ({usage_percent:.1f}%)")
    
    except KeyboardInterrupt:
        print("\nStopping storage simulator...")
        simulator.stop()
    except Exception as e:
        print(f"\nError in main: {e}")
        simulator.stop()
    finally:
        print("Simulation ended")