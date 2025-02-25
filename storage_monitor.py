import os
import pymongo
from datetime import datetime, timedelta, timezone

class StorageMetricsCollector:
    def __init__(self, mongo_uri: str, directories: list):
        self.directories = directories
        self.mongo_client = pymongo.MongoClient(mongo_uri)
        self.db = self.mongo_client['storage_monitoring']
        self.snapshots_collection = self.db['file_snapshots']
        self.metrics_collection = self.db['storage_metrics']

    def get_file_list(self, directory: str) -> list:
        """Fetch file details (size, modification time) in a directory."""
        file_data = []
        total_size = 0
        total_files = 0

        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    stats = os.stat(file_path)
                    total_size += stats.st_size
                    total_files += 1
                    file_data.append({
                        "file_path": file_path,
                        "size": round(stats.st_size, 5),
                        "last_modified": datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc)
                    })
                except Exception as e:
                    print(f"Error accessing {file_path}: {str(e)}")
        
        return file_data, round(total_size, 5), total_files

    def store_snapshot(self):
        """Take a snapshot of all files and store in MongoDB."""
        current_time = datetime.now(timezone.utc)

        for directory in self.directories:
            file_list, total_size, total_files = self.get_file_list(directory)
            self.snapshots_collection.insert_one({
                "timestamp": current_time,
                "directory": directory,
                "total_size": round(total_size, 5),
                "total_files": total_files,
                "files": file_list
            })

    def compute_file_changes(self, prev_snapshot, current_snapshot):
        """Compare two snapshots to determine added, deleted, and modified files."""
        prev_files_dict = {f['file_path']: f for f in prev_snapshot.get('files', [])}
        current_files_dict = {f['file_path']: f for f in current_snapshot.get('files', [])}

        files_added = 0
        files_deleted = 0
        files_modified = 0
        size_added = 0
        size_deleted = 0
        size_modified = 0

        # Identify added files
        for file_path, file_info in current_files_dict.items():
            if file_path not in prev_files_dict:
                files_added += 1
                size_added += file_info['size']
            elif file_info['size'] != prev_files_dict[file_path]['size']:
                files_modified += 1
                size_modified += file_info['size'] - prev_files_dict[file_path]['size']

        # Identify deleted files
        for file_path, file_info in prev_files_dict.items():
            if file_path not in current_files_dict:
                files_deleted += 1
                size_deleted += file_info['size']

        return (
            files_added, files_deleted, files_modified,
            round(size_added, 5), round(size_deleted, 5), round(size_modified, 5)
        )

    def process_metrics(self):
        """Calculate file changes and store in MongoDB."""
        current_time = datetime.now(timezone.utc)

        for directory in self.directories:
            # Get the last two snapshots
            snapshots = list(self.snapshots_collection.find({"directory": directory}).sort("timestamp", -1).limit(2))

            if len(snapshots) < 2:
                print(f"Not enough snapshots for {directory}. Skipping calculation.")
                continue

            prev_snapshot, current_snapshot = snapshots[1], snapshots[0]

            # Compute file changes
            files_added, files_deleted, files_modified, size_added, size_deleted, size_modified = self.compute_file_changes(
                prev_snapshot, current_snapshot
            )

            # Compute total space used
            current_space = round(current_snapshot.get("total_size", 0) / (1024 ** 3), 5)  # Convert to GB
            total_files = current_snapshot.get("total_files", 0)

            # Store computed metrics in MongoDB
            self.metrics_collection.insert_one({
                "timestamp": current_time,
                "directory": directory,
                "files_added": files_added,
                "files_deleted": files_deleted,
                "files_modified": files_modified,
                "size_added_gb": round(size_added / (1024 ** 3), 5),
                "size_deleted_gb": round(size_deleted / (1024 ** 3), 5),
                "size_modified_gb": round(size_modified / (1024 ** 3), 5),
                "total_files": total_files,
                "current_space_gb": current_space
            })
            
            print("details of each directory")

            # Print current status
            print(f"""
Timestamp: {current_time}
Directory: {directory}
Total Files: {total_files}
Current Space Used: {current_space} GB
Files Added: {files_added} ({round(size_added / (1024 ** 3), 5)} GB)
Files Deleted: {files_deleted} ({round(size_deleted / (1024 ** 3), 5)} GB)
Files Modified: {files_modified} ({round(size_modified / (1024 ** 3), 5)} GB)
-------------------""")

    def close_connection(self):
        """Close MongoDB connection."""
        self.mongo_client.close()


def main():
    mongo_uri = "mongodb+srv://hpecpp:zz14OLIaQG7sC3cL@cluster0.nuoab.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    directories = ['/info', '/scratch', '/projects', '/customers']

    collector = StorageMetricsCollector(mongo_uri, directories)
    
    # Step 1: Store file snapshot
    collector.store_snapshot()
    
    # Step 2: Calculate file changes & store in storage_metrics
    collector.process_metrics()
    
    collector.close_connection()


if __name__ == "__main__":
    main()
