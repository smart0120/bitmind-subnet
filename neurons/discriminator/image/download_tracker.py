"""
SQLite database for tracking dataset download and extraction status.
"""

import sqlite3
import os
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime


class DownloadTracker:
    """Track dataset download and extraction status in SQLite database."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize download tracker.
        
        Args:
            db_path: Path to SQLite database file (default: ./datasets_cache/download_tracker.db)
        """
        if db_path is None:
            db_path = Path("./datasets_cache/download_tracker.db")
        else:
            db_path = Path(db_path)
        
        # Ensure parent directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create downloads table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS downloads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                status TEXT NOT NULL,
                images_extracted INTEGER DEFAULT 0,
                output_dir TEXT,
                downloaded_at TIMESTAMP,
                extracted_at TIMESTAMP,
                error_message TEXT,
                UNIQUE(dataset_name, file_name)
            )
        """)
        
        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_dataset_file 
            ON downloads(dataset_name, file_name)
        """)
        
        conn.commit()
        conn.close()
    
    def is_downloaded(self, dataset_name: str, file_name: str) -> bool:
        """
        Check if a file has been successfully downloaded and extracted.
        
        Args:
            dataset_name: Dataset name (e.g., "bitmind/ffhq-256")
            file_name: File name (e.g., "train-00000.parquet")
        
        Returns:
            True if file is downloaded and extracted successfully
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT status FROM downloads
            WHERE dataset_name = ? AND file_name = ?
        """, (dataset_name, file_name))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0] == "completed"
        return False
    
    def get_downloaded_files(self, dataset_name: str) -> List[Tuple[str, int]]:
        """
        Get list of successfully downloaded files for a dataset.
        
        Args:
            dataset_name: Dataset name
        
        Returns:
            List of tuples (file_name, images_extracted)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT file_name, images_extracted FROM downloads
            WHERE dataset_name = ? AND status = 'completed'
            ORDER BY file_name
        """, (dataset_name,))
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def get_image_paths(self, dataset_name: str, output_dir: Path) -> List[str]:
        """
        Get all image paths for a dataset from the database.
        
        Args:
            dataset_name: Dataset name
            output_dir: Output directory where images are stored
        
        Returns:
            List of image file paths
        """
        downloaded_files = self.get_downloaded_files(dataset_name)
        image_paths = []
        
        # Find all images in output directory
        if output_dir.exists():
            # Images are stored as: {dataset_name}_{hash}.jpg
            dataset_prefix = dataset_name.replace("/", "_")
            image_files = list(output_dir.glob(f"{dataset_prefix}_*.jpg")) + \
                         list(output_dir.glob(f"{dataset_prefix}_*.png"))
            image_paths = [str(p) for p in image_files]
        
        return image_paths
    
    def mark_downloading(self, dataset_name: str, file_name: str, file_type: str, output_dir: str):
        """
        Mark a file as being downloaded.
        
        Args:
            dataset_name: Dataset name
            file_name: File name
            file_type: File type (parquet, zip, jpg, etc.)
            output_dir: Output directory path
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO downloads
            (dataset_name, file_name, file_type, status, output_dir, downloaded_at)
            VALUES (?, ?, ?, 'downloading', ?, ?)
        """, (dataset_name, file_name, file_type, output_dir, datetime.now()))
        
        conn.commit()
        conn.close()
    
    def mark_completed(self, dataset_name: str, file_name: str, images_extracted: int):
        """
        Mark a file as successfully downloaded and extracted.
        
        Args:
            dataset_name: Dataset name
            file_name: File name
            images_extracted: Number of images extracted from this file
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE downloads
            SET status = 'completed',
                images_extracted = ?,
                extracted_at = ?
            WHERE dataset_name = ? AND file_name = ?
        """, (images_extracted, datetime.now(), dataset_name, file_name))
        
        conn.commit()
        conn.close()
    
    def mark_failed(self, dataset_name: str, file_name: str, error_message: str):
        """
        Mark a file download/extraction as failed.
        
        Args:
            dataset_name: Dataset name
            file_name: File name
            error_message: Error message
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE downloads
            SET status = 'failed',
                error_message = ?
            WHERE dataset_name = ? AND file_name = ?
        """, (error_message, dataset_name, file_name))
        
        conn.commit()
        conn.close()
    
    def get_dataset_stats(self, dataset_name: str) -> dict:
        """
        Get statistics for a dataset.
        
        Args:
            dataset_name: Dataset name
        
        Returns:
            Dictionary with stats: total_files, completed_files, failed_files, total_images
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_files,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_files,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_files,
                SUM(images_extracted) as total_images
            FROM downloads
            WHERE dataset_name = ?
        """, (dataset_name,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'total_files': result[0] or 0,
                'completed_files': result[1] or 0,
                'failed_files': result[2] or 0,
                'total_images': result[3] or 0
            }
        return {'total_files': 0, 'completed_files': 0, 'failed_files': 0, 'total_images': 0}
    
    def clear_dataset(self, dataset_name: str):
        """
        Clear all records for a dataset (useful for force re-download).
        
        Args:
            dataset_name: Dataset name
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM downloads WHERE dataset_name = ?
        """, (dataset_name,))
        
        conn.commit()
        conn.close()
