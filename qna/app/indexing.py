import logging
import os
import time
import uuid

from pydantic import BaseModel
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

from app.db import save_indexing


async def run_indexing_manually():
    indexing_id = str(uuid.uuid4())
    await save_indexing(indexing_id, 'started', 'manually')


async def run_indexing_auto():
    indexing_id = str(uuid.uuid4())
    await save_indexing(indexing_id, 'started', 'auto')


file_endings = ('.pdf', '.docx', '.csv')


# Модель для представления событий изменения файлов
class FileChangeEvent(BaseModel):
    event_type: str
    src_path: str
    timestamp: float


class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(file_endings):
            logger.info(f'File modified: {event.src_path}')

    def on_created(self, event):
        if event.src_path.endswith(file_endings):
            logger.info(f'File created: {event.src_path}')

    def on_deleted(self, event):
        if event.src_path.endswith(file_endings):
            logger.info(f'File deleted: {event.src_path}')


def start_observer():
    path = os.getenv('DATASET_DIR_PATH')
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    logger.info(f'Started file watching in directory: {path}')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
