import asyncio
import logging
import os
import time
import uuid

from pydantic import BaseModel
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app.db import save_indexing, set_indexing
from app.pipeline import index_store

logger = logging.getLogger(__name__)


async def run_indexing(indexing_id: str):
    try:
        await index_store()
        await set_indexing(indexing_id, 'finished')
    except Exception as exc:
        logger.error(f'Indexing failed for indexing_id="{indexing_id}": {exc}')
        await set_indexing(indexing_id, 'failed')


async def run_indexing_manually():
    indexing_id = str(uuid.uuid4())
    await save_indexing(indexing_id, 'started', 'manually')
    logger.info(f'Run manually indexing: indexing_id="{indexing_id}"')
    asyncio.create_task(run_indexing(indexing_id))


async def run_indexing_auto():
    indexing_id = str(uuid.uuid4())
    await save_indexing(indexing_id, 'started', 'auto')
    logger.info(f'Run auto indexing: indexing_id="{indexing_id}"')
    asyncio.create_task(run_indexing(indexing_id))


def run_async_task(task):
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.create_task(task)
    else:
        loop.run_until_complete(task)


file_endings = ('.pdf', '.docx', '.csv')


# Модель для представления событий изменения файлов
class FileChangeEvent(BaseModel):
    event_type: str
    src_path: str
    timestamp: float


class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(file_endings):
            logger.info(f'File modified: {event.src_path}')
            run_async_task(run_indexing_auto())

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(file_endings):
            logger.info(f'File created: {event.src_path}')
            run_async_task(run_indexing_auto())

    def on_deleted(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(file_endings):
            logger.info(f'File deleted: {event.src_path}')


def start_observer():
    path = os.getenv('DATASET_DIR_PATH')
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    logger.info(f'Started file watching in directory: {path}')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
