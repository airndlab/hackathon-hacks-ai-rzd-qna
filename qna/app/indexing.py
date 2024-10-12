import uuid

from app.db import save_indexing


async def run_indexing_manually():
    indexing_id = str(uuid.uuid4())
    await save_indexing(indexing_id, 'started', 'manually')


async def run_indexing_auto():
    indexing_id = str(uuid.uuid4())
    await save_indexing(indexing_id, 'started', 'auto')
