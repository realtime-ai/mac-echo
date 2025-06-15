

import asyncio
from typing import AsyncIterator, TypeVar


T = TypeVar('T')


class QueueIterator(AsyncIterator[T]):
    def __init__(self, queue: asyncio.Queue[T]):
        self.queue = queue

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            item = await self.queue.get()
            if item is None:  # 使用 None 作为结束标志
                raise StopAsyncIteration
            return item
        finally:
            self.queue.task_done()
