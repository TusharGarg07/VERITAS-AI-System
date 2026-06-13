from typing import Dict, Any
from veritas.utils.logger import logger

class ContextManager:
    def __init__(self):
        self._context: Dict[str, Any] = {}

    def get_context(self, session_id: str) -> Dict[str, Any]:
        try:
            return self._context.get(session_id, {})
        except Exception as e:
            logger.error(f"Failed to get context for {session_id}: {str(e)}")
            return {}

    def update_context(self, session_id: str, data: Dict[str, Any]):
        try:
            if session_id not in self._context:
                self._context[session_id] = {}
            self._context[session_id].update(data)
        except Exception as e:
            logger.error(f"Failed to update context for {session_id}: {str(e)}")

context_manager = ContextManager()
