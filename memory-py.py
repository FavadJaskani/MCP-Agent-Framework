from typing import Dict, Any, List, Optional
import json
from datetime import datetime

class MemoryTool:
    """Tool for storing and retrieving information from long-term memory."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.name = "memory"
        self.description = "Store and retrieve information from long-term memory"
        self.storage_path = storage_path
        self.memory = {}  # In-memory storage (would use a database in production)
    
    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute a memory operation."""
        if action == "store":
            return self._store_memory(**kwargs)
        elif action == "retrieve":
            return self._retrieve_memory(**kwargs)
        elif action == "search":
            return self._search_memory(**kwargs)
        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}. Supported actions: store, retrieve, search"
            }
    
    def _store_memory(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Store information in memory."""
        entry = {
            "value": value,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.memory[key] = entry
        
        return {
            "status": "success",
            "key": key,
            "message": f"Successfully stored information under key: {key}"
        }
    
    def _retrieve_memory(self, key: str) -> Dict[str, Any]:
        """Retrieve information from memory."""
        if key not in self.memory:
            return {
                "status": "error",
                "error": f"No information found for key: {key}"
            }
        
        entry = self.memory[key]
        
        return {
            "status": "success",
            "key": key,
            "value": entry["value"],
            "metadata": entry["metadata"],
            "timestamp": entry["timestamp"]
        }
    
    def _search_memory(self, query: str) -> Dict[str, Any]:
        """Search for information in memory."""
        # This would use vector search in a real implementation
        # For this example, just do a simple text search
        results = []
        
        for key, entry in self.memory.items():
            value_str = str(entry["value"])
            if query.lower() in value_str.lower():
                results.append({
                    "key": key,
                    "value": entry["value"],
                    "metadata": entry["metadata"],
                    "timestamp": entry["timestamp"]
                })
        
        return {
            "status": "success",
            "query": query,
            "results": results,
            "count": len(results)
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "action": "The memory operation to perform (store, retrieve, search)",
                "key": "The key to store or retrieve information",
                "value": "The value to store (for store action)",
                "metadata": "Additional metadata for the stored information",
                "query": "Search query (for search action)"
            }
        }