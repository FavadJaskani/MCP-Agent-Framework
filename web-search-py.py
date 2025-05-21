from typing import Dict, Any, List, Optional

class WebSearchTool:
    """Tool for performing web searches."""
    
    def __init__(self, api_key: Optional[str] = None, search_engine: str = "google"):
        self.name = "web_search"
        self.description = "Search the web for information"
        self.api_key = api_key
        self.search_engine = search_engine
    
    def execute(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Perform a web search."""
        # In a real implementation, this would call a search API
        # For this example, return mock results
        results = [
            {
                "title": f"Result {i+1} for: {query}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a snippet of text from search result {i+1} for the query: {query}..."
            }
            for i in range(num_results)
        ]
        
        return {
            "query": query,
            "results": results,
            "num_results": len(results)
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "query": "The search query string",
                "num_results": "Number of results to return (default: 5)"
            }
        }