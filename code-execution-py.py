import subprocess
from typing import Dict, Any, Optional

class CodeExecutionTool:
    """Tool for executing code in various languages."""
    
    def __init__(self, allowed_languages: Optional[list] = None):
        self.name = "code_execution"
        self.description = "Execute code in various languages"
        self.allowed_languages = allowed_languages or ["python", "javascript", "bash"]
    
    def execute(self, code: str, language: str = "python", timeout: int = 10) -> Dict[str, Any]:
        """Execute code and return the result."""
        if language not in self.allowed_languages:
            return {
                "status": "error",
                "error": f"Language {language} is not supported. Allowed languages: {', '.join(self.allowed_languages)}"
            }
        
        # IMPORTANT: In a real implementation, this would need security measures
        # like sandboxing to prevent malicious code execution
        
        # For this example, only simulate the execution
        return self._simulate_execution(code, language)
    
    def _simulate_execution(self, code: str, language: str) -> Dict[str, Any]:
        """Simulate code execution (for safety in this example)."""
        return {
            "status": "success",
            "language": language,
            "output": f"[Simulated {language} execution output for safety]\nCode would run in a sandbox in a real implementation.",
            "execution_time": 0.5
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "code": "The code to execute",
                "language": f"The programming language (allowed: {', '.join(self.allowed_languages)})",
                "timeout": "Maximum execution time in seconds (default: 10)"
            }
        }