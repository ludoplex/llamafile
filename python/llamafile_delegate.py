#!/usr/bin/env python3
"""
llamafile_delegate.py - Python helper for delegating tasks to llamafile.

Designed for AI agents (like Claude) to offload simple tasks to a local LLM,
reducing API costs while maintaining quality for complex reasoning.

Example usage:
    from llamafile_delegate import LlamafileDelegate
    
    delegate = LlamafileDelegate(
        llamafile_path="./llamafile",
        model_path="./model.gguf"
    )
    
    # Simple prompt
    result = delegate.ask("Reformat this as bullet points: ...")
    
    # Generate JSON
    data = delegate.generate_json("user with name, email, role")
    
    # Summarize
    summary = delegate.summarize(long_text, max_words=100)
    
    # Format for platform
    formatted = delegate.format_for_platform(text, "discord")

Part of the ludoplex/llamafile fork.
"""
import subprocess
import json
import shutil
from pathlib import Path
from typing import Optional, Union, Dict, Any


class LlamafileDelegate:
    """
    Delegate simple tasks to a local llamafile instance.
    
    Designed for integration with AI agents to reduce API costs by
    handling simple tasks locally while preserving complex reasoning
    for cloud models.
    """
    
    # Task categories - delegate these to local LLM
    DELEGATABLE = [
        "text formatting",
        "simple summaries (<500 words)",
        "JSON/YAML generation",
        "template filling",
        "data extraction",
        "bullet point conversion",
        "platform formatting",
    ]
    
    # Keep these for cloud LLM (Claude, GPT, etc.)
    KEEP_CLOUD = [
        "complex reasoning",
        "multi-step planning",
        "tool use decisions",
        "code generation with context",
        "creative writing",
        "nuanced analysis",
    ]
    
    def __init__(
        self,
        llamafile_path: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
        gpu_layers: int = 99,
        timeout: int = 60,
    ):
        """
        Initialize the delegate.
        
        Args:
            llamafile_path: Path to llamafile binary. Auto-detects if None.
            model_path: Path to GGUF model. Required if not baked into llamafile.
            gpu_layers: Number of layers to offload to GPU (default: 99 = all).
            timeout: Timeout in seconds for inference (default: 60).
        """
        self.llamafile_path = self._find_llamafile(llamafile_path)
        self.model_path = self._find_model(model_path)
        self.gpu_layers = gpu_layers
        self.timeout = timeout
        
    def _find_llamafile(self, path: Optional[Union[str, Path]]) -> Path:
        """Find llamafile binary."""
        if path:
            return Path(path)
        
        # Try common locations
        workspace = Path(__file__).parent
        candidates = [
            workspace / "bin" / "llamafile.exe",
            workspace / "bin" / "llamafile",
            Path("./llamafile"),
            Path("./llamafile.exe"),
            Path.home() / ".openclaw" / "workspace" / "bin" / "llamafile.exe",
            Path.home() / ".local" / "bin" / "llamafile",
            Path("/usr/local/bin/llamafile"),
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        # Try PATH
        found = shutil.which("llamafile")
        if found:
            return Path(found)
        
        raise FileNotFoundError(
            "llamafile not found. Provide path or add to PATH."
        )
    
    def _find_model(self, path: Optional[Union[str, Path]]) -> Optional[Path]:
        """Find GGUF model file."""
        if path:
            return Path(path)
        
        # Try common locations
        workspace = Path(__file__).parent
        candidates = [
            workspace / "models",
            Path.home() / ".openclaw" / "workspace" / "models",
            Path.home() / ".cache" / "llamafile" / "models",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                # Find first .gguf file
                for gguf in candidate.glob("*.gguf"):
                    return gguf
        
        return None  # Model might be baked into llamafile
    
    def _build_prompt(self, prompt: str) -> str:
        """Wrap prompt in Qwen chat template."""
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    def _run(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str:
        """Run inference."""
        chat_prompt = self._build_prompt(prompt)
        
        cmd = [str(self.llamafile_path)]
        
        if self.model_path:
            cmd.extend(["-m", str(self.model_path)])
        
        cmd.extend([
            "-p", chat_prompt,
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "--no-display-prompt",
            "-ngl", str(self.gpu_layers),
            "-r", "<|im_end|>",
        ])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding='utf-8',
                errors='replace',  # Handle any encoding issues gracefully
            )
            
            if result.returncode != 0 and result.stderr:
                # Log error but don't crash
                import sys
                print(f"[llamafile warning] {result.stderr[:200]}", file=sys.stderr)
            
            if result.stdout:
                return result.stdout.strip()
            return ""
            
        except subprocess.TimeoutExpired:
            return "[error: inference timeout]"
        except Exception as e:
            return f"[error: {str(e)[:100]}]"
    
    def ask(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str:
        """
        Send a simple prompt to the local LLM.
        
        Args:
            prompt: The prompt to send.
            max_tokens: Maximum tokens to generate (default: 200).
            temperature: Sampling temperature (default: 0.7).
            
        Returns:
            The model's response.
        """
        return self._run(prompt, max_tokens, temperature)
    
    def generate_json(self, description: str) -> Dict[str, Any]:
        """
        Generate JSON from a description.
        
        Args:
            description: Natural language description of desired JSON.
            
        Returns:
            Parsed JSON as a dict, or {"error": ..., "raw": ...} on failure.
        """
        prompt = f"""Generate valid JSON for: {description}
Output only the JSON, no explanation:"""
        
        response = self._run(prompt, max_tokens=500, temperature=0.3)
        
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        
        return {"error": "Failed to parse JSON", "raw": response}
    
    def summarize(self, text: str, max_words: int = 100) -> str:
        """
        Summarize text.
        
        Args:
            text: Text to summarize.
            max_words: Maximum words in summary (default: 100).
            
        Returns:
            Summary string.
        """
        prompt = f"""Summarize this in {max_words} words or less:

{text}

Summary:"""
        return self._run(prompt, max_tokens=max_words * 2)
    
    def format_for_platform(self, text: str, platform: str) -> str:
        """
        Reformat text for a specific platform.
        
        Args:
            text: Text to reformat.
            platform: Target platform (discord, whatsapp, telegram, twitter).
            
        Returns:
            Reformatted text.
        """
        prompt = f"""Reformat this text for {platform}. Follow platform conventions:
- Discord: No markdown tables, wrap links in <>
- WhatsApp: No headers, use *bold* for emphasis
- Telegram: Markdown OK, keep concise
- Twitter: Max 280 chars, add hashtags

Text: {text}

Output only the reformatted text:"""
        return self._run(prompt, max_tokens=300)
    
    def extract_data(self, text: str, fields: list) -> Dict[str, Any]:
        """
        Extract structured data from text.
        
        Args:
            text: Source text to extract from.
            fields: List of field names to extract.
            
        Returns:
            Dict with extracted fields.
        """
        fields_str = ", ".join(fields)
        prompt = f"""Extract these fields from the text: {fields_str}

Text: {text}

Output as JSON:"""
        return self.generate_json(f"extracted data with fields: {fields_str}")


# Convenience functions for quick use
_default_delegate: Optional[LlamafileDelegate] = None


def get_delegate(**kwargs) -> LlamafileDelegate:
    """Get or create the default delegate."""
    global _default_delegate
    if _default_delegate is None:
        _default_delegate = LlamafileDelegate(**kwargs)
    return _default_delegate


def ask_local(prompt: str, **kwargs) -> str:
    """Quick access to ask()."""
    return get_delegate().ask(prompt, **kwargs)


def generate_json(description: str) -> Dict[str, Any]:
    """Quick access to generate_json()."""
    return get_delegate().generate_json(description)


def summarize(text: str, max_words: int = 100) -> str:
    """Quick access to summarize()."""
    return get_delegate().summarize(text, max_words)


def format_for_platform(text: str, platform: str) -> str:
    """Quick access to format_for_platform()."""
    return get_delegate().format_for_platform(text, platform)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: llamafile_delegate.py <prompt>")
        print("       llamafile_delegate.py --test")
        sys.exit(1)
    
    if sys.argv[1] == "--test":
        print("Testing llamafile delegation...")
        delegate = LlamafileDelegate()
        result = delegate.ask("Say 'Delegation works!' in exactly 2 words.")
        print(f"Response: {result}")
    else:
        prompt = " ".join(sys.argv[1:])
        delegate = LlamafileDelegate()
        print(delegate.ask(prompt))
