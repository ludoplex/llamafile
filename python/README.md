# llamafile Python Delegation

Python helper for AI agents to delegate simple tasks to a local llamafile instance, reducing API costs while preserving complex reasoning for cloud models.

## Installation

```bash
# Copy to your project or add to PYTHONPATH
cp llamafile_delegate.py /your/project/
```

## Quick Start

```python
from llamafile_delegate import LlamafileDelegate

# Initialize with your llamafile and model
delegate = LlamafileDelegate(
    llamafile_path="./llamafile.exe",
    model_path="./qwen2.5-7b-instruct.gguf"
)

# Simple prompt
result = delegate.ask("Reformat this as bullet points: item1, item2, item3")

# Generate JSON from description
data = delegate.generate_json("user profile with name, email, and role")
# Returns: {'name': '...', 'email': '...', 'role': '...'}

# Summarize text
summary = delegate.summarize(long_article, max_words=50)

# Format for specific platform
tweet = delegate.format_for_platform(announcement, "twitter")
```

## Convenience Functions

For quick one-off calls without managing a delegate instance:

```python
from llamafile_delegate import ask_local, generate_json, summarize

result = ask_local("What is 2+2?")
data = generate_json("a shopping list with 3 items")
summary = summarize(long_text)
```

## Use Cases

### Delegate to Local LLM
- Text formatting and cleanup
- Simple summaries (< 500 words)
- JSON/YAML generation
- Template variable substitution
- Data extraction from structured text
- Platform-specific formatting

### Keep for Cloud LLM (Claude, GPT, etc.)
- Complex multi-step reasoning
- Tool use and API coordination
- Code generation with context
- Creative writing
- Nuanced analysis

## Configuration

```python
delegate = LlamafileDelegate(
    llamafile_path="./llamafile",     # Path to binary
    model_path="./model.gguf",        # Path to GGUF model
    gpu_layers=99,                    # GPU offload (99 = all)
    timeout=60,                       # Inference timeout in seconds
)
```

## Chat Template

The delegate uses Qwen's chat template by default:
```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

This works well with Qwen, Mistral, and most instruction-tuned models.

## License

Apache 2.0 (same as llamafile)
