from functools import reduce
import json
import os
from pathlib import Path

from groq import Groq
import requests
from sambanova import SambaNova

if os.environ.get("GITHUB_ACTOR") is None:
    import dotenv
    dotenv.load_dotenv()


_MIN_TOKENS = 6_000
_TAGS = os.environ.get("TAGS", "v0.0.0")[1:]
_API_TIMEOUT = int(os.environ.get("API_TIMEOUT", "5"))
_SAMBANOVA_MODEL_ENDPOINT = os.environ.get("SAMBANOVA_MODEL_ENDPOINT")
_SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
_SAMBANOVA_CHAT_MODEL = os.environ.get("SAMBANOVA_CHAT_MODEL", "gpt-oss-120b")
_OPENROUTER_MODEL_ENDPOINT = os.environ.get("OPENROUTER_MODEL_ENDPOINT")
_OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
_GROQ_MODEL_ENDPOINT = os.environ.get("GROQ_MODEL_ENDPOINT")
_GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
_GROQ_CHAT_MODEL = os.environ.get("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")


# Fail fast if critical keys are missing
if not (_SAMBANOVA_API_KEY and _OPENROUTER_API_KEY and _GROQ_API_KEY):
    raise RuntimeError("Missing required API keys in environment")


def get_sambanova_models() -> list[dict]:
    """
    Fetch and filter SambaNova models with sufficient context length.

    Returns:
        List of dicts with keys: id, provider, context_length, max_completion_tokens.
    """
    print("Getting SambaNova models...")
    try:
        response = requests.get(_SAMBANOVA_MODEL_ENDPOINT, timeout=_API_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException:
        return []

    models = []
    for model in response.json().get("data", []):
        context_length = int(model.get("context_length") or 0)
        max_comp_tokens = int(model.get("max_completion_tokens") or 0)
        if context_length >= _MIN_TOKENS and max_comp_tokens >= _MIN_TOKENS:
            models.append({
                "id": model.get("id"),
                "provider": "sambanova",
                "context_length": f"{context_length // 1000}k",
                "max_completion_tokens": f"{max_comp_tokens // 1000}k",
            })
    return models


def get_openrouter_models() -> list[dict]:
    """
    Fetch and filter OpenRouter models with sufficient context length.

    Returns:
        List of dicts with keys: id, provider, context_length, max_completion_tokens.
    """
    print("Getting OpenRouter models...")
    try:
        response = requests.get(
            _OPENROUTER_MODEL_ENDPOINT,
            headers={
                "Authorization": f"Bearer {_OPENROUTER_API_KEY}",
            },
            timeout=_API_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException:
        return []

    models = []
    for model in response.json().get("data", []):
        model_id = model.get("id")
        if ":free" not in model_id:
            continue
        context_length = int(model.get("context_length") or 0)
        top = model.get("top_provider") or {}
        max_comp_tokens = int(top.get("max_completion_tokens") or 0)
        if context_length >= _MIN_TOKENS and max_comp_tokens >= _MIN_TOKENS:
            models.append({
                "id": model_id,
                "provider": "openrouter",
                "context_length": f"{context_length // 1000}k",
                "max_completion_tokens": f"{max_comp_tokens // 1000}k",
            })
    return models


def get_groq_models() -> list[dict]:
    """
    Fetch and filter Groq models with sufficient context length.

    Returns:
        List of dicts with keys: id, provider, context_length, max_completion_tokens.
    """
    print("Getting Groq models...")
    try:
        response = requests.get(
            _GROQ_MODEL_ENDPOINT,
            headers={
                "Authorization": f"Bearer {_GROQ_API_KEY}",
            },
            timeout=_API_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException:
        return []

    models = []
    for model in response.json().get("data", []):
        context_length = int(model.get("context_window") or model.get("context_length") or 0)
        max_comp_tokens = int(model.get("max_completion_tokens") or 0)
        if context_length >= _MIN_TOKENS and max_comp_tokens >= _MIN_TOKENS:
            models.append({
                "id": model.get("id"),
                "provider": "groq",
                "context_length": f"{context_length // 1000}k",
                "max_completion_tokens": f"{max_comp_tokens // 1000}k",
            })
    return models


def _ask_to_get_coding_partners(models: list[dict]) -> None:
    """
    Send the aggregated model list to an LLM and write the resulting "config.yaml" file.
    Tries Groq first; on failure it falls back to SambaNova.

    Args:
        models: List of model dictionaries gathered from all providers.
    """
    yaml_sample = f"""
        name: Continue Elite Configs
        version: {_TAGS}
        schema: v1
        models:
        - name: 1.1 Fast | Llama-4-Scout | Groq | 131k/8k
            provider: groq
            model: meta-llama/llama-4-scout-17b-16e-instruct
            apiKey: ${{ env.GROQ_API_KEY }}
    """
    system_prompts = [
        "You are a Software Engineer."
        "You are creating a remote config file config.yml for Continue vs-code extension.",
        "Models should be made to 3 sections (order them by most applicable for that section).",
        "1st section - Fast coding chatter, for quick code changes, handle small code snippets. Can hold small context length.",
        "2nd section - Code review chatter, for code documentation, refactors, code fixes, can hold multiple files, git diffs, not small yet not massive context length.",
        "3rd section - Thinking, big brains and with deep code analysis with massive context models.",
        "Each sections can have 5 models each. The config.yaml main keys are: name, version, schema, models.",
        f"Version should be {_TAGS}.",
        "Model keys are: name, provider, model, apiKey.",
        "Provider values: openrouter, sambanova, groq (lower‑case).",
        "apiKey is should be on either in ${{ secrets.SAMBANOVA_API_KEY }} or ${{ secrets.OPENROUTER_API_KEY }} or ${{ secrets.GROQ_API_KEY }}) only, since it's in YAML file",
        "Name format: 'Title | Model Name | Provider | context_length/max_completion_tokens'.",
        "Example Name format: '1.1 Fast | Llama-4-Scout | Groq | 131k/8k'",
        "Example Name format: '2.1 Refactor Specialist | Kimi-k2 | Groq | 262k/16k'",
    ]
    user_prompt = (
        "Using the following models list, build the Continue config.yaml. "
        "Reply with NOTHING except the finished yaml file. No thoughts, no markdown fences.\n\n"
        f"Sample yaml:\n\n{yaml_sample}\n\nModels:\n\n{models}\n\n"
        "Prioritize Groq and SambaNova models over OpenRouter when model duplicates exist."
        "Avoid rate-limited or congested OpenRouter endpoints."
        "Don't use models that requires terms acceptance or other none direct use of model kinda shit."
        "Only use text to text models. No text to speech and other things. No 'language' on the model"
    )

    messages = [{"role": "system", "content": prompt} for prompt in system_prompts] + [
        {"role": "user", "content": user_prompt}
    ]
    response_content = ""
    try:
        print("Sending prompt to Groq...")
        client = Groq(
            api_key=_GROQ_API_KEY
        )
        chat_completion = client.chat.completions.create(model=_GROQ_CHAT_MODEL, messages=messages)
        response_content = chat_completion.choices[0].message.content
        print("Response: Groq [OK]")
    except Exception as e:
        print(f"Groq error: {e}")
        try:
            print("Failed getting response from Groq. Sending prompt to SambaNova instead...")
            client = SambaNova(
                base_url="https://api.sambanova.ai/v1",
                api_key=_SAMBANOVA_API_KEY,
            )
            chat_completion = client.chat.completions.create(model=_SAMBANOVA_CHAT_MODEL, messages=messages)
            response_content = chat_completion.choices[0].message.content
            print("Response: SambaNova [OK]")
        except Exception as e:
            print(f"SambaNova error: {e}")
    _write_text_file(Path("config.yaml"), response_content)


def _write_text_file(path: Path, text: str) -> None:
    """
    Write text to path, creating parent directories as needed.

    Args:
        path: Destination file path.
        text: Content to write.

    Raises:
        OSError: If the file cannot be written.
    """
    if text:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        except OSError as e:
            print(f"Error writing to {path!r}: {e}")


def create_json(filename: str, models: list) -> list:
    """
    Serialize models as pretty-printed JSON.

    Args:
        filename: Target file name (e.g., ``sambanova.json``).
        models: List of model dictionaries to serialize.

    Raises:
        OSError: Propagated from the underlying file write if it fails.
    """
    if models:
        try:
            file_path = Path(f"models/{filename}")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(json.dumps(models, indent=4))
        except OSError as e:
            print(f"Error writing to {filename!r}: {e}")
    return models


if __name__ == "__main__":
    all_models = []
    all_models.extend(create_json("sambanova.json", get_sambanova_models()))
    all_models.extend(create_json("openrouter.json", get_openrouter_models()))
    all_models.extend(create_json("groq.json", get_groq_models()))
    _ask_to_get_coding_partners(all_models)