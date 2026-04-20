import json
import os
from pathlib import Path
import time

import click
from groq import Groq
import requests
from requests.adapters import HTTPAdapter
from sambanova import SambaNova
from urllib3.util.retry import Retry

if os.environ.get("GITHUB_ACTOR") is None:
    import dotenv
    dotenv.load_dotenv()

_MIN_TOKENS = 6_000
_TAGS = os.environ.get("TAGS", "v0.0.0")[1:]
_API_TIMEOUT = int(os.environ.get("API_TIMEOUT", "5"))
_MODELS_FOLDER = "models"
_MODELS_EXT = "json"

_SAMBANOVA_MODEL_ENDPOINT = os.environ.get("SAMBANOVA_MODEL_ENDPOINT")
_SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
_SAMBANOVA_CHAT_MODEL = os.environ.get("SAMBANOVA_CHAT_MODEL", "gpt-oss-120b")

_OPENROUTER_MODEL_ENDPOINT = os.environ.get("OPENROUTER_MODEL_ENDPOINT")
_OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

_GROQ_MODEL_ENDPOINT = os.environ.get("GROQ_MODEL_ENDPOINT")
_GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
_GROQ_CHAT_MODEL = os.environ.get("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")

_MODEL_USER_PROMPT_URL = os.environ.get("MODEL_USER_PROMPT_URL")
_MODEL_SYSTEM_PROMPT_URL = os.environ.get("MODEL_SYSTEM_PROMPT_URL")

# Configure HTTP session with retries
_HTTP_SESSION = requests.Session()
_ADAPTER = HTTPAdapter(
    max_retries=Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[408, 429, 500, 502, 503, 504],
    )
)
_HTTP_SESSION.mount("https://", _ADAPTER)
_HTTP_SESSION.mount("http://", _ADAPTER)

# Fail fast if critical keys are missing
if not (_SAMBANOVA_API_KEY and _OPENROUTER_API_KEY and _GROQ_API_KEY):
    raise RuntimeError("Missing required API keys in environment")


def get_sambanova_models() -> list[dict]:
    """
    Fetch and filter SambaNova models that meet the minimum token requirements.

    Returns:
        List of dictionaries containing model information with keys:
        id, provider, context_length, max_completion_tokens.
    """
    click.echo("Getting SambaNova models...")
    try:
        response = _HTTP_SESSION.get(_SAMBANOVA_MODEL_ENDPOINT, timeout=_API_TIMEOUT)
        response.raise_for_status()
        models_data = response.json()
    except requests.RequestException as error:
        click.echo(f"SambaNova API error: {error}", err=True)
        return []

    models = []
    for model in models_data.get("data", []):
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
    Fetch and filter OpenRouter models that meet the minimum token requirements.

    Returns:
        List of dictionaries containing model information with keys:
        id, provider, context_length, max_completion_tokens.
    """
    click.echo("Getting OpenRouter models...")
    try:
        response = _HTTP_SESSION.get(
            _OPENROUTER_MODEL_ENDPOINT,
            headers={
                "Authorization": f"Bearer {_OPENROUTER_API_KEY}",
            },
            timeout=_API_TIMEOUT,
        )
        response.raise_for_status()
        models_data = response.json()
    except requests.RequestException as error:
        click.echo(f"OpenRouter API error: {error}", err=True)
        return []

    models = []
    for model in models_data.get("data", []):
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
        List of dictionaries containing model information with keys:
        id, provider, context_length, max_completion_tokens.
    """
    click.echo("Getting Groq models...")
    try:
        response = _HTTP_SESSION.get(
            _GROQ_MODEL_ENDPOINT,
            headers={
                "Authorization": f"Bearer {_GROQ_API_KEY}",
            },
            timeout=_API_TIMEOUT,
        )
        response.raise_for_status()
        models_data = response.json()
    except requests.RequestException as error:
        click.echo(f"Groq API error: {error}", err=True)
        return []

    models = []
    for model in models_data.get("data", []):
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


def build_continue_yaml(models: list[dict]) -> str:
    """
    Generate a config.yaml file by sending model data to an LLM.

    Retrieves system and user prompt from URLs

    Args:
        models: List of model dictionaries gathered from all providers.
    """
    try:
        response = _HTTP_SESSION.get(_MODEL_SYSTEM_PROMPT_URL, timeout=_API_TIMEOUT)
        response.raise_for_status()
        system_prompts = response.text.split("\n")
        system_prompts.append(f"Version should be {_TAGS}.")

        response = _HTTP_SESSION.get(_MODEL_USER_PROMPT_URL, timeout=_API_TIMEOUT)
        response.raise_for_status()
        user_prompt = f"{response.text}\nModels:\n{models}\n"
    except requests.RequestException as error:
        click.echo(f"Model Prompt error: {error}", err=True)
        system_prompts = []
        user_prompt = ""

    messages = [{"role": "system", "content": prompt} for prompt in system_prompts] + [{"role": "user", "content": user_prompt}]
    try:
        click.echo("Sending prompt to Groq...")
        client = Groq(api_key=_GROQ_API_KEY)
        reply = client.chat.completions.create(model=_GROQ_CHAT_MODEL, messages=messages)
        click.echo("Response: Groq [OK]")
        return reply.choices[0].message.content
    except Exception as error:
        click.echo(f"Groq API error: {error}", err=True)
        try:
            click.echo("Failed getting response from Groq. Sending prompt to SambaNova instead...")
            client = SambaNova(base_url="https://api.sambanova.ai/v1", api_key=_SAMBANOVA_API_KEY)
            reply = client.chat.completions.create(model=_SAMBANOVA_CHAT_MODEL, messages=messages)
            click.echo("Response: SambaNova [OK]")
            return reply.choices[0].message.content
        except Exception as error:
            click.echo(f"SambaNova API error: {error}", err=True)
            return ""


def _write_file(path: Path, text: str) -> None:
    """
    Write text content to a file, creating parent directories as needed.

    Args:
        path: Destination file path.
        text: Content to write.

    Raises:
        OSError: If the file cannot be written.
    """
    if text:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError as error:
            click.echo(f"Error writing to {path!r}: {error}", err=True)
            raise


def _dump_json(name: str, models: list) -> None:
    """
    Serialize models as pretty-printed JSON.

    Args:
        name: Target file name.
        models: List of model dictionaries to serialize.
    """
    try:
        file_path = Path(f"{_MODELS_FOLDER}/{name}.{_MODELS_EXT}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(models, indent=4))
    except OSError as error:
        click.echo(f"Error writing to {name!r}.json: {error}", err=True)


def _models_equal(old: list[dict], new: list[dict]) -> bool:
    """
    Compare two model lists for equality, ignoring order.

    Args:
        old: Previously stored model list.
        new: Newly fetched model list.

    Returns:
        True if both lists contain the same id/provider pairs.
    """
    key = lambda m: (m["id"], m["provider"])
    return sorted(old, key=key) == sorted(new, key=key)


def _json_changed(provider: str, fresh_models: list[dict]) -> bool:
    """
    Determine whether the JSON file for a provider needs to be updated.

    Args:
        provider: Provider name (e.g., "sambanova").
        fresh_models: Newly fetched model list for the provider.

    Returns:
        True if the file does not exist, cannot be read, or its content
        differs from fresh_models.
    """
    json_path = Path(f"{_MODELS_FOLDER}/{provider}.{_MODELS_EXT}")
    if not json_path.exists():
        return True
    try:
        existing = json.loads(json_path.read_text())
    except Exception:
        return True
    return not _models_equal(existing, fresh_models)


@click.command()
@click.option("--force-update", "--force", is_flag=True, default=False, help="Force update configuration.")
def update(force_update: bool) -> None:
    """
    Main entry‑point for the CLI command.

    Fetches models from all providers, writes JSON files when they change,
    and (re)generates config.yaml when needed.

    Args:
        force_update (bool): When true forces regeneration of config.yaml.
    """
    start = time.perf_counter()
    changed = False
    all_models = []

    providers = [
        ("sambanova", get_sambanova_models),
        ("openrouter", get_openrouter_models),
        ("groq", get_groq_models)
    ]

    for provider_name, provider_func in providers:
        models = provider_func()
        if _json_changed(provider_name, models):
            _dump_json(provider_name, models)
            changed = True
        all_models.extend(models)

    if changed or force_update:
        yaml = build_continue_yaml(all_models)
        _write_file(Path("config.yaml"), yaml)
    else:
        click.echo("No provider changes detected - config.yaml left untouched.")

    elapsed = time.perf_counter() - start
    click.echo("=" * 60)
    click.echo(f"✨ Process completed in {elapsed:.2f}s")
    click.echo("=" * 60)

if __name__ == "__main__":
    update()