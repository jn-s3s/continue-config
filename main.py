import json
import os
from pathlib import Path
import time

import click
import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

if os.environ.get("GITHUB_ACTOR") is None:
    import dotenv
    dotenv.load_dotenv()

_MIN_TOKENS = 6_000
_TAGS = os.environ.get("TAGS", "v0.0.0")[1:]
_API_TIMEOUT = int(os.environ.get("API_TIMEOUT", "5"))
_MODELS_FOLDER = "models"
_MODELS_EXT = "json"

_SAMBANOVA_API_BASE_URL = os.environ.get("SAMBANOVA_API_BASE_URL")
_SAMBANOVA_API_ENDPOINT = os.environ.get("SAMBANOVA_API_ENDPOINT")
_SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
_SAMBANOVA_CHAT_MODEL = os.environ.get("SAMBANOVA_CHAT_MODEL", "gpt-oss-120b")

_OPENROUTER_API_BASE_URL = os.environ.get("OPENROUTER_API_BASE_URL")
_OPENROUTER_API_ENDPOINT = os.environ.get("OPENROUTER_API_ENDPOINT")
_OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

_GROQ_API_BASE_URL= os.environ.get("GROQ_API_BASE_URL")
_GROQ_API_ENDPOINT = os.environ.get("GROQ_API_ENDPOINT")
_GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
_GROQ_CHAT_MODEL = os.environ.get("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")

_GEMINI_API_BASE_URL = os.environ.get("GEMINI_API_BASE_URL")
_GEMINI_API_ENDPOINT = os.environ.get("GEMINI_API_ENDPOINT")
_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
_GEMINI_CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash")

_MODEL_USER_PROMPT_URL = os.environ.get("MODEL_USER_PROMPT_URL")
_MODEL_SYSTEM_PROMPT_URL = os.environ.get("MODEL_SYSTEM_PROMPT_URL")

# Fail fast if critical keys are missing
if not (_SAMBANOVA_API_KEY and _OPENROUTER_API_KEY and _GROQ_API_KEY and _GEMINI_API_KEY):
    raise RuntimeError("Missing required API keys in environment")

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


def get_sambanova_models() -> list[dict | None]:
    """
    Fetch and filter SambaNova models that meet the minimum token requirements.

    Returns:
        List of dictionaries containing model information.
    """
    try:
        response = _HTTP_SESSION.get(_SAMBANOVA_API_BASE_URL + _SAMBANOVA_API_ENDPOINT, timeout=_API_TIMEOUT)
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


def get_openrouter_models() -> list[dict | None]:
    """
    Fetch and filter OpenRouter models that meet the minimum token requirements.

    Returns:
        List of dictionaries containing model information.
    """
    try:
        response = _HTTP_SESSION.get(
            _OPENROUTER_API_BASE_URL + _OPENROUTER_API_ENDPOINT,
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


def get_groq_models() -> list[dict | None]:
    """
    Fetch and filter Groq models with sufficient context length.

    Returns:
        List of dictionaries containing model information.
    """
    try:
        response = _HTTP_SESSION.get(
            _GROQ_API_BASE_URL + _GROQ_API_ENDPOINT,
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


def get_gemini_models() -> list[dict | None]:
    """
    Fetch and filter Gemini models that meet the minimum token requirements.

    Returns:
        List of dictionaries containing model information.
    """
    try:
        response = _HTTP_SESSION.get(
            _GEMINI_API_BASE_URL + _GEMINI_API_ENDPOINT,
            params={
                "key": _GEMINI_API_KEY
            },
            timeout=_API_TIMEOUT
        )
        response.raise_for_status()
        models_data = response.json()
    except requests.RequestException as error:
        click.echo(f"Gemini API error: {error}", err=True)
        return []

    models = []
    for model in models_data.get("models", []):
        context_length = int(model.get("inputTokenLimit") or 0)
        max_comp_tokens = int(model.get("outputTokenLimit") or 0)
        if context_length >= _MIN_TOKENS and max_comp_tokens >= _MIN_TOKENS:
            models.append({
                "id": model.get("name").replace("models/", ""),
                "provider": "gemini",
                "context_length": f"{context_length // 1000}k",
                "max_completion_tokens": f"{max_comp_tokens // 1000}k",
            })
    return models


def build_continue_yaml(models: list[dict]) -> str:
    """
    Generate a config.yaml file by sending model data to an LLM.

    The function retrieves system and user prompts from the URLs defined in the environment,
    then queries the configured LLM providers in order until a non‑empty response is received.

    Args:
        models: List of model dictionaries gathered from all providers.

    Returns:
        The generated YAML content as a string, or ``None`` if all providers fail to produce a result.
    """

    system_prompts, user_prompts = _get_model_prompts(models)

    generators = [
        _ask_gemini_to_generate_config,
        _ask_groq_to_generate_config,
        _ask_sambanova_to_generate_config,
    ]

    for gen in generators:
        result = gen(system_prompts, user_prompts)
        if result:
            return result
    return None


def _get_model_prompts(models: list[dict]) -> tuple[list, str]:
    """
    Fetch the system and user prompt templates from the URLs configured in the environment and inject the current model list.

    Args:
        models: The aggregated list of model dictionaries.

    Returns:
        A tuple containing:
        * system_prompts - a list of system‑prompt lines.
        * user_prompts - a single string that combines the user‑prompt template with a formatted representation of models.
    """
    click.echo("Fetching system and user prompt ...")
    try:
        response = _HTTP_SESSION.get(_MODEL_SYSTEM_PROMPT_URL, timeout=_API_TIMEOUT)
        response.raise_for_status()
        system_prompts = response.text.split("\n")
        system_prompts.append(f"Version should be {_TAGS}.")
        response = _HTTP_SESSION.get(_MODEL_USER_PROMPT_URL, timeout=_API_TIMEOUT)
        response.raise_for_status()
        user_prompts = f"{response.text}\nModels:\n{models}\n"
    except requests.RequestException as error:
        click.echo(f"Generating prompt error: {error}", err=True)
        system_prompts = []
        user_prompts = ""

    return system_prompts, user_prompts


def _ask_groq_to_generate_config(system_prompts: list, user_prompts: str):
    """
    Send system and user prompts to Groq for generating a config.

    Args:
        system_prompts: List of system prompts.
        user_prompts: User prompts string.

    Returns:
        The generated config content as a string.
    """
    prompts = [{"role": "system", "content": prompt} for prompt in system_prompts] + [{"role": "user", "content": user_prompts}]
    try:
        click.echo("Sending system and user prompt to Groq ...")
        from groq import Groq
        client = Groq(api_key=_GROQ_API_KEY)
        reply = client.chat.completions.create(model=_GROQ_CHAT_MODEL, messages=prompts)
        click.echo("Response: Groq [OK]")
        return reply.choices[0].message.content
    except Exception as error:
        click.echo(f"Groq error: {error}", err=True)
        return ""


def _ask_sambanova_to_generate_config(system_prompts: list, user_prompts: str):
    """
    Send system and user prompts to SambaNova for generating a config.

    Args:
        system_prompts: List of system prompts.
        user_prompts: User prompts string.

    Returns:
        The generated config content as a string.
    """
    prompts = [{"role": "system", "content": prompt} for prompt in system_prompts] + [{"role": "user", "content": user_prompts}]
    try:
        click.echo("Sending system and user prompt to SambaNova ...")
        from sambanova import SambaNova
        client = SambaNova(base_url=_SAMBANOVA_API_BASE_URL, api_key=_SAMBANOVA_API_KEY)
        reply = client.chat.completions.create(model=_SAMBANOVA_CHAT_MODEL, messages=prompts)
        click.echo("Response: SambaNova [OK]")
        return reply.choices[0].message.content
    except Exception as error:
        click.echo(f"SambaNova error: {error}", err=True)
        return ""


def _ask_gemini_to_generate_config(system_prompts: list, user_prompts: str):
    """
    Send system and user prompts to Gemini for generating a config.

    Args:
        system_prompts: List of system prompts.
        user_prompts: User prompts string.

    Returns:
        The generated config content as a string.
    """
    try:
        click.echo("Sending system and user prompt to Gemini ...")
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=_GEMINI_API_KEY)

        response = client.models.generate_content(
            model=_GEMINI_CHAT_MODEL,
            config=types.GenerateContentConfig(
                system_instruction='. '.join(system_prompts)
            ),
            contents=user_prompts
        )
        print(response.text)
        click.echo("Response: Gemini [OK]")
        return response.text
    except Exception as error:
        click.echo(f"Gemini error: {error}", err=True)
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
    click.echo("Generating new configurations ...")
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
        ("groq", get_groq_models),
        ("gemini", get_gemini_models),
    ]

    with tqdm(total=len(providers), unit="providers", desc="Fetching and updating models", colour="green") as pbar:
        for provider_name, provider_func in providers:
            pbar.set_postfix_str(f"Provider: {provider_name}")
            models = provider_func()
            if _json_changed(provider_name, models):
                _dump_json(provider_name, models)
                changed = True
            all_models.extend(models)
            pbar.update(1)

    if changed or force_update:
        yaml = build_continue_yaml(all_models)
        _write_file(Path("config.yaml"), yaml)
    else:
        click.echo("No provider models changes detected - config.yaml left untouched.")

    elapsed = time.perf_counter() - start
    click.echo("=" * 60)
    click.echo(f"✨ Process completed in {elapsed:.2f}s")
    click.echo("=" * 60)


if __name__ == "__main__":
    update()