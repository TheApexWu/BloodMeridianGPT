"""
Unified LLM generation interface.

Usage:
    from lib.api_clients import generate
    text = generate("gpt-4o", prompt="Write in McCarthy's style...", max_tokens=600)

All responses are cached to disk. Reruns do not re-query APIs.
"""

import hashlib
import json
import time
from pathlib import Path

from lib.config import MODELS, RESULTS_DIR, get_api_key, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE

CACHE_DIR = RESULTS_DIR / "api_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(model_key: str, prompt: str, temperature: float, max_tokens: int) -> str:
    """Deterministic hash for a request."""
    blob = json.dumps({
        "model_key": model_key,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _load_cache(key: str) -> str | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["response"]
    return None


def _save_cache(key: str, model_key: str, prompt: str, response: str):
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps({
        "model_key": model_key,
        "prompt": prompt[:200],
        "response": response,
    }, indent=2))


def generate(
    model_key: str,
    prompt: str,
    system: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    use_cache: bool = True,
) -> str:
    """
    Generate text from any registered model.

    Args:
        model_key: Key from config.MODELS (e.g. "gpt-4o", "claude-sonnet")
        prompt: User prompt
        system: Optional system prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        use_cache: If True, return cached response when available

    Returns:
        Generated text string
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Choose from: {list(MODELS.keys())}")

    spec = MODELS[model_key]
    provider = spec["provider"]
    model_id = spec["model_id"]

    # Check cache
    cache_key = _cache_key(model_key, prompt, temperature, max_tokens)
    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached

    # Dispatch to provider
    if provider == "openai":
        response = _generate_openai(model_id, prompt, system, temperature, max_tokens)
    elif provider == "anthropic":
        response = _generate_anthropic(model_id, prompt, system, temperature, max_tokens)
    elif provider == "google":
        response = _generate_google(model_id, prompt, system, temperature, max_tokens)
    elif provider == "ollama":
        response = _generate_ollama(model_id, prompt, system, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Cache and return
    _save_cache(cache_key, model_key, prompt, response)
    return response


def _generate_openai(model_id, prompt, system, temperature, max_tokens):
    from openai import OpenAI
    client = OpenAI(api_key=get_api_key("openai"))
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def _generate_anthropic(model_id, prompt, system, temperature, max_tokens):
    import anthropic
    client = anthropic.Anthropic(api_key=get_api_key("anthropic"))
    kwargs = {
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return resp.content[0].text


def _generate_google(model_id, prompt, system, temperature, max_tokens):
    from google import genai
    client = genai.Client(api_key=get_api_key("google"))
    config = genai.types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    if system:
        config.system_instruction = system
    resp = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=config,
    )
    return resp.text


def _generate_ollama(model_id, prompt, system, temperature, max_tokens):
    import ollama as ol
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = ol.chat(
        model=model_id,
        messages=messages,
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    return resp["message"]["content"]


def generate_batch(
    model_key: str,
    prompts: list[str],
    system: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    delay: float = 1.0,
    use_cache: bool = True,
) -> list[str]:
    """
    Generate from a list of prompts with rate-limiting delay between calls.
    Cached calls don't trigger the delay.
    """
    results = []
    for prompt in prompts:
        cache_key = _cache_key(model_key, prompt, temperature, max_tokens)
        is_cached = _load_cache(cache_key) is not None
        response = generate(
            model_key, prompt, system=system,
            temperature=temperature, max_tokens=max_tokens,
            use_cache=use_cache,
        )
        results.append(response)
        if not is_cached and delay > 0:
            time.sleep(delay)
    return results
