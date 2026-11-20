"""Chat model providers behind one function: complete(system, user) -> LLMResult."""

from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from rag.config import get_settings


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=30))
def complete(system: str, user: str, model: str | None = None, max_tokens: int | None = None) -> LLMResult:
    s = get_settings()
    model = model or s.llm_model
    max_tokens = max_tokens or s.llm_max_tokens

    if s.llm_provider == "anthropic":
        import anthropic

        resp = anthropic.Anthropic().messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=s.llm_temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResult(text, resp.usage.input_tokens, resp.usage.output_tokens, model)

    if s.llm_provider == "openai":
        from openai import OpenAI

        client = OpenAI(
            base_url=s.llm_base_url,
            # Ollama ignores the key, but the SDK insists on one being set
            api_key=s.llm_api_key or s.openai_api_key or ("ollama" if s.llm_base_url else None),
            timeout=300,  # local models on CPU are slow
        )
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=s.llm_temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return LLMResult(
            resp.choices[0].message.content or "",
            resp.usage.prompt_tokens,
            resp.usage.completion_tokens,
            model,
        )

    return _fake(system, user, model)


def _fake(system: str, user: str, model: str) -> LLMResult:
    """Offline stand-in for tests: cites source [1] if any context was given."""
    if "JUDGE" in system:
        text = '{"faithful": true, "correct": true, "reason": "fake judge"}'
    elif "<source id=" in user:
        text = "According to the manual, see the referenced section [1]."
    else:
        text = "I don't know based on the provided manuals."
    return LLMResult(text, len(user) // 4, len(text) // 4, "fake")
