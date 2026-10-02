"""Model-free COSY target-language validation and fixed prompt instructions."""


class UnsupportedLanguageError(ValueError):
    """A nonempty language string is outside the COSY language contract."""


class InvalidLanguageTypeError(ValueError):
    """A language must be a string or None."""


_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ru": "Russian",
}
_ALIASES = {name.casefold(): code for code, name in _LANGUAGE_NAMES.items()}
_ALIASES.update({code: code for code in _LANGUAGE_NAMES})
_ALIASES["deutsch"] = "de"


def normalize_language(language=None):
    """Resolve an explicit COSY language or its German default, without guessing."""
    if language is None:
        return "de"
    if not isinstance(language, str):
        raise InvalidLanguageTypeError("Language must be a string or null.")
    value = language.strip().casefold()
    if not value:
        return "de"
    try:
        return _ALIASES[value]
    except KeyError:
        raise UnsupportedLanguageError(
            "Language is not supported by this model."
        ) from None


def instruction_prompt(language, style_prompt=None):
    """Compose instructions before EOP without altering target/reference text."""
    instruction = (
        "Sprich auf Deutsch mit standarddeutscher Aussprache."
        if language == "de"
        else f"Speak in {_LANGUAGE_NAMES[language]}."
    )
    prompt = "You are a helpful assistant. " + instruction
    if style_prompt:
        prompt += " " + style_prompt
    return prompt + "<|endofprompt|>"
