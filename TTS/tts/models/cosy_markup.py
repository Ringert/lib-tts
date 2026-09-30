"""Bounded COSY-only markup planning, with no model or audio imports."""

import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass

PREFIX = "markup:v1:"
STYLE_PROMPTS = {
    "neutral": "Speak in a calm, neutral voice.",
    "loud": "Please say a sentence as loudly as possible.",
    "soft": "Please say a sentence in a very soft voice.",
    "angry": "请非常生气地说一句话。",
    "scream": "Please say a sentence as loudly as possible. Shout angrily with a tense, forceful voice.",
}


# Registered controls in pinned CosyVoice3Tokenizer (074ca6d). Kept as
# lightweight lexical data: importing its tokenizer would load model packages.
_NATIVE_CONTROLS = frozenset(
    """<|endoftext|> <|im_start|> <|im_end|> <|endofprompt|> [breath] <strong> </strong>
    [noise] [laughter] [cough] [clucking] [accent] [quick_breath] <laughter> </laughter>
    [hissing] [sigh] [vocalized-noise] [lipsmack] [mn] <|endofsystem|> [AA] [AA0] [AA1]
    [AA2] [AE] [AE0] [AE1] [AE2] [AH] [AH0] [AH1] [AH2] [AO] [AO0] [AO1] [AO2] [AW] [AW0]
    [AW1] [AW2] [AY] [AY0] [AY1] [AY2] [B] [CH] [D] [DH] [EH] [EH0] [EH1] [EH2] [ER] [ER0]
    [ER1] [ER2] [EY] [EY0] [EY1] [EY2] [F] [G] [HH] [IH] [IH0] [IH1] [IH2] [IY] [IY0]
    [IY1] [IY2] [JH] [K] [L] [M] [N] [NG] [OW] [OW0] [OW1] [OW2] [OY] [OY0] [OY1] [OY2]
    [P] [R] [S] [SH] [T] [TH] [UH] [UH0] [UH1] [UH2] [UW] [UW0] [UW1] [UW2] [V] [W] [Y]
    [Z] [ZH] [a] [ai] [an] [ang] [ao] [b] [c] [ch] [d] [e] [ei] [en] [eng] [f] [g] [h] [i]
    [ian] [in] [ing] [iu] [ià] [iàn] [iàng] [iào] [iá] [ián] [iáng] [iáo] [iè] [ié] [iòng]
    [ióng] [iù] [iú] [iā] [iān] [iāng] [iāo] [iē] [iě] [iōng] [iū] [iǎ] [iǎn] [iǎng] [iǎo]
    [iǒng] [iǔ] [j] [k] [l] [m] [n] [o] [ong] [ou] [p] [q] [r] [s] [sh] [t] [u] [uang]
    [ue] [un] [uo] [uà] [uài] [uàn] [uàng] [uá] [uái] [uán] [uáng] [uè] [ué] [uì] [uí]
    [uò] [uó] [uā] [uāi] [uān] [uāng] [uē] [uě] [uī] [uō] [uǎ] [uǎi] [uǎn] [uǎng] [uǐ]
    [uǒ] [vè] [w] [x] [y] [z] [zh] [à] [ài] [àn] [àng] [ào] [á] [ái] [án] [áng] [áo] [è]
    [èi] [èn] [èng] [èr] [é] [éi] [én] [éng] [ér] [ì] [ìn] [ìng] [í] [ín] [íng] [ò] [òng]
    [òu] [ó] [óng] [óu] [ù] [ùn] [ú] [ún] [ā] [āi] [ān] [āng] [āo] [ē] [ēi] [ēn] [ēng] [ě]
    [ěi] [ěn] [ěng] [ěr] [ī] [īn] [īng] [ō] [ōng] [ōu] [ū] [ūn] [ǎ] [ǎi] [ǎn] [ǎng] [ǎo]
    [ǐ] [ǐn] [ǐng] [ǒ] [ǒng] [ǒu] [ǔ] [ǔn] [ǘ] [ǚ] [ǜ]""".split()  # noqa: SIM905 - compact table of pinned tokenizer controls
)


def _has_native_controls(value):
    return any(
        token in _NATIVE_CONTROLS for token in re.findall(r"<[^>]*>|\[[^\]]*\]", value)
    )


class MarkupError(ValueError):
    """Safe error categories; never contain supplied text or parser diagnostics."""


@dataclass(frozen=True)
class Speech:
    text: str
    style: str


@dataclass(frozen=True)
class Pause:
    ms: int


@dataclass(frozen=True)
class SpeechPlan:
    segments: tuple[Speech | Pause, ...]


def parse_style(text: str, style: str | None) -> SpeechPlan | str | None:
    if style is None:
        return None
    if not isinstance(style, str) or len(style) > 1000:
        raise MarkupError("Invalid style length or type.")
    if not style.startswith(PREFIX):
        return style.strip() or None
    xml = style[len(PREFIX) :]
    if "<!" in xml or "<?" in xml:
        raise MarkupError("XML declarations and directives are unsupported.")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        raise MarkupError("Malformed speech markup.") from None
    if root.tag != "speech" or root.attrib:
        raise MarkupError("Expected one unqualified speech root.")
    leaves = []
    elements = 0
    pauses = 0
    pause_ms = 0

    def add_text(value, style_name, emphasis):
        if value:
            # Native control tokens must only originate in this planner.
            if _has_native_controls(value):
                raise MarkupError("Native backend controls are unsupported.")
            leaves.append((value, style_name, emphasis))

    def walk(node, name="neutral", emphasis=False, depth=1):
        nonlocal elements, pauses, pause_ms
        elements += 1
        if depth > 8 or elements > 64:
            raise MarkupError("Markup structure limit exceeded.")
        if node.tag == "style":
            if set(node.attrib) != {"name"} or node.get("name") not in STYLE_PROMPTS:
                raise MarkupError("Unsupported style or attributes.")
            name = node.get("name")
        elif node.tag == "pause":
            value = node.get("ms", "")
            if set(node.attrib) != {"ms"} or not re.fullmatch(r"[0-9]+", value):
                raise MarkupError("Pause requires integer milliseconds.")
            ms = int(value)
            pauses += 1
            pause_ms += ms
            if not 1 <= ms <= 2000 or pauses > 12 or pause_ms > 5000:
                raise MarkupError("Pause limit exceeded.")
            if len(node) or node.text:
                raise MarkupError("Pause must be empty.")
            leaves.append(Pause(ms))
            return
        elif node.tag == "lengthen":
            raise MarkupError("Vowel lengthening is not supported.")
        elif node.tag == "emphasis":
            if node.attrib:
                raise MarkupError("Unsupported emphasis attributes.")
            emphasis = True
        elif node is not root:
            raise MarkupError("Unsupported speech element.")
        if node.tag in {"style", "emphasis"} and not "".join(node.itertext()).strip():
            raise MarkupError("Speech ranges must contain text.")
        add_text(node.text, name, emphasis)
        for child in node:
            walk(child, name, emphasis, depth + 1)
            add_text(child.tail, name, emphasis)

    walk(root)
    plain = "".join(item[0] for item in leaves if not isinstance(item, Pause))
    if _has_native_controls(plain):
        raise MarkupError("Native backend controls are unsupported.")
    if plain.strip() != text.strip():
        raise MarkupError("Markup text does not match target text.")
    segments = []
    current_name = None
    current = []
    current_emphasis = False
    offset = 0

    def flush():
        nonlocal current, current_emphasis
        if current:
            if current_emphasis:
                current.append("</strong>")
            segments.append(Speech("".join(current), current_name))
        current = []
        current_emphasis = False

    def word_character(char):
        return unicodedata.category(char)[0] in "LMN" or char in "_'’‐-"

    def boundary():
        if (
            0 < offset < len(plain)
            and word_character(plain[offset - 1])
            and word_character(plain[offset])
        ):
            raise MarkupError("Style changes and pauses must lie at word boundaries.")

    for item in leaves:
        if isinstance(item, Pause):
            boundary()
            flush()
            segments.append(item)
            current_name = None
            continue
        value, name, emphasis = item
        if current_name is not None and name != current_name:
            boundary()
            flush()
        current_name = name
        if emphasis != current_emphasis:
            current.append("<strong>" if emphasis else "</strong>")
            current_emphasis = emphasis
        current.append(value)
        offset += len(value)
    flush()
    segments = [
        s
        for s in segments
        if isinstance(s, Pause) or re.sub(r"</?strong>", "", s.text).strip()
    ]
    count = sum(isinstance(s, Speech) for s in segments)
    if not 1 <= count <= 12:
        raise MarkupError("Speech segment limit exceeded.")
    return SpeechPlan(tuple(segments))
