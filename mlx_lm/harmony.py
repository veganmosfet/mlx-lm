import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class HarmonyAdapterConfig:
    """
    Configuration for adapting OpenAI Harmony formatted prompts and responses.
    """
    reasoning: str = "high"
    valid_channels: str = "analysis, commentary, final"


@dataclass
class HarmonyMessage:
    role: str = ""
    channel: Optional[str] = None
    recipient: Optional[str] = None
    constraint: Optional[str] = None
    content: str = ""
    closing: Optional[str] = None


@dataclass
class HarmonyParsedResponse:
    text: str
    tool_call_payloads: List[str] = field(default_factory=list)
    finish_reason: Optional[str] = None
    analysis: List[str] = field(default_factory=list)


class HarmonyResponseParser:
    """
    Parses harmony-formatted completion tokens into structured messages.
    """

    def __init__(self, tokenizer, special_token_ids: Dict[str, int]):
        self.tokenizer = tokenizer
        self.special_token_ids = special_token_ids
        self.messages: List[HarmonyMessage] = []

    def parse(self, tokens: Sequence[int]) -> HarmonyParsedResponse:
        self.messages = []
        if not tokens:
            return HarmonyParsedResponse(text="")

        current: Optional[HarmonyMessage] = None
        section: Optional[str] = None
        buffers = {"role": [], "channel": [], "constraint": [], "content": []}

        def reset_buffers():
            buffers["role"] = []
            buffers["channel"] = []
            buffers["constraint"] = []
            buffers["content"] = []

        def ensure_current():
            nonlocal current
            if current is None:
                current = HarmonyMessage(role="assistant")

        def decode(buffer_name: str) -> str:
            data = buffers[buffer_name]
            if not data:
                return ""
            return self.tokenizer.decode(data, skip_special_tokens=True)

        def apply_header():
            ensure_current()
            role_text = decode("role").strip()
            channel_text = decode("channel").strip()

            def parse_header_value(value: str):
                if not value:
                    return "", None
                if "to=" not in value:
                    return value, None
                prefix, _, suffix = value.partition("to=")
                return prefix.strip(), suffix.strip()

            role_value, role_recipient = parse_header_value(role_text)
            if role_value:
                current.role = role_value
            if role_recipient:
                current.recipient = role_recipient

            channel_value, channel_recipient = parse_header_value(channel_text)
            if channel_value:
                current.channel = channel_value
            if channel_recipient:
                cleaned_recipient = channel_recipient
                if current.channel and cleaned_recipient.endswith(current.channel):
                    cleaned_recipient = cleaned_recipient[: -len(current.channel)].strip()
                current.recipient = cleaned_recipient
            constraint_text = decode("constraint").strip()
            if constraint_text:
                current.constraint = constraint_text

        def finalize_message(closing_token: str):
            nonlocal current, section
            if current is None:
                reset_buffers()
                section = None
                return
            apply_header()
            content_text = decode("content")
            if content_text:
                current.content += content_text
            current.closing = closing_token
            self.messages.append(current)
            current = None
            section = None
            reset_buffers()

        special = self.special_token_ids
        for token in tokens:
            if token == special.get("<|start|>"):
                finalize_message("<|end|>")
                current = HarmonyMessage()
                section = "role"
                reset_buffers()
                continue
            if token == special.get("<|channel|>"):
                ensure_current()
                section = "channel"
                continue
            if token == special.get("<|constrain|>"):
                ensure_current()
                section = "constraint"
                continue
            if token == special.get("<|message|>"):
                ensure_current()
                apply_header()
                section = "content"
                continue
            if token == special.get("<|end|>"):
                finalize_message("<|end|>")
                continue
            if token == special.get("<|return|>"):
                finalize_message("<|return|>")
                continue
            if token == special.get("<|call|>"):
                finalize_message("<|call|>")
                continue

            if section is None:
                continue
            buffers[section].append(token)

        finalize_message("<|end|>")
        parsed = self._summarize_messages()
        return parsed

    def _summarize_messages(self) -> HarmonyParsedResponse:
        analysis: List[str] = []
        visible_segments: List[str] = []
        tool_payloads: List[str] = []
        finish_reason: Optional[str] = None

        for message in self.messages:
            if message.role != "assistant":
                continue
            closing = message.closing or ""
            channel = (message.channel or "").strip()
            content = message.content
            if closing == "<|call|>":
                payload = self._tool_payload(message)
                if payload is not None:
                    tool_payloads.append(payload)
                continue
            if channel == "analysis":
                if content.strip():
                    analysis.append(content.strip())
                continue
            if content:
                visible_segments.append(content)

        if tool_payloads and not visible_segments:
            finish_reason = "tool_calls"

        combined_text = "\n".join(
            segment.strip() if isinstance(segment, str) else ""
            for segment in visible_segments
            if segment
        ).strip()
        return HarmonyParsedResponse(
            text=combined_text,
            tool_call_payloads=tool_payloads,
            finish_reason=finish_reason,
            analysis=analysis,
        )

    def _tool_payload(self, message: HarmonyMessage) -> Optional[str]:
        name = (message.recipient or "").strip()
        if not name:
            return None
        raw_args = (message.content or "").strip()
        try:
            arguments = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            arguments = raw_args
        payload = {
            "name": name,
            "arguments": arguments,
        }
        return json.dumps(payload)


class HarmonyAdapter:
    """
    Converts OpenAI compatible chat payloads to the Harmony prompt format and
    parses responses back to OpenAI compatible structures.
    """

    STOP_TOKENS = ("<|return|>", "<|call|>")

    def __init__(self, tokenizer, config: HarmonyAdapterConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.special_token_ids = {
            "<|start|>": self._token_id("<|start|>"),
            "<|end|>": self._token_id("<|end|>"),
            "<|message|>": self._token_id("<|message|>"),
            "<|channel|>": self._token_id("<|channel|>"),
            "<|constrain|>": self._token_id("<|constrain|>"),
            "<|return|>": self._token_id("<|return|>"),
            "<|call|>": self._token_id("<|call|>"),
        }
        self.response_parser = HarmonyResponseParser(tokenizer, self.special_token_ids)
        self.call_token_id = self.special_token_ids.get("<|call|>")
        self.return_token_id = self.special_token_ids.get("<|return|>")

    def parse_tokens(self, tokens: Sequence[int]) -> HarmonyParsedResponse:
        return self.response_parser.parse(tokens)

    def _token_id(self, token: str) -> Optional[int]:
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, list):
                if token_id:
                    return token_id[0]
                return None
            if isinstance(token_id, int) and token_id >= 0:
                return token_id
        return None
