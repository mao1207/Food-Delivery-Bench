# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Union, List, Any, Dict
import base64, io, os, time, logging
import random  # ✅ for jitter backoff
import numpy as np
from PIL import Image
from openai import OpenAI
import httpx  # ✅ for explicit timeout

class BaseModel:
    """
    Minimal, reusable VLM/LLM caller.

    Responsibilities:
      - Accepts arbitrary system/user text with optional images.
      - Builds a Chat Completions request.
      - Returns the model's text response (str).
      - No project-specific prompts, parsing, or ReAct logic.

    Key methods:
      - generate(system, user, images, **inference_kwargs) -> str

    Notes:
      - Images can be np.ndarray, PIL.Image, bytes, local file path, or URL/data URL string.
      - 'reasoning_effort' is optional, mostly useful for GPT-5 family models.
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        model: str = "gpt-4o-mini",
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        rate_limit_per_min: Optional[int] = 20,
        supports_vision: Optional[bool] = True,
        reasoning_effort: Optional[str] = None,  # {"minimal","low","medium","high"} for GPT-5 family
        logger: Optional[logging.Logger] = None,
        http_timeout_s: float = 60.0,          # ✅ new: explicit timeout
        referer: str = "https://example.com",  # ✅ new: OpenRouter recommended header
        app_title: str = "DeliveryBench-Scorer",  # ✅ new: OpenRouter recommended header
    ):
        # ✅ Add timeout + OpenRouter recommended headers for stability
        print(f"base_url: {url}, api_key: {api_key}")
        self.client = OpenAI(
            base_url=url,
            api_key=api_key,
            timeout=httpx.Timeout(http_timeout_s),
            default_headers={
                "HTTP-Referer": referer,
                "X-Title": app_title,
            },
        )
        self.model = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.rate_limit_per_min = rate_limit_per_min
        self.supports_vision = bool(supports_vision)
        self.reasoning_effort = reasoning_effort
        self.logger = logger or self._default_logger()

    # --------- Public entrypoint ----------
    def generate(
        self,
        *,
        system: Optional[Union[str, List[str]]] = None,
        user: Optional[str] = None,
        images: Optional[Union[Any, List[Any]]] = None,
        n: int = 1,
        retry: int = 3,
        # per-call overrides:
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        rate_limit_per_min: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        extra_messages: Optional[List[Dict[str, Any]]] = None,  # power users can inject extra chat turns
        **kwargs,
    ) -> str:
        """
        Args:
          system: str or list[str] for system instructions (optional).
          user: the main user text (optional if you only want to send images).
          images: optional image or list of images (<=3 used), see _to_image_part().
          n: number of candidates (default 1) — returns first.
          retry: number of retry attempts on transient errors.
          max_tokens/temperature/top_p/rate_limit_per_min/reasoning_effort: per-call overrides.
          extra_messages: advanced escape hatch to inject additional chat messages.

        Returns:
          str: assistant message content ('' if missing).
        """
        mtoks = self.max_tokens if max_tokens is None else int(max_tokens)
        temp = self.temperature if temperature is None else float(temperature)
        tp = self.top_p if top_p is None else float(top_p)
        rate = self.rate_limit_per_min if rate_limit_per_min is None else rate_limit_per_min

        messages: List[Dict[str, Any]] = []

        # System content (supports single or list)
        if system:
            if isinstance(system, str):
                messages.append({"role": "system", "content": system})
            else:
                for s in system:
                    messages.append({"role": "system", "content": s})

        # Build user content block (text + images)
        user_content: List[Dict[str, Any]] = []
        if user:
            user_content.append({"type": "text", "text": user})

        if images:
            if not isinstance(images, (list, tuple)):
                images = [images]
            # keep at most 3
            for img in list(images)[:3]:
                if img is None:
                    continue
                if not self.supports_vision:
                    # if model doesn't support vision, silently skip images (or raise)
                    continue
                user_content.append(self._to_image_part(img))

        # Add user message only if there's something to send
        if user_content:
            messages.append({"role": "user", "content": user_content})

        # Optional extra messages (advanced)
        if extra_messages:
            messages.extend(extra_messages)

        # Reasoning effort hook (only if model supports it; safe to omit)
        effort = reasoning_effort if reasoning_effort is not None else self.reasoning_effort
        effort_allowed = {"minimal", "low", "medium", "high"}
        use_effort = (
            effort in effort_allowed
            and ("gpt-5" in (self.model or "").lower())
            and ("gpt-5-chat-latest" not in (self.model or "").lower())
        )

        last_err = None
        for attempt in range(1, int(retry) + 1):
            try:
                # simple RPM pacing
                if rate:
                    time.sleep(max(0.0, 60.0 / float(rate)))

                create_kwargs = dict(
                    model=self.model,
                    messages=messages,
                    max_tokens=mtoks,
                    temperature=temp,
                    top_p=tp,
                    n=n,
                )
                if use_effort:
                    create_kwargs["reasoning_effort"] = effort

                # passthrough any extra OpenAI params
                create_kwargs.update(kwargs)

                resp = self.client.chat.completions.create(**create_kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                self.logger.error(f"[BaseModel] attempt {attempt} failed: {e}")
                last_err = e
                # ✅ exponential backoff + jitter to reduce flapping on transient failures
                backoff = min(10.0, (2 ** attempt) * 0.5) + random.uniform(0, 0.5)
                time.sleep(backoff)

        raise RuntimeError(f"generate() failed after {retry} tries: {last_err}")

    # --------- Image helpers ----------
    def _to_image_part(self, image_obj: Any) -> dict:
        """
        Converts many image types to Chat Completions 'image_url' parts.

        Supports:
          - numpy.ndarray (H,W[,3/4])
          - PIL.Image.Image
          - bytes/bytearray
          - str: http(s) URL / data URL / local file path / bare base64
        """
        # numpy array
        if isinstance(image_obj, np.ndarray):
            b64 = self._ndarray_to_b64jpeg(image_obj)
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

        # bytes
        if isinstance(image_obj, (bytes, bytearray)):
            b64 = base64.b64encode(image_obj).decode("utf-8")
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

        # string-like: URL / data URL / local file / bare b64
        if isinstance(image_obj, str):
            s = image_obj.strip()
            if s.startswith("http://") or s.startswith("https://") or s.startswith("data:image"):
                return {"type": "image_url", "image_url": {"url": s}}
            if os.path.isfile(s):
                with open(s, "rb") as f:
                    b = f.read()
                b64 = base64.b64encode(b).decode("utf-8")
                return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            # fallback: treat as raw base64
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{s}"}}

        # PIL
        if isinstance(image_obj, Image.Image):
            b64 = self._pil_to_b64jpeg(image_obj)
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

        # last resort: stringified
        return {"type": "image_url", "image_url": {"url": str(image_obj)}}

    @staticmethod
    def _ndarray_to_b64jpeg(arr: np.ndarray, quality: int = 90) -> str:
        if arr.ndim == 2:
            mode = "L"
        elif arr.ndim == 3 and arr.shape[2] == 3:
            mode = "RGB"
        elif arr.ndim == 3 and arr.shape[2] == 4:
            mode = "RGBA"
        else:
            raise ValueError(f"Unsupported ndarray shape: {arr.shape}")
        img = Image.fromarray(arr.astype(np.uint8), mode=mode)
        return BaseModel._pil_to_b64jpeg(img, quality=quality)

    @staticmethod
    def _pil_to_b64jpeg(img: Image.Image, quality: int = 90) -> str:
        # normalize to RGB/JPEG
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _default_logger() -> logging.Logger:
        logger = logging.getLogger("BaseModel")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            h = logging.StreamHandler()
            fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
            h.setFormatter(fmt)
            logger.addHandler(h)
        return logger
