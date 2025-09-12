# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Union, List, Any
import base64, io, os, time
import numpy as np
from PIL import Image
from openai import OpenAI
from Base.Prompt import get_system_prompt


class BaseModel:
    """
    极简 VLM 调用器：
    - 你在 DeliveryMan 里准备完整 prompt（含各分区），以及三张图（全局/局部/第一视角）
    - 这里仅负责把 prompt + images 发给模型，返回文本响应
    - 不拼 system prompt，不做 ReAct/JSON 解析，不做动作解析
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        model: str = "gpt-4o-mini",
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 1.0,
        rate_limit_per_min: Optional[int] = 20,
        supports_vision: Optional[bool] = None,
    ):
        self.client = OpenAI(base_url=url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.rate_limit_per_min = rate_limit_per_min

        # 自动判断是否支持视觉；也可在 __init__ 里手动传 supports_vision 覆盖
        if supports_vision is None:
            m = model.lower()
            self.supports_vision = any(
                key in m
                for key in [
                    "gpt-4o", "4.1", "vision", "vlm", "llava", "gemini", "gpt-o"
                ]
            )
        else:
            self.supports_vision = bool(supports_vision)

    # --------- 公共入口：只要 prompt + images，返回文本 ---------
    def generate(
        self,
        user_prompt: str,
        images: Optional[Union[Any, List[Any]]] = None,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: int = 1,
        retry: int = 8,
        rate_limit_per_min: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        user_prompt: 已在 DeliveryMan 里拼好的完整文本
        images: [img1, img2, img3]（可传少于3张或 None）
        返回：模型文本（str）
        """
        max_tokens = self.max_tokens if max_tokens is None else int(max_tokens)
        temperature = self.temperature if temperature is None else float(temperature)
        top_p = self.top_p if top_p is None else float(top_p)
        rate = self.rate_limit_per_min if rate_limit_per_min is None else rate_limit_per_min

        messages = [{"role": "system", "content": get_system_prompt()}]

        # 组装 user content
        user_content = [{"type": "text", "text": user_prompt}]

        if images:
            if not isinstance(images, (list, tuple)):
                images = [images]
            for img in list(images)[:3]:  # 只取前三张
                if img is None:
                    continue
                if not self.supports_vision:
                    # 给出一个温和的提示，但仍然继续仅文本调用（也可直接 raise）
                    # raise ValueError(f"Model {self.model} does not support vision but images were provided.")
                    continue
                user_content.append(self._to_image_part(img))

        messages.append({"role": "user", "content": user_content})

        last_err = None
        for i in range(1, retry + 1):
            try:
                # if rate:
                #     time.sleep(max(0.0, 60.0 / float(rate)))
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    **kwargs,
                )
                print(resp)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                print(f"[BaseModel] generate attempt {i} failed: {e}")

        raise RuntimeError(f"VLM generate failed after {retry} tries: {last_err}")

    # --------- 内部：把任意图片对象转为 chat.completions 可接受的 image part ---------
    def _to_image_part(self, image_obj: Any) -> dict:
        """
        支持：
        - numpy.ndarray (H,W,3) or (H,W) -> JPEG base64 data URL
        - str: 
            * 以 "http"/"https"/"data:image" 开头：直接作为 URL
            * 其他：当作本地文件路径读取 -> data URL
        - bytes: 原始图像二进制 -> data URL
        """
        # numpy 数组
        if isinstance(image_obj, np.ndarray):
            b64 = self._ndarray_to_b64jpeg(image_obj)
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

        # bytes
        if isinstance(image_obj, (bytes, bytearray)):
            b64 = base64.b64encode(image_obj).decode("utf-8")
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

        # 字符串：URL / data URL / 本地路径
        if isinstance(image_obj, str):
            s = image_obj.strip()
            if s.startswith("http://") or s.startswith("https://") or s.startswith("data:image"):
                return {"type": "image_url", "image_url": {"url": s}}
            # 当作本地文件
            if os.path.isfile(s):
                with open(s, "rb") as f:
                    b = f.read()
                b64 = base64.b64encode(b).decode("utf-8")
                return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            # 兜底：当作 data URL（如果传进来已经是 b64 且没带前缀）
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{s}"}}

        # PIL Image
        if isinstance(image_obj, Image.Image):
            b64 = self._pil_to_b64jpeg(image_obj)
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

        # 其他类型：尝试用 str(image) 兜底
        s = str(image_obj)
        return {"type": "image_url", "image_url": {"url": s}}

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
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")