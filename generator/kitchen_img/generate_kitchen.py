"""Generate photorealistic kitchen images using Google Gemini API.

Ported from the pipeline branch's kitchen_image_gen/ Next.js implementation.

Requirements:
    pip install google-generativeai

Environment variable:
    GEMINI_API_KEY=your_api_key_here

Model used: gemini-2.0-flash-exp (supports image generation response modality).
The pipeline branch used gemini-3.1-flash-image-preview (TypeScript SDK);
update MODEL_NAME below if your API access uses a different model.
"""

import base64
import os

MODEL_NAME = "gemini-2.0-flash-exp"

PROMPT_PREFIX = (
    "Generate a photorealistic image of a commercial kitchen interior. "
    "The image should show the floor clearly with good perspective. "
    "No people or animals should be present. "
)

PROMPT_TEMPLATES = [
    "Large commercial kitchen with white tile floor, stainless steel counters, overhead fluorescent lighting, wide angle shot from corner",
    "Restaurant kitchen with dark stone floor, industrial shelving, warm tungsten lighting, shot from doorway perspective",
    "Small cafe kitchen with checkered linoleum floor, wooden counters, natural window light, slightly elevated camera angle",
    "Industrial kitchen with concrete floor, metal prep tables, harsh overhead LED panels, straight-on view",
    "Hotel kitchen with gray tile floor, marble counters, mixed warm and cool lighting, diagonal perspective",
    "Bakery kitchen with terracotta tile floor, wooden workbenches, pendant lights, wide shot",
    "Fast food kitchen with red tile floor, stainless steel equipment, bright fluorescent lights, low angle shot",
    "School cafeteria kitchen with beige vinyl floor, institutional counters, ceiling grid lights, wide angle",
]


def generate_kitchen_image(prompt: str, api_key: str) -> dict:
    """Generate a kitchen image using Google Gemini API.

    Args:
        prompt: Descriptive prompt appended to the standard prefix.
        api_key: Google Gemini API key.

    Returns:
        dict with keys:
            image_bytes (bytes): Raw image data.
            mime_type (str): e.g. 'image/png' or 'image/jpeg'.

    Raises:
        ImportError: If google-generativeai package is not installed.
        ValueError: If Gemini returns no image.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai package is required. "
            "Install with: pip install google-generativeai"
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=MODEL_NAME)

    full_prompt = PROMPT_PREFIX + prompt
    response = model.generate_content(
        full_prompt,
        generation_config={"response_modalities": ["IMAGE", "TEXT"]},
    )

    for part in response.candidates[0].content.parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return {
                "image_bytes": base64.b64decode(inline.data),
                "mime_type": getattr(inline, "mime_type", None) or "image/png",
            }

    raise ValueError(
        "Gemini did not return an image. "
        "Ensure GEMINI_API_KEY is valid and the model supports image generation. "
        f"Model used: {MODEL_NAME}"
    )
