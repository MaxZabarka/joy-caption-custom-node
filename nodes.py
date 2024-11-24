import requests
import numpy as np
import base64
import io
from PIL import Image
from easy_nodes import ComfyNode, StringInput, NumberInput, Choice, ImageTensor
import os

list_of_things = []


@ComfyNode()
def joy_caption_alpha_two(
    image: ImageTensor,
    caption_type: str = Choice(
        [
            "Descriptive",
            "Descriptive (Informal)",
            "Training Prompt",
            "MidJourney",
            "Booru tag list",
            "Booru-like tag list",
            "Art Critic",
            "Product Listing",
            "Social Media Post",
        ]
    ),
    caption_length: str = Choice(
        ["any", "very short", "short", "medium-length", "long", "very long"]
        + [str(i) for i in range(20, 261, 10)]
    ),
    extra_person_character: bool = False,
    extra_changeable_attributes: bool = True,
    extra_lighting: bool = False,  # "Include information about lighting."
    extra_camera_angle: bool = False,
    extra_watermark: bool = False,
    extra_jpeg_artifacts: bool = False,
    extra_camera_details: bool = False,
    extra_keep_pg: bool = False,
    extra_no_resolution: bool = False,
    extra_aesthetic_quality: bool = False,
    extra_composition_style: bool = False,
    extra_no_text: bool = True,
    extra_depth_of_field: bool = False,
    extra_lighting_sources: bool = False,
    extra_no_ambiguous: bool = True,
    extra_content_rating: bool = True,
    extra_most_important: bool = True,
    top_p: float = NumberInput(
        0.9, 0, 1, step=0.01
    ),
    temperature: float = NumberInput(
        0.6, 0, 2, step=0.1
    ),
    max_tokens: int = NumberInput(300, 0, 1000, step=1),
    name: str = StringInput(""),
    custom_prompt: str = StringInput(""),
    endpoint: str = StringInput("https://api.runpod.ai/v2/4mw06685te9vzh/runsync"),
) -> str:
    extras = []
    if extra_person_character:
        extras.append(
            "If there is a person/character in the image you must refer to them as {name}."
        )
    if extra_changeable_attributes:
        extras.append(
            "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style)."
        )
    if extra_lighting:
        extras.append("Include information about lighting.")
    if extra_camera_angle:
        extras.append("Include information about camera angle.")
    if extra_watermark:
        extras.append("Include information about whether there is a watermark or not.")
    if extra_jpeg_artifacts:
        extras.append(
            "Include information about whether there are JPEG artifacts or not."
        )
    if extra_camera_details:
        extras.append(
            "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc."
        )
    if extra_keep_pg:
        extras.append("Do NOT include anything sexual; keep it PG.")
    if extra_no_resolution:
        extras.append("Do NOT mention the image's resolution.")
    if extra_aesthetic_quality:
        extras.append(
            "You MUST include information about the subjective aesthetic quality of the image from low to very high."
        )
    if extra_composition_style:
        extras.append(
            "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry."
        )
    if extra_no_text:
        extras.append("Do NOT mention any text that is in the image.")
    if extra_depth_of_field:
        extras.append(
            "Specify the depth of field and whether the background is in focus or blurred."
        )
    if extra_lighting_sources:
        extras.append(
            "If applicable, mention the likely use of artificial or natural lighting sources."
        )
    if extra_no_ambiguous:
        extras.append("Do NOT use any ambiguous language.")
    if extra_content_rating:
        extras.append("Include whether the image is sfw, suggestive, or nsfw.")
    if extra_most_important:
        extras.append("ONLY describe the most important elements of the image.")

    image = image[0]

    if len(image.shape) == 2:
        image = image.unsqueeze(-1)

    if image.shape[-1] == 1:
        image = torch.cat([image] * 3, axis=-1)

    image = image.cpu().numpy()

    pil_image = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8))
    # pil_image = Image.fromarray(image[0].numpy(), mode="RGB")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")

    pil_image.save("test.png")

    buffer.seek(0)

    base64_image = base64.b64encode(buffer.read()).decode("utf-8")
    print(base64_image[:100])

    payload = {
        "input": {
            "base64_image": base64_image,
            "caption_type": caption_type,
            "caption_length": caption_length,
            "extras": extras,
            "name": name,
            "custom_prompt": custom_prompt,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "temperature": temperature,
        }
    }

    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    response.raise_for_status()
    return response.text
