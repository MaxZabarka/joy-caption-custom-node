import requests
from easy_nodes import ComfyNode, StringInput, NumberInput, Choice

# Define the list of all extras as a constant
EXTRAS_OPTIONS = [
    "If there is a person/character in the image you must refer to them as {name}.",
    "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
    "Include information about lighting.",
    "Include information about camera angle.",
    "Include information about whether there is a watermark or not.",
    "Include information about whether there are JPEG artifacts or not.",
    "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
    "Do NOT include anything sexual; keep it PG.",
    "Do NOT mention the image's resolution.",
    "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
    "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
    "Do NOT mention any text that is in the image.",
    "Specify the depth of field and whether the background is in focus or blurred.",
    "If applicable, mention the likely use of artificial or natural lighting sources.",
    "Do NOT use any ambiguous language.",
    "Include whether the image is sfw, suggestive, or nsfw.",
    "ONLY describe the most important elements of the image.",
]


@ComfyNode()
def send_image_to_endpoint(
    base64_image: str = StringInput("Base64 encoded image data"),
    caption_type: str = Choice(["Descriptive", "Analytical", "Creative"]),
    caption_length: str = Choice(["short", "medium", "long"]),
    extras: list[str] = [
        Choice(["False", "True"]) for description in EXTRAS_OPTIONS
    ],
    name: str = StringInput("Enter a name if applicable"),
    custom_prompt: str = StringInput("Enter a custom prompt"),
    max_tokens: int = NumberInput(300, 0, 1000),
    top_p: float = NumberInput(
        0.9, 0, 1, step=0.01, display="Top P (nucleus sampling)"
    ),
    temperature: float = NumberInput(
        0.6, 0, 2, step=0.1, display="Temperature (creativity)"
    ),
    endpoint: str = StringInput("API Endpoint"),
) -> dict:
    """
    Send image and parameters to the endpoint and return the response.
    """
    # Prepare the extras field by filtering enabled options
    enabled_extras = [
        description
        for choice, description in zip(extras, EXTRAS_OPTIONS)
        if choice == "True"
    ]

    # Prepare the payload
    payload = {
        "input": {
            "base64_image": base64_image,
            "caption_type": caption_type,
            "caption_length": caption_length,
            "extras": enabled_extras,
            "name": name,
            "custom_prompt": custom_prompt,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "temperature": temperature,
        }
    }
    return {}
