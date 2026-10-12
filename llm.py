import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY is missing from the .env file."
    )


client = Groq(api_key=api_key)

MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)


def call_groq(
    system_prompt,
    user_prompt,
    max_tokens=512,
):
    """
    Send a token-controlled request to Groq.

    GPT-OSS uses internal reasoning tokens, so it needs
    more than 250 completion tokens even for short SQL.
    """

    # GPT-OSS may use part of the completion allowance
    # for internal reasoning.
    effective_max_tokens = max(max_tokens, 512)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_completion_tokens=effective_max_tokens,

            # Keep reasoning short.
            reasoning_effort="low",

            # Do not return reasoning content.
            include_reasoning=False,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

    except Exception as error:
        raise RuntimeError(
            f"Groq request failed using model "
            f"'{MODEL_NAME}': {error}"
        ) from error

    choice = response.choices[0]
    content = choice.message.content

    if response.usage:
        print(
            "\nGroq token usage:"
            f"\n  Model:  {MODEL_NAME}"
            f"\n  Input:  {response.usage.prompt_tokens}"
            f"\n  Output: {response.usage.completion_tokens}"
            f"\n  Total:  {response.usage.total_tokens}"
            f"\n  Finish: {choice.finish_reason}"
        )

    if not content or not content.strip():
        if choice.finish_reason == "length":
            raise RuntimeError(
                "The model reached its output-token limit "
                "before producing the final SQL."
            )

        raise RuntimeError(
            "Groq returned an empty final response."
        )

    return content.strip()