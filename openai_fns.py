from typing import Optional, Type, TypeVar

from openai import OpenAI
from openai.types.chat import ChatCompletion, ParsedChatCompletion
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

client = OpenAI()


def call_gpt(
    msg: str,
    only_return_text: bool = False,
    model: str = "gpt-4o-mini",
    system_prompt: Optional[str] = None,
) -> ChatCompletion | str:
    if system_prompt:
        msgs = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": msg},
        ]
    else:
        msgs = [
            {"role": "user", "content": msg},
        ]
    response = client.chat.completions.create(model=model, messages=msgs)
    if only_return_text:
        return response.choices[0].message.content
    else:
        return response


def call_gpt_strict_response(
    msg: str,
    response_format: Type[T],
    only_return_object: bool = False,
    model: str = "gpt-4o-mini",
    system_prompt: Optional[str] = None,
) -> ParsedChatCompletion[T]:
    if system_prompt:
        msgs = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": msg},
        ]
    else:
        msgs = [
            {"role": "user", "content": msg},
        ]
    response = client.beta.chat.completions.parse(
        model=model, messages=msgs, response_format=response_format
    )
    if only_return_object:
        return response_format.model_validate(response.choices[0].message.parsed)
    return response


def get_response_text(response: ChatCompletion) -> str:
    return response.choices[0].message.content


if __name__ == "__main__":
    from typing import List

    class listOfStrings(BaseModel):
        numbers: List[str]

    print(call_gpt("What's up?", only_return_text=True))
    response = call_gpt_strict_response(
        "give me a list of numbers",
        response_format=listOfStrings,
        only_return_object=True,
    )
    print(response)
    print(type(response))
