from ollama import chat, ChatResponse

from resource import IfHighlyAbout, USER_PROMPT, KEY_INFO_PROMPT


def classify_if_match(paper_text: str) -> bool:
    response: ChatResponse = chat(
        model='qwen2.5',
        messages=[
            {
                'role': 'user',
                'content': USER_PROMPT + str(paper_text),
            },
        ],
        format=IfHighlyAbout.model_json_schema()
    )
    response_result = IfHighlyAbout.model_validate_json(response.message.content)
    return response_result.result

def extract_key_information(paper_text: str) -> str:
    if not paper_text:
        return ""
    response: ChatResponse = chat(
        model='qwen2.5',
        messages=[{"role": "user", "content": KEY_INFO_PROMPT + "\n\n" + paper_text}]
    )
    return response.message.content or ""