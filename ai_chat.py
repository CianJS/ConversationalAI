import json
from openai import OpenAI
import os

from utils.conversation_log import recall_conversation

env = os.environ
open_ai_key = env['OPENAI_API_KEY']
client = OpenAI(
    api_key=open_ai_key
)

def generate_response(user_input):
    # 캐릭터 기억 가져오기
    messages = recall_conversation()
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
        temperature=0.7,
    )

    answer = ''
    try:
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                answer += chunk.choices[0].delta.content
    except Exception as e:
        answer = e

    return answer
