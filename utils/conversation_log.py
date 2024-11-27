import os
import json

base_path = os.path.dirname(os.path.abspath(__file__))
identity_path = os.path.join(base_path, '../config/character.txt')
conversation_log_path = os.path.join(base_path, '../data/conversationLog.json')

def get_character_config():
    with open(identity_path, "r", encoding="utf-8") as f:
        character_config = f.read()
    return {"role": "system", "content": character_config}


def recall_conversation():
    conversation_log = []
    character_config = get_character_config()
    conversation_log.append(character_config)
    conversation_log.append({'role': 'system', 'content': '아래 내용은 그동안의 대화 내용입니다.'})

    with open(conversation_log_path, 'r') as h:
        for i in json.load(h)['history']:
            conversation_log.append(i)
    
    conversation_log.append({'role': 'system', 'content': '여기까지가 그동안 진행했던 대화의 내용입니다.'})
    return conversation_log


def remember_conversation(new_logs):
    conversation = {}

    with open(conversation_log_path, 'r') as h:
        conversation = json.load(h)
    
    for l in new_logs:
        conversation['history'].append(l)

    with open(conversation_log_path, 'w', encoding='utf-8') as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)
