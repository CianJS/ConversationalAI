import json
from utils.conversation_log import recall_conversation
import tiktoken # for token counting
import numpy as np
from collections import defaultdict


encoding = tiktoken.get_encoding("cl100k_base")

# not exact!
# simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(message, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    num_tokens += tokens_per_message
    for key, value in message.items():
        num_tokens += len(encoding.encode(value))
        if key == "name":
            num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(message):
    num_tokens = 0
    if message["role"] == "assistant":
        num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def print_distribution(values, name):
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")


def find_errors(dataset):
    # Format error checks
    format_errors = defaultdict(int)

    for message in dataset:
        if not isinstance(message, dict):
            format_errors["data_type"] += 1
            continue
            
        if not message:
            format_errors["missing_messages_list"] += 1
            continue
            
        if "role" not in message or "content" not in message:
            format_errors["message_missing_key"] += 1
        
        if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
            format_errors["message_unrecognized_key"] += 1
        
        if message.get("role", None) not in ("system", "user", "assistant", "function"):
            format_errors["unrecognized_role"] += 1
            
        content = message.get("content", None)
        function_call = message.get("function_call", None)
        
        if (not content and not function_call) or not isinstance(content, str):
            format_errors["missing_content"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")


def run(dataset):
    # Warnings and tokens counts
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for message in dataset:
        if not message["role"] == "system":
            n_missing_system += 1
        if not message["role"] == "user":
            n_missing_user += 1
        n_messages.append(len(message))
        convo_lens.append(num_tokens_from_messages(message))
        assistant_message_lens.append(num_assistant_tokens_from_messages(message))
        
    print("Num examples missing system message:", n_missing_system)
    print("Num examples missing user message:", n_missing_user)
    print_distribution(n_messages, "num_messages_per_example")
    print_distribution(convo_lens, "num_total_tokens_per_example")
    print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
    n_too_long = sum(l > 16385 for l in convo_lens)
    print(f"\n{n_too_long} examples may be over the 16,385 token limit, they will be truncated during fine-tuning")

if __name__ == "__main__":
    dataset = recall_conversation()
    find_errors(dataset)
    run(dataset)