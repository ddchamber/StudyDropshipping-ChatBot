import json
import os
import math
import re
from typing import List
from dotenv import load_dotenv
import boto3
import re
from langchain import hub

# Access the environment variables
load_dotenv()
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION")
discord_token = os.getenv("DISCORD_TOKEN")

aws_client = boto3.client(
    "bedrock-runtime",
    region_name="us-west-2",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

# Set file paths
input_path = "/Users/dan/calpoly/BusinessAnalytics/GSB570GENAI/studyDropshipping/structured_output.json"
output_path = "/Users/dan/calpoly/BusinessAnalytics/GSB570GENAI/studyDropshipping/qa_threads.json"

# Load the structured input data
with open(input_path, "r", encoding="utf-8") as f:
    structured = json.load(f)

# Split into batches of 250 messages
batch_size = 250
structured_batches = [structured[i:i+batch_size] for i in range(0, len(structured), batch_size)]

# Utility functions for prompt construction
def build_summary_prompt(batch: List[dict]) -> str:
    return f"""
You are an expert assistant at organizing conversational data into coherent Q&A threads.

Below is a collection of Discord message threads with:
- Root messages (questions/comments/ideas)
- Associated replies
- User identification for each message

Your goal is to transform this raw data into well-structured Q&A pairs that preserve the complete context and information flow. Please:

1. Group messages based on semantic coherence and conversational flow
2. Preserve ALL specific information (URLs, tool names, instructions, code snippets)
3. Consolidate related discussions from the same user, even if they appear as separate threads
4. Maintain the natural question-answer dynamic, including follow-ups and clarifications

Important guidance:
- Consider the full context beyond just punctuation or reply structure
- Messages without question marks can still be questions needing answers
- Pay special attention to who is speaking - MikeChamberlin and DomDei are likely providing authoritative answers
- Messages from other users typically represent questions or requests

When determining thread boundaries:
- Multiple messages from the same user within a brief timeframe likely belong together
- Related follow-ups, clarifications and evolving discussions should be consolidated
- Prioritize conversational coherence over strict reply structure
- Avoid creating separate threads when messages clearly continue a previous discussion

For each coherent thread, please output:

---
Q&A Thread
Question/Topic: [Original question or discussion topic]
Thread: All messages connected to original question in chronological order (This is important so make sure the flow of the conversation is preserved like a human would converse)

---

Only include messages that form part of a coherent conversation. Discard isolated messages without meaningful context or replies.
Please do not include any system messages or instructions in your output.
Remember to maintain the entire original message content, including any URLs or code snippets. You are not summarizing anything, just recording the conversation.

Here are the messages to organize:
{batch}
"""

def build_structuring_prompt(thread_text: str, thread_id: int) -> str:
    return f"""
You are a dropshipping support assistant.

Below is a Discord Q/A thread. Your job is to:
1. Identify the main question.
2. Preserve the full conversation as `full_thread`, do not summarize the responses ever. Keep all information given.
3. Assign it to a category from this list:
   - Product Research
   - Website Customization
   - Sourcing & Suppliers
   - Shopify Setup / Apps
   - Organic Advertising
   - Paid Advertising
   - Mindset / Motivation
   - General Beginner Questions

Output only a JSON object in this format:

{{
  "id": "qa-thread-{thread_id:03d}",
  "header": "...",
  "content": "...",
  "category": "..."
  "source": "discord",
}}

Here is the thread:
--------------------
{thread_text}
--------------------
"""

### Connect to AWS Bedrock
# Select the Bedrock model
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

# Define generation parameters
model_kwargs =  { 
    "max_tokens": 2048, # maximum tokens to return
    "temperature": 0.0, # creativity
    "top_k": 250,       # restrict to top k tokens
    "top_p": 0.9,       # only sample from set of tokens w/ probability ≤ 0.9
    "stop_sequences": ["\n\nHuman"],
}

# Instantiate the ChatBedrock wrapper
from langchain_aws import ChatBedrock
llm = ChatBedrock(
    client=aws_client,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

# The hub provides sample templates for each agent type
prompt = hub.pull("hwchase17/structured-chat-agent") # this is a template pre-tuned for building a structured chat agent

structured_threads = []
thread_id_counter = 1

for batch_index, batch in enumerate(structured_batches):
    print(f"Processing batch {batch_index + 1} of {len(structured_batches)}...")

    # Step 1: Use Claude to group the batch into conversational threads
    try:
        summary_prompt = build_summary_prompt(batch)
        response = llm.invoke(summary_prompt)
    except Exception as e:
        print(f"Error invoking Claude for batch {batch_index + 1}: {e}")
        continue

    # Step 2: Extract individual threads from Claude's output
    thread_blocks = re.split(r"(?m)^\s*---\s*$", response.content)
    thread_blocks = [block.strip() for block in thread_blocks if block.strip()]

    print(f"Found {len(thread_blocks)} threads in batch {batch_index + 1}")

    # Step 3: Format each thread into the desired JSON structure
    for thread_text in thread_blocks:
        struct_prompt = build_structuring_prompt(thread_text, thread_id_counter)
        try:
            result = llm.invoke(struct_prompt)
            qa = json.loads(result.content)
            structured_threads.append(qa)
            thread_id_counter += 1
        except json.JSONDecodeError:
            print(f"Could not parse JSON for thread {thread_id_counter}")
        except Exception as e:
            print(f"Error structuring thread {thread_id_counter}: {e}")

# Step 4: Save all results to the output JSON file
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(structured_threads, f, indent=2, ensure_ascii=False)

print(f"Done. {len(structured_threads)} Q&A threads written to: {output_path}")




