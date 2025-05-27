import discord
import sqlite3
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import torch
import boto3

torch.set_num_threads(1)

# --- Z-score filter ---
def calculate_zscores(cosine_scores):
    mean = np.mean(cosine_scores)
    std_deviation = np.std(cosine_scores, ddof=1)
    return [(x - mean) / std_deviation if std_deviation != 0 else 0 for x in cosine_scores]

# --- Environment & Keys ---
load_dotenv()
discord_token = os.getenv("DISCORD_TOKEN")
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION")
channel = 1375597403750797493

# --- AWS Claude Setup ---
aws_client = boto3.client(
    "bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

model_id = "anthropic.claude-3-haiku-20240307-v1:0"
model_kwargs = {
    "max_tokens": 2048,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 0.9,
    "stop_sequences": ["\n\nHuman"],
}

from langchain_aws import ChatBedrock
llm = ChatBedrock(client=aws_client, model_id=model_id, model_kwargs=model_kwargs)

# --- Discord Setup ---
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Load embeddings & index map
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = np.load("models/embeddings.npy")  # shape: (N, 384), already normalized
with open("models/id_map.txt") as f:
    id_map = f.read().splitlines()

conversation_history = []
last_user_question = {"text": None}  # use a mutable dict so you can update it inside the event
last_query_data = {
    "question": None,
    "top_threads": None,
    "formatted_history": None,
    "z_scores": None,
    "cosine_scores": None,
    "gap": None,
}

@client.event
async def on_ready():
    print(f"Bot is online as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.channel.id != channel:
        return# Ignore messages not from the allowed channel

    try:
        user_input = message.content.strip().lower()

        if user_input == "/show_context":
            if not last_query_data["formatted_history"]:
                await message.channel.send("⚠️ No recent question to show context for.")
                return

            debug_output = f"Auto Z-Score Threshold Based on Dropoff\n\n"
            debug_output += f"Largest gap = {last_query_data['gap']:.2f}\n"
            debug_output += f"Keeping top {len(last_query_data['top_threads'])} threads\n\n"

            for idx, cos_sim, z in last_query_data["top_threads"]:
                debug_output += f"• {id_map[idx]} | z={z:.2f} | cos_sim={cos_sim:.3f}\n"

            debug_output += "\n\nFormatted Context Sent to Claude:\n\n"
            debug_output += last_query_data["formatted_history"]

            await message.channel.send(f"```{debug_output[:1950]}```")
            return

        # --- REAL QUESTION: Run new embedding and match ---
        user_question = message.content.strip()
        print(f"New user question: {user_question}")

        query_emb = model.encode(user_question).reshape(1, -1)
        query_emb = query_emb / np.linalg.norm(query_emb)

        print("Calculating cosine similarity")
        cosine_scores = cosine_similarity(query_emb, embeddings)[0]
        top_k = 20
        top_indices = np.argsort(cosine_scores)[::-1][:top_k]

        z_scores = calculate_zscores(cosine_scores[top_indices])

        sorted_z = sorted(zip(top_indices, cosine_scores[top_indices], z_scores), key=lambda x: x[2], reverse=True)
        max_gap, best_cutoff_idx = 0, 0
        for i in range(1, len(sorted_z)):
            gap = sorted_z[i - 1][2] - sorted_z[i][2]
            if gap > max_gap:
                max_gap = gap
                best_cutoff_idx = i
        top_threads = sorted_z[:best_cutoff_idx]

        print(f"Using top {len(top_threads)} threads (dropoff at gap = {max_gap:.2f})")

        filtered_threads = []
        for idx, cos_sim, z in top_threads:
            thread_id = id_map[idx]
            conn = sqlite3.connect("data/threads.db")
            cursor = conn.cursor()
            cursor.execute("SELECT header, content FROM threads WHERE id = ?", (thread_id,))
            result = cursor.fetchone()
            conn.close()
            if result:
                filtered_threads.append((result[0], result[1]))

        formatted_history = "\n\n".join([f"Closest Q: {header}\nA: {content}" for header, content in filtered_threads])

        # 🔁 Store results for reuse by /show_context
        last_query_data["question"] = user_question
        last_query_data["top_threads"] = top_threads
        last_query_data["z_scores"] = z_scores
        last_query_data["cosine_scores"] = cosine_scores[top_indices]
        last_query_data["gap"] = max_gap
        last_query_data["formatted_history"] = formatted_history

        formatted_conversation = ""
        for turn in conversation_history[-5:]: 
            formatted_conversation += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"


        # --- Prompt to Claude ---
        rag_prompt = f"""
<Role>
You are an expert in helping complete beginners become successful dropshippers. You specialize in: Product Research, Website Overview and Customization, Sourcing and Suppliers, TRUST Dropshipping Group, Mindset, Organic Advertising, Paid Advertising, Shopify Apps
You have learned from an 8-hour course by Mike and Dom, as well as the Study Dropshipping Discord community FAQ threads. You speak clearly, simply, and with enthusiasm, always aiming to help users understand and succeed. You are capable of: Guiding users to their goals, explaining concepts without skipping key details, diving deeper if asked, providing examples when relevant. 
</Role>

<Task Flow>
When a user asks a question:
Retrieve relevant information from the internal knowledge base (course material or Discord).

Answer the user’s question step-by-step, as clearly as possible. Include only the response section when you print to the customer:
<information> (facts pulled from course/FAQ) </information>  
<goal> (what the user is trying to accomplish) </goal>  
<difficulty> (how hard or easy this task is) </difficulty> 
<response> (what the user will see based on the steps above, make depth based on difficulty)</response>

Include only the response section when you print to the customer.
</Task Flow>

<Business Specifics>
Your ultimate goal is to help users move forward in their dropshipping journey—ideally to the point where they: Sign up for a Shopify trial using the company’s affiliate link, and then start a paid subscription. 
By making the process simple, clear, and motivating, you increase the chance that users will become successful dropshippers and long-term subscribers. 
</Business Specifics>

<Examples>
Q: How do I choose a winning product?
A:
 <information>
 Winning products usually share a few characteristics:
- They're easy to make eye-catching videos for (think: “TikTok-worthy”).
- They spark emotional reactions—especially controversy or surprise.
- They solve a problem or make life easier.
 </information>
<goal> To identify a product that will perform well in ads and convert customers. </goal>
<difficulty> Medium – Requires practice and testing, but tools and examples help. </difficulty>
<response> Great question — picking the right product is one of the biggest steps in getting traction with dropshipping.
Start by looking for products that catch attention fast. Ask yourself: Would this stop me from scrolling on TikTok or Instagram? Products that are visually interesting, solve a real problem, or stir some emotion (like surprise or controversy) tend to perform the best in ads.
Next, think about how unique the product feels. If it’s already everywhere, it’ll be tough to stand out. And finally, check the numbers — look for something you can sell for 3x what it costs you to source.
It might take a few tries to land on the right one, but with research tools and inspiration from what’s already working for others, you’ll be able to spot the patterns.
Want help brainstorming or validating a product you’re thinking about?
</response>
</Examples>

<Conversation History>
{formatted_conversation}
</Conversation History>

<User Question and Information from database>
A user just asked this question:
"{user_question}"

Relevant Threads:
{formatted_history}
</User Question and Information from database>


<Reiteration>
You are a friendly, professional dropshipper who wants to grow the community through free, helpful, and clear advice. Be excited to help, break things down step-by-step, and always aim to get the user closer to taking action.
Only return the <response> without the tags.
</Reiteration>
"""  
        print("Calling Claude")
        claude_response = llm.invoke(rag_prompt)

        await message.channel.send(claude_response.content.strip())

        conversation_history.append({
            "question": user_question,
            "answer": claude_response.content.strip()
        })

    except Exception as e:
        print("ERROR:", e)
        await message.channel.send("Something went wrong.")

client.run(discord_token)
