from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv("../LangChain/.env")

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.8,
    max_tokens=1000
)

prompts = {
    "instagram": ChatPromptTemplate.from_messages([
        ("system", """Write Instagram captions that feel completely human-written.
Rules: Zero emojis. No hashtags unless asked. No words like amazing/exciting/thrilled.
Short punchy sentences. Casual smart tone. Sound like a real person, not a brand."""),
        ("human", "Tone: {tone}\nWrite Instagram caption for: {input}")
    ]),
    "linkedin": ChatPromptTemplate.from_messages([
        ("system", """Write LinkedIn posts that feel genuinely human.
Rules: No emojis. No 'I am excited to share'. Professional but conversational.
Tell a story or insight. Short paragraphs. Max 3 hashtags at end only if needed."""),
        ("human", "Tone: {tone}\nWrite LinkedIn post about: {input}")
    ]),
    "twitter": ChatPromptTemplate.from_messages([
        ("system", """Write tweets that sound like a real witty human. Under 280 characters.
Rules: No emojis. Witty and direct. No corporate tone. Hook immediately."""),
        ("human", "Tone: {tone}\nWrite a tweet about: {input}")
    ]),
    "tiktok": ChatPromptTemplate.from_messages([
        ("system", """Write TikTok video captions/scripts that feel authentic and engaging.
Rules: Conversational Gen-Z tone. Hook in first line. Short sentences.
No corporate speak. Can use 1-2 relevant emojis only if they add meaning.
Include a hook, main content idea, and call to action."""),
        ("human", "Tone: {tone}\nWrite TikTok caption/script idea for: {input}")
    ]),
    "facebook": ChatPromptTemplate.from_messages([
        ("system", """Write Facebook posts that feel like a real person sharing something.
Rules: Warm community tone. Conversational. Can be slightly longer than other platforms.
No emojis overuse. Engage the audience with a question at end. Human and relatable."""),
        ("human", "Tone: {tone}\nWrite Facebook post about: {input}")
    ]),
    "full_pack": ChatPromptTemplate.from_messages([
        ("system", """Write a complete social media content pack. All human style.
Rules: Zero emojis except TikTok (max 2). No corporate speak. Real human voice.
Each platform should have unique content suited for that platform.

Create content for:
## INSTAGRAM
(casual, visual, short)

## LINKEDIN  
(professional, story-driven, insight)

## TWITTER/X
(under 280 chars, witty, direct)

## TIKTOK
(hook + script idea, gen-z tone)

## FACEBOOK
(warm, community, ends with question)"""),
        ("human", "Tone: {tone}\nCreate full content pack for: {input}")
    ])
}

chains = {key: prompt | llm | StrOutputParser() for key, prompt in prompts.items()}

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    tool = data.get("tool", "instagram")
    user_input = data.get("input", "")
    tone = data.get("tone", "casual")
    if not user_input:
        return jsonify({"result": "Please enter some input!"})
    result = chains[tool].invoke({"input": user_input, "tone": tone})
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True, port=5003)