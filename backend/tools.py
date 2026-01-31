# Tools (web, pdf, html, OCR, LLM, chroma)

from ddgs import DDGS
from groq import Groq
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

def search_web(query: str, max_results=5):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(r.get("body", ""))
    except Exception as e:
        print(f"Error while searching web: {e}")
    return results

def call_llm(prompt: str):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.3
    )
    return response.choices[0].message.content

def call_llm_stream(prompt: str):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        stream=True,  # ✅ enable streaming
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# if __name__ == "__main__":
#     print(search_web("What is LangGraph?"))
    # print(call_llm("What is LangGraph?"))