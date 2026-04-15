from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

def get_video_id(url):
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    elif "youtube.com" in url:        
        parsed = urlparse(url)
        return parse_qs(parsed.query).get('v', [None])[0]
    else:
        return None

print("=" * 50)
print("YouTube Video Summarizer")
print("=" * 50)

video_url = input("Enter URL: \n")
video_id = get_video_id(video_url)

yt_transcript_api = YouTubeTranscriptApi()

if video_id is None:
    print("Invalid URL!")
    exit()

transcript = yt_transcript_api.fetch(video_id)

combined_transcript = "No caption/transcription found!"
if transcript is not None:
    combined_transcript = ""
    for snippet in transcript:
        combined_transcript += snippet.text + " "

gemini_api_key = os.getenv("GOOGLE_API_KEY")

system_prompt = ChatPromptTemplate.from_template("""
Summarize this YouTube video transcript in a clear, 
structured format.
                                                 
Video URL: {video_url}

Transcript:
{transcript_text}

Provide:
1. A concise 2-3 paragraph summary and a suitable title.
2. Key topics covered (bullet points)
3. Main takeaways (numbered list)
4. Whether the video is worth watching and for whom

Keep it concise but informative.
""")

llm = ChatGoogleGenerativeAI(
    api_key=gemini_api_key,
    model="gemini-2.5-flash",
    temperature=0.7
)

llm_chain = system_prompt | llm

print("=" * 50)
print("Requesting AI to summarize the video...")
print("=" * 50)

response = llm_chain.invoke({
    "transcript_text": combined_transcript,
    "video_url": video_url
})

print("Received response from AI successfully.")

print("=" * 50)
print("Summary:")
print("=" * 50)
print(response.content)