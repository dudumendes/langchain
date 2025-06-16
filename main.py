from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

number_of_days = 7
number_of_children = 2
activity = "beach"

prompt = f"Create a {number_of_days}-day travel itinerary for a family with {number_of_children} children who like {activity}"

print(prompt) 

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a travel planning assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=500,
    temperature=0.7)

print(response.choices[0].message.content)