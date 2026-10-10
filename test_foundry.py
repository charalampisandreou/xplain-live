import json
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import OpenAI

ENDPOINT = "https://plhack-foundry-candreou-resource.openai.azure.com/openai/v1/"
DEPLOYMENT = "gpt-4.1-mini"

token_provider = get_bearer_token_provider(
    AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
)
client = OpenAI(base_url=ENDPOINT, api_key=token_provider)

SYSTEM = (
    "You are a football commentator. Use only the facts given. "
    "Convert xP into a plain percentage chance (e.g. 'a 12% pass'); never say 'expected pass value'. "
    "Say 'pass' or 'ball played', never 'throw'. "
    "Write ONE sentence of at most 18 words, present tense. "
    "Return JSON: {\"line_en\": str, \"line_es\": str, \"numbers_used\": [str]}."
)
facts = {"xP": 0.12, "completed": True, "progressive": True,
         "drivers": ["long distance", "under pressure"]}

resp = client.chat.completions.create(
    model=DEPLOYMENT,
    messages=[{"role": "system", "content": SYSTEM},
              {"role": "user", "content": json.dumps(facts)}],
    response_format={"type": "json_object"},
    max_completion_tokens=150,
)
print(json.loads(resp.choices[0].message.content))