import json
import os

import boto3

# AWS configuration - set via environment variables
# Do not hardcode credentials in source code
os.environ['AWS_ACCESS_KEY'] = os.getenv('AWS_ACCESS_KEY', '')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY', '')
os.environ['AWS_SESSION_TOKEN'] = os.getenv('AWS_SESSION_TOKEN', '')
os.environ['AWS_REGION'] = os.getenv('AWS_REGION', 'us-east-1')


def call_llm(system_prompt: str, messages):
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        aws_session_token=os.environ['AWS_SESSION_TOKEN'],
        region_name=os.environ['AWS_REGION'],
    )

    model_id = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
    content_type = 'application/json'
    accept = 'application/json'

    # Body
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "system": system_prompt,
        "messages": messages
    })

    # Run Bedrock API
    response = bedrock_client.invoke_model(
        modelId=model_id,
        contentType=content_type,
        accept=accept,
        body=body
    )

    response_body = json.loads(response.get('body').read())
    return response_body['content'][0]['text']
