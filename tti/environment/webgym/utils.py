import base64
import re
import os
import json
import time
import logging
import numpy as np
from PIL import Image
import boto3
import fitz
from .utils_webarena import fetch_browser_info, fetch_page_accessibility_tree,\
                    parse_accessibility_tree, clean_accesibility_tree
import re

def replace_ec2_address(text, new_address_dict):
    # Regular expression pattern to match the EC2 address
    pattern = r'ec2-\d{2,3}-\d{1,3}-\d{1,3}-\d{1,3}\.us-west-2\.compute\.amazonaws\.com'
    text = re.sub(pattern, "WEBARENA_HOST", text)
    pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    text = re.sub(pattern, "WEBARENA_HOST", text)

    # Replace the old EC2 address with the new one
    if "7780" in text:
        new_address = new_address_dict.shopping_admin
    elif "7770" in text:
        new_address = new_address_dict.shopping
    elif "9999" in text:
        new_address = new_address_dict.reddit
    elif "8023" in text:
        new_address = new_address_dict.gitlab
    elif "3000" in text:
        new_address = new_address_dict.map
        #text = text.replace(":3000",":443")

    result = re.sub("WEBARENA_HOST", new_address, text)
    
    return result


def resize_image(image_path):
    image = Image.open(image_path)
    width, height = image.size

    if min(width, height) < 512:
        return image
    elif width < height:
        new_width = 512
        new_height = int(height * (new_width / width))
    else:
        new_height = 512
        new_width = int(width * (new_height / height))

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    resized_image.save(image_path)


# base64 encoding
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def pdf_to_base64_pngs(pdf_path):

    doc = fitz.open(pdf_path)
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(f"{pdf_path}_{page_num}.png")

    base64_pngs = []
    for page_num in range(doc.page_count):
        base64_png = encode_image(f"{pdf_path}_{page_num}.png")
        base64_pngs.append(base64_png)
        os.remove(f"{pdf_path}_{page_num}.png")
    
    return base64_pngs

def get_pdf_retrieval_ans_from_claude(pdf_path, task, region_name="us-east-1", aws_key_id=None, aws_secret_key=None):
    logging.info("You download a PDF file that will be retrieved using the Claude API.")
    base64_pngs = pdf_to_base64_pngs(pdf_path)

    content = [{'type':'text', 'text': task}]
    for encoded_png in base64_pngs[:20]:
        content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': encoded_png}})
    messages = [
        {'role': 'user', 'content': content}
    ]
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": messages,
        "temperature": 0.4,
        "top_p": 0.7,
        'system' : "You are a helpful assistant that can analyze the content of a PDF file and give an answer that matches the given task, or retrieve relevant content that matches the task.",
    }
    if aws_key_id and aws_secret_key:
        client = boto3.client(service_name="bedrock-runtime", region_name=region_name, aws_access_key_id=aws_key_id, aws_secret_access_key=aws_secret_key)
    else:
        client = boto3.client(service_name="bedrock-runtime", region_name=region_name)
    response = client.invoke_model(
        modelId = "anthropic.claude-3-sonnet-20240229-v1:0",
        body=json.dumps(request_body)
    )
    result = json.loads(response.get("body").read())
    return result['content'][0]['text']

def get_pdf_retrieval_ans_from_assistant(client, pdf_path, task):
    logging.info("You download a PDF file that will be retrieved using the Assistant API.")
    file = client.files.create(
        file=open(pdf_path, "rb"),
        purpose='assistants'
    )
    logging.info("Create assistant...")
    assistant = client.beta.assistants.create(
        instructions="You are a helpful assistant that can analyze the content of a PDF file and give an answer that matches the given task, or retrieve relevant content that matches the task.",
        model="gpt-4-1106-preview",
        tools=[{"type": "retrieval"}],
        file_ids=[file.id]
    )
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=task,
        file_ids=[file.id]
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    while True:
        # Retrieve the run status
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status == 'completed':
            break
        time.sleep(2)
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    messages_text = messages.data[0].content[0].text.value
    file_deletion_status = client.beta.assistants.files.delete(
        assistant_id=assistant.id,
        file_id=file.id
    )
    logging.info(file_deletion_status)
    assistant_deletion_status = client.beta.assistants.delete(assistant.id)
    logging.info(assistant_deletion_status)
    return messages_text
