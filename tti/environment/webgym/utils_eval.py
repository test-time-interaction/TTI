import argparse
import os
import json
import time
import re
import base64
import anthropic
from typing import Any
from copy import deepcopy
from tti.environment.webgym.helper_functions import (
    llm_fuzzy_match,
    llm_ua_match,
)
from nltk.tokenize import word_tokenize
import logging
import traceback
import collections
import urllib
import html

"""Implements helper functions to assist evaluation cases where other evaluators are not suitable."""
import json
from typing import Any
from urllib.parse import urlparse

import requests
from openai import OpenAI

REDDIT = "http://WEBARENA_HOST:PORT"
GITLAB = "http://WEBARENA_HOST:PORT"
SHOPPING_ADMIN = "http://WEBARENA_HOST:PORT/admin"
SHOPPING = "http://WEBARENA_HOST:PORT"

import re
import asyncio
import logging
import os
import random
import time

# import aiolimiter
import openai
from tqdm.asyncio import tqdm_asyncio

def shopping_get_auth_token(url) -> str:
    response = requests.post(
        url=f"{url}/rest/default/V1/integration/admin/token",
        headers={"content-type": "application/json"},
        data=json.dumps(
            {
                "username": "emma.lopez@gmail.com",
                "password": "Password.123",
            }
        ),
    )
    token: str = response.json()
    return token


def shopping_get_latest_order_url(url) -> str:
    """Get the latest order url from the shopping website."""

    header = {
        "Authorization": f"Bearer {shopping_get_auth_token(url)}",
        "Content-Type": "application/json",
    }

    params = {
        "searchCriteria[sortOrders][0][field]": "created_at",
        "searchCriteria[sortOrders][0][direction]": "DESC",
        "searchCriteria[pageSize]": "1",
    }

    response = requests.get(
        f"{url}/rest/V1/orders", params=params, headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()["items"][0]
    order_id = int(response_obj["increment_id"])
    order_url = f"{url}/sales/order/view/order_id/{order_id}/"
    return order_url


def shopping_get_sku_latest_review_author(sku: str, url) -> str:
    """Get the latest review for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token(url)}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{url}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    author: str = response_obj[-1]["nickname"]
    return author


def shopping_get_sku_latest_review_rating(sku: str, url) -> str:
    """Get the latest review for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token(url)}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{url}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    assert response_obj[0]["ratings"][0]["rating_name"] == "Rating"
    rating: str = str(response_obj[-1]["ratings"][0]["percent"])
    return rating


def reddit_get_post_url(url: str) -> str:
    """Get the post url"""
    # Url is http://domain/f/subreddit/post_id/...
    # get domain, subreddit, post_id
    domain = urlparse(url).netloc
    tok_url = urlparse(url).path.split("/")
    # not a valid post/comment url, return the url as is
    if len(tok_url) < 4:
        return url
    if tok_url[1] != "f":
        return url
    subreddit = urlparse(url).path.split("/")[2]
    post_id = urlparse(url).path.split("/")[3]
    scheme = urlparse(url).scheme
    post_url = f"{scheme}://{domain}/f/{subreddit}/{post_id}/"
    return post_url


def gitlab_get_project_memeber_role(driver: None, account_name: str) -> str:
    # get the account index
    try:
        # Get account index via JS
        account_idx = driver.execute_script(
            """
            const accountName = arguments[0];
            const elements = document.querySelectorAll("td[data-label='Account'] span.gl-avatar-labeled-sublabel");
            let index = -1;
            for (let i = 0; i < elements.length; i++) {
                if (elements[i].outerText === "@" + accountName) {
                    index = i;
                    break;
                }
            }
            return index;
            """,
            account_name
        )
    
        if account_idx == -1:
            return ""
    
        # Get the role using the index
        role = driver.execute_script(
            """
            const index = arguments[0];
            const roleElements = document.querySelectorAll("td.col-max-role span");
            return roleElements[index]?.outerText || "";
            """,
            account_idx
        )
    
    except Exception as e:
        print("[GITLAB EVAL ERROR]", e)
        role = ""
    
    return role



USER_PROMPT = """TASK: <task>
Result Response: <answer>
<num> screenshots at the end: """

USER_REFERENCE_PROMPT = """TASK: <task>
Result Response: <answer>
Reference Response: <reference>
<num> screenshots at the end: """


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class StringEvaluator():
    """Check whether the answer is correct with:
    exact match: the answer is exactly the same as the reference answer
    must include: each phrase in the reference answer must be included in the answer
    fuzzy match: the answer is similar to the reference answer, using LLM judge
    """

    def __init__(self):
        self.client = None

    def clean_answer(self, answer: str) -> str:
        answer = answer.strip()
        if answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]
        elif answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        return answer.lower()

    def exact_match(self, ref: str, pred: str) -> float:
        return float(
            self.clean_answer(pred)
            == self.clean_answer(ref)
        )

    def must_include(self, ref: str, pred: str, tokenize: bool = False) -> float:
        clean_ref = self.clean_answer(ref)
        clean_pred = self.clean_answer(pred)
        # tokenize the answer if the ref is a single word
        # prevent false positive (e.g, 0)
        if (
            tokenize
            and len(clean_ref) == 1
            and len(word_tokenize(clean_ref)) == 1
        ):
            tok_pred = word_tokenize(clean_pred)
            return float(clean_ref in tok_pred)
        else:
            return float(clean_ref in clean_pred)

    def fuzzy_match(self, ref: str, pred: str, intent: str, client) -> float:
        return llm_fuzzy_match(pred, ref, intent)#, client)

    def ua_match(self, ref: str, pred: str, intent: str, client) -> float:
        return llm_ua_match(pred, ref, intent)#, client)

    def __call__(
        self,
        task_content,
        answer,
        eval_config,
        driver
    ) -> float:
        pred = self.clean_answer(answer)
        
        score = 1.0
        for approach, value in eval_config["reference_answers"].items():
            match approach:
                case "exact_match":
                     
                    score *= self.exact_match(value, pred)

                case "must_include":
                    assert isinstance(value, list)
                    for must_value in value:
                        score *= self.must_include(
                            ref=must_value,
                            pred=pred,
                            tokenize=(len(value) == 1),
                        )
                    
                case "fuzzy_match":
                    intent = task_content
                    if value == "N/A":
                        # if the instruction only asks the model to generate N/A when encountering an unachievable task
                        # without more concrete reasons
                        score *= self.exact_match(ref=value, pred=pred)
                        # if the instruction also asks the model to generate the reason why the task is unachievable
                        # this should be the default as it will prevent false positive N/A`
                        if score != 1:
                            score = 1.0 * self.ua_match(
                                intent=task_content,
                                ref=eval_config["string_note"],
                                pred=pred,
                                client=self.client,
                            )
                    else:
                        assert isinstance(value, list)
                        # for reference in value:
                        score *= self.fuzzy_match(
                                ref=", ".join(value), pred=pred, intent=intent, client=self.client,
                            )
        return score
        
def replace_ip_and_port(target_url, url_to_modify):
    # Extract IP and port from target URL
    ip_port_pattern = r'http://([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+):([0-9]+)'
    if "443" in target_url:
        target_ip = "ec2-98-84-50-50.compute-1.amazonaws.com"
        target_port = "443"
    else:      
        target_match = re.search(ip_port_pattern, target_url)
            
        if not target_match:
            return url_to_modify
        
        target_ip = target_match.group(1)
        target_port = target_match.group(2)

    url_to_modify = url_to_modify.replace("WEBARENA_HOST", target_ip).replace("PORT", target_port)
    # Replace IP and port in the second URL
    modified_url = re.sub(
        ip_port_pattern,
        f'http://{target_ip}:{target_port}',
        url_to_modify
    )
    return modified_url

class URLEvaluator():
    """Check URL matching"""

    def __call__(
            self,
            task_content,
            answer,
            eval_config,
            driver
            ):
        def clean_url(url: str) -> str:
            url = str(url)
            url = url.rstrip("/")
            return url

        def parse_url(url: str) -> tuple[str, dict[str, list[str]]]:
            """Parse a URL into its base, path, and query components."""
            parsed_url = urllib.parse.urlparse(url)
            base_path = parsed_url.netloc + parsed_url.path
            query = urllib.parse.parse_qs(parsed_url.query)
            return base_path, query      

        def parse_urls(
            urls: list[str],
        ) -> tuple[list[str], dict[str, set[str]]]:
            """Parse a list of URLs."""
            base_paths = []
            queries = collections.defaultdict(set)
            for url in urls:
                base_path, query = parse_url(url)
                base_paths.append(base_path)
                for k, v in query.items():
                    queries[k].update(v)
            return base_paths, queries

        pred = clean_url(driver.current_url)
        ref_urls = replace_ip_and_port(eval_config["webarena_starting_url"], eval_config["reference_url"]).split(" |OR| ")
        ref_urls = [clean_url(url) for url in ref_urls]
        matching_rule = eval_config.get("url_note", "GOLD in PRED")
        
        if matching_rule == "GOLD in PRED":
            ref_base_paths, ref_queries = parse_urls(ref_urls)
            pred_base_paths, pred_query = parse_url(pred)

            base_score = float(
                any(
                    [
                        ref_base_path in pred_base_paths
                        for ref_base_path in ref_base_paths
                    ]
                )
            )
            query_score = 1.0
            for k, possible_values in ref_queries.items():
                query_score *= float(
                    any(
                        possible_ref_value in pred_query.get(k, [])
                        for possible_ref_value in possible_values
                    )
                )
            score = base_score * query_score
        else:
            raise ValueError(f"Unknown matching rule: {matching_rule}")
        return score

class HTMLContentEvaluator():
    """Check whether the contents appear in the page"""

    def clean_answer(self, answer: str) -> str:
        answer = answer.strip()
        if answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]
        elif answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        return answer.lower()

    def exact_match(self, ref: str, pred: str) -> float:
        return float(
            self.clean_answer(pred)
            == self.clean_answer(ref)
        )

    def must_include(self, ref: str, pred: str, tokenize: bool = False) -> float:
            clean_ref = self.clean_answer(ref)
            clean_pred = self.clean_answer(pred)
            # tokenize the answer if the ref is a single word
            # prevent false positive (e.g, 0)
            if (
                tokenize
                and len(clean_ref) == 1
                and len(word_tokenize(clean_ref)) == 1
            ):
                tok_pred = word_tokenize(clean_pred)
                return float(clean_ref in tok_pred)
            else:
                return float(clean_ref in clean_pred)

    def __call__(
            self,
            task_content,
            answer,
            eval_config,
            driver
            ):
        targets = eval_config["program_html"]
        score = 1.0
        for target in targets:
            target_url = target["url"]
            if target_url.startswith("func"):
                func = target_url.split("func:")[1]
                func = func.replace("__last_url__", driver.current_url)
                if "shopping" in str(func):
                    if "get_latest_order_url" in str(func):
                        print("get_latest_order_url", func, replace_ip_and_port(eval_config["webarena_starting_url"], SHOPPING))
                        target_url = shopping_get_latest_order_url(replace_ip_and_port(eval_config["webarena_starting_url"], SHOPPING))
                    else: 
                        raise ValueError(f"[Unknown func1: {func}]")
                else:
                    target_url = eval(func)
            target_url = replace_ip_and_port(eval_config["webarena_starting_url"], target_url)
            
            locator = target["locator"]

            if target_url != "last":
                try:
                    driver.get(target_url)
                    time.sleep(3)
                except:
                    print("[EVAL PROGRAM HTML FAIL]", target_url)
            
            if not locator.strip():
                selected_element = driver.page_source
            elif locator.startswith("document.") or locator.startswith("[...document."):
                if "prep_actions" in target:
                    try:
                        for prep_action in target["prep_actions"]:
                            driver.execute_script(f"return {prep_action}")
                    except Exception:
                        pass
                try:
                    selected_element = str(driver.execute_script(f"return {locator}"))
                    if not selected_element:
                        selected_element = ""
                except Exception:
                    selected_element = ""
            elif locator.startswith("func:"):
                func = locator.split("func:")[1]
                func = func.replace("__page__", "driver")  
                if "shopping" in str(func):
                    if "get_latest_order_url" in str(func):
                        selected_element = shopping_get_latest_order_url(replace_ip_and_port(eval_config["webarena_starting_url"], SHOPPING))
                    else: 
                        argument = func[re.search("'", func).end():]
                        argument = argument[:re.search("'", argument).start()]
                        if "author" in func:
                            selected_element = shopping_get_sku_latest_review_author(argument, replace_ip_and_port(eval_config["webarena_starting_url"], SHOPPING))
                        elif "review" in func:
                            selected_element = shopping_get_sku_latest_review_rating(argument, replace_ip_and_port(eval_config["webarena_starting_url"], SHOPPING))
                        else:
                            raise ValueError(f"[Unknown func: {func}]")
                else:
                    selected_element = eval(func)
            else:
                raise ValueError(f"[Unknown locator: {locator}]")
                        
            selected_element = html.unescape(selected_element)

            if "exact_match" in target["required_contents"]:
                required_contents = target["required_contents"]["exact_match"]
                cur_score = self.exact_match(ref=required_contents, pred=selected_element)
                score *= float(cur_score)
            elif "must_include" in target["required_contents"]:
                required_contents = target["required_contents"]["must_include"]
                assert isinstance(required_contents, list)
                for content in required_contents:
                    content_or = content.split(" |OR| ")
                    cur_score = any([self.must_include(ref=content, pred=selected_element, tokenize=False) for content in content_or])
                    score *= float(cur_score)
            else:
                raise ValueError(f"Unknown required_contents: {target['required_contents'].keys()}")
        return score

import threading
import concurrent

def webarena_batch_eval(trajectories, batch_obs, batch_eval_info, batch_env):
    job_args = []
    for i in range(len(trajectories)):
        
        obs, eval_info, env = batch_obs[i], batch_eval_info[i], batch_env.envs[i]
        if env.task is None or len(trajectories[i]) == 0:
            job_args.append(None)
            continue
        eval_config = env.task['eval']
        task_content = env.task['ques']
            
        answer = eval_info.get('answer', 'N/A')
        eval_config["webarena_starting_url"] = obs["starting_url"]
            
        driver = env.driver_task
        job_args.append((task_content, answer, eval_config, driver))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        jobs = [executor.submit(webarena_eval, jargs[0], jargs[1], jargs[2], jargs[3], batch_env.verbose) if jargs is not None else None for jargs in job_args]
        rewards = [job.result() if job is not None else None for job in jobs]

    for i in range(len(trajectories)):   
        if rewards[i] is None:
            continue
        if rewards[i] == -1:
            continue
        trajectories[i][-1]['reward'] = rewards[i]
    return trajectories

def webarena_eval(task_content, answer, eval_config, driver, verbose):

    try:
        eval_types = eval_config["eval_types"]
        evaluators = []
        for eval_type in eval_types:
            match eval_type:
                case "string_match":
                    evaluators.append(StringEvaluator())
                case "url_match":
                    evaluators.append(URLEvaluator())
                case "program_html":
                    evaluators.append(HTMLContentEvaluator())
                case _:
                    raise ValueError(f"eval_type {eval_type} is not supported")
    
        score = 1.0
        for evaluator in evaluators:
            cur_score = evaluator(task_content, answer, eval_config, driver)
            score *= cur_score
        if verbose:
            logging.info(f"[WEBARENA EVAL SUCCEED] Task: {task_content} Answer: {answer} Config: {eval_config} Result: {score}")
    except Exception as e:
        print("[EVAL ERROR]", e)
        print(traceback.format_exc())
        score = -1
        if verbose:
            logging.info(f"[WEBARENA EVAL FAIL] Task: {task_content} Answer: {answer} Config: {eval_config} Result: {score}")
    return score

def auto_eval_by_claude_console(it_messages, process_dir, img_path, anthropic_api_key, api_model, img_num, task, evaluator_prompt):
    # Optionally extract a reference answer if provided.
    reference = None
    if task.get('eval') is not None and task['eval'] is not None and task['eval'].get('reference_answer_raw_annotation') is not None:
        reference = task['eval']['reference_answer_raw_annotation']
    
    if len(it_messages) == 0:
        return None

    # Extract the task content.
    task_info = it_messages[0]["content"]
    if isinstance(task_info, list):
        task_info = task_info[0]["text"]
    assert 'Now given a task' in task_info, "Task content not found"
    pattern = r"Now given a task:(.+?)Please interact with"
    matches = re.search(pattern, task_info)
    task_content = matches.group(1).strip() if matches else ""

    # Extract the answer content.
    ans_info = it_messages[-1]["content"]
    if 'Action: ANSWER' not in ans_info:
        return 0
    pattern_ans = r"ANSWER[; ]+\[?(.[^\]]*)\]?"
    matches_ans = re.search(pattern_ans, ans_info)
    answer_content = matches_ans.group(1).strip() if matches_ans else ""

    # Gather the most recent screenshot images.
    screenshots = [int(f[10:].split('.png')[0]) for f in os.listdir(process_dir) if '.png' in f]
    screenshots.sort()
    screenshots = screenshots[-img_num:]
    
    whole_content_img = []
    for screenshot_id in screenshots:
        cur_img_path = os.path.join(process_dir, f'screenshot{screenshot_id}.png')
        b64_img = encode_image(cur_img_path)
        whole_content_img.append({
            'type': 'image',
            'source': {'type': 'base64', 'media_type': 'image/png', 'data': b64_img}
        })

    # Prepare the full prompt for evaluation.
    user_prompt_tmp = USER_PROMPT.replace('<task>', task_content)
    user_prompt_tmp = user_prompt_tmp.replace('<answer>', answer_content)
    user_prompt_tmp = user_prompt_tmp.replace('<num>', str(img_num))
    
    messages = [
        {
            'role': 'user',
            'content': (
                [{'type': 'text', 'text': user_prompt_tmp}]
                + whole_content_img +
                [{'type': 'text', 'text': "Your verdict:\n"}]
            )
        }
    ]
    
    # Initialize the Anthropic client.
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    response = client.messages.create(
        model=api_model,
        max_tokens=1000,
        system=evaluator_prompt,
        thinking={"type": "enabled", "budget_tokens": 500},
        messages=messages,
        temperature=0  # Use temperature=0 for deterministic output; adjust if needed.
    )

    # Extract the text from the response.
    # Depending on the Anthropic API version, you may have a 'completion' key
    # or a content list. Adjust as needed.
    claude_3_res = response.get("completion", "")
    if not claude_3_res and isinstance(response.get("content"), list):
        # Fallback: use the text from the first content item.
        claude_3_res = response["content"][0].get("text", "")

    # (Optional) Replace inline image sources with a dummy URL for logging.
    print_message = messages[0]
    for idx in range(len(print_message['content'])):
        if print_message['content'][idx]['type'] == 'image':
            print_message['content'][idx]['source'] = {"url": "data:image/png;base64, b64_img"}
    
    # Determine the evaluation result based on the output text.
    auto_eval_res = 1 if ("SUCCESS" in claude_3_res and "NOT SUCCESS" not in claude_3_res) else 0
    return auto_eval_res, claude_3_res

def get_eval_prompt_gemma3(it_messages, process_dir, img_num):
    if len(it_messages) == 0:
        print("ERROR: No messages found for evaluation")
        return ""

    # Extract the task content.
    task_info = it_messages[0]["content"]
    if isinstance(task_info, list):
        task_info = task_info[0]["text"]
    assert 'Now given a task' in task_info, "Task content not found"
    
    pattern = r"Now given a task:(.+?)Please interact with"
    matches = re.search(pattern, task_info)
    task_content = matches.group(1).strip() if matches else ""

    # Extract the answer content.
    ans_info = it_messages[-1]["content"]
    if 'Action: ANSWER' not in ans_info:
        print("ERROR: No ANSWER found for evaluation")
        return 0
    pattern_ans = r"ANSWER[; ]+\[?(.[^\]]*)\]?"
    matches_ans = re.search(pattern_ans, ans_info)
    answer_content = matches_ans.group(1).strip() if matches_ans else ""

    # Gather the most recent screenshot images.
    screenshots = [int(f[10:].split('.png')[0]) for f in os.listdir(process_dir) if '.png' in f]
    screenshots.sort()
    screenshots = screenshots[-img_num:]
    
    whole_content_img = []
    for screenshot_id in screenshots:
        cur_img_path = os.path.join(process_dir, f'screenshot{screenshot_id}.png')
        b64_img = encode_image(cur_img_path)
        whole_content_img.append({
            'type': 'image',
            'source': {'type': 'base64', 'media_type': 'image/png', 'data': b64_img}
        })

    # Prepare the full prompt for evaluation.
    user_prompt_tmp = USER_PROMPT.replace('<task>', task_content)
    user_prompt_tmp = user_prompt_tmp.replace('<answer>', answer_content)
    user_prompt_tmp = user_prompt_tmp.replace('<num>', str(img_num))
    
    messages = [
        {
            'role': 'user',
            'content': (
                [{'type': 'text', 'text': user_prompt_tmp}]
                + whole_content_img +
                [{'type': 'text', 'text': "Your verdict:\n"}]
            )
        }
    ]
    
    return messages
