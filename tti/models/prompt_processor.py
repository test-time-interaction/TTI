import re
import json
import base64
import os
from collections import defaultdict
import logging

def extract_information(text):
    patterns = {
        "click": r"Click \[?(\d+)\]?.*",
        "type": r"Type \[?(\d+)\]?[; ]+\[?(.[^\]]*)\]?.*",
        "scroll": r"Scroll (?:\[?(\d+|WINDOW)\]?[; ]+)?\[?(up|down)\]?.*",
        "wait": r"^Wait",
        "goback": r"^GoBack",
        "bing": r"^Bing",
        "answer": r"ANSWER[; ]+\[?(.[^\]]*)\]?.*"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            if key in ["click", "wait", "goback", "bing"]:
                # no content
                return key, match.groups()
            else:
                num = match.group(1)
                if key == "scroll" and num is None:
                    num = "WINDOW"
                return key, {"number": num, "content": match.group(2)} if key in ["type", "scroll"] else {"content": match.group(1)}
    return None, None
    
class PromptProcessor:
    def __init__(self, prompt_path, evaluator_prompt_path, max_attached_imgs=5, verbose=False):
        """
        Initialize the PromptProcessor class.
        
        Args:
            max_attached_imgs (int): Maximum number of images to attach to messages.
        """
        self.max_attached_imgs = max_attached_imgs
        self.double_check = False
        self.verbose = verbose
        self.multi_agent = []

        self.use_image = True # False
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_templates = json.load(f)
        logging.info(f"Load prompt from path: {prompt_path}")
             
        with open(evaluator_prompt_path, "r") as fb:
            self.evaluator_prompt = fb.read()
        self.evaluator_prompt += "\nNow analyze the following case.\n\nTask: {task_goal}\n{reference_answer}URL of the last webpage: {url}\nAccessibility tree of the last webpage:\n{accessibility_tree}\nResult response: {answer}\nLast {num} screenshots upon task completion:"

        logging.info(f"Multi-agent: {self.multi_agent}")
        

    def process_batch_evaluation(self, batch_observation, batch_eval_info):
        """
        Process batch observations for evaluation, tracking which indices contain valid data.
        
        Returns:
            tuple: (batch_msgs, valid_indices) where valid_indices tracks which original indices 
                have valid observations and are included in batch_msgs
        """
        batch_msgs = []
        valid_indices = []  # Track which indices have valid observations
        
        for i in range(len(batch_observation)):
            # Only process valid observations
            if batch_observation[i] is not None and batch_eval_info[i] is not None:
                try:
                    msg = self.process_evaluation(batch_observation[i], batch_eval_info[i])
                    batch_msgs.append(msg)
                    valid_indices.append(i)  # Track the original index
                except Exception as e:
                    print(f"Error processing evaluation for observation {i}: {str(e)}")
                    # Don't include failed processing in the results
        
        return batch_msgs, valid_indices

    def create_dummy_evaluation_message(self):
        """Create a dummy evaluation message for None observations"""
        # Use placeholders for all required fields
        dummy_task = "Task was not completed or failed"
        dummy_url = "http://example.com"
        dummy_tree = "No accessibility tree available"
        dummy_answer = "N/A"
        dummy_ref_answer = ""
        
        # Create a message using the evaluator prompt template
        msg = self.evaluator_prompt.replace("{url}", dummy_url)\
                                .replace("{task_goal}", dummy_task)\
                                .replace("{accessibility_tree}", dummy_tree)\
                                .replace("{answer}", dummy_answer)\
                                .replace("{num}", "0")\
                                .replace("{reference_answer}", dummy_ref_answer)
        
        msg_format = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': msg + "\nYour verdict:\n"}
                ]
            }
        ]
        return msg_format

    def process_evaluation(self, observation, eval_info):
        # Handle the case where observation or eval_info is None
        if observation is None or eval_info is None:
            return self.create_dummy_evaluation_message()
        
        # Get task_dir with a default value if not present
        task_dir = observation.get('task_dir', '')
        if not task_dir:
            # If task_dir is empty, create a dummy message
            return self.create_dummy_evaluation_message()
        
        answer = eval_info.get('answer', 'N/A')
        reference_answer = eval_info.get('reference_answer', '')
        if reference_answer is None:
            reference_answer = ''
        else:
            reference_answer = 'The reference answer is: ' + reference_answer + "\n"
        
        # Check if task_dir exists and handle errors
        try:
            screenshots = [int(f.split("/")[-1].split('.png')[0].replace("screenshot","")) for f in os.listdir(task_dir) if '.png' in f]
            screenshots.sort()
            num = min(self.max_attached_imgs, len(screenshots))
            screenshots = screenshots[-num:]

            task_goal = observation.get('task', '')
            accessibility_tree = observation.get('tree', '')
            url = observation.get('url', '')
            msg = self.evaluator_prompt.replace("{url}", url).replace("{task_goal}", task_goal).replace("{accessibility_tree}", accessibility_tree).replace("{answer}", answer).replace("{num}", str(num)).replace("{reference_answer}", reference_answer)
        
            whole_content_img = []
            for screenshot_id in screenshots:
                cur_img_path = os.path.join(task_dir, f'screenshot{screenshot_id}.png')
                whole_content_img.append({
                            'type': 'image', 
                        'source': {
                                'type': 'path', 'path': cur_img_path}
                        })
        
            msg_format = [
            {
                'role': 'user',
                'content': (
                    [{'type': 'text', 'text': msg}]
                    + whole_content_img +
                    [{'type': 'text', 'text': "Your verdict:\n"}]
                )
            }
            ]
            return msg_format
        except Exception as e:
            print(f"Error creating evaluation message: {str(e)}")
            return self.create_dummy_evaluation_message()

    def process_batch_observation(self, batch_observation, batch_history, time_step=1):
        batch_msgs = []
        for i in range(len(batch_observation)):
            batch_msgs.append(self.process_observation(batch_observation[i], batch_history[i], time_step))
        return batch_msgs
        
    def process_observation(self, observation, history, time_step=1):
        """
        Process an observation dictionary and create a formatted message.
        
        Args:
            observation (dict): Dictionary containing observation data including:
                - task (dict): Task information
                - image (str): Path to screenshot
                - web_name (str): Name of the website being browsed
                - history (str): Interaction history
                - tree (str): Accessibility tree information
                - url (str): Current URL
                - pdf_obs (str, optional): PDF observation text
                - warn_obs (str, optional): Warning observation text
                - fail_obs (str, optional): Failure observation text
            time_step (int): Current time step in the interaction
            history (list, optional): List of previous messages
        
        Returns:
            dict: Formatted message for the LLM
        """
        # Extract values from observation
        task_goal = observation.get('task', '')
        image_path = observation.get('image')
        accessibility_tree = observation.get('tree', '')
        url = observation.get('url', '')
        pdf_obs = observation.get('pdf_obs', '')
        warn_obs = observation.get('warn_obs', '')
        fail_obs = observation.get('fail_obs', '')
        task_domain = observation.get('web_name', '')

        # Format initial or observation message
        if time_step == 1:
            history = self.format_initial_message(task_goal, task_domain, url, image_path, accessibility_tree, fail_obs)
                
        else:
            new_msg = self.format_observation_message(task_goal, url, pdf_obs, warn_obs, image_path, accessibility_tree, fail_obs)
            if "Action: ANSWER" in history[-1]['content'][0]['text'] or "Action:\nANSWER" in history[-1]['content'][0]['text']:     
                new_msg['content'][0]['text'] += "\n\nImportant: You returned an answer in the last step. Let's pause, check the web page, and think again. If you still think the task is finished, double-check your answer, revise it if need, and return a final answer. If not, continue the task." 
            
            if len(history) >= 3:
                history = self.clip_messages(history)
            history.append(new_msg)

        if fail_obs:
            history[-1]['content'][0]['text'] = history[-1]['content'][0]['text'] + fail_obs
        if self.verbose:       
            print(history[-1]['content'][0]['text'][re.search("Current URL:",history[-1]['content'][0]['text'] ).start():])
                
            with open(observation['task_dir'] + f"/msg{time_step}.json", "w") as f:
                json.dump(history, f, indent=4)
            
        return history
    
    def format_initial_message(self, task_goal, task_domain, url, image_path, accessibility_tree, fail_obs):
        """
        Format the initial message for the LLM.
        
        Args:
            task_goal (dict): Task information (should contain 'ques' key)
            url (str): Current URL
            web_img_b64 (str): Base64 encoded screenshot image
            accessibility_tree (str, optional): Accessibility tree information
            
        Returns:
            dict: Formatted message for the LLM
        """
        if task_domain in self.prompt_templates["hint"].keys():
            hint = self.prompt_templates["hint"][task_domain]
        else:
            hint = self.prompt_templates["hint"]["general"]
        init_msg = self.prompt_templates["initial"].replace("{url}", url).replace("{task_goal}", task_goal).replace("{accessibility_tree}", accessibility_tree).replace("{hint}", hint)
        
        init_msg_format = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        'role': 'user',
        'content': [
            {'type': 'text', 'text': init_msg},
        ]
    }
]        
        if self.use_image and image_path:
            init_msg_format[-1]['content'].append({
                'type': 'image', 
                'source': {
                            'type': 'path', 'path': image_path}
            })
        
        return init_msg_format
    
    def format_observation_message(self, task_goal, url, pdf_obs, warn_obs, image_path, accessibility_tree, fail_obs):
        """
        Format an observation message for the LLM.
        
        Args:
            pdf_obs (str): PDF observation text, if any
            warn_obs (str): Warning observation text, if any
            web_img_b64 (str): Base64 encoded screenshot image
            accessibility_tree (str, optional): Accessibility tree information
            
        Returns:
            dict: Formatted observation message for the LLM
        """
        if not pdf_obs:
            base_text = self.prompt_templates["observation"].replace("{url}", url).replace("{task_goal}", task_goal).replace("{accessibility_tree}", accessibility_tree).replace("{warn_obs}", warn_obs)
        else:
            base_text = self.prompt_templates["pdf_observation"].replace("{url}", url).replace("{task_goal}", task_goal).replace("{accessibility_tree}", accessibility_tree).replace("{warn_obs}", warn_obs).replace("{pdf_obs}", pdf_obs)
        
        curr_msg = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': base_text},
            ]
        }
        
        if self.use_image and image_path:
            curr_msg['content'].append({
                'type': 'image', 
                'source': {
                            'type': 'path', 'path': image_path}
            })
        
        return curr_msg
    

    def clip_messages(self, messages):
        """
        Clip the messages to ensure no more than max_attached_imgs images are included.
        
        Args:
            messages (list): List of messages to clip.
            
        Returns:
            list: Clipped messages.
        """
        img_count = 0
        user_msg_indices = []
        
        # Find user messages with images
        for i, msg in enumerate(messages):
            if msg['role'] == 'user':
                if isinstance(msg['content'], list):
                    for j, item in enumerate(msg['content']):
                        if item.get('type') == 'text':
                            if "Observation omitted for previous steps." in messages[i]['content'][j]['text']:
                                continue
                            if "Now solve the following task." in messages[i]['content'][j]['text']:
                                messages[i]['content'][j]['text'] = messages[i]['content'][j]['text'][:re.search("Screenshot of current viewpoint:", messages[i]['content'][j]['text']).start()] + "Observation omitted for previous steps. See attachment for screenshot."
                            else:
                                messages[i]['content'][j]['text'] = "Observation omitted for previous steps. See attachment for screenshot."
                                
                        elif 'image' in item.get('type'):
                            img_count += 1
        if img_count > self.max_attached_imgs:
            for i, msg in enumerate(messages):
                if msg['role'] == 'user':
                    if isinstance(msg['content'], list):
                        for j, item in enumerate(msg['content']):
                            if 'image' in item.get('type'):
                                del messages[i]['content'][j]
                                img_count -= 1
                                messages[i]['content'][0]['text'] = messages[i]['content'][0]['text'].replace("See attachment for screenshot.", "").strip()
                        if img_count <= self.max_attached_imgs:
                            break
        return messages
    

    def process_batch_response(self, batch_response, batch_observation=None):
        action_keys, infos, messages = [], [], []
        for i in range(len(batch_response)):
            action_key, info, message = self.process_llm_response(batch_response[i], batch_observation[i])
            action_keys.append(action_key)
            infos.append(info)
            messages.append(message)
        return action_keys, infos, messages
        
    def process_llm_response(self, response, observation):
        """
        Process the LLM response to extract action information.
        
        Args:
            response (str): The LLM response text.
            
        Returns:
            tuple: (action_key, info) extracted from the response or (None, None) if format is invalid.
        """
        message = {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': response},
            ]
        }

        if self.verbose:
            print("-"*20, "RESPONSE", "-"*20)
            print(response)
            
        # Check if response has required sections
        for tag in self.prompt_templates["pattern"].split("|"):
            if tag not in response:
                return None, None, message
            
        # Extract the action section
        sections = re.split(self.prompt_templates["pattern"], response)
        action = sections[-1].strip() 
        found = True

        if "click" in action.lower() or "type" in action.lower():
            action_id = re.findall(r'\[(\d+)\]', action)
            if not action_id:
                return None, None, message
    
            action_id = int(action_id[0])
            tree = observation['tree']
     
        action_key, info = extract_information(action)
        
        message = {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': response},
            ]
        }
        return action_key, info, message
 
    
    def _encode_image(self, image_path):
        """
        Encode an image to base64.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            str: Base64 encoded image string.
        """        
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
