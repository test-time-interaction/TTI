import platform
import random
import time
import json
import re
import os
import shutil
import logging
import threading
from PIL import Image
import boto3
import numpy as np
import gym
from tti.misc import colorful_print
from time import sleep
import gc
import torch
from tqdm import tqdm

from selenium import webdriver 
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

from .utils import get_pdf_retrieval_ans_from_claude

import re
from .utils import replace_ec2_address
import traceback
from .utils_webarena import webarena_login, WEBARENA_DOMAINS

IN_VIEWPORT_RATIO_THRESHOLD = 0.6
IGNORED_ACTREE_PROPERTIES = (
    "focusable",
    "editable",
    "readonly",
    "level",
    "settable",
    "multiline",
    "invalid","url"
)

def driver_config(force_device_scale, headless, download_dir):
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    if force_device_scale:
        options.add_argument("--force-device-scale-factor=1")
    if headless:
        options.add_argument("--headless")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
    options.add_experimental_option(
        "prefs", {
            "download.default_directory": download_dir,
            "plugins.always_open_pdf_externally": True,
        }
    )
    return options


def exec_action_click(info, web_ele, driver_task):
    driver_task.execute_script("arguments[0].scrollIntoView({block: 'center'});", web_ele)
    time.sleep(1)  # Give time for scrolling to complete
    
    rect = driver_task.execute_script("""
        var rect = arguments[0].getBoundingClientRect();
        return {
            'x': rect.left + (rect.width / 2),
            'y': rect.top + (rect.height / 2),
            'width': rect.width,
            'height': rect.height
        };
    """, web_ele)

    x, y = rect['x'], rect['y']


    driver_task.execute_script(f"""
        var evt = new MouseEvent('click', {{
            'view': window,
            'bubbles': true,
            'cancelable': true,
            'clientX': {x},
            'clientY': {y}
        }});
        document.elementFromPoint({x}, {y}).dispatchEvent(evt);
    """)
    
    time.sleep(10)


def exec_action_type(info, web_ele, driver_task):
    warn_obs = ""
    type_content = info['content']
    driver_task.execute_script("arguments[0].scrollIntoView({block: 'center'});", web_ele)
    time.sleep(1)  # Give time for scrolling to complete  

    try:
        ele_tag_name = web_ele.tag_name.lower()
        ele_type = web_ele.get_attribute("type")
    except:
        ele_tag_name = ""
        ele_type = ""
        
    if (ele_tag_name != 'input' and ele_tag_name != 'textarea') or (ele_tag_name == 'input' and ele_type not in ['text', 'search', 'password', 'email', 'tel']):
        warn_obs = f"note: The web element you're trying to type may not be a textbox, and its tag name is <{ele_tag_name}>, type is {ele_type}."
        
    try:
        # Not always work to delete
        web_ele.clear()
        # Another way to delete
        if platform.system() == 'Darwin':
            web_ele.send_keys(Keys.COMMAND + "a")
        else:
            web_ele.send_keys(Keys.CONTROL + "a")
        web_ele.send_keys(" ")
        web_ele.send_keys(Keys.BACKSPACE)
    except Exception as e:
        pass
    
    # Get element properties and ensure it's visible and enabled
    element_properties = driver_task.execute_script("""
        var element = arguments[0];
        var rect = element.getBoundingClientRect();
        var computedStyle = window.getComputedStyle(element);
        
        return {
            'x': rect.left + (rect.width / 2),
            'y': rect.top + (rect.height / 2),
            'width': rect.width,
            'height': rect.height,
            'isVisible': computedStyle.visibility !== 'hidden' && 
                         computedStyle.display !== 'none' && 
                         rect.width > 0 && 
                         rect.height > 0,
            'isEnabled': !element.disabled,
            'tagName': element.tagName.toLowerCase(),
            'type': element.type || ''
        };
    """, web_ele)
    
    # Clear the field first (more reliable)
    driver_task.execute_script("arguments[0].value = '';", web_ele)
    time.sleep(0.5)

    for i in range(3):
        try:
        
            x, y = element_properties['x'], element_properties['y']
        
        
            driver_task.execute_script(f"""
                var evt = new MouseEvent('click', {{
                    'view': window,
                    'bubbles': true,
                    'cancelable': true,
                    'clientX': {x},
                    'clientY': {y}
                }});
                document.elementFromPoint({x}, {y}).dispatchEvent(evt);
            """)
            time.sleep(2)
        
            try:
                driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea' && e.target.type != 'search') {e.preventDefault();}};""")
            except:
                pass
        
            driver_task.execute_script("""
                var element = arguments[0];
                var text = arguments[1];
                var index = 0;
                
                // Clear existing content
                element.value = '';
                
                // Function to type character by character
                function typeNextChar() {
                    if (index >= text.length) {
                        // Dispatch final events when typing is done
                        element.dispatchEvent(new Event('input', { bubbles: true }));
                        element.dispatchEvent(new Event('change', { bubbles: true }));
                        return;
                    }
                    
                    // Append the next character
                    element.value += text[index];
                    
                    // Dispatch input event for this character
                    element.dispatchEvent(new Event('input', { bubbles: true }));
                    
                    // Move to next character
                    index++;
                    setTimeout(typeNextChar, 10);  // 10ms between characters
                }
                
                // Start typing
                typeNextChar();
            """, web_ele, type_content)
            
            # Calculate wait time based on content length (minimum 1s)
            wait_time = max(1, len(type_content) * 0.01 + 0.5)  # 10ms per char + 500ms buffer
            time.sleep(wait_time)
            
            # Press Enter using multiple approaches
            try:
                # Method 1: JavaScript Enter event with detailed properties
                driver_task.execute_script("""
                    var element = arguments[0];
                    
                    // First check if it's a textarea
                    var isMultiLine = (element.tagName.toLowerCase() === 'textarea');
                    
                    // For single-line inputs, Enter typically submits forms
                    if (!isMultiLine) {
                        // Try to find parent form
                        var form = element.form;
                        if (form) {
                            // Try native form submit
                            try { form.submit(); return; } catch(e) {}
                            
                            // Try event-based form submit
                            try { 
                                form.dispatchEvent(new Event('submit', {bubbles: true, cancelable: true}));
                                return;
                            } catch(e) {}
                        }
                    }
                    
                    // If form submit didn't work or it's a textarea, send Enter key events
                    ['keydown', 'keypress', 'keyup'].forEach(function(eventType) {
                        var event = new KeyboardEvent(eventType, {
                            key: 'Enter',
                            code: 'Enter',
                            keyCode: 13,
                            which: 13,
                            bubbles: true,
                            cancelable: true,
                            composed: true,  // For Shadow DOM
                            charCode: (eventType === 'keypress') ? 13 : 0
                        });
                        element.dispatchEvent(event);
                    });
                """, web_ele)
            except:
                # Method 2: Fallback to Selenium's native send_keys
                from selenium.webdriver.common.keys import Keys
                try:
                    web_ele.send_keys(Keys.ENTER)
                except:
                    # Method 3: Use ActionChains
                    from selenium.webdriver.common.action_chains import ActionChains
                    ActionChains(driver_task).send_keys(Keys.ENTER).perform()
            
            time.sleep(10)  # Wait for any form submission or page changes
            break
        except Exception as e:
            if i >= 2:            
                raise NotImplementedError
            time.sleep(5)
    return warn_obs

def exec_action_click(info, web_ele, driver_task):
    driver_task.execute_script("arguments[0].setAttribute('target', '_self')", web_ele)
    web_ele.click()
    time.sleep(10)


def exec_action_type(info, web_ele, driver_task):
    warn_obs = ""
    type_content = info['content']

    try:
        ele_tag_name = web_ele.tag_name.lower()
        ele_type = web_ele.get_attribute("type")
    except:
        ele_tag_name = ""
        ele_type = ""
        
    if (ele_tag_name != 'input' and ele_tag_name != 'textarea') or (ele_tag_name == 'input' and ele_type not in ['text', 'search', 'password', 'email', 'tel']):
        warn_obs = f"note: The web element you're trying to type may not be a textbox, and its tag name is <{ele_tag_name}>, type is {ele_type}."
        
    try:
        # Not always work to delete
        web_ele.clear()
        # Another way to delete
        if platform.system() == 'Darwin':
            web_ele.send_keys(Keys.COMMAND + "a")
        else:
            web_ele.send_keys(Keys.CONTROL + "a")
        web_ele.send_keys(" ")
        web_ele.send_keys(Keys.BACKSPACE)
    except Exception as e:
        pass

    for i in range(3):
        try:
            actions = ActionChains(driver_task)
            actions.click(web_ele).perform()
            actions.pause(2)
        
            try:
                driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea' && e.target.type != 'search') {e.preventDefault();}};""")
            except:
                pass
        
            actions.send_keys(type_content)
            actions.pause(2)
        
            actions.send_keys(Keys.ENTER)
            actions.perform()
            time.sleep(10)
            break
        except Exception as e:
            if i >= 2:            
                raise NotImplementedError
            time.sleep(5)
    return warn_obs

def exec_action_scroll(info, web_eles, driver_task, window_height):
    scroll_ele_number = info['number']
    scroll_content = info['content']
    if scroll_ele_number == "WINDOW":
        if scroll_content == 'down':
            driver_task.execute_script(f"window.scrollBy(0, {window_height*2//3});")
        else:
            driver_task.execute_script(f"window.scrollBy(0, {-window_height*2//3});")
    else:
        if int(scroll_ele_number) <= len(web_eles):
            scroll_ele_number = int(scroll_ele_number)
            web_ele = web_eles[scroll_ele_number]
        else:
            raise NotImplementedError
            
        actions = ActionChains(driver_task)
        driver_task.execute_script("arguments[0].focus();", web_ele)
        if scroll_content == 'down':
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_DOWN).key_up(Keys.ALT).perform()
        else:
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_UP).key_up(Keys.ALT).perform()
    time.sleep(10)
    

class WebBroswerGym(gym.Env):
    def __init__(self,
                 tasks, env_config,
                download_dir = 'downloads', # download directory, need exist
                output_dir = 'results', # need exist
                region = 'us-west-2',
                aws_key_id = None,
                aws_secret_key = None, batch_id=0, is_test=False
                 ):
        self.tasks = tasks
        self.env_config = env_config
        self.download_dir_ini = download_dir
        self.ini_dir = output_dir
        self.region = region
        self.aws_key_id = aws_key_id
        self.aws_secret_key = aws_secret_key
        self.batch_id = batch_id
        self.is_test = is_test
        self.starting_url = "" 
        self.verbose = env_config.verbose if hasattr(env_config, "verbose") else False
        self.fix_box_color = env_config.fix_box_color if hasattr(env_config, "fix_box_color") else True
        
        self.max_iter = env_config.max_iter if hasattr(env_config, "max_iter") else 20
        self.window_width = env_config.window_width if hasattr(env_config, "window_width") else 1024
        self.window_height = env_config.window_height if hasattr(env_config, "window_height") else 768
        self.force_device_scale = env_config.force_device_scale if hasattr(env_config, "force_device_scale") else True
        self.headless = env_config.headless if hasattr(env_config, "headless") else True
        self.webarena_host = env_config.webarena_host if hasattr(env_config, "webarena_host") else ""
        self.use_rich_actree = env_config.use_rich_actree if hasattr(env_config, "use_rich_actree") else True  
        self.current_viewport_only = env_config.current_viewport_only if hasattr(env_config, "current_viewport_only") else True
        self.batch_size = env_config.batch_size if hasattr(env_config, "batch_size") else 4

        self.task = None
        self.time_step = 0
        self.driver_task = None
        self.terminated = False
        self.min_try = env_config.min_try if hasattr(env_config, "min_try") else 1
        self.tries = env_config.min_try if hasattr(env_config, "min_try") else 1
        self.num_containers_per_machine = env_config.num_containers_per_machine if hasattr(env_config, "num_containers_per_machine") else 1

    def step(self, action):
        try:
            return self._step(action)
        except Exception as e:
            if self.verbose:
                logging.error('[ERROR] STEP ENV')
                logging.error(e)
                logging.error(traceback.format_exc())
            self.close()
            return None

    def _step(self, action):
        if self.time_step >= self.max_iter:
            self.terminated = True
        if self.terminated:
            return None
        self.time_step += 1
        Terminated = False
        Reward = 0
        eval_info = {"action_success": True}
        action_key, info = action["action_key"], action["info"]
    
        self.fail_obs = ""
        self.pdf_obs = ""
        self.warn_obs = ""
        
        # execute action
        try:
            window_handle_task = self.driver_task.current_window_handle
            self.driver_task.switch_to.window(window_handle_task)

            if action_key == 'click':
                click_ele_number = int(info[0])
                if click_ele_number >= len(self.web_eles):
                    raise NotImplementedError
                else:
                    web_ele = self.web_eles[click_ele_number]
                
                try:
                    ele_tag_name = web_ele.tag_name.lower()
                    ele_type = web_ele.get_attribute("type")
                except:
                    ele_tag_name = ""
                    ele_type = ""

                try:
                    exec_action_click(info, web_ele, self.driver_task)
                except:
                    if hasattr(web_ele, 'id'):
                        web_ele = self.driver_task.find_element(By.ID, web_ele.id)
                    exec_action_click(info, web_ele, self.driver_task)
                
                # deal with PDF file
                current_files = sorted(os.listdir(self.download_dir))
                if current_files != self.download_files:
                    # wait for download finish
                    time.sleep(10)
                    current_files = sorted(os.listdir(self.download_dir))

                    current_download_file = [pdf_file for pdf_file in current_files if pdf_file not in self.download_files and pdf_file.endswith('.pdf')]
                    if current_download_file:
                        print('start to solve pdf')
                        pdf_file = current_download_file[0]
                        pdf_file_path = os.path.join(self.download_dir, pdf_file)
                        try:
                            pdf_obs = get_pdf_retrieval_ans_from_claude(pdf_file_path, self.task['ques'], region_name=self.region, aws_key_id=self.aws_key_id, aws_secret_key=self.aws_secret_key)
                        except:
                            pdf_obs = ""
                        shutil.copy(pdf_file_path, self.task_dir)
                        self.pdf_obs = "You downloaded a PDF file, I ask the Assistant API to answer the task based on the PDF file and get the following response: " + pdf_obs
                        print("pdf solved", pdf_obs)

                    self.download_files = current_files

                if ele_tag_name == 'button' and ele_type == 'submit':
                    time.sleep(10)

            elif action_key == 'wait':
                time.sleep(10)

            elif action_key == 'type':
                type_ele_number = int(info['number'])
                if type_ele_number > len(self.web_eles):
                    raise NotImplementedError
                else:
                    web_ele = self.web_eles[type_ele_number]
                
                try:
                    self.warn_obs = exec_action_type(info, web_ele, self.driver_task)
                except:
                    if hasattr(web_ele, 'id'):
                        web_ele = self.driver_task.find_element(By.ID, web_ele.id)
                    self.warn_obs = exec_action_type(info, web_ele, self.driver_task)
                     
                if 'wolfram' in self.task['web']:
                    time.sleep(10)

            elif action_key == 'scroll':
                exec_action_scroll(info, self.web_eles, self.driver_task, self.window_height)
                
            elif action_key == 'goback':
                self.driver_task.back()
                time.sleep(30)

            elif action_key == 'bing':
                self.driver_task.get('https://www.bing.com/')
                time.sleep(60)

            elif action_key == 'answer':
                
                if self.verbose:
                    logging.info(f"[TASK FINISHED] " + info['content'])
                self.tries -= 1  # Set tries to 0 to force termination
                if self.tries <= 0:
                    Terminated = True
                    eval_info['answer'] = info['content'] if info['content'] else "N/A"
                    eval_info['reference_answer'] = None
                    if self.task.get('eval') and self.task['eval'] and self.task['eval'].get('reference_answer_raw_annotation'):
                        eval_info['reference_answer'] = self.task['eval']['reference_answer_raw_annotation']
                    if self.task.get('reference_answer'):
                        eval_info['reference_answer'] = self.task['reference_answer']

            else:
                raise NotImplementedError

        except Exception as e:
            eval_info["action_success"] = False
            if self.verbose:
                logging.error('[ERROR] ACTION CANNOT BE EXECUTED - ERROR: ' + str(e))
                logging.error('[ERROR] ACTION_KEY: ' + str(action_key))
                logging.error('[ERROR] RAW_ACTION: ' + str(action))
                logging.error(str(info))
                logging.error(traceback.format_exc())
            self.fail_obs = "\n\nImportant: the action you have chosen in the last round is invalid because either the specified element description does not match the webpage display, or the element is not interactive, or the output format is wrong. You should revise the action."
            if len(self.web_eles) == 0:
                self.fail_obs = "\n\nImportant: the action you have chosen in the last round is invalid. The current webpage cannot proceed further. You must GoBack."
            
        
        if self.time_step >= self.max_iter:
            Terminated = True

        if Terminated:
            obs = self.get_observation()
            self.close()
            self.terminated = True
            return obs, Reward, Terminated, eval_info
        
    
        try:
            _, self.web_eles, self.tree, _, all_union_bounds, all_texts = self.get_web_element_rect()
        except Exception as e:
            if self.verbose:
                logging.error('[ERROR] SET-OF-MARK')
                logging.error(e)
                logging.error(traceback.format_exc())
            return self.get_observation(), Reward, True, eval_info
         
        self.img_path = os.path.join(self.task_dir, 'screenshot{}.png'.format(self.time_step))
        screenshot_success = False
        for screenshot_attempt in range(5):  # Try up to 5 times
            try:
                self.driver_task.save_screenshot(self.img_path)
                # Verify image was correctly saved and is not empty/corrupted
                if os.path.exists(self.img_path):  # Ensure file is at least 1KB
                    screenshot_success = True
                    break
            except Exception as e:
                if self.verbose:
                    logging.error(f"[ERROR] GET SCREENSHOT (attempt {screenshot_attempt+1}/5)")
                    logging.error(e)
            time.sleep(1)  # Wait before retry
        
        # If we couldn't get a screenshot after all attempts, terminate this trajectory
        if not screenshot_success:
            if self.verbose:
                logging.error("[CRITICAL] Failed to capture initial screenshot after multiple attempts - terminating trajectory")
            self.close()
            return None
        
        if self.use_rich_actree:
            try:
                self.tree, self.web_eles = self.get_actree(all_union_bounds, all_texts, self.web_eles, self.current_viewport_only)
            except Exception as e:
                if self.verbose:
                    logging.error("[ERROR] GET ACTREE")
                    logging.error(traceback.format_exc())
                                
        self.url = self.driver_task.current_url
            
        for attr in ['url', 'tree']:
            for map_pattern in self.url_mapping:
                setattr(self, attr, getattr(self, attr).replace(map_pattern[0], map_pattern[1]))
        return self.get_observation(), Reward, Terminated, eval_info
        

    def reset(self, num_round=0):
        """
        Reset the environment with a timeout to prevent hanging.
        
        Args:
            task_id: Optional ID of the task to use
            task_new: Optional new task to use
            
        Returns:
            Observation or None if reset fails or times out
        """
        reset_ready = threading.Event()
        reset_result = [None]
        reset_error = [None]
        
        def reset_thread():
            try:
                # Store original reset logic in a local variable
                result = self._reset_impl(num_round)
                reset_result[0] = result
                reset_ready.set()
            except Exception as e:
                logging.error(f"[ERROR] reset thread: {e}")
                logging.error(traceback.format_exc())
                reset_error[0] = e
                reset_ready.set()
        
        thread = threading.Thread(target=reset_thread)
        thread.daemon = True
        thread.start()
        
        # Wait for reset to complete or timeout
        if not reset_ready.wait(300):
            logging.error("Reset operation timed out after 5 minutes")
            self.close()
            return None
        
        # Check if reset encountered an error
        if reset_error[0] is not None:
            logging.error(f"Reset encountered an error: {reset_error[0]}")
            self.close()
            return None
        
        return reset_result[0]
    
    def _reset_impl(self, num_round=0):
        """
        Implementation of reset logic with improved driver cleanup and error handling.
        """
        self.terminated = False
        self.time_step = 0
        self.tries = self.min_try
        current_time = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
        self.output_dir = os.path.join(self.ini_dir, current_time)
        os.makedirs(self.output_dir, exist_ok=True)
        self.download_dir = os.path.join(self.download_dir_ini, current_time)
        os.makedirs(self.download_dir, exist_ok=True)

        # Added batching protection - extra cooldown if we're in a later batch
        if num_round > 0:
            time.sleep(2)  # Allow resources to stabilize between batches
            gc.collect()   # Force garbage collection

        # Clean up previous driver if it exists (but don't call quit())
        if hasattr(self, "driver_task") and self.driver_task is not None:
            try:
                self.driver_task = None
            except Exception as e:
                if self.verbose:
                    logging.error(f"[ERROR] Previous driver cleanup: {e}")
            # Force garbage collection
            gc.collect()
            
        if self.is_test:
            cur_task_id = num_round * self.batch_size + self.batch_id
            if cur_task_id >= len(self.tasks):
                self.terminated = True
                return None
            task = self.tasks[cur_task_id % len(self.tasks)]
        else:
            task = random.choice(self.tasks)
            if self.verbose:
                logging.info("[RESET RANDOM TASK]")
        
        self.task = task
        self.url = self.task['web']
        self.url_mapping = [(self.url, self.url)]

        self.task_dir = os.path.join(self.output_dir, 'task{}'.format(task["id"]))
        
        # Improved directory management
        try:
            if os.path.exists(self.task_dir):
                shutil.rmtree(self.task_dir)
                # Add reasonable timeout for directory removal
                max_wait = 15  # seconds
                wait_start = time.time()
                while os.path.exists(self.task_dir) and time.time() - wait_start < max_wait:
                    colorful_print(f"task dir {self.task_dir} is still exist, waiting for 1 second", fg='red')
                    time.sleep(1)
                # Force removal if still exists
                if os.path.exists(self.task_dir):
                    try:
                        os.system(f"rm -rf {self.task_dir}")
                    except Exception as e:
                        logging.error(f"Force removal error: {e}")
        except Exception as e:
            logging.error(f"Directory cleanup error: {e}")
            
        # Ensure directory exists
        os.makedirs(self.task_dir, exist_ok=True)
        
        # Create Driver with better retry logic
        options = driver_config(self.force_device_scale, self.headless, self.download_dir)
        
        for i in range(3):
            try:
                self.driver_task = webdriver.Chrome(options=options)
                self.driver_task.set_script_timeout(120)
                break
            except Exception as e:
                if i >= 2:
                    if self.verbose:
                        logging.error('[ERROR] DRIVER FAILURE')
                        logging.error(e)
                        logging.error(traceback.format_exc())
                    self.close()
                    return None
                time.sleep(2 * (i + 1))  # Exponential backoff

        # About window size
        self.driver_task.set_window_size(self.window_width, self.window_height)

        if self.webarena_host and self.task['web_name'] in WEBARENA_DOMAINS:
            self.task['ques'] = self.task['ques'].replace("subreddit", "subforum").replace("sub-reddit", "subforum").replace("reddit", "postmill").replace("Reddit", "postmill")
            success, self.url_mapping, self.url = webarena_login(self.task['web_name'], self.url, self.driver_task, self.webarena_host, batch_id=self.batch_id, num_containers_per_machine=self.num_containers_per_machine)
            if not success:
                if self.verbose:
                    logging.error("[ERROR] LOGIN FAIL")
                self.close()
                return None 
            if self.verbose:
                logging.info(f"[LOGIN SUCCESS] {self.task['web_name']} {self.batch_id}")
                
        retry_time = 0
        while True:
            try:
                self.driver_task.get(self.url)
                break
            except Exception as e:
                if self.verbose:
                    logging.error(f"[ERROR] GET URL: {e}")
                time.sleep(2)
                retry_time += 1
                if retry_time > 2:
                    if self.verbose:
                        logging.error(f'[ERROR] DRIVER LOADING {self.task["web"]}')
                    self.terminated = True
                    return None

        try:
            self.driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea') {e.preventDefault();}};""")
        except Exception as e:
            logging.error('Driver error when adding event listener.')
            logging.error(e)

        # We only deal with PDF file
        try:
            for filename in os.listdir(self.download_dir):
                file_path = os.path.join(self.download_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            if self.verbose:
                logging.error(f'[ERROR] Download dir cleanup: {e}')

        self.download_files = []

        self.fail_obs = ""  # When error execute the action
        self.pdf_obs = ""  # When download PDF file
        self.warn_obs = ""  # Type warning

        try:
            _, self.web_eles, self.tree, _, all_union_bounds, all_texts = self.get_web_element_rect()
        except Exception as e:
            if self.verbose:
                logging.error('[ERROR] SET-OF-MARK')
                logging.error(e)
                logging.error(traceback.format_exc())
            self.close()
            return None

        self.img_path = os.path.join(self.task_dir, 'screenshot{}.png'.format(self.time_step))
        screenshot_success = False
        for screenshot_attempt in range(5):  # Try up to 5 times
            try:
                self.driver_task.save_screenshot(self.img_path)
                # Verify image was correctly saved and is not empty/corrupted
                if os.path.exists(self.img_path):  # Ensure file is at least 1KB
                    screenshot_success = True
                    break
            except Exception as e:
                if self.verbose:
                    logging.error(f"[ERROR] GET SCREENSHOT (attempt {screenshot_attempt+1}/5)")
                    logging.error(e)
            time.sleep(1)  # Wait before retry
        
        # If we couldn't get a screenshot after all attempts, terminate this trajectory
        if not screenshot_success:
            if self.verbose:
                logging.error("[CRITICAL] Failed to capture initial screenshot after multiple attempts - terminating trajectory")
            self.close()
            return None

        if self.use_rich_actree:
            try:
                self.tree, self.web_eles = self.get_actree(all_union_bounds, all_texts, self.web_eles, self.current_viewport_only)
            except Exception as e:
                if self.verbose:
                    logging.error("[ERROR] GET ACTREE")
                    logging.error(traceback.format_exc())
        
        for attr in ['url', 'tree']:
            for map_pattern in self.url_mapping:
                setattr(self, attr, getattr(self, attr).replace(map_pattern[0], map_pattern[1]))
        self.starting_url = self.url_mapping[-1][0]
        return self.get_observation(), None

    def get_observation(self):
        observation = {
            'image': self.img_path,
            'task': self.task['ques'],
            'web_name': self.task['web_name'],
            'url': self.url,
            'tree': self.tree,
            'pdf_obs' : self.pdf_obs, 
            'warn_obs': self.warn_obs,
            'fail_obs': self.fail_obs,
            'task_dir': self.task_dir,
            'starting_url': self.starting_url
    
        }
        return observation

    def close(self):
        """Improved browser cleanup with error handling"""
        self.terminated = True
        self.tries = self.min_try

    
    def clear_labels(self):
        js_clear_script = """
    (function() {
        // Select all elements that were added as labels
        // We can identify them by their fixed position, outline style, and zIndex
        const labelElements = document.querySelectorAll('div[style*="outline"][style*="fixed"][style*="z-index: 2147483647"]');
        
        // Remove each label element from the DOM
        labelElements.forEach(function(element) {
            if (element && element.parentNode) {
                element.parentNode.removeChild(element);
            }
        });
        
        // Return the number of elements removed
        return labelElements.length;
    })();
    """
        removed_count = self.driver_task.execute_script(js_clear_script)
        
        return removed_count
        
    def get_actree(self, rects, texts, elements, current_viewport_only=True):
        self.clear_labels()
        accessibility_tree = self.driver_task.execute_cdp_cmd('Accessibility.getFullAXTree', {})["nodes"]
        seen_ids = set()
        _accessibility_tree = []
        for node in accessibility_tree:
            if node["nodeId"] not in seen_ids:
                _accessibility_tree.append(node)
                seen_ids.add(node["nodeId"])
        accessibility_tree = _accessibility_tree
        
        nodeid_to_cursor = {}
        for cursor, node in enumerate(accessibility_tree):
            nodeid_to_cursor[node["nodeId"]] = cursor
            # usually because the node is not visible etc
            if "backendDOMNodeId" not in node:
                node["union_bound"] = None
                continue
            if node["role"]["value"] == "RootWebArea":
                # always inside the viewport
                node["union_bound"] = [0, 0, 1, 1]
            else:
                try:
                    backend_node_id = node["backendDOMNodeId"]
                    
                    # Use Runtime.getRemoteObject to get a remote object reference
                    remote_object = self.driver_task.execute_cdp_cmd('DOM.resolveNode', {'backendNodeId': backend_node_id})
                    object_id = remote_object['object']['objectId']
             
                    box_model = self.driver_task.execute_cdp_cmd('DOM.getBoxModel', {
                    'objectId': object_id
                })
                except:
                    node["union_bound"] = None
                    continue

                # The content box is what we usually refer to as the bounding box
                content_box = box_model.get('model', {}).get('content', [])
                if not content_box or len(content_box) != 8:
                    node["union_bound"] = None
                else:
                    x1, y1 = content_box[0], content_box[1]  # top-left
                    x2, y2 = content_box[2], content_box[3]  # top-right
                    x3, y3 = content_box[4], content_box[5]  # bottom-right
                    
                    # Calculate width and height
                    width = x2 - x1   # or x3 - x4
                    height = y3 - y1  # or y4 - y2
                    node["union_bound"] = [x1, y1, width, height]
                    
        # filter nodes that are not in the current viewport
        truncated = False
        scroll_up = False
        scroll_down = False

        full_tree, full_all_union_bounds, full_rect_dict = self.node_list_to_tree(accessibility_tree, rects, texts)
        if current_viewport_only:

            def remove_node_in_graph(node) -> None:
                # update the node information in the accessibility tree
                nodeid = node["nodeId"]
                node_cursor = nodeid_to_cursor[nodeid]
                parent_nodeid = node["parentId"]
                children_nodeids = node["childIds"]
                parent_cursor = nodeid_to_cursor[parent_nodeid]
                parent_node = accessibility_tree[parent_cursor]
                for prop in parent_node.get("properties", []):
                    if prop.get("name") == "multiselectable":
                        if accessibility_tree[parent_cursor]["parentId"] != "[REMOVED]":
                            return

                # update the children of the parent node
                assert (
                    accessibility_tree[parent_cursor].get("parentId", "Root")
                    is not None
                )
                # remove the nodeid from parent's childIds
                index = accessibility_tree[parent_cursor]["childIds"].index(
                    nodeid
                )
                accessibility_tree[parent_cursor]["childIds"].pop(index)
                # Insert children_nodeids in the same location
                for child_nodeid in children_nodeids:
                    accessibility_tree[parent_cursor]["childIds"].insert(
                        index, child_nodeid
                    )
                    index += 1
                # update children node's parent
                for child_nodeid in children_nodeids:
                    child_cursor = nodeid_to_cursor[child_nodeid]
                    accessibility_tree[child_cursor][
                        "parentId"
                    ] = parent_nodeid
                # mark as removed
                accessibility_tree[node_cursor]["parentId"] = "[REMOVED]"

            for node in accessibility_tree:
                if not node["union_bound"]:
                    remove_node_in_graph(node)
                    continue

                [x, y, width, height] = node["union_bound"]

                if y < 0:
                    scroll_up = True
                if y > self.window_height:
                    scroll_down = True

                # invisible node
                if width == 0 or height == 0:
                    remove_node_in_graph(node)
                    continue

                in_viewport_ratio = self.get_element_in_viewport_ratio(
                    elem_left_bound=float(x),
                    elem_top_bound=float(y),
                    width=float(width),
                    height=float(height),
                )

                if in_viewport_ratio < IN_VIEWPORT_RATIO_THRESHOLD:
                    truncated=True
                    remove_node_in_graph(node)

            accessibility_tree = [
                node
                for node in accessibility_tree
                if node.get("parentId", "Root") != "[REMOVED]"
            ]
            tree, all_union_bounds, rect_dict = self.node_list_to_tree(accessibility_tree, rects, texts)

            if truncated:
                if scroll_up and scroll_down:
                    tree = tree +"\nYou can scroll up or down to see more."
                elif scroll_down:
                    tree = tree +"\nYou can scroll down to see more. Page has not yet reached the bottom."
                elif scroll_up:
                    tree = tree +"\nYou can scroll up to see more."
        # print("[FULL VS. PARTIAL]", len(full_tree), len(tree))

        if len(full_tree) <= 5 * len(tree) and len(full_tree) < 12000:
            tree = full_tree
            all_union_bounds = full_all_union_bounds
            rect_dict = full_rect_dict
        tree, elements = self.clean_tree(tree, elements, rects, texts, all_union_bounds, rect_dict)
        
        return tree, elements

    def node_list_to_tree(self, accessibility_tree, rects, texts):
        node_id_to_idx = {}
        for idx, node in enumerate(accessibility_tree):
            node_id_to_idx[node["nodeId"]] = idx

        obs_nodes_info = {}
        all_union_bounds = {}
        count = len(rects)        
        
        rect_dict = np.zeros(count)

        def dfs(idx: int, obs_node_id: str, depth: int) -> str:
            tree_str = ""
            node = accessibility_tree[idx]
            indent = "\t" * depth
            valid_node = True
            try:
                role = node["role"]["value"]
                name = node["name"]["value"]
                if isinstance(name, str) and len(name) > 500:
                    name = name[:500] + "..."
                node_str = f"[{obs_node_id}] {role} {repr(name)}"
                properties = []
                for property in node.get("properties", []):
                    try:
                        if property["name"] in IGNORED_ACTREE_PROPERTIES:
                            continue
                        p_val = property["value"]["value"]
                        if isinstance(p_val, str) and len(p_val) > 500:
                            p_val = p_val[:500] + "..."
                        properties.append(
                            f'{property["name"]}: {p_val}'
                        )
                    except KeyError:
                        pass

                if properties:
                    node_str += " " + " ".join(properties)

                # check valid
                if not node_str.strip():
                    valid_node = False

                # empty generic node
                if not name.strip():
                    if not properties:
                        if role in [
                            "generic",
                            "img",
                            "list",
                            "strong",
                            "paragraph",
                            "banner",
                            "navigation",
                            "Section",
                            "LabelText",
                            "Legend",
                            "listitem",
                        ]:
                            valid_node = False
                    elif role in ["listitem"]:
                        valid_node = False

                if valid_node:
                    all_union_bounds[obs_node_id]=node["union_bound"]
                    tree_str += f"{indent}{node_str}"
                    obs_nodes_info[obs_node_id] = {
                        "backend_id": node["backendDOMNodeId"],
                        "union_bound": node["union_bound"],
                        "text": node_str,
                    }
                    if node["union_bound"] in rects:
                        ridx = rects.index(node["union_bound"])
                        if texts[ridx] in node_str:
                            rect_dict[ridx]=1

            except Exception as e:
                valid_node = False

            for _, child_node_id in enumerate(node["childIds"]):
                if child_node_id not in node_id_to_idx:
                    continue
                # mark this to save some tokens
                child_depth = depth + 1 if valid_node else depth
                child_str = dfs(
                    node_id_to_idx[child_node_id], child_node_id, child_depth
                )
                if child_str.strip():
                    if tree_str.strip():
                        tree_str += "\n"
                    tree_str += child_str

            return tree_str

        tree = dfs(0, accessibility_tree[0]["nodeId"], 0)
        return tree, all_union_bounds, rect_dict
        
    def clean_tree(self, tree, elements, rects, texts, all_union_bounds, rect_dict):
        bound_dict = {}
        cleaned_tree = []
        last_info = ''
        count = len(rects)        
        
        rect_dict2 = np.zeros(count)
        allids= []
        for ele in elements:
            allids.append(ele.id)

        cbox_depth = -2
        opt_depth = -2
        last_depth = 0
        try:
            id_prefix = ".".join(elements[-1].id.split(".")[:-1])
        except:
            if self.verbose:
                logging.error('[ERROR] NO ELEMENTS FOUND IN SCREENSHOT')
                logging.error(tree)
            id_prefix = ""
            
        tree = re.sub(r'\\u[eE][0-9a-fA-F]{3}', '', tree)
    
        # Remove non-breaking space sequences
        tree = re.sub(r'\\xa0', ' ', tree)
        
        # Handle actual Unicode characters if present
        tree = re.sub(r'[\uE000-\uF8FF]', '', tree)
        tree = re.sub(r'[\xa0]', ' ', tree)
        
        # Replace multiple spaces with a single space
        # But preserve tabs and newlines
        tree = re.sub(r'[ \f\r\v]+', ' ', tree).replace(" ' ", " '").replace(" '\n", "'\n")
        for tr in tree.split("\n"):
            neg_id = re.findall(r'\[-(\d+)\]',tr)
            if neg_id:
                continue
            ori_id = re.findall(r'\[(\d+)\]',tr)
            if ori_id:
                newid = id_prefix + "." + ori_id[0]
                
                if "listbox" in tr:
                    cbox_depth = len(tr[:re.search(r'\[', tr).start()])
                if "] option '" in tr:
                    opt_depth = len(tr[:re.search(r'\[', tr).start()])
                    if opt_depth != cbox_depth + 1:
                        continue    

                try:
                    cur_info = tr[re.search(r"'", tr).end():].strip()
                    cur_info = cur_info[:re.search(r"'", cur_info).start()].strip()
                except:
                    try:
                        cur_info = tr[re.search("\"", tr).end():].strip()
                        cur_info = cur_info[:re.search("\"", cur_info).start()].strip()
                    except:
                        cur_info = tr[re.search(r'\]', tr).end():].strip()
                cur_depth = len(tr[:re.search(r'\[', tr).start()])
                
                bound = all_union_bounds[ori_id[0]]
                bound_dict[newid] = all_union_bounds[ori_id[0]]

                if newid in allids:
                    ridx = allids.index(newid)
                    rect_dict2[ridx] = 1
                    tr = tr[:re.search(r'\[', tr).end()] + str(ridx) + "]: " + tr[re.search(r'\]', tr).end():].strip()
                    
                    last_info = cur_info 
                    last_depth = cur_depth
                    cleaned_tree.append(tr.strip())
                    continue
                    
                if "listbox" in tr or "] option '" in tr or "RootWebArea" in tr:
                    last_info = cur_info 
                    last_depth = cur_depth
                    tr = tr[:re.search(r'\[', tr).end()] + str(count) + "]: " + tr[re.search(r'\]', tr).end():].strip()
                    count += 1
                    newid = id_prefix + "." + ori_id[0]
                    element = WebElement(self.driver_task, newid)
                    elements.append(element)
                    cleaned_tree.append(tr.strip())
                    continue
                    
                if ("''" in tr) or "LineBreak '\\n'" in tr:
                    continue
                if "button" not in tr and "input" not in tr and "textarea" not in tr and "link" not in tr and "listbox" not in tr and "menu" not in tr and "option" not in tr and "box" not in tr:
                    if (cur_info in last_info):
                        continue
                tr = tr[:re.search(r'\[', tr).end()] + str(count) + "]: " + tr[re.search(r'\]', tr).end():].strip()
                count += 1
                newid = id_prefix + "." + ori_id[0]
                element = WebElement(self.driver_task, newid)
                elements.append(element)
                last_info = cur_info    
                last_depth = cur_depth
                cleaned_tree.append(tr.strip())
                
        tree = ";\t".join(cleaned_tree).strip()
        return tree, elements
        
    def get_element_in_viewport_ratio(self,
        elem_left_bound: float,
        elem_top_bound: float,
        width: float,
        height: float
    ) -> float:
        elem_right_bound = elem_left_bound + width
        elem_lower_bound = elem_top_bound + height

        win_left_bound = 0
        win_right_bound = self.window_width
        win_top_bound = 0
        win_lower_bound = self.window_height

        # Compute the overlap in x and y axes
        overlap_width = max(
            0,
            min(elem_right_bound, win_right_bound)
            - max(elem_left_bound, win_left_bound),
        )
        overlap_height = max(
            0,
            min(elem_lower_bound, win_lower_bound)
            - max(elem_top_bound, win_top_bound),
        )

        # Compute the overlap area
        ratio = overlap_width * overlap_height / (width * height)
        return ratio
        
    # interact with webpage and add rectangles on elements
    def get_web_element_rect(self):

        if self.fix_box_color:
            selected_function = "getFixedColor"
        else:
            selected_function = "getRandomColor"
    
        js_script = """
            let labels = [];
    
            function markPage() {
                var bodyRect = document.body.getBoundingClientRect();
    
                var items = Array.prototype.slice.call(
                    document.querySelectorAll('*')
                ).map(function(element) {
                    var vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
                    var vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);
                    
                    var rects = [...element.getClientRects()].filter(bb => {
                    var center_x = bb.left + bb.width / 2;
                    var center_y = bb.top + bb.height / 2;
                    var elAtCenter = document.elementFromPoint(center_x, center_y);
    
                    return elAtCenter === element || element.contains(elAtCenter) 
                    }).map(bb => {
                    const rect = {
                        left: Math.max(0, bb.left),
                        top: Math.max(0, bb.top),
                        right: Math.min(vw, bb.right),
                        bottom: Math.min(vh, bb.bottom)
                    };
                    return {
                        ...rect,
                        width: rect.right - rect.left,
                        height: rect.bottom - rect.top
                    }
                    });
    
                    var area = rects.reduce((acc, rect) => acc + rect.width * rect.height, 0);
    
                    return {
                    element: element,
                    include: 
                        (element.tagName === "INPUT" || element.tagName === "TEXTAREA" || element.tagName === "SELECT") ||
                        (element.tagName === "BUTTON" || element.tagName === "A" || (element.onclick != null) || window.getComputedStyle(element).cursor == "pointer") ||
                        (element.tagName === "IFRAME" || element.tagName === "VIDEO" || element.tagName === "LI" || element.tagName === "TD" || element.tagName === "OPTION")
                    ,
                    area,
                    rects,
                    text: element.textContent.trim().replace(/\s{2,}/g, ' '),
                    coords: rects.map(r => ({left: r.left, top: r.top, right: r.right, bottom: r.bottom})) // return coordinate here
                    };
                }).filter(item =>
                    item.include && (item.area >= 20)
                );
    
                // Only keep inner clickable items
                // first delete button inner clickable items
                const buttons = Array.from(document.querySelectorAll('button, a, input[type="button"], div[role="button"]'));
    
                //items = items.filter(x => !buttons.some(y => y.contains(x.element) && !(x.element === y) ));
                items = items.filter(x => !buttons.some(y => items.some(z => z.element === y) && y.contains(x.element) && !(x.element === y) ));
                items = items.filter(x => 
                    !(x.element.parentNode && 
                    x.element.parentNode.tagName === 'SPAN' && 
                    x.element.parentNode.children.length === 1 && 
                    x.element.parentNode.getAttribute('role') &&
                    items.some(y => y.element === x.element.parentNode)));
    
                items = items.filter(x => !items.some(y => x.element.contains(y.element) && !(x == y)))
    
                // Function to generate random colors
                function getRandomColor(index) {
                    var letters = '0123456789ABCDEF';
                    var color = '#';
                    for (var i = 0; i < 6; i++) {
                    color += letters[Math.floor(Math.random() * 16)];
                    }
                    return color;
                }
    
                function getFixedColor(index) {
                    var color = '#000000'
                    return color
                }
                //function getFixedColor(index){
                //    var colors = ['#FF0000', '#00FF00', '#0000FF', '#000000']; // Red, Green, Blue, Black
                //    return colors[index % 4];
                //}
                
    
                // Lets create a floating border on top of these elements that will always be visible
                items.forEach(function(item, index) {
                    item.rects.forEach((bbox) => {
                    newElement = document.createElement("div");
                    var borderColor = COLOR_FUNCTION(index);
                    newElement.style.outline = `2px dashed ${borderColor}`;
                    newElement.style.position = "fixed";
                    newElement.style.left = bbox.left + "px";
                    newElement.style.top = bbox.top + "px";
                    newElement.style.width = bbox.width + "px";
                    newElement.style.height = bbox.height + "px";
                    newElement.style.pointerEvents = "none";
                    newElement.style.boxSizing = "border-box";
                    newElement.style.zIndex = 2147483647;
                    // newElement.style.background = `${borderColor}80`;
                    
                    // Add floating label at the corner
                    var label = document.createElement("span");
                    label.textContent = index;
                    label.style.position = "absolute";
                    //label.style.top = "-19px";
                    label.style.top = Math.max(-19, -bbox.top) + "px";
                    //label.style.left = "0px";
                    label.style.left = Math.min(Math.floor(bbox.width / 5), 2) + "px";
                    label.style.background = borderColor;
                    label.style.color = "white";
                    label.style.padding = "2px 4px";
                    label.style.fontSize = "12px";
                    label.style.borderRadius = "2px";
                    newElement.appendChild(label);
                    
                    document.body.appendChild(newElement);
                    labels.push(newElement);
                    // item.element.setAttribute("-ai-label", label.textContent);
                    });
                })
    
                // For the first way
                // return [labels, items.map(item => ({
                //     rect: item.rects[0] // assuming there's at least one rect
                // }))];
    
                // For the second way
                return [labels, items]
            }
            return markPage();""".replace("COLOR_FUNCTION", selected_function)
        rects, items_raw = self.driver_task.execute_script(js_script)
        
        format_ele_text = []
        all_union_bounds = []
        all_texts = []  
        
        for web_ele_id in range(len(items_raw)):
            element = items_raw[web_ele_id]['element']
            element_id = self.driver_task.execute_script("return arguments[0];", element)
            all_union_bounds.append([items_raw[web_ele_id]["rects"][0]["left"],items_raw[web_ele_id]["rects"][0]["top"],items_raw[web_ele_id]["rects"][0]["width"], items_raw[web_ele_id]["rects"][0]["height"]])
            # Get the nodeId using Chrome DevTools Protocol
            
            label_text = items_raw[web_ele_id]['text']
            ele_tag_name = element.tag_name
            ele_type = element.get_attribute("type")
            ele_aria_label = element.get_attribute("aria-label")
            input_attr_types = ['text', 'search', 'password', 'email', 'tel']
    
            if not label_text:
                if (ele_tag_name.lower() == 'input' and ele_type in input_attr_types) or ele_tag_name.lower() == 'textarea' or (ele_tag_name.lower() == 'button' and ele_type in ['submit', 'button']):
                    if ele_aria_label:
                        format_ele_text.append(f"[{web_ele_id}]: <{ele_tag_name}> \"{ele_aria_label}\";")
                        all_texts.append(ele_aria_label)
                    else:
                        format_ele_text.append(f"[{web_ele_id}]: <{ele_tag_name}> \"{label_text}\";" )
                        all_texts.append(label_text)
                else:
                    all_texts.append("")
    
            elif label_text and len(label_text) < 200:
                if not ("<img" in label_text and "src=" in label_text):
                    if ele_tag_name in ["button", "input", "textarea"]:
                        if ele_aria_label and (ele_aria_label != label_text):
                            format_ele_text.append(f"[{web_ele_id}]: <{ele_tag_name}> \"{label_text}\", \"{ele_aria_label}\";")
                            all_texts.append(ele_aria_label)
                        else:
                            format_ele_text.append(f"[{web_ele_id}]: <{ele_tag_name}> \"{label_text}\";")
                            all_texts.append(label_text)
                    else:
                        if ele_aria_label and (ele_aria_label != label_text):
                            format_ele_text.append(f"[{web_ele_id}]: \"{label_text}\", \"{ele_aria_label}\";")
                            all_texts.append(ele_aria_label)
                        else:
                            format_ele_text.append(f"[{web_ele_id}]: \"{label_text}\";")
                            all_texts.append(label_text)
                else:
                    all_texts.append("")
            else:
                all_texts.append("") 
    
        format_ele_text = '\t'.join(format_ele_text)

        return rects, [web_ele['element'] for web_ele in items_raw], format_ele_text, items_raw, all_union_bounds, all_texts

import concurrent
class BatchedWebEnv():
    def __init__(self,
                tasks, 
                download_dir = 'downloads', # download directory, need exist
                output_dir = 'results', # need exist
                env_config = None,
                is_test = False,
                region = 'us-west-2',
                aws_key_id = None,
                aws_secret_key = None
                 ):
        
        self.is_test = is_test
        self.tasks = tasks
        self.env_config = env_config
        self.ssh_key_path = env_config.ssh_key_path if hasattr(env_config, "ssh_key_path") else "/home/ubuntu/.ssh/id_rsa"
        self.download_dir = download_dir
        self.output_dir = output_dir
        self.webarena_host = env_config.webarena_host if hasattr(env_config, "webarena_host") else ""
        self.batch_size = env_config.batch_size if hasattr(env_config, "batch_size") else 4
        self.use_webarena_eval = (self.webarena_host and env_config.use_webarena_eval) if hasattr(env_config, "use_webarena_eval") else False
        self.max_iter = env_config.max_iter if hasattr(env_config, "max_iter") else 20
        self.verbose = env_config.verbose if hasattr(env_config, "verbose") else False        
        
        self.envs = []
                        
        for i in range(self.batch_size):
            os.makedirs(os.path.join(self.output_dir, f'batch{i}'), exist_ok=True)
            os.makedirs(os.path.join(self.download_dir, f'batch{i}'), exist_ok=True)
            env = WebBroswerGym(tasks, env_config, 
                                os.path.join(self.download_dir, f'batch{i}'), 
                                os.path.join(self.output_dir, f'batch{i}'),
                                region=region,
                                aws_key_id=aws_key_id,
                                aws_secret_key=aws_secret_key, batch_id=i, is_test=is_test)
            self.envs.append(env)
        
        self.connection_pool_size = min(16, self.batch_size * 2)  # Limit connection pool size
        os.environ['PYTHONUTF8'] = '1'  # Ensure proper encoding
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['WEBDRIVER_CONNECTION_POOL_SIZE'] = str(self.connection_pool_size)
            
    def reset_server(self):
        """Reset server between batches to clear zombie processes"""
        try:
            # Force kill Chrome/ChromeDriver processes
            os.system("pkill -9 -f '(chrome|chromedriver)' || true")
            
            # Also kill any pending Selenium processes
            os.system("pkill -9 -f selenium || true")
            
            # Clear any port locks that might be preventing new connections
            os.system("sudo fuser -k 9515/tcp || true")  # Default ChromeDriver port
            
            # Wait longer for processes to terminate
            time.sleep(5)
            
            # Force clear any system-wide resource leaks
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print("Server reset complete - all browser processes terminated")
        except Exception as e:
            print(f"[ERROR] SERVER RESET: {e}")
        

    def reset(self, num_round=0):
        batch_size = self.batch_size
            
        results = [None] * self.batch_size
        
        # Divide into smaller batches for staggered initialization
        for batch_start in tqdm(range(0, self.batch_size, batch_size), desc="Resetting Environments"):
            batch_end = min(self.batch_size, batch_start + batch_size)
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                jobs = [executor.submit(self.envs[i].reset, num_round) 
                    for i in range(batch_start, batch_end)]
                for i, job in enumerate(jobs):
                    results[batch_start + i] = job.result()
            
            # Wait between batches to avoid resource contention
            time.sleep(5)
        return results
    
    
    def step(self, actions):
        assert len(actions) == self.batch_size
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(env.step, action) for env, action in zip(self.envs, actions)]
            results = [job.result() for job in jobs]

        return results
