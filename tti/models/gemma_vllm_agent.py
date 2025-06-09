from PIL import Image
import json
import time
import torch
import asyncio
import aiohttp
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed.parallel_state import destroy_model_parallel
import time
import concurrent.futures
import requests
import json
import time
import concurrent.futures
import torch
import deepspeed
import gc
import ray
import base64
import time
import re
from transformers import Gemma3ForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from tti.misc import merge_dicts
import os
from vllm import LLM, SamplingParams
import threading
import logging        # For logging messages and errors
import json
import time
import torch
import asyncio
import aiohttp
from copy import deepcopy


class GemmaVllmAgent(nn.Module):
    def __init__(self, policy_lm, config=None, 
                 use_lora=False, vllm_tensor_parallel_size=1):
        """
        Create a GemmaAgent that wraps a gemma3 model.
        
        Parameters:
            policy_lm (str): HuggingFace model identifier (e.g. "Qwen/Qwen2.5-VL-7B-Instruct")
            config: A config object with attributes such as temperature and max_new_tokens.
            use_q4, use_lora, use_anyres: Additional flags (kept for signature compatibility; not used here).
        """
        super(GemmaVllmAgent, self).__init__()
        self.model = None
        self.train_model = None
        
        self.policy_lm = policy_lm
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.config = config
        self.temperature = config.temperature if config is not None else 1.0
        self.max_new_tokens = config.max_new_tokens if config is not None else 128
        self.use_lora = use_lora
        self.mode = None
        self.max_prompt_length = 16000
        self.updated_model_path = None

        self.sampling_params = SamplingParams(temperature=self.temperature, top_p = 0.95, min_p = 0.0,
                    max_tokens=self.max_new_tokens, logprobs=True,
                    stop_token_ids=[1, 106])

        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm)
        self.processor = AutoProcessor.from_pretrained(policy_lm)
            
    def enter_infer_mode(self, updated_model_path=None):
        """
        Transition from training to inference mode by properly cleaning up resources
        and initializing vLLM.
        """
        # Skip if already in infer mode
        if self.mode == "infer":
            return
            
        previous_mode = self.mode        
        
        # Initialize vLLM only on rank 0
        self.mode = "infer"  # Update mode before vLLM init to prevent race conditions

        if deepspeed.comm.get_rank() == 0:
            print(f"Transitioning to inference mode from {previous_mode}")
            # Kill any existing Ray processes first
            try:
                os.system("pkill -9 -f ray")
                os.system("pkill -9 -f vllm")
                time.sleep(3)  # Give time for processes to terminate
            except Exception as e:
                print(f"Warning when killing processes: {e}")
            
            # Set environment variables for vLLM
            os.environ["VLLM_USE_V1"] = "1"
            os.environ["VLLM_WORKER_USE_RAY"] = "0"
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            
            # Attempt to initialize vLLM with conservative settings
            try:        
                model_path = self.policy_lm if updated_model_path is None else updated_model_path   
                self.updated_model_path = model_path
                self.model = LLM(
                    model=model_path,
                    max_model_len=self.max_prompt_length,
                    tensor_parallel_size=self.vllm_tensor_parallel_size, 
                    gpu_memory_utilization=0.9,
                    enforce_eager=True,  # Avoid CUDA graphs
                    disable_log_stats=True,
                    limit_mm_per_prompt={"image": 4}
                )
                print(f"vLLM initialization successful from {model_path}")
            except Exception as e:
                print(f"Error initializing vLLM: {e}")
                import traceback
                traceback.print_exc()
                # Don't raise - continue with a null model
                self.model = None
        else:
            # Non-rank 0 processes just update their state
            self.model = None
        

    def enter_train_mode(self, updated_model_path=None):
        """
        Transition to training mode with robust cleanup of inference resources.
        """
        previous_mode = self.mode       

        if previous_mode == "train":
            return
        
        # Step 1: Clean up vLLM/Ray resources
        if previous_mode == "infer" and deepspeed.comm.get_rank() == 0:
            print(f"Transitioning to training mode from {previous_mode}")
            # Clean up vLLM resources if they exist
            if hasattr(self, "model") and self.model is not None:
                try:
                    destroy_model_parallel()
                    try:
                        del self.model.llm_engine.driver_worker
                    except Exception as e:
                        print(f"Warning deleting driver_worker")
                    del self.model
                    self.model = None
                    print("Cleaned up vLLM model")
                except Exception as e:
                    print(f"Warning during vLLM cleanup: {e}")
                    self.model = None
            
            # Kill any Ray processes
            try:
                if ray.is_initialized():
                    ray.shutdown()
                os.system("pkill -f ray")
                os.system("pkill -f vllm")
                time.sleep(3)  # Give processes time to terminate
            except Exception as e:
                print(f"Warning during Ray cleanup: {e}")
        
        # Clear GPU memory on all ranks
        gc.collect()
        torch.cuda.empty_cache()
        
        # Step 2: Update mode to train
        self.mode = "train"
        
        # Step 3: Load the HF model on all ranks
        model_path = self.policy_lm if updated_model_path is None else updated_model_path
        rank = deepspeed.comm.get_rank()

        if self.train_model is None:
            try:
                # Load with minimal memory usage
                self.train_model = Gemma3ForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                )
                print(f"Successfully loaded HF model for training on rank {rank}")
            except Exception as e:
                print(f"Error loading model on rank {rank}: {e}")
                import traceback
                traceback.print_exc()
                self.train_model = None
        else:
            print(f"Use existing on rank {rank}")
        
    
    # 1. Update get_gemma_vllm_prompts in GemmaVllmAgent class
    def get_gemma_hf_prompts(self, unprocessed_observation, system_prompt=None, transfer_image_to_base64=True):
        observation = deepcopy(unprocessed_observation)
        
        # Process each item in observation
        for obs_idx, obs in enumerate(observation):
            for round_idx, round_copy in enumerate(obs):
                if round_copy['role'] == "user":
                    for i in range(len(round_copy['content'])):
                        content_item = round_copy['content'][i]
                                            
                        # Process source if present
                        if 'source' in content_item:
                            # Handle image paths
                            if content_item['source'].get('type') == 'path':
                                img_path = content_item['source']['path']
                                observation[obs_idx][round_idx]['content'][i]['type'] = "image"
                                if transfer_image_to_base64:
                                    with open(img_path, "rb") as image_file:
                                        img_data = base64.b64encode(image_file.read()).decode('utf-8')
                                    observation[obs_idx][round_idx]['content'][i]['image'] = f"data:image/png;base64,{img_data}"
                                else:
                                    observation[obs_idx][round_idx]['content'][i]['image'] = img_path
                                del observation[obs_idx][round_idx]['content'][i]['source']
                                
                            # Handle base64 data
                            elif 'data' in content_item['source']:
                                observation[obs_idx][round_idx]['content'][i]['type'] = "image_url"
                                observation[obs_idx][round_idx]['content'][i]['image_url'] = {"url": f"data:image/png;base64,{content_item['source']['data']}"}
                                del  observation[obs_idx][round_idx]['content'][i]['source'] 
        return observation
        
    # 1. Update get_gemma_vllm_prompts in GemmaVllmAgent class
    def get_gemma_vllm_prompts(self, unprocessed_observation, system_prompt=None, transfer_image_to_base64=True):
        
        observation = deepcopy(unprocessed_observation)
        
        # Process each item in observation
        for obs_idx, obs in enumerate(observation):
            for round_idx, round_copy in enumerate(obs):
                if round_copy['role'] == "user":
                    for i in range(len(round_copy['content'])):
                        content_item = round_copy['content'][i]
                        
                        # Process source if present
                        if 'source' in content_item:
                            # Handle image paths
                            if content_item['source'].get('type') == 'path':
                                img_path = content_item['source']['path']
                                observation[obs_idx][round_idx]['content'][i]['type'] = "image_url"
                                if transfer_image_to_base64:
                                    with open(img_path, "rb") as image_file:
                                        img_data = base64.b64encode(image_file.read()).decode('utf-8')
                                    observation[obs_idx][round_idx]['content'][i]['image_url'] = {"url": f"data:image/png;base64,{img_data}"}
                                else:
                                    observation[obs_idx][round_idx]['content'][i]['image_url'] = {"url": img_path}
                                del observation[obs_idx][round_idx]['content'][i]['source']
                                
                            # Handle base64 data
                            elif 'data' in content_item['source']:
                                observation[obs_idx][round_idx]['content'][i]['type'] = "image_url"
                                observation[obs_idx][round_idx]['content'][i]['image_url'] = {"url": f"data:image/png;base64,{content_item['source']['data']}"}
                                del  observation[obs_idx][round_idx]['content'][i]['source']
        return observation
        
    def get_action(self, unprocessed_messages):
        """
        Get action from the model with improved error handling and logging
        """
        gen_ready = threading.Event()
        gen_result = [None]
        gen_error = [None]
        
        def generate_thread():
            with torch.no_grad():
                try:
                    # Process messages with better error handling
                    messages = self.get_gemma_vllm_prompts(unprocessed_messages)
                    bsz = len(messages)
                    
                    # Log message processing results
                    print(f"Processed {len(messages)} message lists")
                                            
                    # Try to get a response from the model
                    try:
                        response = self.model.chat(
                            messages=messages,
                            sampling_params=self.sampling_params,
                            chat_template=None,
                        )
                        
                    except Exception as e:
                        print("error in vLLM chat", e)
                        # might be too long output error
                        outputs = []
                        probs = []
                        trunc_messages = []
                        for msg_idx, msg in enumerate(messages):
                            total_len = 0
                            for msg_ in msg:
                                total_len += len(self.processor.tokenizer([msg_['content'][0]['text']])["input_ids"][0])
    
                            if total_len > self.max_prompt_length * 0.8:
                                
                                msg[-1]['content'][0]['text'] = msg[1]['content'][0]['text'][:re.search("Task: ", msg[1]['content'][0]['text']).start()] + msg[-1]['content'][0]['text'][re.search("Task: ", msg[-1]['content'][0]['text']).start():]
                                trunc_input_ids = self.processor.tokenizer([msg[-1]['content'][0]['text']])["input_ids"][0][:int(self.max_prompt_length * 0.8)]
                                trunc_input = self.processor.tokenizer.batch_decode([trunc_input_ids])[0]
                                msg[-1]['content'][0]['text'] = msg[-1]['content'][0]['text'][:len(trunc_input)]
                                trunc_messages.append([(msg[0])] + [(msg[-1])])
                                
                            else:
                                trunc_messages.append((msg))
    
                        response = self.model.chat(
                        messages=trunc_messages,
                        sampling_params=self.sampling_params,
                            chat_template=None,
                    )
                        
                    print(f"Received {len(response)} responses from model")
                    outputs = []
                    probs = []
                    for i, out in enumerate(response):
                        try:
                            
                            generated_text = out.outputs[0].text
                            prob = out.outputs[0].logprobs
                            p = 0
                            for tok_prob in prob:
                                for k, v in tok_prob.items():
                                    p += v.logprob
                                
                            outputs.append(generated_text)
                            probs.append(p / len(prob))
    
                            # print(f"Successfully processed and validated output {i}")
                        except Exception as e:
                            print(f"Error processing output {i}: {e}")
                            outputs.append("Thought: There was an error processing the model output.\nAction: Wait")
                            probs.append(-100)
    
                    if len(outputs) > bsz:
                        outputs = outputs[-bsz:]
                        probs = probs[-bsz:]

                    gen_result[0] = (outputs, probs)
                    gen_ready.set()
                    return 
                        
                except Exception as e:
                    print(f"CRITICAL ERROR in get_action: {e}")
                    import traceback
                    traceback.print_exc()
                    # Return a fallback response that will never be empty
                    gen_result[0] = (["Thought: There was a critical error in the model.\nAction: Wait"] * len(messages), [0] * len(messages))
                    gen_ready.set()
                    return
        thread = threading.Thread(target=generate_thread)
        thread.daemon = True
        thread.start()
        
        
        # Wait for generation to complete or timeout
        timeout_seconds = 300
        if not gen_ready.wait(timeout_seconds):
            logging.error(f"vLLM generation timed out after {timeout_seconds} seconds")

            logging.info("Attempting to reinitialize vLLM model after timeout")
            self.enter_train_mode()
            self.enter_infer_mode(self.updated_model_path)
            logging.info("Retrying generation with reinitialized model")
            try:
                generate_thread()
                return gen_result[0][0], gen_result[0][1]
            except Exception as e:
                logging.error(f"Retry generation failed: {e}")
                return ["Thought: There was a critical error in the model.\nAction: Wait"] * len(messages), [0] * len(messages)

        return gen_result[0][0], gen_result[0][1]


    def get_log_prob(self, messages, actions):
        """
        Compute log probabilities for actions with proper gradient flow and teacher-forcing shift.
        """ 
        messages = self.get_gemma_hf_prompts(messages, transfer_image_to_base64=False)

        # Apply chat template with proper multimodal handling.
        texts = [self.processor.apply_chat_template(message, tokenize=False) for message in messages]
        images = [self._process_vision_info(message) for message in messages]
        
        # Ensure each action ends with an EOS token.
        for i in range(len(actions)):
            actions[i] = "<start_of_turn>model\n" + actions[i] + "<end_of_turn>"
        
        # Append the actions to the texts.
        for i in range(len(texts)):
            texts[i] += actions[i]
        
        # Tokenize the whole batch (texts + images).
        try:
            batch = self.processor(
                text=texts, 
                images=images, 
                return_tensors="pt", 
                padding=True
            ).to(self.train_model.device)
        except Exception as e:
            print(f"Error during tokenization: {e}")
            print(f"message: {messages}")
        
        # Also tokenize actions on their own to later know lengths and create a mask for valid tokens.
        tokenized_actions = self.processor.tokenizer(actions, padding=True)
        tokenized_actions_mask = torch.tensor(tokenized_actions["attention_mask"]).to(self.train_model.device)
        
        # Forward pass (with gradient tracking).
        outputs = self.train_model(**batch)
        logits = outputs.logits  # Shape: [batch_size, seq_length, vocab_size]
        
        # Scale logits by temperature and compute log softmax over the vocabulary.
        logits = logits / self.temperature
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Determine the maximum action length (from the tokenized actions).
        max_action_length = tokenized_actions_mask.size(1)
        
        # The appended actions are at the end of the sequence.
        # For teacher forcing, we use the logits corresponding to each token position 
        # (except the very last one) to predict the next token.
        teacher_forcing_logits = log_probs[:, -max_action_length:-1, :]  
        # teacher_forcing_logits has shape: [batch_size, max_action_length - 1, vocab_size]
        
        # Prepare teacher forcing targets by shifting the tokenized actions by one (i.e. dropping the first token).
        # This aligns predictions at time t with target token at t+1.
        action_input_ids = torch.tensor(tokenized_actions["input_ids"]).to(self.train_model.device)
        teacher_forcing_targets = action_input_ids[:, 1:]  # Shape: [batch_size, max_action_length - 1]
        
        # Create a mask of tokens to exclude
        bsz, seq_len = tokenized_actions_mask.shape
        shifted_tokenized_actions_mask = torch.zeros_like(tokenized_actions_mask, dtype=tokenized_actions_mask.dtype, device=tokenized_actions_mask.device)
        shift_amount = 4
        shifted_tokenized_actions_mask[:, shift_amount:] = tokenized_actions_mask[:, :(seq_len - shift_amount)]
        teacher_forcing_mask = shifted_tokenized_actions_mask[:, 1:]  # Shape: [batch_size, max_action_length - 1]
                
        # Gather the log probabilities corresponding to the teacher forcing targets.
        action_log_probs = teacher_forcing_logits.gather(
            dim=2, index=teacher_forcing_targets.unsqueeze(-1)
        ).squeeze(-1)
        # Now, action_log_probs has shape: [batch_size, max_action_length - 1]
        
        # Zero out the padded tokens and sum the log probabilities for each example.
        summed_log_probs = (action_log_probs * teacher_forcing_mask).sum(dim=1)
        
        # Divide by the number of valid (non-padded) tokens to get the average log probability.
        valid_token_counts = teacher_forcing_mask.sum(dim=1)
        avg_log_probs = summed_log_probs / valid_token_counts

        return avg_log_probs

        
    def _process_vision_info(self, messages: list[dict]) -> list[Image.Image]:
        image_inputs = []
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                content = [content]

            for element in content:
                if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                    image_path = element['image']
                    image = Image.open(image_path)
                    image_inputs.append(image.convert("RGB"))
        return image_inputs
        
