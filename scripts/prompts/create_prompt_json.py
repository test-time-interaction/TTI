import json
import os

def create_webvoyager_prompt_templates(filename):
    """Create a JSON file with prompt templates for WebVoyager."""
    
    actor_template = {
        "initial": """Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. In each iteration, you will receive an observation that includes the accessibility tree of the webpage and a screenshot of the current viewpoint. The accessbility tree contains information about the web elements and their properties. The screenshot will feature numerical labels placed in the TOP LEFT corner of web elements in th current viewpoint.
Carefully analyze the webpage information to identify the numerical label corresponding to the web element that requires interaction, then follow the guidelines and choose one of the following actions:
1. Click a web element.
2. Delete existing content in a textbox and then type content. 
3. Scroll up or down the whole window.
4. Go back, returning to the previous webpage.
5. Navigate to Bing's homepage.
6. Answer. This action should only be chosen when all questions in the task have been solved.

Correspondingly, action should STRICTLY follow the format specified by one of the following lines:
Click [numerical_label]
Type [numerical_label] [content]
Scroll [up/down]
GoBack
Bing
ANSWER [content]

Some examples are:
Click [8]
Type [22] [Boston]
Scroll [down]
Bing
ANSWER [06516]

Key guidelines you MUST follow:
* Action guidelines *
1. The predicted action should be based on elements as long as it's accessibility tree OR screenshot. Sometimes, accessibility tree or screenshot captures more elements than the other, but it's fine to use either one.
2. To input text for search bars, no need to click textbox first, directly type content. After typing, the system automatically hits 'ENTER' key.
3. When a complex task involves multiple questions or steps, select 'ANSWER' only at the very end, after addressing all of these questions or steps. Double check the formatting requirements in the task when ANSWER. Always think twice before using 'ANSWER' action!!!
4. When specifying the content for 'Type' and 'ANSWER' actions, be sure to wrap the content with '[]'.
5. Use `GoBack` to return to the previous state, use it when you find the previous action incorrect. 
6. When you see a pop-up page, you should immediately `GoBack` to the previous page.
7. Use `Bing` when you need to navigate to a different website or search for new information.

Your reply should strictly follow the format:

Thought: Your reasoning trace. A good practice is to follow this format:
- Observation summary: where are you at now? list all elements that are related to the task goal. e.g. if you're trying to filter something out, list all filters visible.
- Planning: what sequence of actions do you need take to achieve the task goal? give a high-level overview of the steps you need to take.
- Possible actions: to achieve that plan, what are potential actions you need to do immediately and what's their effect? List at least 3 actions and analyze each of them.
Action: Based on this reasoning, identify the single most optimal action. You should output it in the format specified above ("...STRICTLY follow the format...").

After you issue an action, the user will execute it and provide a new observation. Now solve the following task.

Task: {task_goal}

Current URL: {url}

Screenshot of current viewpoint: attached

Accessibility tree of current viewpoint:
{accessibility_tree}""",
        
        # Add hint structure to match WebArena format
        "hint": {
            "general": "- Always save progress through appropriate buttons (Save, Submit, Post, etc.)\n- Always remember to interact with dropdown options after expanding\n- Clear filters before setting new ones\n- Use `Bing` to navigate to Bing when you need to search for information or visit a different website"
        },
        
        "observation": "Task: {task_goal}\nCurrent URL: {url}\nScreenshot of current viewpoint: attached\nAccessibility tree of current viewpoint:\n{accessibility_tree}",
        
        "pdf_observation": "Task: {task_goal}\nPDF: {pdf_obs}",
        
        "error": "The action you have chosen cannot be executed. Please double-check if your output contains the wrong numerical label or action or action format. Then revise it.",
        "tree_indicator": "\nAccessibility tree of current viewpoint:",
        "pattern": r'Thought:|Action:'
    }

    # Write templates to JSON file
    file_path = os.path.join("./", filename + ".json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(actor_template, f, indent=2)
    
    print(f"WebVoyager prompt templates JSON file created: {file_path}")
    
if __name__ == "__main__":
    create_webvoyager_prompt_templates("webvoyager")