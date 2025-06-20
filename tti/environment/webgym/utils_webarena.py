from typing import Any, TypedDict
import re
import logging
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

REDDIT = "http://WEBARENA_HOST:9999"
MAP = "http://WEBARENA_HOST:3000"#443"
GITLAB = "http://WEBARENA_HOST:8023"
SHOPPING_ADMIN = "http://WEBARENA_HOST:7780/admin"
SHOPPING = "http://WEBARENA_HOST:7770"

WEBARENA_DOMAINS = ['reddit', 'gitlab', 'shopping_admin', 'shopping', 'map']

def webarena_login(web_name, url, driver_task, webarena_host, batch_id, num_containers_per_machine):
    batch_id = batch_id % num_containers_per_machine
    match web_name:
        case 'reddit':
            username = "MarvelsGrantMan136"
            password = "test1234"
            for _ in range(3):
                try:
                    port_num = str(9999 + batch_id)
                    reddit = REDDIT.replace("WEBARENA_HOST", webarena_host.reddit).replace("9999", port_num)
                    url_mapping = [(reddit, "http://postmill.xyz"), ("reddit", "postmill"), ("Reddit", "Postmill")]
                    url = url.replace(":9999", ":" + port_num)
                    driver_task.get(f"{reddit}/login")
                    WebDriverWait(driver_task, 10).until(EC.presence_of_element_located((By.XPATH, "//label[text()='Username']/following-sibling::input")))
                    username_field = driver_task.find_element(By.XPATH, "//label[text()='Username']/following-sibling::input")
                    username_field.send_keys(username)
                    password_field = driver_task.find_element(By.XPATH, "//label[text()='Password']/following-sibling::input")
                    password_field.send_keys(password)
                    login_button = WebDriverWait(driver_task, 10).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[@type='submit' and normalize-space(text())='Log in']")))
                    login_button.click()
                    
                    time.sleep(5)
                    break
                except Exception as e:
                    if _ >= 2:
                        logging.error('[ERROR] REDDIT LOGIN')
                        logging.error(e)
                        return False, None, None
                    time.sleep(5)
        case 'gitlab':
            username = "byteblaze"
            password = "hello1234"

            for _ in range(3):
                try:
                    port_num = str(8023 + batch_id)
                    gitlab = GITLAB.replace("WEBARENA_HOST", webarena_host.gitlab).replace("8023", port_num)
                    url_mapping = [(gitlab, "http://gitlab.com")]
                    url = url.replace(":8023", ":" + port_num)
                    driver_task.get(f"{gitlab}/users/sign_in")

                    WebDriverWait(driver_task, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='username-field']"))
                    )

                    username_field = driver_task.find_element(By.CSS_SELECTOR, "[data-testid='username-field']")
                    username_field.click()
                    username_field.clear()
                    username_field.send_keys(username + Keys.TAB)

                    password_field = driver_task.find_element(By.CSS_SELECTOR, "[data-testid='password-field']")
                    password_field.clear()
                    password_field.send_keys(password)
        
                    sign_in_button = WebDriverWait(driver_task, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-testid='sign-in-button']"))
                    )
                    sign_in_button.click()

                    time.sleep(10)
                    break
                except Exception as e:
                    try:
                        port_num = str(8023 + batch_id)
                        gitlab = GITLAB.replace("WEBARENA_HOST", webarena_host.gitlab).replace("8023", port_num)
                        url_mapping = [(gitlab, "http://gitlab.com")]
                        url = url.replace(":8023", ":" + port_num)
                        driver_task.get(f"{gitlab}/users/sign_in")
                        
                        username_field = driver_task.find_element(By.DATA_TEST_ID, "username-field")
                        username_field.click()
                        username_field.send_keys(username)
                        username_field.send_keys(Keys.TAB)
                        
                        # Find and interact with password field
                        password_field = driver_task.find_element(By.DATA_TEST_ID, "password-field")
                        password_field.send_keys(password)
                        
                        # Find and click the sign-in button
                        sign_in_button = driver_task.find_element(By.DATA_TEST_ID, "sign-in-button")
                        sign_in_button.click()

                    except Exception as e:
                        if _ >= 2:
                            logging.error('[ERROR] GITLAB LOGIN')
                            logging.error(e)
                            return False, None, None
                        time.sleep(5)

        case 'shopping_admin':
            username = "admin"
            password = "admin1234"
            
            for _ in range(3):
                try:
                    port_num = str(7780 + batch_id)                    
                    shopping_admin = SHOPPING_ADMIN.replace("WEBARENA_HOST", webarena_host.shopping_admin).replace("7780", port_num)
                    url_mapping = [("metis.lti.cs.cmu.edu", webarena_host.shopping_admin), (shopping_admin, "http://luma.com/admin")]
                    driver_task.get(f"{shopping_admin}")
                    url = url.replace(":7780", ":" + port_num)

                    username_field = driver_task.find_element(By.CSS_SELECTOR, "[placeholder='user name']")
                    username_field.send_keys(username) 

                    password_field = driver_task.find_element(By.CSS_SELECTOR, "[placeholder='password']")
                    password_field.send_keys(password)  

                    sign_in_button = WebDriverWait(driver_task, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "button.action-login.action-primary"))
                    )
                    sign_in_button.click()
                    time.sleep(5)
                    break
                except Exception as e:
                    if _ >= 2:
                        logging.error('[ERROR] CMS LOGIN')
                        logging.error(e)
                        return False, None,None
                    time.sleep(5)               

        case 'shopping':
            username = "emma.lopez@gmail.com"
            password = "Password.123"

            for _ in range(3):
                try:
                    port_num = str(7770 - batch_id)                    
                    shopping = SHOPPING.replace("WEBARENA_HOST", webarena_host.shopping).replace("7770", port_num)
                    url_mapping = [("metis.lti.cs.cmu.edu", webarena_host.shopping), (shopping, "http://onestopmarket.com")]
                    url = url.replace(":7770", ":" + port_num)
                    
                    driver_task.get(f"{shopping}/customer/account/login/")

                    email_input = driver_task.find_element(By.ID, "email")
                    email_input.send_keys(username)

                    password_input = driver_task.find_element(By.ID, "pass")
                    password_input.send_keys(password)

                    sign_in_button = driver_task.find_element(By.ID, "send2")
                    sign_in_button.click()

                    time.sleep(5)
                    break
                except Exception as e:
                    if _ >= 2:
                        logging.error('[ERROR] SHOPPING LOGIN')
                        logging.error(e)
                        return False, None, None
                    time.sleep(5)
        case 'map':
            map_site = MAP.replace("WEBARENA_HOST", webarena_host.map)
            url_mapping = [(map_site, "http://www.openstreetmap.org")]
    return True, url_mapping, url
    
class AccessibilityTreeNode(TypedDict):
    nodeId: str
    ignored: bool
    role: dict[str, Any]
    chromeRole: dict[str, Any]
    name: dict[str, Any]
    properties: list[dict[str, Any]]
    childIds: list[str]
    parentId: str
    backendDOMNodeId: str
    frameId: str
    bound: list[float] | None
    union_bound: list[float] | None
    offsetrect_bound: list[float] | None


class BrowserConfig(TypedDict):
    win_top_bound: float
    win_left_bound: float
    win_width: float
    win_height: float
    win_right_bound: float
    win_lower_bound: float
    device_pixel_ratio: float


class BrowserInfo(TypedDict):
    DOMTree: dict[str, Any]
    config: BrowserConfig

IGNORED_ACTREE_PROPERTIES = (
    "focusable",
    "editable",
    "readonly",
    "level",
    "settable",
    "multiline",
    "invalid",
)

AccessibilityTree = list[AccessibilityTreeNode]

IN_VIEWPORT_RATIO_THRESHOLD = 0.6


def fetch_browser_info(
    # page: Page,
    browser,
) -> BrowserInfo:
    # extract domtree
    tree = browser.execute_cdp_cmd(
        "DOMSnapshot.captureSnapshot",
        {
            "computedStyles": [],
            "includeDOMRects": True,
            "includePaintOrder": True,
        },
    )

    # calibrate the bounds, in some cases, the bounds are scaled somehow
    bounds = tree["documents"][0]["layout"]["bounds"]
    b = bounds[0]
    n = b[2] / browser.get_window_size()["width"]
    bounds = [[x / n for x in bound] for bound in bounds]
    tree["documents"][0]["layout"]["bounds"] = bounds

    win_top_bound = browser.execute_script("return window.pageYOffset;")
    win_left_bound = browser.execute_script("return window.pageXOffset;")
    win_width = browser.execute_script("return window.screen.width;")
    win_height = browser.execute_script("return window.screen.height;")
    win_right_bound = win_left_bound + win_width
    win_lower_bound = win_top_bound + win_height
    device_pixel_ratio = browser.execute_script("return window.devicePixelRatio;")
    assert device_pixel_ratio == 1.0, "devicePixelRatio is not 1.0"

    config: BrowserConfig = {
        "win_top_bound": win_top_bound,
        "win_left_bound": win_left_bound,
        "win_width": win_width,
        "win_height": win_height,
        "win_right_bound": win_right_bound,
        "win_lower_bound": win_lower_bound,
        "device_pixel_ratio": device_pixel_ratio,
    }

    # assert len(tree['documents']) == 1, "More than one document in the DOM tree"
    info: BrowserInfo = {"DOMTree": tree, "config": config}

    return info




def get_element_in_viewport_ratio(
    elem_left_bound: float,
    elem_top_bound: float,
    width: float,
    height: float,
    config: BrowserConfig,
) -> float:
    elem_right_bound = elem_left_bound + width
    elem_lower_bound = elem_top_bound + height

    win_left_bound = 0
    win_right_bound = config["win_width"]
    win_top_bound = 0
    win_lower_bound = config["win_height"]

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
    ratio = overlap_width * overlap_height / width * height
    return ratio




def get_bounding_client_rect(
    browser, backend_node_id: str
) -> dict[str, Any]:
    try:
        remote_object = browser.execute_cdp_cmd(
            "DOM.resolveNode", {"backendNodeId": int(backend_node_id)}
        )
        remote_object_id = remote_object["object"]["objectId"]
        response = browser.execute_cdp_cmd(
            "Runtime.callFunctionOn",
            {
                "objectId": remote_object_id,
                "functionDeclaration": """
                    function() {
                        if (this.nodeType == 3) {
                            var range = document.createRange();
                            range.selectNode(this);
                            var rect = range.getBoundingClientRect().toJSON();
                            range.detach();
                            return rect;
                        } else {
                            return this.getBoundingClientRect().toJSON();
                        }
                    }
                """,
                "returnByValue": True,
            },
        )
        return response
    except:
        return {"result": {"subtype": "error"}}


def fetch_page_accessibility_tree(
    info: BrowserInfo,
    browser,
    # client: CDPSession,
    current_viewport_only: bool,
) -> AccessibilityTree:
    accessibility_tree: AccessibilityTree = browser.execute_cdp_cmd(
        "Accessibility.getFullAXTree", {}
    )["nodes"]

    # a few nodes are repeated in the accessibility tree
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
        backend_node_id = str(node["backendDOMNodeId"])
        if node["role"]["value"] == "RootWebArea":
            # always inside the viewport
            node["union_bound"] = [0.0, 0.0, 10.0, 10.0]
        else:
            response = get_bounding_client_rect(
                browser, backend_node_id
            )
            if response.get("result", {}).get("subtype", "") == "error":
                node["union_bound"] = None
            else:
                x = response["result"]["value"]["x"]
                y = response["result"]["value"]["y"]
                width = response["result"]["value"]["width"]
                height = response["result"]["value"]["height"]
                node["union_bound"] = [x, y, width, height]

    # filter nodes that are not in the current viewport
    if current_viewport_only:

        def remove_node_in_graph(node: AccessibilityTreeNode) -> None:
            # update the node information in the accessibility tree
            nodeid = node["nodeId"]
            node_cursor = nodeid_to_cursor[nodeid]
            parent_nodeid = node["parentId"]
            children_nodeids = node["childIds"]
            parent_cursor = nodeid_to_cursor[parent_nodeid]
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

        config = info["config"]
        for node in accessibility_tree:
            if not node["union_bound"]:
                remove_node_in_graph(node)
                continue

            [x, y, width, height] = node["union_bound"]

            # invisible node
            if width == 0 or height == 0:
                remove_node_in_graph(node)
                continue

            in_viewport_ratio = get_element_in_viewport_ratio(
                elem_left_bound=float(x),
                elem_top_bound=float(y),
                width=float(width),
                height=float(height),
                config=config,
            )

            if in_viewport_ratio < IN_VIEWPORT_RATIO_THRESHOLD:
                remove_node_in_graph(node)

        accessibility_tree = [
            node
            for node in accessibility_tree
            if node.get("parentId", "Root") != "[REMOVED]"
        ]

    return accessibility_tree


def parse_accessibility_tree(
    accessibility_tree: AccessibilityTree,
) -> tuple[str, dict[str, Any]]:
    """Parse the accessibility tree into a string text"""
    node_id_to_idx = {}
    for idx, node in enumerate(accessibility_tree):
        node_id_to_idx[node["nodeId"]] = idx

    obs_nodes_info = {}

    def dfs(idx: int, obs_node_id: str, depth: int) -> str:
        tree_str = ""
        node = accessibility_tree[idx]
        indent = "\t" * depth
        valid_node = True
        try:
            role = node["role"]["value"]
            name = node["name"]["value"]
            node_str = f"[{obs_node_id}] {role} {repr(name)}"
            properties = []
            for property in node.get("properties", []):
                try:
                    if property["name"] in IGNORED_ACTREE_PROPERTIES:
                        continue
                    properties.append(
                        f'{property["name"]}: {property["value"]["value"]}'
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
                tree_str += f"{indent}{node_str}"
                obs_nodes_info[obs_node_id] = {
                    "backend_id": node["backendDOMNodeId"],
                    "union_bound": node["union_bound"],
                    "text": node_str,
                }

        except:
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

    tree_str = dfs(0, accessibility_tree[0]["nodeId"], 0)
    return tree_str, obs_nodes_info


def clean_accesibility_tree(tree_str: str) -> str:
    """further clean accesibility tree"""
    clean_lines: list[str] = []
    for line in tree_str.split("\n"):
        if "statictext" in line.lower():
            prev_lines = clean_lines[-3:]
            pattern = r"\[\d+\] StaticText '([^']+)'"

            match = re.search(pattern, line)
            if match:
                static_text = match.group(1)
                if all(
                    static_text not in prev_line
                    for prev_line in prev_lines
                ):
                    clean_lines.append(line)
        else:
            clean_lines.append(line)

    return "\n".join(clean_lines)
