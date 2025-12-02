---
layout: note
title: "Breaking down ComfyUI to build Virgil"
last_modified_at: 2025-11-13 12:00:00 +0530
categories: [Tech, Stable Diffusion, Tutorial]
image: assets/cover_images/img6.webp
---

I have been building [Virgil] with my brother ([Dhanajaya]), We love tech, ai and art. and we have always been dishearted by how complex most ai generation platforms are either too complex (comfyUI) or ask for a subscription for their simpler UI.

That's why we set out to build virgil, one platform where you could create with AI without limitations and as simple as possible.

This is me breaking down ComfyUI and how I went about it, hopefully it helps you. Feel free to reach out to me with any questions. I have also created BabyComfy which is a minimal abstraction and simplification of ComfyUI, we are building virgil on top of this.

(Also do check out Virgil and give me your thoughts on it!!!)

## How I went about it

ComfyUI is a huge and complex code base to say the least. So I did the following first

1. I have been using ComfyUI for a while so I know the problems and beauty of it, so if you haven't used it yet. I will recommend installinng it and trying it yourself.
2. I forked the repo locally, opened claude code and gave the following prompt `I wish to write a blog breaking down this repository completely, I will also create diagrams in exalidraw to explain parts of it. Additionally I would like to make minimal replica while reading it. Create a reading_guide.md that I can follow. ` (Here is the guide it gave me.)
3. I went through the [docs](https://docs.comfy.org/)
4. Toughest part, just keep reading the code and writing it out.

## ComfyUI Overview

{later add}

## Lessons from docs

| "ComfyUI is an environment for building and running generative content workflows. In this context, a workflow is defined as a collection of program objects called nodes that are connected to each other, forming a network. This network is also known as a graph."
from [here](https://docs.comfy.org/development/core-concepts/workflow)

## Foundations

### main.py

1. How can main.py have multiple `if __name__ == "__main__":`?

"""
Looking at main.py:18, main.py:117, and main.py:364, this file actually has three if **name** == "**main**": blocks, which is valid Python but unconventional.

Here's what's happening:

1. Line 18 - Sets environment variables for telemetry
2. Line 117 - Configures CUDA/device settings
3. Line 364 - The main execution block

When Python runs a file directly, it evaluates code from top to bottom. Each if **name** == "**main**": block is checked independently, and since the condition is true (the file is being run  
 directly), all three blocks execute in sequence.

This works, but it's unusual code organization. Typically, you'd see:

- A single if **name** == "**main**": block at the end
- Code organized chronologically within that block

The reason this file is structured this way appears to be:

- Early environment setup needs to happen before imports (lines 18-21 run before the main imports)
- Device configuration needs to run before torch is imported (line 117)
- The actual startup logic runs last (line 364)

This is a deliberate design choice to control the order of initialization - setting certain environment variables before specific modules are imported. While it works, consolidating these into a  
 single block with clear comments would be more conventional.
"""

2. What "def prompt_worker" is doing in a few lines

"""
The prompt_worker function (main.py:170-246) is a background worker that:

1. Runs in an infinite loop pulling workflow execution requests from a queue
2. Executes each prompt using the PromptExecutor (line 202), which processes the node graph and generates outputs
3. Reports completion status back to the server with success/error messages (lines 206-211)
4. Manages memory by periodically running garbage collection and unloading models when flagged (lines 228-244)

Essentially, it's the core execution engine that processes ComfyUI workflows one at a time, running in a separate thread (started at line 339) so the server can remain responsive while
generating images or running other operations.
"""

3. What does "hijack_progress" do?

"""
The hijack_progress function (main.py:256-285) intercepts all progress updates during workflow execution and broadcasts them to connected clients. It:

1. Creates a hook function that captures progress updates (current value, total steps, preview images)
2. Enriches the data by automatically filling in the prompt_id and node_id from execution context if not provided
3. Sends real-time updates to the client via websocket (line 271) so the UI can show progress bars
4. Streams preview images during generation (lines 272-283) so users can see intermediate results
5. Registers this globally (line 285) so all progress callbacks throughout ComfyUI route through this function
   The term "hijack" means it's hooking into the global progress reporting system to ensure all progress events get sent to the web interface, enabling real-time feedback during image generation.
   """

start_comfyui

- Start a new event loop
- Setup a prompt server with that event loop
- Setup Hook Breaker (Why? what does this do?)
- Cuda malloc warning to let the user know that their gpu may not be supported
- Setup_database (what kind? What is this used for?)
- Add hijack_progress on the prompt server so that the progress is tracked
- Setup thread for prompt_worker (what is this thread?) [#! Potential improvement, python 3.14 and so on does not have GIL. Maybe we can make this better or even faster using that]
- Then start all together

"""
Your understanding is mostly correct! Here's the refined version with answers:

start_comfyui

- Start a new event loop (if not provided)
- Setup a prompt server with that event loop ✓
- Setup Hook Breaker - **Saves/restores core functions around custom node loading**
  (Protects against custom nodes that monkey-patch core ComfyUI functions - notice it's called at line 245 in prompt_worker too, restoring after each execution)
- Cuda malloc warning ✓
- Setup_database - **Initializes a database for workflow history/metadata**
  (Uses app/database/db.py, appears to be a feature in development - the error message says "in future the database will be required")
- Add routes to prompt server (line 336) + hijack_progress for real-time tracking ✓
- Setup thread for prompt_worker - **The execution engine thread that processes workflows from the queue**
  (daemon=True means it dies when main thread exits)
- Return the loop, server, and start_all function (doesn't actually start yet - that happens at line 376)

Re: Python 3.13+ free-threading (not 3.14): The potential improvement exists, but:

- Most compute happens in PyTorch/NumPy which already releases the GIL
- The threading here is more about concurrent I/O (queue management, websockets) than CPU parallelism
- Might see marginal gains, but the bottleneck is GPU execution, not Python threading
  """

### server.py

1. What does "@web.middleware" do?

"""

● @web.middleware is a decorator from aiohttp that creates middleware functions - interceptors that process every HTTP request before it reaches the route handler (and can also modify responses).

How it works:

@web.middleware
async def example_middleware(request, handler): # Code here runs BEFORE the route handler
response = await handler(request) # Call the actual route handler # Code here runs AFTER the route handler
return response

In server.py, there are several middlewares:

1. deprecation_warning (lines 56-72) - Logs warnings when deprecated API paths are accessed
2. compress_body (lines 75-85) - Compresses JSON/text responses with gzip if client supports it
3. create_cors_middleware() (lines 88-103) - Adds CORS headers for cross-origin requests
4. create_origin_only_middleware() (lines 131-165) - Security middleware that prevents CSRF attacks by checking Host and Origin headers match on localhost

The middleware chain (configured at lines 187-194) processes requests in order:
Request → cache_control → deprecation_warning → compress_body → CORS/Origin check → Route handler → Response

This pattern lets you add cross-cutting concerns (logging, security, compression) without modifying every route handler individually.
"""

2. What does "is_loopback" do? 

"""

● The is_loopback function (server.py:105-128) checks if a hostname or IP address refers to localhost (a loopback address like 127.0.0.1 or ::1).

  How it works:

  1. Quick check (lines 108-112): Tries to parse the host as an IP address and check if it's loopback directly
    - Returns True for 127.0.0.1, ::1, etc.
  2. Hostname resolution (lines 116-127): If the quick check fails (e.g., host is "localhost" string), it:
    - Resolves the hostname to IP addresses (both IPv4 and IPv6)
    - Checks if ALL resolved IPs are loopback addresses
    - Returns False if ANY resolved IP is NOT loopback

  Why it's needed:

  This function is used in create_origin_only_middleware() (line 146) to enable CSRF protection only for localhost requests:

  loopback = is_loopback(host_domain_parsed.hostname)

  if loopback and host_domain != origin_domain:
      # Block mismatched Host/Origin on localhost
      return web.Response(status=403)

  Security context: When ComfyUI runs on 127.0.0.1, a malicious website could try to send POST requests to http://127.0.0.1:8188 from your browser. The middleware blocks this by ensuring the
  Origin header matches the Host header for loopback addresses, preventing cross-site attacks against your local server.


"""

3. create_origin_only_middleware -> This can be fixed using cookie as mentioned

4. What is a mimetype? 

"""

A MIME type (Multipurpose Internet Mail Extensions type) is a standardized label that identifies the format/type of a file or data. It tells browsers and applications how to handle the content.     

  Format: type/subtype

  Common examples:
  - image/png - PNG image
  - image/jpeg - JPEG image
  - text/html - HTML webpage
  - application/json - JSON data
  - video/mp4 - MP4 video
  - application/octet-stream - Generic binary data (forces download)

  In server.py, mimetypes are used:

  1. Setup (lines 171-173):
  mimetypes.init()
  mimetypes.add_type('application/javascript; charset=utf-8', '.js')
  mimetypes.add_type('image/webp', '.webp')
  Maps file extensions to MIME types.

  2. File serving (line 529):
  content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
  Determines the correct MIME type from the file extension.

  3. Security (lines 532-533):
  if content_type in {'text/html', 'text/javascript', ...}:
      content_type = 'application/octet-stream'  # Forces download
  Prevents browser from executing HTML/JS files by forcing them to download instead.

  Why it matters: MIME types tell the browser whether to display an image, play a video, download a file, or execute JavaScript. Wrong MIME types can cause security vulnerabilities (XSS attacks)      
  or broken functionality.

"""

5. PromptServer is the main class of this function. The way it works is 

* initialize the different components 
* define the middlewares
* `self.app = web.Application(client_max_size=max_upload_size, middlewares=middlewares)` This takes care of starting a web server using aiohttp
* setup web_root to front_end_manager 
* Start routes using aiohttp using ` web.RouteTableDef()` (What do these aiohtpp endpoints do though?)

The different endpoints and understanding what they do 

#### GET

**websocket_handler**

"""
● The websocket_handler function (server.py:213-271) manages real-time bidirectional communication between the ComfyUI server and web clients (the browser UI).

  What it does:

  1. Establishes WebSocket connection (lines 215-216):
  ws = web.WebSocketResponse()
  await ws.prepare(request)
  2. Assigns client ID (lines 217-222):
    - Reuses existing clientId from query params if reconnecting
    - Generates new UUID if new connection
  3. Stores connection (lines 225-227):
  self.sockets[sid] = ws  # Store WebSocket
  self.sockets_metadata[sid] = {"feature_flags": {}}  # Store metadata
  4. Sends initial state (line 231):
    - Queue status and client ID to newly connected client
  5. Handles reconnection (lines 233-234):
    - If client was executing a workflow, re-send current node
  6. Processes incoming messages (lines 239-267):
    - Listens for messages from the client
    - Feature flags negotiation (lines 246-260): Client and server exchange capability information on first message
    - Handles JSON parsing errors gracefully
  7. Cleanup on disconnect (lines 269-270):
    - Removes socket and metadata when connection closes

  Why it's important: This WebSocket connection is how the UI receives real-time updates like:
  - Progress bars (hijack_progress sends updates here)
  - Preview images during generation
  - Queue status changes
  - Execution status

  Without this, the UI would be blind to what's happening on the server!
"""


**get_root**

Get the root path of the front end

**get_embeddings**

get path of embedding models 

**list_model_types**

list available model types 

**get_models**

Get the models which are available

**get_extensions**

Get the available extensions? 

**get_dir_by_type** (not an endpoint This is a helper function used internally by other routes like image_upload)

get directory based on directory type

**compare_image_hash** (helper function)

compare hash of two images to see if it already exists

**image_upload** (helper function)

Upload image to file path

**upload_image**

calls image_upload to upload an image 

**upload_mask**

Save an image then upload the mask 

**view_image**

view an image (Rather complex code, have a look and try to understand what is going on inside)

**view_metadata**

as the name implies

**system_stats**

get the stats of the system running the service


**get_features**

get features?

**get_prompt**

get prompt from the queue

**node_info**

get node types and version

**get_object_info**

get information about a particular node

**get_object_info_node**

get information about a particular group of nodes

**get_history**

get history? (but of what?)

**get_history_prompt_id**

get history based on prompt id 

**get_queue**

get queue (no clue what is happening inside though)
get_queue (/queue) - Returns pending/running workflows

#### POST 

**post_prompt**

Queues workflows for execution

**post_queue**

clear or delete runs in the queue

**post_interrupt**

interrupt running process

**post_free**

free memory

**post_history**

clear or delete history

#### Normal methods 

**setup**

Start a client without timeout

**add_routes**

add routes to enable communication 

**get_queue_info**

get the remaining tasks in the queue

**send**

based on if it is an image or json or bytes, send it. (Where?)

**encode_bytes**

Encode giving message (Why?)

**send_image**

Decode and save image 

**send_image_with_metadata**

Combine image and metadata and send it

**send_bytes**

send data over socket using bytes 

**send_json**

send data over socket using json

**send_sync**

Add message to the queue?

**queue_updated**

Update the queue

**publish_loop**

Get the messages in the queue and publish it

**start**

Starts Multi address 

**start_multi_address**

Start the web server

**add_on_prompt_handler**

add on prompt handler 

**trigger_on_prompt**

Trigger each handler with the json data

**send_progress_text**

send the progress so far

"""
 Corrections needed:

  get_embeddings - Returns list of embedding names (not paths), with file extensions removed

  get_prompt - Returns queue status info, not a prompt from the queue

  node_info - Returns detailed metadata about a node class (inputs, outputs, category, description), not "types and version"

  get_object_info - Returns info about ALL nodes, not a particular one

  get_object_info_node - Returns info about one specific node class, not a group

  get_history - Returns workflow execution history (answered your "but of what?")

  get_queue - Returns pending/running workflows in the queue (answered your confusion)

  send_image - Encodes image and sends it to WebSocket clients as preview (doesn't decode or save)

  queue_updated - Notifies clients about queue status changes (doesn't update the queue itself)

  Answering your questions:
  - send - Sends to connected WebSocket clients
  - encode_bytes - For efficient binary WebSocket protocol (event type + data)

  Everything else is correct!
"""

### execution.py 

**ExecutionResult**
* Enum for result of the execution

**DuplicateNodeError**

**IsChangedCache**
No clue, I assume it checks if the particular node has been cached or not. 

**CacheEntry**
No clue either

**CacheType**
The type of cache to use I presume.

**CacheSet**
Initialize the cache that needs to be set 

What are outputs and objects though? And why do most of them have Hierarchical Cache? 
Also what is `recursive_debug_dump`

How do you choose which function needs to be async and which needs to be sync

**get_input_data**
depending on if this is v3 or not 
get info like prompt, id and other stuff 

The heck? How is this code valid 

```python
for x in inputs:
    input_data = inputs[x]
```

and whichever data is not available mark them as missing 

**resolve_map_node_over_list_results**
counts down the remaining tasks and continues if not done

**_async_map_node_over_list**
No Clue 

**merge_result_data**
merge node execution results

**get_output_data**
run `_async_map_node_over_list` to get the return values 
check if any pending tasks are left 
finally get output from final values using `get_output_from_returns` 

What? Why this flow? Why do we need to get output from return values? 

**get_output_from_returns**
expand the results to get the final output

**format_value**
format the value of input

**execute**
start with getting all the ids 
if a node is async and pending, remove it because it failed 
for pending subgraph results, take the cached value and delete the rest 
if lazy status is pending resolve the node over list (what does this even mean)
execution block sending msg to server for sync execution 

#### PromptExecutor

How is the execute here different from the above one?

"""
orrections and Clarifications

  IsChangedCache

  Your understanding: "checks if the particular node has been cached or not"
  Actually: It caches the results of a node's IS_CHANGED or fingerprint_inputs method. These methods determine if a node's output needs to be recomputed based     
  on its inputs. It's about change detection, not just cache presence.

  CacheEntry

  It's a simple NamedTuple (lines 92-94) that holds:
  - ui: UI-related outputs (what gets displayed to the user)
  - outputs: The actual data outputs passed to downstream nodes

  CacheSet - outputs vs objects

  - outputs: Caches the results of node execution (the data produced)
  - objects: Caches the node instances themselves (the Python objects)

  HierarchicalCache: It's a caching strategy that can have parent-child relationships (important for subgraph execution where nodes can be nested).

  recursive_debug_dump: Debugging method to inspect cache contents.

  The Python code confusion

  for x in inputs:
      input_data = inputs[x]

  This is completely valid Python! When you iterate over a dict, you iterate over its keys. So x is each key, and inputs[x] gets the value. This is basic
  Python dict iteration.

  get_input_data

  Your understanding is too vague. This function:
  1. Gets the INPUT_TYPES schema for the node
  2. For each input, checks if it's a link (connection from another node) or a direct value
  3. If it's a link, retrieves the output from the cached results of the source node
  4. Handles hidden inputs (PROMPT, UNIQUE_ID, etc.)
  5. Returns all input data needed to execute the node

  _async_map_node_over_list

  This is crucial - it handles batch processing. When a node receives list inputs, it can either:
  - Process the entire list at once (INPUT_IS_LIST = True)
  - Process each element separately and merge results

● It also handles async execution, creating tasks for coroutines.

  get_output_data vs get_output_from_returns

  Why this flow? Because nodes can return outputs in different formats:
  - V1 nodes: tuples or dicts with {'ui': ..., 'result': ...}
  - V3 nodes: _NodeOutputInternal objects
  - Subgraph expansion: dicts with 'expand' key

  get_output_from_returns normalizes these different formats into a consistent structure.

  execute function (standalone)

  This executes a single node. Here's the flow:

  1. Check cache - if node result is cached, return it immediately
  2. Three execution paths:
    - Async continuation: Node had pending async tasks, resolve them
    - Subgraph continuation: Node expanded into subgraph, resolve the results
    - Normal execution: Execute the node fresh
  3. Lazy evaluation: If node has check_lazy_status, it can request additional inputs dynamically
  4. Subgraph expansion: Some nodes can dynamically create new nodes (workflows within workflows)
  5. Cache results and return

  PromptExecutor.execute vs standalone execute

  Key difference:
  - PromptExecutor.execute_async: Orchestrates execution of the entire workflow (all nodes)
  - Standalone execute: Executes a single node

  The PromptExecutor:
  - Manages the execution queue
  - Calls the standalone execute for each node
  - Handles execution order via ExecutionList
  - Manages global state (caches, progress, etc.)
  - Sends status updates to the server/UI

  Key Concepts You're Missing

  1. Subgraph expansion: Nodes can dynamically create new sub-workflows
  2. Lazy evaluation: Nodes can request inputs on-demand during execution
  3. Batch processing: The "map over list" concept handles processing lists of inputs
  4. Async execution: Some nodes return coroutines that execute asynchronously

  Async vs Sync Decision

  Functions are async when they:
  - Need to await other async operations
  - Call _async_map_node_over_list (which might create async tasks)
  - Perform I/O or long-running operations that shouldn't block
"""

**reset**

Get's cache (how does it reset it?)

**add_message**

synchronously adds data and event to server

**handle_execution_error**

Send message to the frontend of the error encountered (node error or any kind of error)

**execute**

Wrapper to run async execute synchronously 

**execute_async**

* Put the interupt as False 
* Add message of starting the execution 
* Start torch in inference mode
* Create Dynamic Prompt 
* Reset the progress state 
* Add a progress handler
* Check if the caches have changed
* While the execution list is not empty keep executing it (This execute is the one outside of the class)
* Handle execution error if any
* poll the ram (What?)
* add the outputs to ui outputs

DONE WITH THE CLASS 

**validate_inputs**

Validates the given inputs. (How?)

`r = await validate_inputs(prompt_id, prompt, o_id, validated)` -> Why recursion?

**full_type_name**

No clue what is going on

**validate_prompt**

Checks if a prompt is valid (How?)

#### PromptQueue

**init**
What is mutex? What is threading.Rlock? What is going on yoooooo??????

**put**
Why put it in a heap queue? What is going on yooooooooooo?

**get**


**ExecutionStatus**


**task_done**


**get_current_queue**


**get_current_queue_volatile**


**get_tasks_remaining**


**wipe_queue**


**delete_queue_item**


**get_history**


**wipe_history**


**delete_history_item**


**set_flags**


**get_flags**


## ComfyUI_frontend Overview

<!-- Self note

> https://registry.comfy.org/ -> This is gold for inspiration and putting in virgil (Hiring goldmine as well) -->
