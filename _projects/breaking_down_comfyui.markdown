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
2. I forked the repo locally, opened claude code and gave the following prompt `I wish to write a blog breaking down this repository completely, I will also create diagrams in exalidraw to explain parts of it. Additionally I would like to make minimal replica while reading it. Create a reading_guide.md that I can follow.` (Here is the guide it gave me.)
3. I went through the [docs](https://docs.comfy.org/)
4. Toughest part, just keep reading the code and writing it out.

## ComfyUI Overview

{later add}

## Lessons from docs

| "ComfyUI is an environment for building and running generative content workflows. In this context, a workflow is defined as a collection of program objects called nodes that are connected to each other, forming a network. This network is also known as a graph."
from [here](https://docs.comfy.org/development/core-concepts/workflow)

## Foundations

### main.py ⭐⭐⭐ CRITICAL

"""
  **Purpose:** Application entry point

- Parses command-line arguments
- Sets up model paths
- Loads custom nodes
- Starts the server
- **Run this to start ComfyUI**

  **Importance:** Critical - the main entry point.
"""

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

### server.py ⭐⭐⭐ CRITICAL

"""
  **Purpose:** Main web server (aiohttp)

- WebSocket server for real-time communication
- HTTP endpoints for API
- Handles prompt submission, queue management
- Sends execution updates to frontend
- Image upload/download

  **Importance:** Critical - this is the main server entry point.
"""

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

  Security context: When ComfyUI runs on 127.0.0.1, a malicious website could try to send POST requests to <http://127.0.0.1:8188> from your browser. The middleware blocks this by ensuring the
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
  content_type = mimetypes.guess_type[filename](0) or 'application/octet-stream'
  Determines the correct MIME type from the file extension.

  3. Security (lines 532-533):
  if content_type in {'text/html', 'text/javascript', ...}:
      content_type = 'application/octet-stream'  # Forces download
  Prevents browser from executing HTML/JS files by forcing them to download instead.

  Why it matters: MIME types tell the browser whether to display an image, play a video, download a file, or execute JavaScript. Wrong MIME types can cause security vulnerabilities (XSS attacks)
  or broken functionality.

"""

5. PromptServer is the main class of this function. The way it works is

- initialize the different components
- define the middlewares
- `self.app = web.Application(client_max_size=max_upload_size, middlewares=middlewares)` This takes care of starting a web server using aiohttp
- setup web_root to front_end_manager
- Start routes using aiohttp using `web.RouteTableDef()` (What do these aiohtpp endpoints do though?)

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

### execution.py ⭐⭐⭐ CRITICAL

"""

- Main execution logic
- PromptQueue class
- execute() function
- PromptExecutor class
"""

**ExecutionResult**

- Enum for result of the execution

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

- Put the interupt as False
- Add message of starting the execution
- Start torch in inference mode
- Create Dynamic Prompt
- Reset the progress state
- Add a progress handler
- Check if the caches have changed
- While the execution list is not empty keep executing it (This execute is the one outside of the class)
- Handle execution error if any
- poll the ram (What?)
- add the outputs to ui outputs

DONE WITH THE CLASS

**validate_inputs**

Validates the given inputs. (How?)

`r = await validate_inputs(prompt_id, prompt, o_id, validated)` -> Why recursion?

**full_type_name**

No clue what is going on

**validate_prompt**

Checks if a prompt is valid (How?)

#### PromptQueue

<https://www.troyfawkes.com/learn-python-multithreading-queues-basics/> -> Really helpful to understand all of this
<https://bbc.github.io/cloudfit-public-docs/> -> This too

**init**
What is mutex? What is threading.Rlock? What is going on yoooooo??????
Okay mutex locks the thread so noone does anything else to it

**put**
Why put it in a heap queue? What is going on yooooooooooo?
adds item to the queue

(Should we use PriorityQueue instead of heapq?)

**get**

get item and count of that item as well as update the task counter

**ExecutionStatus**

What is the status of the execution

**task_done**

Start locked thread

Whats going on inside though?

**get_current_queue**

With mutex get the current queue
Why is this slow?

**get_current_queue_volatile**

How is this different from get_current_queue?

**get_tasks_remaining**

Get size of the remaining tasks

**wipe_queue**

Clean the queue

**delete_queue_item**

Remove 1 item from the queue

**get_history**

get history of everything that was ran?

**wipe_history**

delete history

**delete_history_item**

delete history of a specific item (history is a dict)

**set_flags**

set the flag (What flag, where?)

**get_flags**

What?

"""
PromptQueue Methods - Detailed Analysis

  init (lines 1092-1100)

  Your understanding: "mutex locks the thread"

  More precise:

- threading.RLock() is a reentrant lock (can be acquired multiple times by the same thread)
- threading.Condition(self.mutex) is a condition variable - allows threads to wait for
  notifications
- Together they enable thread-safe queue operations where multiple threads can safely add/remove
  items

  Why threading? ComfyUI can have multiple workers/threads executing prompts simultaneously.

  put (lines 1102-1106)

  Your question: "Why put it in a heap queue?"

  Answer: heapq implements a priority queue. Queue items are tuples like (priority, number,
  prompt_id, prompt, extra_data). The heap automatically keeps the highest priority items at the
  front. This means:

- Lower priority numbers = higher priority execution
- Items execute in priority order, not just FIFO

  Should we use PriorityQueue? No, heapq is fine and more lightweight. queue.PriorityQueue is
  thread-safe but adds overhead.

  get (lines 1108-1119)

  Your understanding: "get item and count of that item as well as update the task counter"

  Correction: It doesn't count "that item" - it:

  1. Waits until queue is not empty (blocks with self.not_empty.wait())
  2. Pops the highest priority item from the heap
  3. Assigns it a unique task ID (self.task_counter)
  4. Moves it to currently_running dict
  5. Returns (item, task_id)

  The task_counter is a global incrementing ID, not item-specific.

  ExecutionStatus (lines 1121-1124)

  ✅ Correct - it's a NamedTuple holding execution status info (success/error, whether completed,
  messages).

  task_done (lines 1126-1146)

  Your confusion: "What's going on inside?"

  Here's the flow:

  1. Lock the thread with mutex
  2. Remove the item from currently_running using item_id
  3. Limit history size - if history > 10,000 items, remove the oldest entry
  4. Convert status to dict (if provided)
  5. Optional processing - call process_item to transform the prompt before storing
  6. Store in history dict with:
  - The prompt info
  - Empty outputs dict (filled later)
  - Status dict
  - History result (UI outputs, metadata)
  7. Notify server that queue updated

  get_current_queue (lines 1148-1154)

  Your question: "Why is this slow?"

  Answer: Line 1154 - copy.deepcopy(self.queue) creates a full recursive copy of all queue items.
  Each item contains the entire prompt dict (all nodes, inputs, etc.). This is expensive for large
  queues.

  Why deep copy? To return a safe snapshot that won't change if the queue is modified by another
  thread.

  get_current_queue_volatile (lines 1156-1161)

  Your question: "How is this different?"

  Key difference:

- get_current_queue: Uses copy.deepcopy() - safe but slow
- get_current_queue_volatile: Uses copy.copy() (shallow copy) - fast but potentially unsafe

  "Volatile" means: The returned data might reference objects that other threads are modifying. Safe
  for read-only viewing, not for modification.

  get_tasks_remaining (lines 1163-1165)

  ✅ Correct - Returns len(queue) + len(currently_running).

  wipe_queue (lines 1167-1170)

  ✅ Correct - Clears all pending items from the queue.

  delete_queue_item (lines 1172-1183)

  Your understanding is basically correct, but here's the detail:

- Takes a function parameter (a predicate/filter)
- Iterates through queue looking for an item where function(item) returns True
- When found, removes it and re-heapifies to maintain heap property
- Returns True if deleted, False if not found

  Example usage:
  queue.delete_queue_item(lambda x: x[2] == "prompt_123")  # Delete by prompt_id

  get_history (lines 1185-1210)

  Your understanding: "get history of everything that was ran"

  More precise: Returns execution history with flexible querying:

- prompt_id=None: Get all history (with optional pagination via offset/max_items)
- prompt_id="xyz": Get specific prompt's history
- map_function: Optional transformer to process each history entry before returning

  Not just "everything" - you can filter and paginate.

  wipe_history (lines 1212-1214)

  ✅ Correct - Clears all execution history.

  delete_history_item (lines 1216-1218)

  ✅ Correct - Deletes a specific history entry by ID. Uses pop(id, None) so it doesn't error if ID
  doesn't exist.

  set_flag (lines 1220-1223)

  Your confusion: "What flag, where?"

  Answer: This is a generic signaling mechanism. Flags are stored in self.flags dict as {name: data}.
   Used for:

- Interrupting execution
- Sending control signals between threads
- Example: queue.set_flag("interrupt", True) to stop execution

  After setting, it calls self.not_empty.notify() to wake up any waiting threads.

  get_flags (lines 1225-1232)

  Answer:

- Retrieves all flags from self.flags dict
- reset=True (default): Returns flags and clears them (consume-once pattern)
- reset=False: Returns a copy without clearing (peek pattern)

  Use case: Worker threads periodically call get_flags() to check for interrupt signals or other
  commands.

  ---
  Key Concepts You're Missing

  1. Thread synchronization: This entire class is about coordinating multiple threads safely
  accessing shared data
  2. Priority queue: Items execute by priority, not FIFO
  3. Producer-consumer pattern: Web API puts items, worker threads get them
  4. Condition variables: not_empty.wait() / not_empty.notify() efficiently wake sleeping threads
"""

### **nodes.py** ⭐⭐⭐ CRITICAL

"""
  **Purpose:** Main node definitions for the core ComfyUI workflow system

- Defines all built-in node classes (LoadImage, SaveImage, KSampler, etc.)
- Contains NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
- Imports all core functionality from comfy/ modules
- **This is where ALL nodes are registered and made available to the UI**

  **Important:** This is one of the most important files - it's the bridge between the UI and the backend functionality.
"""

This is probably the most straight forward file in this repo. There are different classes which are the different nodes. The input type i.e what all it takes is defined as a [classmethod], the return type, fuction, and category are constants to help know what they return, what they do, and where should be stored respectively.

Of these the most essential and commonly used nodes are
- cheakpoint loader
- VAE encode
- VAE Decode
- SaveImage
- LoadImage
- PreviewImage
- Clip Text Encode
- VAELoader
- Ksampler

It's easier to just look at the mapping defined in `NODE_DISPLAY_NAME_MAPPINGS`

---

That covers all of the most important root files. Now let us go through the directories in alphabetical order

### **alembic_db/**

mostly irrelevant

"""
  **Purpose:** Database migration system (Alembic)

- Manages SQL schema changes over time
- Run `alembic revision --autogenerate -m "message"` to create migrations

  **Importance:** Only matters if modifying the database schema.
"""

### **api_server/** ⭐⭐ IMPORTANT

"""
  **Purpose:** FastAPI/aiohttp REST API endpoints
  **Structure:**

- `routes/` - API route handlers
  - `routes/internal/` - Internal API endpoints
- `services/` - Business logic services
- `utils/` - API utilities

  **Importance:** Critical for understanding the API and backend architecture.
"""
This mostly did not make any sense to me


### **app/** ⭐⭐ IMPORTANT

"""
  **Purpose:** Application-level services and management
  **Key files:**

- `frontend_management.py` - Manages frontend versioning and downloads
- `user_manager.py` - User authentication and session management
- `model_manager.py` - Model file tracking and organization
- `custom_node_manager.py` - Custom node installation and management
- `subgraph_manager.py` - Workflow subgraph management
- `logger.py` - Logging configuration
- `database/` - SQLite database models (Alembic migrations)

  **Importance:** Application infrastructure - manages users, models, and custom nodes.
"""

Quick definitions of some directories/files 

* database -> I more or less have no clue what this is or why or where it is even used. 
* app_settings.py -> used for getting and saving user settings
* logger.py -> used to setup logging
* custom_node_manager.py -> get the custom nodes from the .json and load them 
* frontend_management.py -> Used to check if the correct front-end package is installed.
* model_manager.py -> get's the model from the given filepath?
* subgraph_manager.py -> used to load subgraphs
* user_manager.py -> 




### **comfy/** ⭐⭐⭐ CRITICAL

"""

  **Purpose:** Core machine learning and model management library
  **Key files:**

- `model_management.py` - GPU/CPU memory management, model loading/unloading
- `model_patcher.py` - Dynamic model patching for LoRAs, control nets
- `sd.py` - Stable Diffusion model implementation
- `samplers.py` - All sampling algorithms (Euler, DPM, etc.)
- `model_base.py` - Base model architectures
- `model_detection.py` - Auto-detect model types from files
- `supported_models.py` - Configuration for all supported model architectures
- `controlnet.py` - ControlNet implementation
- `lora.py` - LoRA loading and application
- `clip_model.py` - CLIP text encoder
- `utils.py` - Utility functions for tensor operations

  **Subdirectories:**

- `ldm/` - Latent Diffusion Model modules
- `text_encoders/` - Various text encoder implementations (T5, CLIP, etc.)
- `k_diffusion/` - Katherine Crowson's k-diffusion sampling
- `extra_samplers/` - Additional sampling methods

  **Importance:** This is the heart of ComfyUI - all ML/AI functionality lives here.
"""
It is quite troublesome to go through this entire directory and try to understand all parts of it. I will recommend pick this flow. 
Choose one model (let's say flux) go to where that is defined ([add path here]) and read how that is ran.

I did the same for Flux as I wanted to have that as the first model in virgil. And it worked out quite well. You can see my implementation details here.


### **comfy_api/** ⭐⭐⭐ CRITICAL

"""
  **Purpose:** V3 ComfyUI API system (new node API)
  **Structure:**

- `latest/` - Latest API version
- `v0_0_1/`, `v0_0_2/` - Versioned APIs
- `internal/` - Internal API utilities
- `feature_flags.py` - Feature toggles
- `version_list.py` - API version management

  **Importance:** Critical for understanding the new V3 node system and API versioning.
"""

Everything beside internal is just talking about types. I do not understand what is going inside of internal though. [FIND_OUT]

### **comfy_api_nodes/** ⭐

"""
  **Purpose:** Nodes that call external Comfy.org APIs

- API nodes for cloud services (model sharing, workflow sharing, etc.)
- Auto-generated from OpenAPI specs
- Contains both staging and production API integration

  **When to care:** Only if working with Comfy.org cloud features or API nodes.
"""

These contain the nodes that call external services. 

Let's understand how we can build a node by picking one model, in this case I chose Gemini.

<!-- Note to self: Learn from these and make one for gemini -->

### **comfy_config/**

"""
  **Purpose:** Configuration parsing and types

- `config_parser.py` - Parse YAML/JSON configs
- `types.py` - Type definitions for configs

  **Importance:** For advanced configuration management.
"""

### **comfy_execution/** ⭐⭐⭐ CRITICAL

"""
  **Purpose:** Execution engine
  **Key files:**

- `graph.py` - DynamicPrompt, ExecutionList, dependency graph
- `caching.py` - Caching strategies (LRU, RAM pressure, etc.)
- `progress.py` - Progress tracking and reporting
- `validation.py` - Input validation
- `utils.py` - Execution utilities

  **Importance:** EXTREMELY CRITICAL - this is the execution engine that runs workflows.
"""



### **comfy_extras/** ⭐⭐ IMPORTANT

"""
  **Purpose:** Extra/experimental nodes and features
  **Contains 80+ specialized node files:**

- `nodes_custom_sampler.py` - Advanced sampling nodes
- `nodes_model_merging.py` - Model merging functionality
- `nodes_latent.py` - Latent space manipulation
- `nodes_mask.py` - Mask operations
- `nodes_flux.py`, `nodes_sd3.py` - Model-specific nodes
- `nodes_hooks.py` - Sampling hooks
- And many more specialized nodes...

  **Importance:** Contains most advanced/experimental features. Check here for specialized functionality.
"""


### **custom_nodes/** ⭐⭐ IMPORTANT
"""
  **Purpose:** User-installed custom node extensions

- `example_node.py.example` - Template for creating custom nodes
- `websocket_image_save.py` - Example WebSocket node
- **Third-party nodes install here**

  **Importance:** This is where community extensions live. Essential for understanding the plugin system.
"""


### **input/**

"""
  **Purpose:** Input files for workflows

- Default location for input images
- `3d/` - 3D model inputs
- `example.png` - Example input image

  **Importance:** Low - just a data directory.
"""


### **middleware/**
"""
  **Purpose:** HTTP middleware for the web server

- `cache_middleware.py` - HTTP caching headers

  **Importance:** Low - only matters for web server optimization.
"""


### **models/** ⭐⭐⭐ CRITICAL
"""
  **Purpose:** All AI model files (checkpoints, LoRAs, etc.)
  **Subdirectories:**

- `checkpoints/` - Main SD models (.safetensors, .ckpt)
- `loras/` - LoRA files
- `vae/` - VAE models
- `controlnet/` - ControlNet models
- `clip/`, `text_encoders/` - Text encoder models
- `unet/`, `diffusion_models/` - Diffusion models
- `upscale_models/` - Upscaling models
- `embeddings/` - Textual inversion embeddings
- `hypernetworks/` - Hypernetwork files
- `clip_vision/` - CLIP vision models
- `gligen/`, `photomaker/`, etc. - Specialized models

  **Importance:** Critical - this is where you put all your models.
"""


### **output/**
"""
  **Purpose:** Generated output files

- Images, videos, and other outputs go here
- Can be configured with `--output-directory`

  **Importance:** Low - just a data directory.
"""


### **script_examples/**
"""
  **Purpose:** Example scripts for using ComfyUI programmatically

- `basic_api_example.py` - Simple API usage
- `websockets_api_example.py` - WebSocket communication
- `websockets_api_example_ws_images.py` - Receiving images via WebSocket

  **Importance:** Useful for learning how to use ComfyUI as a library or via API.
"""


### **tests/**
"""
  **Purpose:** Integration/functional tests

- More comprehensive than unit tests

  **Importance:** For development.
"""


### **tests-unit/**
"""
  **Purpose:** Unit tests using pytest

- Contains test files for various components
- Run with: `pytest tests-unit/`

  **Importance:** Critical for development, not for usage.
"""


### **utils/**
"""
  **Purpose:** General utility functions

- `extra_config.py` - Extra configuration loading
- `install_util.py` - Installation utilities
- `json_util.py` - JSON helpers

  **Importance:** Low - helper utilities.
"""


### **folder_paths.py** ⭐⭐⭐ CRITICAL
"""
  **Purpose:** Path management system

- Manages all model folder paths
- `folder_names_and_paths` - Maps model types to directories
- `get_filename_list()` - Lists files in model folders
- `get_full_path()` - Resolves model file paths
- File caching for performance
- Configurable via command-line args

  **Importance:** Critical - central path management for all models.
"""


### **cuda_malloc.py** ⭐
"""
  **Purpose:** CUDA memory allocator configuration

- Detects GPU models
- Sets PyTorch CUDA malloc backend
- Blacklists GPUs with issues
- Must run BEFORE importing PyTorch

  **Importance:** Important for GPU memory management, especially on problematic GPUs.
"""


### **hook_breaker_ac10a0.py** ⭐
"""
  **Purpose:** Security - prevents custom nodes from hooking core functions

- Saves original function pointers
- Restores them after custom nodes load
- Prevents malicious monkey-patching
- Currently protects: `comfy.model_management.cast_to`

  **Importance:** Security feature to protect core functionality.
"""


### **latent_preview.py** ⭐⭐
"""
  **Purpose:** Generate preview images during sampling

- TAESD-based previews (fast)
- Latent2RGB previews (faster but lower quality)
- Converts latents to displayable images
- Used for real-time progress visualization

  **Importance:** Important for user experience - shows sampling progress.
"""


### **new_updater.py**
"""
  **Purpose:** Updates the Windows standalone package updater scripts

- Only relevant for Windows standalone builds

  **Importance:** Low - only for Windows packaged version.
"""


### **node_helpers.py** ⭐⭐
"""
  **Purpose:** Helper utilities for nodes

- `conditioning_set_values()` - Modify conditioning data
- `pillow()` - Robust PIL image loading
- `hasher()` - Get hashing function
- `string_to_torch_dtype()` - Convert dtype strings
- `image_alpha_fix()` - Fix alpha channels

  **Importance:** Useful utilities used throughout nodes.
"""


### **protocol.py**
"""
  **Purpose:** Binary protocol constants for WebSocket

- `BinaryEventTypes` - Enum for binary message types
  - PREVIEW_IMAGE = 1
  - UNENCODED_PREVIEW_IMAGE = 2
  - TEXT = 3
  - PREVIEW_IMAGE_WITH_METADATA = 4

  **Importance:** Low - just constants for the WebSocket protocol.
"""


### **comfyui_version.py**
"""
  **Purpose:** Version string

- Auto-generated from `pyproject.toml`
- Current version: "0.3.73"

  **Importance:** Low - just version metadata.
"""

## ComfyUI_frontend Overview

<!-- Self note

> https://registry.comfy.org/ -> This is gold for inspiration and putting in virgil (Hiring goldmine as well) -->

## Appendix

### Everything about async & multithreading in python

#### Threading

<https://realpython.com/intro-to-python-threading/>

Till python pie (3.14) we had something called the [GIL] so people created a lot of work arounds to work with multiple threads. (Maybe in a few years this part of the blog will be irrelevant haha).

But what is a thread? Well let's start by first talking about your CPU, if your CPU has 8 cores that means you have 8 threads. These are the brains and most of the operations you run on python are run by CPU (GPU computation is different!). Now due to the dreaded [GIL] ([here](https://www.artima.com/weblogs/viewpost.jsp?thread=214235) Guido van van Rossum (A dope name for a dope creator) talks about GIL)
we could only use 1 thread (brain) at a time, which for most application just works fine.

But why not use all of the brains if I have them, that is what multi-threading let's us do.

[ADD MEME I PAID FOR THE WHOLE METER I AM GOING TO USE THE WHOLE METER]

Now this is what threading means in a traditional sense, but python dont work this way boy.

"""
A thread is a separate flow of execution. This means that your program will have two things happening at once. But for most Python 3 implementations the different threads do not actually execute at the same time: they merely appear to.

It’s tempting to think of threading as having two (or more) different processors running on your program, each one doing an independent task at the same time. That’s almost right. The threads may be running on different processors, but they will only be running one at a time.

Getting multiple tasks running simultaneously requires a non-standard implementation of Python, writing some of your code in a different language, or using multiprocessing which comes with some extra overhead.

Because of the way CPython implementation of Python works, threading may not speed up all tasks. This is due to interactions with the GIL that essentially limit one Python thread to run at a time.

Tasks that spend much of their time waiting for external events are generally good candidates for threading. Problems that require heavy CPU computation and spend little time waiting for external events might not run faster at all.

This is true for code written in Python and running on the standard CPython implementation. If your threads are written in C they have the ability to release the GIL and run concurrently. If you are running on a different Python implementation, check with the documentation too see how it handles threads.

If you are running a standard Python implementation, writing in only Python, and have a CPU-bound problem, you should check out the multiprocessing module instead.

Architecting your program to use threading can also provide gains in design clarity. Most of the examples you’ll learn about in this tutorial are not necessarily going to run faster because they use threads. Using threading in them helps to make the design cleaner and easier to reason about.

So, let’s stop talking about threading and start using it!
"""

<https://www.troyfawkes.com/learn-python-multithreading-queues-basics/>

"""
Use asyncio for many I/O-bound tasks that wait on sockets or files. Prefer threading when you need blocking libraries but light CPU use. Pick multiprocessing for CPU-bound work to bypass the GIL and run tasks in parallel.
"""

Concurency vs parallalism

What does
.gather
.join
.put
.get

these do?

Blog series here was helpful -> <https://bbc.github.io/cloudfit-public-docs/asyncio/asyncio-part-2>

<https://discuss.python.org/t/wrapping-async-functions-for-use-in-sync-code/8606>
<https://realpython.com/async-io-python/>
<https://realpython.com/python-concurrency/>
<https://realpython.com/python-heapq-module/>
