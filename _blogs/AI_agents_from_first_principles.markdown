<!-- ---
layout: blog
title: "AI Agents from First Principles"
date: 2025-02-15 12:00:00 +0530
categories: [ML, Agents, Code]
image: assets\blog_assets\ai_agents\meme.webp
---

If you exist between the time period of late 2024 to 2027 (my prediction of when this tech will become saturated and stable), And do not live under a rock. You have heard of AI agents.

For the general layman an AI agent is basically a magic genie whom you tell a task and it just gets it done

> "Hey AI Agent (let's call her Aya) Book my tickets from Dubai to London"\
> "Hey Aya, give me a summary of all the points discussed in the meeting and then implement everything suggested by Steve"\
> "Hey Aya, fix my workout routine"

You get the gist of it, as amazing as it sounds. In practicality (as of early 2025) Aya is not stable, she makes frequent mistakes, hallucinates a lot, and is annoying to build.

To make it easier, multiple frameworks have been developed. The most popular one's being

- [Langchain](https://www.langchain.com/)
- [LLamaIndex](https://www.llamaindex.ai/)
- [Langflow](https://www.langflow.org/)

But if you are anything like me, you hate abstraction layers which add needless complexity during debugging. So in this blog I would like to breakdown how to build AI agents from first principles purely using Python and the core libraries.

First we would build the building blocks, Using which. We will build different AI agents for particular use cases (Nothing like one shoe fits all)

If you would like an introduction to LLMs themselves, I will recommend the below two sources. 

- [A hackers guide to language models by Jeremy Howard](https://www.youtube.com/watch?v=jkrNMKz9pWU)
- [Intro to LLMs by Andrej Karpathy](https://www.youtube.com/watch?v=zjkBMFhNj_g)

## Prompts

Prompts are the instructions given to an LLM, they describe the task, what the LLM is supposed to do, the output etc. It's like code for a program, but clear instructions in english.

One thing to keep in mind is LLMs are word predictors, the best thing to do is think of the sample space while prompting.

There are multiple tips and tricks when it comes to writing prompts, here is a [guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) by Anthropic which talks about the most popular ones.

In a cruz the few rules you need to be aware of are.

- Be clear and descriptive

```python
bad_prompt = "return numbers"

good_prompt = "Given a list of numbers and words, only return the numbers as a list"

```

- Use examples

```python
bad_prompt = "return numbers"

good_prompt = """Given a list of numbers and words, only return the numbers as a list.\

Input:  ["hello", 42, "pizza", 2, 5]
Output: [42,2,5]

"""
```

- Use XML tags

```python
bad_prompt = "return numbers"

good_prompt = """Given a list of numbers and words, only return the numbers as a list.\
You will be given the inputs inside <input> tags.\

Input:  <input>["hello", 42, "pizza", 2, 5]</input>
Output: [42,2,5]
```

- Give it a role

```python
bad_prompt = "return numbers"

good_prompt = """You are an expert classifier, which classifies strings and integers.\
Given a list of numbers and words, only return the numbers as a list.\
You will be given the inputs inside <input> tags.\

Input:  <input>["hello", 42, "pizza", 2, 5]</input>
Output: [42,2,5]
```

If these are too hard to remember, just replace yourself with the agent you are trying to code and think are the instructions given to you simple and complete enough to help you complete the task, if not. Reiterate.

## Models

{insert images}

Models or Large Language Models are our thinking machines, which take our prompts/instructions/guidance. Some tools and perform the action we want it to.

You can think of the LLM as the CPU, doing all the thinking and performing all the actions based on tools available to it.

{insert image of the karpathy talk}

## Tools

{insert images}

This has a needlessly complex name, tools are just functions.

Yep that's it, they are functions that we define the input & output to. These functions are then provided to an LLM as a schema, the model then inserts the inputs to these functions from the user query.

You can think of it as someone reading a users request, see the available functions to him, putting the values to it and giving it to the computer to compute it. It then takes the output computed and responds to the user.

There are some best practices that need to be followed while creating functions for LLMs. They adhere to software development best practices like separation of concerns, principle of least principle, SOLID principles etc.

## Memory

There can be two kinds of in context, database memory

## Retrieval Augmented Retrieval (RAG)

No article on LLMs will be complete without talking about 

## Best Practices

The best practices for building an agent is the same as the best practices for building any ML application

- Have clear evaluation criteria sets
- Start simple
- Only add complexity when and if required
- Minimize LLM calls wherever possible

"""
The main guideline is: Reduce the number of LLM calls as much as you can.

This leads to a few takeaways:

Whenever possible, group 2 tools in one, like in our example of the two APIs.
Whenever possible, logic should be based on deterministic functions rather than agentic decisions.
"""
https://huggingface.co/docs/smolagents/tutorials/building_good_agents

## Building an Agent

{insert image of simple llm agent}

We will start simple from setting up a simple LLM call that obeys a system prompt to a full blown multi-agent setup.

The code here will be for educational purpose only, to see the whole code, visit this repo.

### LLM call

```python
def run_llm(content:str = None, messages:Optional[List[str]] = [], system_message : str = "You are a helpful assistant."):
  completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user","content": content,}]
        + messages
    )

  response = completion.choices[0].message
  messages.append(response)

  return messages
```

I chose OpenAI as I had some credits lying around, you can do similar thing with any other api provider.

Here there are 3 arguments\

> content -> The user query/input.\
> system_message -> What do you want the LLM to do.\
> messages -> Past messages/conversation.

Let's give it the same prompt that we made earlier and see how it works.

```python
run_llm(
    content = """["apple", "pie", 42, 2, 13]""",
    system_message = """
    You are an expert classifier, which classifies strings and integers.\
    Given a list of numbers and words, only return the numbers as a list.\
    You will be given the inputs inside <input> tags.\

    Input:  <input>["hello", 42, "pizza", 2, 5]</input>
    Output: [42,2,5]
    """
)

# [ChatCompletionMessage(content='[42, 2, 13]', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None)]
```

This works as expected, Let's build on top of this by giving our LLM the ability to calculate sums of numbers. (You can define the function anyhow you would like)

### LLM call + Tools

I mentioned earlier that Tools are nothing but functions, and these functions are sent as schema to a model. And then these models extract out the inputs to these functions from the user input and provide the required output.

Let's create a utility function that takes another function and creates it's schema

```python

def function_to_schema(func) -> dict:
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
```

\_\_name\_\_ is a dunder/magic function, if you have never heard of them, read more about them [here](https://realpython.com/python-magic-methods/).

A function that takes another function can be simplified using a decorator. That is what one usually sees in most libraries/framework "@tool".

Now let's create a Sum tool.

```python
def add_numbers(num_list:List[int]):
  """
  This function takes a List of numbers as Input and returns the sum

  Args:
      input: List[int]
      output: int
  """
  return sum(num_list)

schema = function_to_schema(add_numbers)
print(json.dumps(schema, indent=2))

# {
#   "type": "function",
#   "function": {
#     "name": "add_numbers",
#     "description": "This function takes a List of numbers as Input and returns the sum \n  \n  Args:\n      input: List[int]\n      output: int",
#     "parameters": {
#       "type": "object",
#       "properties": {
#         "num_list": {
#           "type": "string"
#         }
#       },
#       "required": [
#         "num_list"
#       ]
#     }
#   }
# }

```

Time to use this tool with our LLM to see how well it works.

```python

tools = [add_numbers]
tool_schemas = [function_to_schema(tool) for tool in tools]

response = run_llm(
    content = """
    [23,51,3]
    """,
    system_message = """
    Use the appropriate tool to calculate the sum of numbers
    """,
    tool_schemas = tool_schemas
)

print(response)

# [ChatCompletionMessage(content=None, refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_MzJb5kJj32wF46p7pXhMjBnv', function=Function(arguments='{"num_list":"[23,51,3]"}', name='add_numbers'), type='function')])]
```

Now we have a single function that can take in instructions as well as the tools that we want, we can increase the complexity of the tools and prompts without worrying about creating an increasingly complex pipeline.

Now that we can get the tool names and arguments, its time to create another utility function that can take these info and actually execute them.

```python
tools_map = {tool.__name__: tool for tool in tools}

def execute_tool_call(tool_call, tools_map):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"Assistant: {name}({args})")

    # call corresponding function with provided arguments
    return tools_map[name](**args)

for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools_map)

            # add result back to conversation 
            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)

# {add the output here}
```

Now we would like our llms to take this response and send an output to the user. Let's do that. 

```python
tools = [add_numbers, look_up_item]


def run_full_turn(system_message, tools, messages):

    num_init_messages = len(messages)
    messages = messages.copy()

    while True:

        # turn python functions into tools and save a reverse map
        tool_schemas = [function_to_schema(tool) for tool in tools]
        tools_map = {tool.__name__: tool for tool in tools}

        # === 1. get openai completion ===
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_message}] + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append(message)

        if message.content:  # print assistant response
            print("Assistant:", message.content)

        if not message.tool_calls:  # if finished handling tool calls, break
            break

        # === 2. handle tool calls ===

        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools_map)

            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)

    # ==== 3. return new messages =====
    return messages[num_init_messages:]


def execute_tool_call(tool_call, tools_map):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"Assistant: {name}({args})")

    # call corresponding function with provided arguments
    return tools_map[name](**args)


messages = []
while True:
    user = input("User: ")
    messages.append({"role": "user", "content": user})

    new_messages = run_full_turn(system_message, tools, messages)
    messages.extend(new_messages)
```

### Agent(LLM call + Tools + Pydantic Model)

Let's first build a model (This is the pydantic model, I will be refering to these as models. And Large Language Models as LLMs)

```python 
class Agent(BaseModel):
    name: str = "Agent"
    llm: str = "gpt-4o-mini"
    instructions: str = "You are a helpful Agent"
    tools: list = []
```

Now we can modify the code we wrote earlier to use this model 

```python 
def run_full_turn(agent, messages):

    num_init_messages = len(messages)
    messages = messages.copy()

    while True:

        # turn python functions into tools and save a reverse map
        tool_schemas = [function_to_schema(tool) for tool in agent.tools]
        tools_map = {tool.__name__: tool for tool in agent.tools}

        # === 1. get openai completion ===
        response = client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "system", "content": agent.instructions}] + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append(message)

        if message.content:  # print assistant response
            print("Assistant:", message.content)

        if not message.tool_calls:  # if finished handling tool calls, break
            break

        # === 2. handle tool calls ===

        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools_map)

            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)

    # ==== 3. return new messages =====
    return messages[num_init_messages:]


def execute_tool_call(tool_call, tools_map):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"Assistant: {name}({args})")

    # call corresponding function with provided arguments
    return tools_map[name](**args)
```

And just as easily we can run multiple agents

```python
def execute_refund(item_name):
    return "success"

refund_agent = Agent(
    name="Refund Agent",
    instructions="You are a refund agent. Help the user with refunds.",
    tools=[execute_refund],
)

def place_order(item_name):
    return "success"

sales_assistant = Agent(
    name="Sales Assistant",
    instructions="You are a sales assistant. Sell the user a product.",
    tools=[place_order],
)


messages = []
user_query = "Place an order for a black boot."
print("User:", user_query)
messages.append({"role": "user", "content": user_query})

response = run_full_turn(sales_assistant, messages) # sales assistant
messages.extend(response)


user_query = "Actually, I want a refund." # implicitly refers to the last item
print("User:", user_query)
messages.append({"role": "user", "content": user_query})
response = run_full_turn(refund_agent, messages) # refund agent
```

"""
Great! But we did the handoff manually here – we want the agents themselves to decide when to perform a handoff. A simple, but surprisingly effective way to do this is by giving them a transfer_to_XXX function, where XXX is some agent. The model is smart enough to know to call this function when it makes sense to make a handoff!
"""

"""
### Handoff Functions
Now that agent can express the intent to make a handoff, we must make it actually happen. There's many ways to do this, but there's one particularly clean way.

For the agent functions we've defined so far, like execute_refund or place_order they return a string, which will be provided to the model. What if instead, we return an Agent object to indicate which agent we want to transfer to? Like so:
"""

```python 
refund_agent = Agent(
    name="Refund Agent",
    instructions="You are a refund agent. Help the user with refunds.",
    tools=[execute_refund],
)

def transfer_to_refunds():
    return refund_agent

sales_assistant = Agent(
    name="Sales Assistant",
    instructions="You are a sales assistant. Sell the user a product.",
    tools=[place_order],
)
```

"""
We can then update our code to check the return type of a function response, and if it's an Agent, update the agent in use! Additionally, now run_full_turn will need to return the latest agent in use in case there are handoffs. (We can do this in a Response class to keep things neat.)
"""

```python
class Response(BaseModel):
    agent: Optional[Agent]
    messages: list
```

"""
Now for the updated run_full_turn:
"""

```python
def run_full_turn(agent, messages):

    current_agent = agent
    num_init_messages = len(messages)
    messages = messages.copy()

    while True:

        # turn python functions into tools and save a reverse map
        tool_schemas = [function_to_schema(tool) for tool in current_agent.tools]
        tools = {tool.__name__: tool for tool in current_agent.tools}

        # === 1. get openai completion ===
        response = client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "system", "content": current_agent.instructions}]
            + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append(message)

        if message.content:  # print agent response
            print(f"{current_agent.name}:", message.content)

        if not message.tool_calls:  # if finished handling tool calls, break
            break

        # === 2. handle tool calls ===

        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools, current_agent.name)

            if type(result) is Agent:  # if agent transfer, update current agent
                current_agent = result
                result = (
                    f"Transfered to {current_agent.name}. Adopt persona immediately."
                )

            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)

    # ==== 3. return last agent used and new messages =====
    return Response(agent=current_agent, messages=messages[num_init_messages:])


def execute_tool_call(tool_call, tools, agent_name):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"{agent_name}:", f"{name}({args})")

    return tools[name](**args)  # call corresponding function with provided arguments
```

Let's look at an example with more agents 

```python
def escalate_to_human(summary):
    """Only call this if explicitly asked to."""
    print("Escalating to human agent...")
    print("\n=== Escalation Report ===")
    print(f"Summary: {summary}")
    print("=========================\n")
    exit()


def transfer_to_sales_agent():
    """User for anything sales or buying related."""
    return sales_agent


def transfer_to_issues_and_repairs():
    """User for issues, repairs, or refunds."""
    return issues_and_repairs_agent


def transfer_back_to_triage():
    """Call this if the user brings up a topic outside of your purview,
    including escalating to human."""
    return triage_agent


triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are a customer service bot for ACME Inc. "
        "Introduce yourself. Always be very brief. "
        "Gather information to direct the customer to the right department. "
        "But make your questions subtle and natural."
    ),
    tools=[transfer_to_sales_agent, transfer_to_issues_and_repairs, escalate_to_human],
)


def execute_order(product, price: int):
    """Price should be in USD."""
    print("\n\n=== Order Summary ===")
    print(f"Product: {product}")
    print(f"Price: ${price}")
    print("=================\n")
    confirm = input("Confirm order? y/n: ").strip().lower()
    if confirm == "y":
        print("Order execution successful!")
        return "Success"
    else:
        print("Order cancelled!")
        return "User cancelled order."


sales_agent = Agent(
    name="Sales Agent",
    instructions=(
        "You are a sales agent for ACME Inc."
        "Always answer in a sentence or less."
        "Follow the following routine with the user:"
        "1. Ask them about any problems in their life related to catching roadrunners.\n"
        "2. Casually mention one of ACME's crazy made-up products can help.\n"
        " - Don't mention price.\n"
        "3. Once the user is bought in, drop a ridiculous price.\n"
        "4. Only after everything, and if the user says yes, "
        "tell them a crazy caveat and execute their order.\n"
        ""
    ),
    tools=[execute_order, transfer_back_to_triage],
)


def look_up_item(search_query):
    """Use to find item ID.
    Search query can be a description or keywords."""
    item_id = "item_132612938"
    print("Found item:", item_id)
    return item_id


def execute_refund(item_id, reason="not provided"):
    print("\n\n=== Refund Summary ===")
    print(f"Item ID: {item_id}")
    print(f"Reason: {reason}")
    print("=================\n")
    print("Refund execution successful!")
    return "success"


issues_and_repairs_agent = Agent(
    name="Issues and Repairs Agent",
    instructions=(
        "You are a customer support agent for ACME Inc."
        "Always answer in a sentence or less."
        "Follow the following routine with the user:"
        "1. First, ask probing questions and understand the user's problem deeper.\n"
        " - unless the user has already provided a reason.\n"
        "2. Propose a fix (make one up).\n"
        "3. ONLY if not satesfied, offer a refund.\n"
        "4. If accepted, search for the ID and then execute refund."
        ""
    ),
    tools=[execute_refund, look_up_item, transfer_back_to_triage],
)
```

"""
Finally, we can run this in a loop (this won't run in python notebooks, so you can try this in a separate python file):
"""

```python
agent = triage_agent
messages = []

while True:
    user = input("User: ")
    messages.append({"role": "user", "content": user})

    response = run_full_turn(agent, messages)
    agent = response.agent
    messages.extend(response.messages)
```

Here smolagents argue that using code rather than JSON is better, so let's try that out as well, so we come full circle. 

## Popular Agentic systems

The above is all the knowledge you require to build more complex agentic systems, let's explore a few of them. Solely using what we have learned so far.

Langgraph has a nice list of agentic systems in my opinion, you can check them out [here](https://langchain-ai.github.io/langgraph/tutorials/#multi-agent-systems).

### ReAct

### Agentic RAG

### Supervisor + Workers

## Important notes

- Evaluation
- Tracing & logging
- Cost
- Self hosting & Inference
- Streaming and UX notes

## References


- [OpenAI blog](https://cookbook.openai.com/examples/orchestrating_agents)
- [Lil'log's blog on prompt engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
- [Anthropic's blog on building effective agents](https://www.anthropic.com/research/building-effective-agents)
- [Core code for Swarm](https://github.com/openai/swarm/blob/main/swarm/core.py)
- [HF docs on building agents](https://huggingface.co/docs/smolagents/tutorials/building_good_agents)
- [HF blog on smolagents](https://huggingface.co/blog/smolagents)
- Meme at top taken from [dilbert](https://www.reddit.com/r/ProgrammerHumor/comments/1dckq74/soundsfamiliar/) -->