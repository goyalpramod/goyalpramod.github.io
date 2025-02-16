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
def run_llm(content:str = None, messages:Optional[List[str]] = [], tool_schemas:Optional[List[str]] = [], system_message : str = "You are a helpful assistant."):
  completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user","content": content,}]
        + messages,
    tools = tool_schemas,
    ),
    temperature = 0,


  response = completion.choices[0].message
  messages.append(response)

  return messages
```

I chose OpenAI as I had some credits lying around, you can do similar thing with any other api provider.

Here there are 3 arguments\

> content -> The user query/input.\
> system_message -> What do you want the LLM to do.\
> messages -> Past messages/conversation.
> tools ->
> temperature

Let's give it the same prompt that we made earlier and see how it works.

```python
def none_tool():
    """
    does nothing
    """
    pass

run_llm(
    content = """["apple", "pie", 42, 2, 13]""",
    system_message = """
    You are an expert classifier, which classifies strings and integers.\
    Given a list of numbers and words, only return the numbers as a list.\
    You will be given the inputs inside <input> tags.\

    Input:  <input>["hello", 42, "pizza", 2, 5]</input>
    Output: [42,2,5]
    """,
    tools = [none_tool]
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

We can make the above function a bit more dummy proof for dumber models by modifying it as such 

```python
from typing import List, Union
import ast

def add_numbers(num_list: Union[List[int], str]) -> int:
    """
    This function takes either a List of integers or a string representation of a list
    and returns the sum of the numbers.

    Args:
        num_list: List[int] or str - Either a list of integers or a string representing a list
            e.g. "[1, 2, 3]" or [1, 2, 3]

    Returns:
        int: The sum of all numbers in the list

    Raises:
        ValueError: If the string cannot be converted to a list of integers
        SyntaxError: If the string is not properly formatted
    """
    if isinstance(num_list, str):
        try:
            num_list = ast.literal_eval(num_list)
            if not isinstance(num_list, list):
                raise ValueError("String must represent a list")
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid input string format: {e}")

    # Verify all elements are integers
    if not all(isinstance(x, int) for x in num_list):
        raise ValueError("All elements must be integers")

    return sum(num_list)
```

While we are at it, let's create an additional multiply_numbers tool too. 

```python
from typing import List, Union
import ast

def multiply_numbers(num_list: Union[List[int], str]) -> int:
    """
    This function takes either a List of integers or a string representation of a list
    and returns the product of all numbers.

    Args:
        num_list: List[int] or str - Either a list of integers or a string representing a list
            e.g. "[1, 2, 3]" or [1, 2, 3]

    Returns:
        int: The product of all numbers in the list

    Raises:
        ValueError: If the string cannot be converted to a list of integers,
                   if the list is empty, or if any element is not an integer
        SyntaxError: If the string is not properly formatted
    """
    # Handle string input
    if isinstance(num_list, str):
        try:
            num_list = ast.literal_eval(num_list)
            if not isinstance(num_list, list):
                raise ValueError("String must represent a list")
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid input string format: {e}")

    # Check if list is empty
    if not num_list:
        raise ValueError("List cannot be empty")

    # Verify all elements are integers
    if not all(isinstance(x, int) for x in num_list):
        raise ValueError("All elements must be integers")

    # Calculate product using reduce and multiplication operator
    from functools import reduce
    from operator import mul
    return reduce(mul, num_list)
```

Time to use this tool with our LLM to see how well it works.

```python

tools = [add_numbers, multiply_numbers]
tool_schemas = [function_to_schema(tool) for tool in tools]

response = run_llm(
    content = """
    [23,51,321]
    """,
    system_message = """
    Use the appropriate tool to calculate the sum of numbers, and only the tool and nothing else.
    """,
    tool_schemas = tool_schemas
)

print(response)

# [ChatCompletionMessage(content=None, refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_enJSWBrayTgSlFCKgrgw6BGz', function=Function(arguments='{"num_list":"[23,51,321]"}', name='add_numbers'), type='function')])]
```

Now we have a single function that can take in instructions as well as the tools that we want, we can increase the complexity of the tools and prompts without worrying about creating an increasingly complex pipeline.

Now that we can get the tool names and arguments, its time to create another utility function that can take these info and actually execute them.

```python
tools_map = {tool.__name__: tool for tool in tools}
messages = []

def execute_tool_call(tool_call, tools_map):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"Assistant: {name}({args})")

    # call corresponding function with provided arguments
    return tools_map[name](**args)

for tool_call in response[0].tool_calls:
            result = execute_tool_call(tool_call, tools_map)

            # add result back to conversation
            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)

# Assistant: add_numbers({'num_list': '[23,51,321]'})
```

Now we would like our llms to take this response and send an output to the user. Let's do that. 

```python
def run_agent(system_message, tools, messages):

    num_init_messages = len(messages)
    messages = messages.copy()

    while True:

        # turn python functions into tools and save a reverse map
        tool_schemas = [function_to_schema(tool) for tool in tools]
        tools_map = {tool.__name__: tool for tool in tools}

        # === 1. get openai completion ===
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
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
                "content": str(result),
            }
            messages.append(result_message)

    # ==== 3. return new messages =====
    return messages[num_init_messages:]


def execute_tool_call(tool_call, tools_map):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"Assistant: {name}({args})")

    # call corresponding function with provided arguments
    return str(tools_map[name](**args))


tools = [add_numbers]
messages = []
system_message = """
    You are an expert number processor and classifier. Your task is to extract and sum only the numbers from any input, ignoring all non-numeric values.

    Rules:
    1. Only process numeric values (integers)
    2. Ignore all non-numeric values (strings, letters, special characters)
    3. Use the add_numbers function to calculate the sum
    4. Format the input properly before passing to add_numbers

    Examples:
    Input: <input>["hello", 42, "pizza", 2, 5]</input>
    Process: Extract numbers [42, 2, 5]
    Output: 49

    Input: <input>[asj,cg,111,42,2]</input>
    Process: Extract numbers [111, 42, 2]
    Output: 155

    Input: <input>[text, more, 100, words, 50]</input>
    Process: Extract numbers [100, 50]
    Output: 150

    For any input, first extract the numbers, then use add_numbers function to calculate their sum.
    Make sure to format the input as a proper list string with square brackets before passing to add_numbers.
    """


while True:
    user = input("User: ")
    messages.append({"role": "user", "content": user})

    new_messages = run_agent(system_message, tools, messages)
    messages.extend(new_messages)
```

### Agent(LLM call + Tools + Pydantic Model)

Let's first build a model (This is the pydantic model, I will be refering to these as models. And Large Language Models as LLMs)

```python 
class Agent(BaseModel):
    name: str = "Agent"
    llm: str = "gpt-4o-mini"
    system_message: str = "You are a helpful Agent"
    tools: list = []
```

Now we can modify the code we wrote earlier to use this model 

```python 
def run_agent(agent, messages):

    num_init_messages = len(messages)
    messages = messages.copy()

    while True:

        # turn python functions into tools and save a reverse map
        tool_schemas = [function_to_schema(tool) for tool in agent.tools]
        tools_map = {tool.__name__: tool for tool in agent.tools}

        # === 1. get openai completion ===
        response = client.chat.completions.create(
            model=agent.llm,
            messages=[{"role": "system", "content": agent.system_message}] + messages,
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
                "content": str(result),
            }
            messages.append(result_message)

    # ==== 3. return new messages =====
    return messages[num_init_messages:]


def execute_tool_call(tool_call, tools_map):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"Assistant: {name}({args})")

    # call corresponding function with provided arguments
    return str(tools_map[name](**args))
```

And just as easily we can run multiple agents

```python
calculator_add = Agent(
    name="Addition Calculator",
    system_message="""You are an expert number processor. Extract and sum only the numbers from any input, ignoring non-numeric values.
    Example:
    Input: [text, 100, words, 50]
    Process: Extract numbers [100, 50]
    Output: 150""",
    tools=[add_numbers],
)

calculator_multiply = Agent(
    name="Multiplication Calculator",
    system_message="""You are an expert number processor. Extract and multiply only the numbers from any input, ignoring non-numeric values.
    Example:
    Input: [text, 4, words, 5]
    Process: Extract numbers [4, 5]
    Output: 20""",
    tools=[multiply_numbers],
)

messages = []
user_query = "[hello, 10, world, 5, test, 2]"
print("User:", user_query)
messages.append({"role": "user", "content": user_query})

response = run_agent(calculator_add, messages)  # Addition calculator
messages.extend(response)

user_query = "Now multiply these numbers"  # implicitly refers to the numbers from previous input
print("User:", user_query)
messages.append({"role": "user", "content": user_query})
response = run_agent(calculator_multiply, messages)  # Multiplication calculator

# User: [hello, 10, world, 5, test, 2]
# Assistant: add_numbers({'num_list': '[10, 5, 2]'})
# Assistant: The sum of the numbers extracted from the input is 17.
# User: Now multiply these numbers
# Assistant: multiply_numbers({'num_list': '[10, 5, 2]'})
# Assistant: The product of the numbers extracted from the input is 100.
```

"""
Great! But we did the handoff manually here – we want the agents themselves to decide when to perform a handoff. A simple, but surprisingly effective way to do this is by giving them a transfer_to_XXX function, where XXX is some agent. The model is smart enough to know to call this function when it makes sense to make a handoff!
"""

"""
### Handoff Functions
Now that agent can express the intent to make a handoff, we must make it actually happen. There's many ways to do this, but there's one particularly clean way.

For the agent functions we've defined so far, like execute_refund or place_order they return a string, which will be provided to the model. What if instead, we return an Agent object to indicate which agent we want to transfer to? Like so:
"""
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
def run_agent(agent, messages):

    current_agent = agent
    num_init_messages = len(messages)
    messages = messages.copy()

    while True:

        # turn python functions into tools and save a reverse map
        tool_schemas = [function_to_schema(tool) for tool in current_agent.tools]
        tools = {tool.__name__: tool for tool in current_agent.tools}

        # === 1. get openai completion ===
        response = client.chat.completions.create(
            model=agent.llm,
            messages=[{"role": "system", "content": current_agent.system_message}]
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
                "content": str(result),
            }
            messages.append(result_message)

    # ==== 3. return last agent used and new messages =====
    return Response(agent=current_agent, messages=messages[num_init_messages:])


def execute_tool_call(tool_call, tools, agent_name):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"{agent_name}:", f"{name}({args})")

    return str(tools[name](**args))  # call corresponding function with provided arguments
```


## Popular Agentic systems

The above is all the knowledge you require to build more complex agentic systems, let's explore a few of them. Solely using what we have learned so far.

Langgraph has a nice list of agentic systems in my opinion, you can check them out [here](https://langchain-ai.github.io/langgraph/tutorials/#multi-agent-systems).

### ReAct

based on the original [ReAct paper](https://arxiv.org/abs/2210.03629)

```python 
def add_numbers(a: float, b: float) -> float:
    """
    Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The sum of a and b
    """
    return a + b

def subtract_numbers(a: float, b: float) -> float:
    """
    Subtract b from a.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The result of a - b
    """
    return a - b

def multiply_numbers(a: float, b: float) -> float:
    """
    Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The product of a and b
    """
    return a * b

def divide_numbers(a: float, b: float) -> float:
    """
    Divide a by b.
    
    Args:
        a: First number (dividend)
        b: Second number (divisor)
        
    Returns:
        The result of a / b
        
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```

```python
react_system_message = """You are a mathematical reasoning agent that follows the ReAct pattern: Thought, Action, Observation.

For each step of your reasoning:
1. Thought: First explain your thinking
2. Action: Then choose and use a tool
3. Observation: Observe the result

Format your responses as:
Thought: <your reasoning>
Action: <tool_name>(<parameters>)
Observation: <result>
Thought: <your next step>
...

Available tools:
- add_numbers(a, b): Add two numbers
- subtract_numbers(a, b): Subtract b from a
- multiply_numbers(a, b): Multiply two numbers
- divide_numbers(a, b): Divide a by b

Always break down complex calculations into steps using this format."""

react_agent = Agent(
    name="ReActMath",
    llm="gpt-4o-mini",
    system_message=react_system_message,
    tools=[add_numbers, subtract_numbers, multiply_numbers, divide_numbers]
)
```

```python
# Example usage
messages = [{
    "role": "user",
    "content": "Calculate (23 + 7) * 3 - 15"
}]

response = run_agent(react_agent, messages)

# ReActMath: Thought: First, I need to calculate the sum of 23 and 7. Then I will multiply the result by 3, and finally, I will subtract 15 from that product. I'll break this down into steps for clarity. 

# Action: I will first add 23 and 7. 
# functions.add_numbers({ a: 23, b: 7 })

# Observation: Let's perform the addition.
# ReActMath: add_numbers({'a': 23, 'b': 7})
# ReActMath: Thought: The sum of 23 and 7 is 30. Now, I will multiply this result by 3. 

# Action: I will multiply 30 by 3. 
# functions.multiply_numbers({ a: 30, b: 3 }) 

# Observation: Let's perform the multiplication.
# ReActMath: multiply_numbers({'a': 30, 'b': 3})
# ReActMath: Thought: The product of 30 and 3 is 90. Now, I need to subtract 15 from this result. 

# Action: I will subtract 15 from 90. 
# functions.subtract_numbers({ a: 90, b: 15 }) 

# Observation: Let's perform the subtraction.
# ReActMath: subtract_numbers({'a': 90, 'b': 15})
# ReActMath: Thought: The result of subtracting 15 from 90 is 75. Therefore, the final result of the calculation (23 + 7) * 3 - 15 is 75.

# Final Result: 75
```

### Agentic RAG

```python
import requests
from bs4 import BeautifulSoup
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import json

class DocumentStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.documents = []
        self.index = None
        self.dimension = 384

    def add_documents(self, url: str) -> str:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.find(id="mw-content-text")
        if content:
            paragraphs = content.find_all('p')
            self.documents = [p.text for p in paragraphs if len(p.text.split()) > 20]

        embeddings = self.embedder.encode(self.documents)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        return f"Processed {len(self.documents)} documents"

    def search(self, query: str, k: int = 3) -> List[str]:
        query_vector = self.embedder.encode([query])
        D, I = self.index.search(np.array(query_vector).astype('float32'), k)
        return [self.documents[i] for i in I[0]]

# Initialize document store
doc_store = DocumentStore()

def retrieve_documents(url: str):
    """Tool function for document retrieval"""
    return doc_store.add_documents(url)

def search_context(query: str):
    """Tool function for searching documents"""
    return doc_store.search(query)

def check_relevance(context: List[str]) -> bool:
    """
    Tool function to check if retrieved context is relevant using an LLM.
    
    Args:
        context: List of context strings to evaluate
        
    Returns:
        bool: True if context is relevant, False otherwise
    """
    if not context:
        return False
        
    system_message = """You are a relevance checking assistant. 
    Evaluate if the given context is relevant and substantial enough to answer questions.
    Return only 'true' or 'false'."""
    
    prompt = f"""Evaluate if this context is relevant and substantial (contains meaningful information):

    Context: {context}

    Return only 'true' or 'false'."""

    messages = run_llm(
        content=prompt,
        system_message=system_message
    )
    
    # Get the last message which contains the LLM's response
    result = messages[-1].content.lower().strip()
    return result == 'true'

def rewrite_query(query: str, context: List[str]) -> str:
    """
    Tool function to rewrite a query based on context using an LLM.
    
    Args:
        query: Original query to rewrite
        context: List of context strings to use for rewriting
        
    Returns:
        str: Rewritten query incorporating context
    """
    system_message = """You are a query rewriting assistant.
    Your task is to rewrite the original query to incorporate relevant context.
    Maintain the original intent while making it more specific based on the context."""
    
    prompt = f"""Original Query: {query}

Available Context: {context}

Rewrite the query to be more specific using the context. 
Maintain the original intent but make it more precise."""

    messages = run_llm(
        content=prompt,
        system_message=system_message
    )
    
    # Get the last message which contains the rewritten query
    return messages[-1].content.strip()

rag_agent = Agent(
    name="RAG Agent",
    system_message="""You are an intelligent RAG agent that follows a specific workflow:
    1. First, determine if you need to retrieve documents
    2. If yes, use the retrieval tool
    3. Check the relevance of retrieved documents
    4. Either rewrite the query or generate an answer
    5. Always cite your sources from the context

    Be explicit about each step you're taking.""",
    tools=[retrieve_documents, search_context, check_relevance, rewrite_query],
    llm="gpt-4o-mini"
)

def run_rag_agent(agent, messages):
    num_init_messages = len(messages)
    messages = messages.copy()

    while True:
        # Get tool schemas and tool map
        tool_schemas = [function_to_schema(tool) for tool in agent.tools]
        tools = {tool.__name__: tool for tool in agent.tools}

        # Make API call with system message and history
        response = client.chat.completions.create(
            model=agent.llm,
            messages=[{"role": "system", "content": agent.system_message}] + messages,
            tools=tool_schemas,
        )
        
        # Get and append the assistant's message
        message = response.choices[0].message
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": message.tool_calls if hasattr(message, "tool_calls") else None
        })

        if message.content:
            print(f"{agent.name}:", message.content)

        if not hasattr(message, "tool_calls") or not message.tool_calls:
            break

        # Handle tool calls
        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools, agent.name)
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })

    return messages[num_init_messages:]

# Main execution
messages = []

# First, index a document
url = "https://en.wikipedia.org/wiki/Alan_Turing"
messages.append({
    "role": "user",
    "content": f"Please retrieve and index this document: {url}"
})

while True:
    try:
        response = run_rag_agent(rag_agent, messages)
        messages.extend(response)

        user_input = input("\nUser (type 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

        messages.append({"role": "user", "content": user_input})

    except Exception as e:
        print(f"Error occurred: {e}")
        break
```

### Supervisor + Workers

```python
# Math Genius Tools
def solve_equation(equation: str) -> str:
    """
    Solves mathematical equations.
    Args:
        equation: str - A mathematical equation to solve
            e.g. "2x + 5 = 13" or "integrate(x^2)"
    Returns:
        str: The solution to the equation
    """
    # In a real implementation, this would use a math solving library
    return f"Solution for equation: {equation}"

def perform_statistics(data: str) -> str:
    """
    Performs statistical analysis on given data.
    Args:
        data: str - Data in format "[1,2,3,4]" or "1,2,3,4"
    Returns:
        str: Statistical analysis including mean, median, mode, std dev
    """
    # In a real implementation, this would use numpy/pandas
    return f"Statistical analysis of: {data}"

# Code Writer Tools
def generate_code(specification: str, language: str) -> str:
    """
    Generates code based on specifications.
    Args:
        specification: str - Description of what the code should do
        language: str - Programming language to use
    Returns:
        str: Generated code
    """
    return f"Generated {language} code for: {specification}"

def review_code(code: str) -> str:
    """
    Reviews code for best practices and potential issues.
    Args:
        code: str - Code to review
    Returns:
        str: Code review comments
    """
    return f"Code review for: {code}"

def transfer_to_math_genius():
    """Transfer to math genius agent for complex calculations."""
    return math_genius_agent

def transfer_to_code_writer():
    """Transfer to code writer agent for programming tasks."""
    return code_writer_agent

def transfer_to_supervisor():
    """Transfer back to supervisor agent."""
    return supervisor_agent

# Agent Definitions
math_genius_agent = Agent(
    name="Math Genius",
    llm="gpt-4o",
    system_message=(
        "You are a mathematical genius capable of solving complex equations "
        "and performing advanced statistical analysis. Always show your work "
        "step by step. If a task is outside your mathematical expertise, "
        "transfer back to the supervisor."
    ),
    tools=[solve_equation, perform_statistics, transfer_to_supervisor]
)

code_writer_agent = Agent(
    name="Code Writer",
    llm="gpt-4o",
    system_message=(
        "You are an expert programmer capable of writing efficient, clean code "
        "and providing detailed code reviews. Always explain your code and "
        "include comments. If a task is outside your programming expertise, "
        "transfer back to the supervisor."
    ),
    tools=[generate_code, review_code, transfer_to_supervisor]
)

supervisor_agent = Agent(
    name="Supervisor",
    llm="gpt-4o",
    system_message=(
        "You are a supervisor agent responsible for delegating tasks to specialized agents. "
        "For mathematical problems, delegate to the Math Genius. "
        "For programming tasks, delegate to the Code Writer. "
        "Ensure all responses are complete and coordinate between agents when needed."
    ),
    tools=[transfer_to_math_genius, transfer_to_code_writer]
)

# Example usage
def process_request(user_input: str):
    """
    Process user request through the multi-agent system.
    Args:
        user_input: str - User's question or request
    Returns:
        List of messages containing the conversation
    """
    messages = [{"role": "user", "content": user_input}]
    response = run_full_turn(supervisor_agent, messages)
    return response.messages

# Interactive testing loop
if __name__ == "__main__":
    print("Welcome to the Multi-Agent System!")
    print("You can interact with a supervisor who will delegate to either:")
    print("1. Math Genius - for mathematical problems")
    print("2. Code Writer - for programming tasks")
    print("Type 'exit', 'quit', or 'bye' to end the conversation\n")

    agent = supervisor_agent
    messages = []

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
                
            messages.append({"role": "user", "content": user_input})
            response = run_full_turn(agent, messages)
            agent = response.agent
            messages.extend(response.messages)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Resetting conversation...")
            messages = []
            agent = supervisor_agent
```

## Important notes

- Evaluation
- Tracing & logging
- Cost
- Self hosting & Inference
- Streaming and UX notes
- Security 

## References


- [OpenAI blog](https://cookbook.openai.com/examples/orchestrating_agents)
- [Lil'log's blog on prompt engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
- [Anthropic's blog on building effective agents](https://www.anthropic.com/research/building-effective-agents)
- [Core code for Swarm](https://github.com/openai/swarm/blob/main/swarm/core.py)
- [HF docs on building agents](https://huggingface.co/docs/smolagents/tutorials/building_good_agents)
- [HF blog on smolagents](https://huggingface.co/blog/smolagents)
- Meme at top taken from [dilbert](https://www.reddit.com/r/ProgrammerHumor/comments/1dckq74/soundsfamiliar/)



Here smolagents argue that using code rather than JSON is better, so let's try that out as well, so we come full circle.  -->
