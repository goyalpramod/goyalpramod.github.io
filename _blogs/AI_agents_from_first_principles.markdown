<!-- ---
layout: blog
title: "AI Agents from first principles"
date: 2025-01-3 12:00:00 +0530
categories: [personal, technology]
image: { add image }
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

But if you are anything like me, you hate abstraction layers which add needless complexity during debugging. So in this blog I would like to breakdown how to build systems like Aya from first principles purely using Python and the core libraries.

- [A hackers guide to language models by Jeremy Howard](https://www.youtube.com/watch?v=jkrNMKz9pWU)
- [Intro to LLMs by Andrej Karpathy](https://www.youtube.com/watch?v=zjkBMFhNj_g)

First we would build the building blocks, Using which. We will build different systems like Aya for particular usecases (Nothing like one shoe fits all)

### Prompts

Prompting are the instructions given to an LLM, they describe the task, what the LLM is supposed to do, the output etc. It's like code for a program, but clear instructions in english.

One thing to keep in mind is LLMs are word predictors, the best thing to do is think of the sample space while prompting.

There are multiple tips and tricks when it comes to writing prompts, here is a [guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) by Anthropic which talks about the most popular ones

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

If these are too hard to remember, just replace yourself with the Agent you are trying to code and think if the instructions given to you are simple and complete enough to help you with the task, if not. Reiterate.

### Models

{insert images}

Models or Large Language Models are our thinking machines, which take our prompts/instructions/guidance. Some tools and perform the action we want it to.

You can think of the LLM as the CPU, doing all the thinking and performing all the actions based on tools available to it.

{insert image of the karpathy talk}

### Tools

{insert images}

This has a needlessly complex name, tools are just functions.

Yep that's it, they are functions that we define the input & output to. These functions are then provided to an LLM as a schema, the model then inserts the inputs to these functions from the user query.

You can think of it as someone reading a users request, see the available functions to him, putting the values to it and giving it to the computer to compute it. It then takes the output computed and responds to the user.

There are some best practices that need to be followed while creating functions for LLMs. They adhere to software development best practices like seperation of concers, principle of least principle, SOLID principles etc.

### Memory

There can be two kinds of in context, database memory

### Best Practices

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

I have chosed OpenAI as I had some credits lying around, you can do similar thing with any other api provider.

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

\_\_name\_\_ is a dunder/magic function, if you have never heard of them, read more about them here.

A function that takes another function can be simplied using a decorator. That is what one usually sees in most libraries/framework "@tool".

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



```

## References -->
