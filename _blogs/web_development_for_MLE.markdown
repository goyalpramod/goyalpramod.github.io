<!-- ---
layout: blog
title: "Web development for MLE"
date: 2025-02-9 12:00:00 +0530
categories: [personal, technology, learning, webd]
image: assets/cover_images/img5.webp
---

My biggest limitations as an MLE had been that I wasn't proficient in Web Development. I have tried to transition into it back in the day a few times. But have always stopped midway either due to lack of time or interest.

Also given the pace of AI innovation, I very rarely had a few months in which I could dedicate time to learning web development.

But funny enough, AI has caught up to webd and it has become easier to develop websites than ever before using tools like [cursor](https://cursor.sh/), [v0](https://v0.dev/), [replit](https://replit.com/) etc.

So I have set a challenge for myself to learn web development in 15 days. And here I will share everything I learn each day.

## The Goal

Build a web application that allows users to upload images and apply aesthetic shader effects, with features like:

- Modern, responsive UI
- Real-time image processing
- Multiple shader templates
- User-friendly controls
- Proper deployment and optimization

## Essential Resources

1. **Modern JavaScript & TypeScript**

   - [javascript.info](https://javascript.info/) - Comprehensive modern JavaScript tutorial
   - [TypeScript Deep Dive](https://basarat.gitbook.io/typescript/) - In-depth TypeScript guide

2. **React & Modern Web Development**

   - [React Dev](https://react.dev/) - Official React documentation
   - [Josh Comeau's Blog](https://www.joshwcomeau.com/) - Modern React patterns and CSS
   - [Patterns.dev](https://www.patterns.dev/) - Modern web architecture patterns

3. **WebGL & Shaders**

   - [WebGL Fundamentals](https://webglfundamentals.org/) - Essential WebGL concepts
   - [Book of Shaders](https://thebookofshaders.com/) - GLSL shader programming

4. **Backend & Deployment**
   - [Node.js Best Practices](https://github.com/goldbergyoni/nodebestpractices)
   - [AWS Getting Started](https://aws.amazon.com/getting-started/)

## Roadmap

### Week 1: Foundations & Frontend

#### Day 1: Modern JavaScript & TypeScript

- **Early Morning (3 hours)**:
  - Complete [javascript.info](https://javascript.info/) sections:
    - Modern JavaScript fundamentals
    - Promises, async/await
    - Modules
- **Night (3 hours)**:
  - [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html):
    - Basic types
    - Interfaces
    - Functions
  - Practice: Build a simple TypeScript utility library

#### Day 2: React Fundamentals

- **Early Morning (3 hours)**:
  - [React Quick Start](https://react.dev/learn):
    - Components
    - Props
    - State
- **Night (3 hours)**:
  - Learn React Hooks
  - Build a simple image upload component
  - Practice with useEffect and useState

#### Day 3: Advanced React & State Management

- **Early Morning (3 hours)**:
  - [Zustand](https://docs.pmnd.rs/zustand/getting-started/introduction) for state management
  - React context API
- **Night (3 hours)**:
  - File handling in React
  - Image processing basics
  - Build image preview functionality

#### Day 4: CSS & Styling

- **Early Morning (3 hours)**:
  - [TailwindCSS](https://tailwindcss.com/docs) fundamentals
  - Responsive design principles
- **Night (3 hours)**:
  - Build responsive UI components
  - Implement dark mode
  - Add loading states and animations

#### Day 5: WebGL Basics

- **Early Morning (3 hours)**:
  - [WebGL Fundamentals](https://webglfundamentals.org/) basics
  - Understanding shaders
- **Night (3 hours)**:
  - Setup WebGL context in React
  - Basic shader implementation

### Week 2: Advanced Concepts & Project Development

#### Day 6-7: Weekend Deep Dive (12 hours each)

- **Day 6**: Shader Development

  - [Book of Shaders](https://thebookofshaders.com/) fundamentals
  - Implement basic image effects
  - Build shader template system

- **Day 7**: Frontend Integration
  - Connect all components
  - Implement shader controls
  - Add error handling
  - Optimize performance

#### Day 8: Backend Setup

- **Early Morning (3 hours)**:
  - Setup Express.js server
  - Configure TypeScript for backend
- **Night (3 hours)**:
  - Implement file upload API
  - Add input validation

#### Day 9: Advanced Backend

- **Early Morning (3 hours)**:
  - Image processing on server
  - Caching implementation
- **Night (3 hours)**:
  - Error handling
  - Rate limiting
  - Security best practices

#### Day 10: Testing & Optimization

- **Early Morning (3 hours)**:
  - Setup testing with Vitest
  - Write component tests
- **Night (3 hours)**:
  - Performance optimization
  - Lazy loading
  - Image optimization

### Week 3: Deployment & Polish

#### Day 11: AWS & Infrastructure

- **Early Morning (3 hours)**:
  - AWS S3 setup for image storage
  - CloudFront CDN configuration
- **Night (3 hours)**:
  - Environment configuration
  - Security best practices

#### Day 12: CI/CD & Monitoring

- **Early Morning (3 hours)**:
  - Setup GitHub Actions
  - Implement automated testing
- **Night (3 hours)**:
  - Error monitoring
  - Performance monitoring
  - Analytics setup

#### Day 13-14: Weekend Polish (12 hours each)

- **Day 13**: Final Features

  - Additional shader effects
  - User preferences
  - Progressive enhancement

- **Day 14**: Production Readiness
  - Performance testing
  - Cross-browser testing
  - SEO optimization
  - Documentation

#### Day 15: Launch Preparation

- Final testing
- Documentation completion
- Deployment checklist
- Performance benchmarking

## Note

It's not like I am starting from absolute scratch, I have created this roadmap specifically tailored to me given my
previous experience with HTML, CSS and JS.

Additionally I have been working as an SDE for quite some time now, So I have experience with docker, CI/CD, pipelines etc

As well as, I have done backend development using fast-api and Django in the past.

## Daily Progress

### Day 1

Learning resource -> javascript.info

**Introduction**

- JS is a safe language, that runs on the browser engine.
- There are specifications and manuals which are specified in [ECMA](https://ecma-international.org/publications-and-standards/standards/ecma-262/). These are the rules and guidelines of using JS. There is also [MDN docs](https://developer.mozilla.org/en-US/docs/Web/JavaScript) which has extensive documentation on js.
- We can access the developer console by pressing F12. This is mainly used while debugging your js code.

**JS Fundamentals**

> Text inside quote blocks indicate summary taken from section of the book directly, this is just for my own quick reference. I will recommend reading the book itself to get a better idea of what is going on.

> - We can use a `<script>` tag to add JavaScript code to a page.

> - The type and language attributes are not required.

> - A script in an external file can be inserted with `<script src="path/to/script.js"></script>`.

- The naming of variable is mostly similar to python, one of the difference being that Camel Case is used widely instead of snake case.

> - `let` – is a modern variable declaration.

> - `var` – is an old-school variable declaration. Normally we don’t use it at all, but we’ll cover subtle differences from let in the chapter The old "var", just in case you need them.

> - `const` – is like let, but the value of the variable can’t be changed.

> There are 8 basic data types in JavaScript.

> Seven primitive data types:

> - number for numbers of any kind: integer or floating-point, integers are limited by ±(253-1).
> - bigint for integer numbers of arbitrary length.
> - string for strings. A string may have zero or more characters, there’s no separate single-character type.
> - boolean for true/false.
> - null for unknown values – a standalone type that has a single value null.
> - undefined for unassigned values – a standalone type that has a single value undefined.
> - symbol for unique identifiers.

> And one non-primitive data type:

> object for more complex data structures.

> The typeof operator allows us to see which type is stored in a variable.

> Usually used as typeof x, but typeof(x) is also possible.

> Returns a string with the name of the type, like "string".

> For null returns "object" – this is an error in the language, it’s not actually an object.


```typescript
// Daily learning summary
interface DayProgress {
  date: "13/02/25";
  topics: string["introduction to JS", "IDE", "Manuals and Specifications", "Developer console", "JS Fundamentals along with Syntax"];
  resources: string["javascript.info",];
  challenges: string["1.Show an alert"];
  solutions: string["1.Use alert inside a script"];
  nextSteps: string[];
}
```


### Day 2

**Continuing JS fundamentals**

>  The difference is that AND returns the first falsy value while OR returns the first truthy one.
- We can use labels to get out of nested for loops
- you can append an input with `+` for type conversion 
- The `nullish coalescing operator ??` is a boolean type check in js just for null/undefined 
- Full summary of JS Fundamentals [here](https://javascript.info/javascript-specials)

**Code quality**



React 

Started react from [here](https://react.dev/learn/tutorial-tic-tac-toe#setup-for-the-tutorial)

## Day 3 

**Objects**

>Functions that are stored in object properties are called “methods”.
>Methods allow objects to “act” like object.doSomething().
>Methods can reference the object as this.
>The value of this is defined at run-time.
>
>When a function is declared, it may use this, but that this has no value until the function is called.
>A function can be copied between objects.
>When a function is called in the “method” syntax: object.method(), the value of this during the call is object.
>Please note that arrow functions are special: they have no this. When this is accessed inside an arrow function, it is taken from outside.

- This is just like self in python. Mostly, new is a type of constructor.

>The optional chaining ?. syntax has three forms:
>
>obj?.prop – returns obj.prop if obj exists, otherwise undefined.
>obj?.[prop] – returns obj[prop] if obj exists, otherwise undefined.
>obj.method?.() – calls obj.method() if obj.method exists, otherwise returns undefined.
>As we can see, all of them are straightforward and simple to use. The ?. checks the left part for null/undefined and allows the evaluation to proceed if it’s not so.
>
>A chain of ?. allows to safely access nested properties.
>
>Still, we should apply ?. carefully, only where it’s acceptable, according to our code logic, that the left part doesn’t exist. So that it won’t hide programming errors from us, if they occur.
>

- Completed Objects: the basics
- Starting with Data types

**Data types**

Destructuring assignment allows for instantly mapping an object or array onto many variables.

The full object syntax:

let {prop : varName = defaultValue, ...rest} = object
This means that property prop should go into the variable varName and, if no such property exists, then the default value should be used.

Object properties that have no mapping are copied to the rest object.

The full array syntax:

let [item1 = defaultValue, item2, ...rest] = array
The first item goes to item1; the second goes into item2, and all the rest makes the array rest.

It’s possible to extract data from nested arrays/objects, for that the left side must have the same structure as the right one.

- `...` is like args, and list*.
-  -->