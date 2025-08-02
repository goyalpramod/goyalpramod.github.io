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

- `this` is just like self in python. Mostly, new is a type of constructor.

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

## Day 4

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

- closures are nice 

- Understanding lexical Environments. Very important thing. Cause weird bugs if you dont get 
  [ADD_IMAGE]

- functions have properties like length, name etc. Much like dunder methods in python 

> Functions are objects.
> 
> Here we covered their properties:
> 
> name – the function name. Usually taken from the function definition, but if there’s none, JavaScript tries to guess it from the context (e.g. an assignment).
> length – the number of arguments in the function definition. Rest parameters are not counted.
> If the function is declared as a Function Expression (not in the main code flow), and it carries the name, then it is called a Named Function Expression. The name can be used inside to reference itself, for recursive calls or such.
> 
> Also, functions may carry additional properties. Many well-known JavaScript libraries make great use of this feature.
> 
> They create a “main” function and attach many other “helper” functions to it. For instance, the jQuery library creates a function named $. The lodash library creates a function _, and then adds _.clone, _.keyBy and other properties to it (see the docs when you want to learn more about them). Actually, they do it to lessen their pollution of the global space, so that a single library gives only one global variable. That reduces the possibility of naming conflicts.
> 
> So, a function can do a useful job by itself and also carry a bunch of other functionality in properties.

## Day 5

I learned about the following topics. But I have a feeling they aren't really helpful in webd. But rather very specific JS things. 
Future will tell if I am correct as I build projects. 

> Scheduling: setTimeout and setInterval
> Decorators and forwarding, call/apply
> Function binding
> Arrow functions revisited

> arrows functions have no this, it takes it from outside. (whichever scope comes first)

> Arrow functions:
> 
> Do not have this
> Do not have arguments
> Can’t be called with new
> They also don’t have super, but we didn’t study it yet. We will on the chapter Class inheritance


> In JavaScript, all objects have a hidden [[Prototype]] property that’s either another object or null.
> We can use obj.__proto__ to access it (a historical getter/setter, there are other ways, to be covered soon).
> The object referenced by [[Prototype]] is called a “prototype”.
> If we want to read a property of obj or call a method, and it doesn’t exist, then JavaScript tries to find it in the prototype.
> Write/delete operations act directly on the object, they don’t use the prototype (assuming it’s a data property, not a setter).
> If we call obj.method(), and the method is taken from the prototype, this still references obj. So methods always work with the current object even if they are inherited.
> The for..in loop iterates over both its own and its inherited properties. All other key/value-getting methods only operate on the object itself.


> The basic class syntax looks like this:
> 
> class MyClass {
>   prop = value; // property
> 
>   constructor(...) { // constructor
>     // ...
>   }
> 
>   method(...) {} // method
> 
>   get something(...) {} // getter method
>   set something(...) {} // setter method
> 
>   [Symbol.iterator]() {} // method with computed name (symbol here)
>   // ...
> }
> MyClass is technically a function (the one that we provide as constructor), while methods, getters and setters are written to MyClass.prototype.
> 
> In the next chapters we’ll learn more about classes, including inheritance and other features.

## Day 6

- I finished studying about classes in js. I wish to wrap JS as soon as possible so I can start building stuff asap.

## Day 7

> The async keyword before a function has two effects:
> 
> Makes it always return a promise.
> Allows await to be used in it.
> The await keyword before a promise makes JavaScript wait until that promise settles, and then:
> 
> If it’s an error, an exception is generated — same as if throw error were called at that very place.
> Otherwise, it returns the result.
> Together they provide a great framework to write asynchronous code that is easy to both read and write.
> 
> With async/await we rarely need to write promise.then/catch, but we still shouldn’t forget that they are based on promises, because sometimes (e.g. in the outermost scope) we have to use these methods. Also Promise.all is nice when we are waiting for many tasks simultaneously.
>
> Proxy is a wrapper around an object, that forwards operations on it to the object, optionally trapping some of them.

>It can wrap any kind of object, including classes and functions.
>
>The syntax is:
>
>let proxy = new Proxy(target, {
>  /* traps */
>});
>…Then we should use proxy everywhere instead of target. A proxy doesn’t have its own properties or methods. It traps an operation if the trap is provided, otherwise forwards it to target object.
>
>We can trap:
>
>Reading (get), writing (set), deleting (deleteProperty) a property (even a non-existing one).
>Calling a function (apply trap).
>The new operator (construct trap).
>Many other operations (the full list is at the beginning of the article and in the docs).
>That allows us to create “virtual” properties and methods, implement default values, observable objects, function decorators and so much more.
>
>We can also wrap an object multiple times in different proxies, decorating it with various aspects of functionality.
>
>The Reflect API is designed to complement Proxy. For any Proxy trap, there’s a Reflect call with same arguments. We should use those to forward calls to target objects.
>
>Proxies have some limitations:
>
>Built-in objects have “internal slots”, access to those can’t be proxied. See the workaround above.
>The same holds true for private class fields, as they are internally implemented using slots. So proxied method calls must have the target object as this to access them.
>Object equality tests === can’t be intercepted.
>Performance: benchmarks depend on an engine, but generally accessing a property using a simplest proxy takes a few times longer. In practice that only matters for some “bottleneck” objects though.

>Currying is a transform that makes f(a,b,c) callable as f(a)(b)(c). 

>DOM specification
>Describes the document structure, manipulations, and events, see https://dom.spec.whatwg.org.
>CSSOM specification
>Describes stylesheets and style rules, manipulations with them, and their binding to documents, see https://www.w3.org/TR/cssom-1/.
>HTML specification
>Describes the HTML language (e.g. tags) and also the BOM (browser object model) – various browser functions: setTimeout, alert, location and so on, see https://html.spec.whatwg.org. It takes the DOM specification and extends it with many additional properties and methods.
>Additionally, some classes are described separately at https://spec.whatwg.org/.
>
>Please note these links, as there’s so much to learn that it’s impossible to cover everything and remember it all.
>
>When you’d like to read about a property or a method, the Mozilla manual at https://developer.mozilla.org/en-US/ is also a nice resource, but the corresponding spec may be better: it’s more complex and longer to read, but will make your fundamental knowledge sound and complete.
>
>To find something, it’s often convenient to use an internet search “WHATWG [term]” or “MDN [term]”, e.g https://google.com?q=whatwg+localstorage, https://google.com?q=mdn+localstorage.
>
>Now, we’ll get down to learning the DOM, because the document plays the central role in the UI.
- Started with ts as well. It do be simple honestly. -->