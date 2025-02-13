---
layout: blog
title: "Web development for MLE"
date: 2025-02-10 12:00:00 +0530
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

## Daily Progress Template

```typescript
// Daily learning summary
interface DayProgress {
  date: string;
  topics: string[];
  resources: string[];
  challenges: string[];
  solutions: string[];
  nextSteps: string[];
}
```
