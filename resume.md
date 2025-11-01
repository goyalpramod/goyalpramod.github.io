---
layout: default
title: Resume
permalink: /resume/
---

<style>
  .resume-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem 1rem;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    line-height: 1.7;
  }

  .resume-header {
    text-align: center;
    margin-bottom: 3rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid #e0e0e0;
  }

  .resume-name {
    font-size: 2.5rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: #1a1a1a;
    letter-spacing: -0.02em;
  }

  .resume-contact {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 1.5rem;
    margin-bottom: 1rem;
    font-size: 0.95rem;
  }

  .resume-contact a {
    color: #555;
    text-decoration: none;
    transition: color 0.2s ease;
  }

  .resume-contact a:hover {
    color: #000;
  }

  .resume-social {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 1rem;
    font-size: 0.9rem;
  }

  .resume-social a {
    color: #666;
    text-decoration: none;
    transition: color 0.2s ease;
  }

  .resume-social a:hover {
    color: #000;
  }

  .resume-intro {
    max-width: 700px;
    margin: 0 auto 3rem;
    padding: 2rem;
    background: #f9f9f9;
    border-radius: 8px;
    text-align: center;
  }

  .resume-intro p {
    margin: 0 0 0.75rem 0;
    font-size: 1rem;
    color: #555;
    line-height: 1.6;
  }

  .resume-intro p:last-child {
    margin-bottom: 0;
  }

  .resume-intro strong {
    color: #1a1a1a;
    font-weight: 600;
  }

  .resume-section {
    margin-bottom: 3rem;
  }

  .section-title {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
    color: #1a1a1a;
    letter-spacing: -0.01em;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e0e0e0;
  }

  .resume-item {
    margin-bottom: 2rem;
  }

  .item-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.25rem;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .item-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a1a1a;
  }

  .item-date {
    font-size: 0.9rem;
    color: #666;
    font-style: italic;
  }

  .item-subtitle {
    font-size: 0.95rem;
    color: #555;
    margin-bottom: 0.5rem;
  }

  .item-detail {
    font-size: 0.95rem;
    color: #666;
    margin-bottom: 0.25rem;
  }

  .item-description {
    font-size: 0.95rem;
    color: #555;
    margin-top: 0.5rem;
  }

  .item-list {
    margin: 0.5rem 0 0 1.2rem;
    padding: 0;
    list-style-type: none;
  }

  .item-list li {
    position: relative;
    padding-left: 1rem;
    margin-bottom: 0.4rem;
    font-size: 0.95rem;
    color: #555;
  }

  .item-list li::before {
    content: "–";
    position: absolute;
    left: 0;
    color: #999;
  }

  .skills-grid {
    display: grid;
    gap: 0.75rem;
  }

  .skill-item {
    font-size: 0.95rem;
    color: #555;
  }

  .skill-label {
    font-weight: 600;
    color: #1a1a1a;
    display: inline-block;
    min-width: 140px;
  }

  .publication-title {
    font-style: italic;
    color: #1a1a1a;
    font-size: 0.95rem;
    margin-bottom: 0.25rem;
  }

  .publication-authors {
    font-size: 0.9rem;
    color: #666;
  }

  .item-link {
    color: #0066cc;
    text-decoration: none;
    border-bottom: 1px solid transparent;
    transition: border-color 0.2s ease;
  }

  .item-link:hover {
    border-bottom-color: #0066cc;
  }

  /* Dark mode support */
  [data-theme="dark"] .resume-name,
  [data-theme="dark"] .section-title,
  [data-theme="dark"] .item-title,
  [data-theme="dark"] .skill-label,
  [data-theme="dark"] .publication-title {
    color: #e0e0e0;
  }

  [data-theme="dark"] .resume-contact a,
  [data-theme="dark"] .resume-social a,
  [data-theme="dark"] .item-subtitle,
  [data-theme="dark"] .item-description,
  [data-theme="dark"] .item-list li,
  [data-theme="dark"] .skill-item {
    color: #b0b0b0;
  }

  [data-theme="dark"] .resume-contact a:hover,
  [data-theme="dark"] .resume-social a:hover {
    color: #fff;
  }

  [data-theme="dark"] .item-date,
  [data-theme="dark"] .item-detail,
  [data-theme="dark"] .publication-authors {
    color: #888;
  }

  [data-theme="dark"] .resume-header,
  [data-theme="dark"] .section-title {
    border-bottom-color: #333;
  }

  [data-theme="dark"] .resume-intro {
    background: #1a1a1a;
  }

  [data-theme="dark"] .resume-intro p {
    color: #b0b0b0;
  }

  [data-theme="dark"] .resume-intro strong {
    color: #e0e0e0;
  }

  /* Mobile responsive */
  @media (max-width: 768px) {
    .resume-container {
      padding: 1.5rem 1rem;
    }

    .resume-name {
      font-size: 2rem;
    }

    .resume-contact {
      flex-direction: column;
      gap: 0.5rem;
    }

    .item-header {
      flex-direction: column;
      align-items: flex-start;
    }

    .skill-label {
      display: block;
      min-width: auto;
      margin-bottom: 0.25rem;
    }
  }
</style>

<div class="resume-container">
  <!-- Header -->
  <header class="resume-header">
    <h1 class="resume-name">Pramod Goyal</h1>
    <div class="resume-contact">
      <span>goyalpramod1729[at]gmail.com</span>
    </div>
    <div class="resume-social">
      <a href="https://linkedin.com/in/goyalpramod" target="_blank" rel="noopener noreferrer">LinkedIn</a>
      <span style="color: #ddd;">|</span>
      <a href="https://github.com/goyalpramod" target="_blank" rel="noopener noreferrer">GitHub</a>
      <span style="color: #ddd;">|</span>
      <a href="https://youtube.com/@goyal_pramod" target="_blank" rel="noopener noreferrer">YouTube</a>
      <span style="color: #ddd;">|</span>
      <a href="https://x.com/goyal__pramod" target="_blank" rel="noopener noreferrer">X</a>
      <span style="color: #ddd;">|</span>
      <a href="https://instagram.com/goyal__pramod" target="_blank" rel="noopener noreferrer">Instagram</a>
    </div>
  </header>

  <!-- Personal Introduction -->
  <section class="resume-intro">
    <p>I am always happy to talk to dreamers. If you would like to work with me or just have a quick chat, feel free to email me at <strong>goyalpramod1729[at]gmail.com</strong>. Seriously, drop me a text. I am very happy to meet new people.</p>
    <p>You can check out my other socials if you would like to learn a bit more about me!</p>
  </section>

  <!-- Education -->
  <section class="resume-section">
    <h2 class="section-title">Education</h2>
    
    <div class="resume-item">
      <div class="item-header">
        <div class="item-title">University of Maryland, College Park</div>
        <div class="item-date">2025-27</div>
      </div>
      <div class="item-subtitle">Master's in Artificial Intelligence</div>
    </div>

    <div class="resume-item">
      <div class="item-header">
        <div class="item-title">National Institute of Technology, Rourkela</div>
        <div class="item-date">2020-24</div>
      </div>
      <div class="item-subtitle">Bachelor of Technology in Electronics and Instrumentation Engineering</div>
      <div class="item-detail">CGPA: 8.13</div>
    </div>

    <div class="resume-item">
      <div class="item-header">
        <div class="item-title">Mother's Public School</div>
        <div class="item-date">2018-20</div>
      </div>
      <div class="item-subtitle">Central Board of Secondary Education (C.B.S.E)</div>
      <div class="item-detail">Percentage: 95.4%</div>
    </div>
  </section>

  <!-- Publications -->
  <section class="resume-section">
    <h2 class="section-title">Publications</h2>
    
    <div class="resume-item">
      <div class="publication-title">Hate Speech and Offensive Content Detection in Indo-Aryan Languages: A Battle of LSTM and Transformers</div>
      <div class="publication-authors">Goyal, P., Narayan, N., Biswal, M., & Panigrahi, A. (2023)</div>
      <div class="item-description">
        <a href="https://arxiv.org/abs/2312.05671" class="item-link">Link to preprint</a>
      </div>
    </div>
  </section>

  <!-- Experience -->
  <section class="resume-section">
    <h2 class="section-title">Experience</h2>
    
    <div class="resume-item">
      <div class="item-header">
        <div class="item-title">Founding AI Developer</div>
        <div class="item-date">May 2025 - August 2025</div>
      </div>
      <div class="item-subtitle">FutForce – Building Conversational Agents</div>
      <ul class="item-list">
        <li>Developed complex Dialogflow CX conversational flows for enterprise chatbot solutions</li>
        <li>Created custom testing and evaluation framework to ensure system reliability and performance</li>
        <li>Scaled application infrastructure to support 50,000+ concurrent users</li>
        <li>Contributed to ERPNext implementation using Frappe framework for business process automation</li>
      </ul>
    </div>

    <div class="resume-item">
      <div class="item-header">
        <div class="item-title">Founding AI Developer</div>
        <div class="item-date">February 2024 - April 2025</div>
      </div>
      <div class="item-subtitle">Dimension – Orchestrating LLMs</div>
      <ul class="item-list">
        <li>Developed and maintained AI infrastructure using Langchain with monitoring via Langsmith and evaluation using DeepEval</li>
        <li>Responsible for developing multiple RAG-based pipelines with a retrieval efficiency of 97 percent</li>
        <li>Improved accuracy of multiple pipelines using prompt engineering and token reduction, decreasing costs by up to 10 times in production</li>
        <li>Fine-tuning models using different techniques like LoRA and PEFT</li>
        <li>Deploying LLMs for efficient inference in servers using vLLM and TensorRT</li>
      </ul>
    </div>

    <div class="resume-item">
      <div class="item-header">
        <div class="item-title">Open Source Contributor</div>
        <div class="item-date">July 2023 - August 2023</div>
      </div>
      <div class="item-subtitle">Code4GovTech – Text2SQL</div>
      <ul class="item-list">
        <li>Selected for the Code4GovTech program to act as an open source contributor to the Text2SQL project</li>
        <li>Responsible for working with large language models (LLMs), setting up tests, and working on token optimization</li>
        <li>Improved accuracy of LLMs from 0.516 to 0.743</li>
      </ul>
      <div class="item-description">
        <a href="#" class="item-link">Repository – Text2SQL</a>
      </div>
    </div>
  </section>
</div>