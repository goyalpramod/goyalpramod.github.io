---
layout: default
title: Resume
permalink: /resume/
---

<style>
  .timeline-container {
    max-width: 1000px;
    margin: 0 auto;
    position: relative;
  }

  .timeline-header {
    text-align: center;
    margin-bottom: 3rem;
  }

  .page-title {
    font-size: 2.5rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: #222;
  }

  .page-description {
    font-size: 1.1rem;
    color: #666;
    max-width: 600px;
    margin: 0 auto;
  }

  .timeline {
    position: relative;
    padding: 2rem 0;
  }

  /* Main timeline line */
  .timeline::before {
    content: '';
    position: absolute;
    left: 50%;
    top: 0;
    bottom: 0;
    width: 2px;
    background: #ddd;
    transform: translateX(-50%);
  }

  .timeline-item {
    position: relative;
    margin-bottom: 3rem;
    width: 100%;
  }

  .timeline-content {
    background: #fff;
    border: 1px solid #eee;
    border-radius: 8px;
    padding: 1.5rem;
    width: 45%;
    position: relative;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
  }

  .timeline-content:hover {
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    transform: translateY(-2px);
  }

  /* Alternating sides */
  .timeline-item:nth-child(odd) .timeline-content {
    margin-left: auto;
    margin-right: 55%;
  }

  .timeline-item:nth-child(even) .timeline-content {
    margin-left: 55%;
    margin-right: auto;
  }

  /* Timeline dots */
  .timeline-dot {
    position: absolute;
    left: 50%;
    top: 2rem;
    width: 16px;
    height: 16px;
    background: #fff;
    border: 3px solid #333;
    border-radius: 50%;
    transform: translateX(-50%);
    z-index: 2;
    transition: all 0.3s ease;
  }

  .timeline-item:hover .timeline-dot {
    background: #333;
    transform: translateX(-50%) scale(1.2);
  }

  /* Arrows pointing to timeline */
  .timeline-content::before {
    content: '';
    position: absolute;
    top: 2rem;
    width: 0;
    height: 0;
    border: 10px solid transparent;
  }

  .timeline-item:nth-child(odd) .timeline-content::before {
    right: -20px;
    border-left-color: #eee;
  }

  .timeline-item:nth-child(even) .timeline-content::before {
    left: -20px;
    border-right-color: #eee;
  }

  .timeline-content::after {
    content: '';
    position: absolute;
    top: 2rem;
    width: 0;
    height: 0;
    border: 9px solid transparent;
  }

  .timeline-item:nth-child(odd) .timeline-content::after {
    right: -18px;
    border-left-color: #fff;
  }

  .timeline-item:nth-child(even) .timeline-content::after {
    left: -18px;
    border-right-color: #fff;
  }

  .timeline-date {
    font-size: 0.9rem;
    color: #666;
    font-weight: 500;
    margin-bottom: 0.5rem;
  }

  .timeline-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #222;
    margin-bottom: 0.5rem;
  }

  .timeline-subtitle {
    font-size: 1rem;
    color: #555;
    margin-bottom: 1rem;
    font-style: italic;
  }

  .timeline-description {
    color: #666;
    line-height: 1.6;
  }

  .timeline-description ul {
    margin: 0.5rem 0;
    padding-left: 1.2rem;
  }

  .timeline-description li {
    margin-bottom: 0.3rem;
  }

  .timeline-link {
    color: #007acc;
    text-decoration: none;
    border-bottom: 1px solid transparent;
    transition: border-color 0.3s ease;
  }

  .timeline-link:hover {
    border-bottom-color: #007acc;
  }

  .timeline-tag {
    display: inline-block;
    background: #f5f5f5;
    color: #555;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.8rem;
    margin: 0.2rem 0.2rem 0 0;
  }

  /* Mobile responsive */
  @media (max-width: 768px) {
    .timeline::before {
      left: 20px;
    }

    .timeline-content {
      width: calc(100% - 60px);
      margin-left: 60px !important;
      margin-right: 0 !important;
    }

    .timeline-dot {
      left: 20px;
    }

    .timeline-content::before,
    .timeline-content::after {
      left: -20px !important;
      right: auto !important;
      border-right-color: #eee !important;
      border-left-color: transparent !important;
    }

    .timeline-content::after {
      border-right-color: #fff !important;
    }

    .page-title {
      font-size: 2rem;
    }
  }

  [data-theme="dark"] .timeline-title {
  color: #fff !important;
}

[data-theme="dark"] .page-title {
  color: #fff !important;
}

[data-theme="dark"] .timeline-content {
  background: #1a1a1a;
  border-color: #333;
  color: #e0e0e0;
}

[data-theme="dark"] .timeline-content::before {
  border-left-color: #333;
  border-right-color: #333;
}

[data-theme="dark"] .timeline-content::after {
  border-left-color: #1a1a1a;
  border-right-color: #1a1a1a;
}

[data-theme="dark"] .timeline::before {
  background: #444;
}

[data-theme="dark"] .timeline-dot {
  background: #1a1a1a;
  border-color: #fff;
}

[data-theme="dark"] .timeline-item:hover .timeline-dot {
  background: #fff;
}
</style>

<div class="timeline-container">
  <header class="timeline-header">
    <h1 class="page-title">My Journey</h1>
    <p class="page-description">A timeline of my education, experiences, and projects that have shaped my path in AI and technology.</p>
    <div class="contact-links" style="margin-top: 1.5rem;">
      <a href="mailto:goyalpramod1729@gmail.com" class="timeline-link" style="margin-right: 2rem;">
        goyalpramod1729[at]gmail.com
      </a>
      <a href="https://linktr.ee/goyalpramod" class="timeline-link" target="_blank" rel="noopener noreferrer">
        All Links: linktr.ee/goyalpramod
      </a>
    </div>
  </header>

  <div class="timeline">
    <!-- Education: University of Maryland -->
    <div class="timeline-item">
      <div class="timeline-dot"></div>
      <div class="timeline-content">
        <div class="timeline-date">2025 - 2027</div>
        <h3 class="timeline-title">Master's in Artificial Intelligence</h3>
        <p class="timeline-subtitle">University of Maryland, College Park</p>
        <div class="timeline-description">
          <p>Currently pursuing advanced studies in AI, focusing on cutting-edge research and applications in machine learning and neural networks.</p>
        </div>
      </div>
    </div>

    <!-- Current Work: FutForce -->
    <div class="timeline-item">
      <div class="timeline-dot"></div>
      <div class="timeline-content">
        <div class="timeline-date">May 2025 - August 2025</div>
        <h3 class="timeline-title">Founding AI Developer</h3>
        <p class="timeline-subtitle">FutForce - Building Conversational Agents</p>
        <div class="timeline-description">
          <ul>
            <li>Developed complex Dialogflow CX conversational flows for enterprise chatbot solutions</li>
            <li>Created custom testing and evaluation framework to ensure system reliability and performance</li>
            <li>Scaled application infrastructure to support 50,000+ concurrent users</li>
            <li>Contributed to ERPNext implementation using Frappe framework for business process automation</li>
          </ul>
        </div>
      </div>
    </div>

    <!-- YouTube Project -->
    <div class="timeline-item">
      <div class="timeline-dot"></div>
      <div class="timeline-content">
        <div class="timeline-date">January 2025 - Present</div>
        <h3 class="timeline-title">ML Mathematics from Scratch</h3>
        <p class="timeline-subtitle">YouTube Channel</p>
        <div class="timeline-description">
          <p>Produced 5 videos on core ML, attracting 350+ subscribers and 5000+ views. Making complex mathematical concepts accessible through visual explanations.</p>
          <div style="margin-top: 0.5rem;">
            <span class="timeline-tag">Education</span>
            <span class="timeline-tag">Mathematics</span>
            <span class="timeline-tag">Machine Learning</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Diffusion Models Blog -->
    <div class="timeline-item">
      <div class="timeline-dot"></div>
      <div class="timeline-content">
        <div class="timeline-date">February 2025</div>
        <h3 class="timeline-title">Diffusion Models from Scratch</h3>
        <p class="timeline-subtitle">Technical Blog Series</p>
        <div class="timeline-description">
          <p>Created comprehensive technical blog series explaining diffusion model mathematics, forward/reverse processes, and loss functions with step-by-step code examples.</p>
          <a href="#" class="timeline-link">Link to the blog</a>
          <div style="margin-top: 0.5rem;">
            <span class="timeline-tag">Deep Learning</span>
            <span class="timeline-tag">Generative AI</span>
            <span class="timeline-tag">Technical Writing</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Dimension Work -->
    <div class="timeline-item">
      <div class="timeline-dot"></div>
      <div class="timeline-content">
        <div class="timeline-date">February 2024 - April 2025</div>
        <h3 class="timeline-title">Founding AI Developer</h3>
        <p class="timeline-subtitle">Dimension - Orchestrating LLMs</p>
        <div class="timeline-description">
          <ul>
            <li>Developed and maintained AI infrastructure using Langchain with monitoring via Langsmith</li>
            <li>Responsible for developing multiple RAG-based pipelines with a retrieval efficiency of 97 percent</li>
            <li>Improved accuracy using prompt engineering and token reduction, decreasing costs by up to 10 times</li>
            <li>Fine-tuning models using LoRA and PEFT techniques</li>
            <li>Deploying LLMs for efficient inference using vLLM and TensorRT</li>
          </ul>
        </div>
      </div>
    </div>

    <!-- ML Paper Implementation -->
    <div class="timeline-item">
      <div class="timeline-dot"></div>
      <div class="timeline-content">
        <div class="timeline-date">December 2024</div>
        <h3 class="timeline-title">ML Paper Implementation</h3>
        <p class="timeline-subtitle">Research Project</p>
        <div class="timeline-description">
          <p>Implemented foundation ML papers from scratch, including attention mechanisms, transformers, and diffusion models.</p>
          <a href="#" class="timeline-link">Link to the repository</a>
          <div style="margin-top: 0.5rem;">
            <span class="timeline-tag">Research</span>
            <span class="timeline-tag">Implementation</span>
            <span class="timeline-tag">Open Source</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Bachelor's Degree -->
    <div class="timeline-item">
      <div class="timeline-dot"></div>
      <div class="timeline-content">
        <div class="timeline-date">2020 - 2024</div>
        <h3 class="timeline-title">Bachelor of Technology</h3>
        <p class="timeline-subtitle">National Institute of Technology, Rourkela</p>
        <div class="timeline-description">
          <p>Electronics and Instrumentation Engineering with CGPA: 8.13</p>
          <p><strong>Relevant Coursework:</strong> Data Structures and Algorithm, Linear Algebra, Statistics and Probability, Introduction to Machine Learning</p>
        </div>
      </div>
    </div>

    <!-- Code4GovTech -->
    <div class="timeline-item">
      <div class="timeline-dot"></div>
      <div class="timeline-content">
        <div class="timeline-date">July 2023 - August 2023</div>
        <h3 class="timeline-title">Open Source Contributor</h3>
        <p class="timeline-subtitle">Code4GovTech - Text2SQL</p>
        <div class="timeline-description">
          <ul>
            <li>Selected for the Code4GovTech program to contribute to the Text2SQL project</li>
            <li>Worked with large language models, setting up tests, and token optimization</li>
            <li>Improved accuracy of LLMs from 0.516 to 0.743</li>
          </ul>
          <a href="#" class="timeline-link">Repository - Text2SQL</a>
        </div>
      </div>
    </div>

    <!-- Publication -->
    <div class="timeline-item">
      <div class="timeline-dot"></div>
      <div class="timeline-content">
        <div class="timeline-date">2023</div>
        <h3 class="timeline-title">Research Publication</h3>
        <p class="timeline-subtitle">Hate Speech Detection Research</p>
        <div class="timeline-description">
          <p><strong>"Hate Speech and Offensive Content Detection in Indo-Aryan Languages: A Battle of LSTM and Transformers"</strong></p>
          <p>Goyal, P., Narayan, N., Biswal, M., & Panigrahi, A.</p>
          <a href="#" class="timeline-link">Link to preprint</a>
          <div style="margin-top: 0.5rem;">
            <span class="timeline-tag">NLP</span>
            <span class="timeline-tag">Research</span>
            <span class="timeline-tag">Publication</span>
          </div>
        </div>
      </div>
    </div>

    <!-- High School -->
    <div class="timeline-item">
      <div class="timeline-dot"></div>
      <div class="timeline-content">
        <div class="timeline-date">2018 - 2020</div>
        <h3 class="timeline-title">High School</h3>
        <p class="timeline-subtitle">Mother's Public School</p>
        <div class="timeline-description">
          <p>Central Board of Secondary Education (C.B.S.E) with 95.4% marks. Foundation in mathematics and sciences that sparked my interest in technology.</p>
        </div>
      </div>
    </div>
  </div>
</div>