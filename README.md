
# A roadmap for AI-driven protein design
<img src="https://github.com/miangoar/ciencia-de-proteinas-basada-en-IA/blob/main/img/roadmap.jpg" alt="Description" width="900" height="auto">

# Content list
- [Tools](./tools): libraries organized into 25 categories
- [Learning resources](./learning_resources): courses, tutorials and useful publications organized into nine categories
- [Databases](./databases): resources to download genomic and protein data organized into 12 categories
- [Youtube](./youtube): recommended channels and videos to learn about proteins, mathematics, and data science
- [Lectures and slides](./video): links to each lecture and to download the slides


# Breve descripcion en Español
Curso gratuito de 10 clases (37 horas y +800 diapositivas) sobre diseño de proteínas basado en Inteligencia Artificial. Para más detalles, consulta la página del curso: 
- <https://miangoar.github.io/teaching/proteins>

# Brief description
Free 10-lecture course (37 hours and +800 slides) on AI-based protein design.

I created this <a href="https://miangoaren.github.io/teaching/proteins">free course</a> consisting of 10 lectures to introduce you to AI-driven protein design.

![webinar](https://github.com/miangoar/AI-driven-protein-design/raw/main/img/roadmap.jpg)

# Contents
- [General description](#general-description)
- [Course organization](#course-organization)
- [Access to the slides](#access-to-the-slides)
- [How to support this project](#how-to-support-this-project)
- [About me](#about-me)

# General description

I want more people to learn how to design proteins using Artificial Intelligence (AI). However, I have encountered three main problems:

1. There is a large amount of information, and it is not clear where to start or which topics are necessary.
2. There are no comprehensive online courses on this topic in Spanish.
3. Courses related to this field are usually expensive for most students in Latin America.

<p>To address these issues, I created this free 37-hour course, distributed across 10 lectures, to introduce you to AI-driven protein design. The course includes two main resources:</p>

<ol>
  <li><a href="https://www.youtube.com/@miangoar_bio">The 10 lectures on YouTube</a></li>
  <li><a href="https://github.com/miangoar/AI-driven-protein-design">A GitHub repository</a> with the following resources:
    <ol type="a">
      <li><a href="https://github.com/miangoar/AI-driven-protein-design/tree/main/tools/readme.md">Tools</a>: libraries organized into 25 categories</li>
      <li><a href="https://github.com/miangoar/AI-driven-protein-design/tree/main/learning_resources/readme.md">Learning resources</a>: courses, tutorials and useful publications organized into nine categories</li>
      <li><a href="https://github.com/miangoar/AI-driven-protein-design/tree/main/databases/readme.md">Databases</a>: resources to download genomic and protein data organized into 12 categories</li>
      <li><a href="https://github.com/miangoar/AI-driven-protein-design/tree/main/video/readme.md">Lectures</a>: links to each lecture and to download the slides</li>
      <li><a href="https://github.com/miangoar/AI-driven-protein-design/tree/main/youtube/readme.md">YouTube</a>: Recommended channels and videos to learn about proteins, mathematics, and data science</li>
    </ol>
  </li>
</ol>

# Course organization

![lectures](https://github.com/miangoar/AI-driven-protein-design/raw/main/img/lectures.png)


The lectures are organized from 01 to 10 to facilitate conceptual understanding. For example, reviewing AlphaFold requires knowledge of structural biology and deep learning, which are covered in detail in their respective lectures. Below is a brief description of each lecture and its topics:

<ol>
  <li><strong><a href="https://www.youtube.com/watch?v=XXXX">Basic computing concepts</a></strong>: how CPUs and GPUs work, as well as the essential software for data analysis.
    <ol type="a">
      <li>Where does your journey begin?</li>
      <li>Hardware
        <ul>
          <li>CPU</li>
          <li>GPU</li>
        </ul>
      </li>
      <li>Software
        <ul>
          <li>Linux/Bash and GitHub</li>
          <li>Python</li>
        </ul>
      </li>
    </ol>
  </li>

  <li><strong><a href="https://www.youtube.com/watch?v=XXXX">Machine learning</a></strong>: what AI is and its subfields, the current capabilities of algorithms, and how a model is trained.
    <ol type="a">
      <li>Current state of AI</li>
      <li>How AI learns
        <ul>
          <li>Patterns</li>
          <li>Machine learning operations (MLOps)</li>
          <li>Learning paradigms</li>
        </ul>
      </li>
      <li>How to train a model
        <ul>
          <li>Data processing</li>
          <li>How to choose a model</li>
          <li>Training process</li>
        </ul>
      </li>
    </ol>
  </li>

  <li><strong><a href="https://www.youtube.com/watch?v=XXXX">Deep learning</a></strong>: how neural networks work, the different types of neural networks, and the software used to work with them.
    <ol type="a">
      <li>Neural networks
        <ul>
          <li>Neurons</li>
          <li>Deep learning</li>
          <li>Loss functions</li>
          <li>Backpropagation</li>
          <li>Optimizers</li>
          <li>Architectures</li>
          <li>Explainability (why) and Interpretability (how)</li>
        </ul>
      </li>
      <li>Deep learning frameworks</li>
    </ol>
  </li>

  <li><strong><a href="https://www.youtube.com/watch?v=XXXX">Transformers and language models</a></strong>: how Transformers and modern language models work.
    <ol type="a">
      <li>Language models</li>
      <li>Transformers
        <ul>
          <li>Original architecture</li>
          <li>BERT and GPT architectures</li>
          <li>Scaling laws</li>
          <li>Pre-training and post-training</li>
          <li>Reinforcement learning</li>
        </ul>
      </li>
      <li>Performance and generalization
        <ul>
          <li>Benchmark saturation</li>
          <li>Hype</li>
        </ul>
      </li>
      <li>How to work with LLMs
        <ul>
          <li>Optimization techniques (for GPU-poors like us)</li>
          <li>Hugging Face and Software 2.0</li>
        </ul>
      </li>
    </ol>
  </li>

  <li><strong><a href="https://www.youtube.com/watch?v=XXXX">Protein structure</a></strong>: principles of structural biology and organization.
    <ol type="a">
      <li>Structural organization
        <ul>
          <li>Amino acids</li>
          <li>Secondary and tertiary structure</li>
          <li>Experimental workflow for structure determination</li>
          <li>Structure Viewers</li>
        </ul>
      </li>
      <li>Classifications
        <ul>
          <li>Folds and domains</li>
          <li>First classification schemes</li>
          <li>Similarity metrics</li>
          <li>Sequence and structural divergence</li>
          <li>Current classifications schemes</li>
        </ul>
      </li>
      <li>The shape of the protein universe
        <ul>
          <li>Uneven distribution</li>
          <li>Complex homologous relationships</li>
          <li>Switch folds</li>
        </ul>
      </li>
    </ol>
  </li>

  <li><strong><a href="https://www.youtube.com/watch?v=XXXX">Protein function</a></strong>: how proteins adopt their structure and how function is regulated.
    <ol type="a">
      <li>Protein folding
        <ul>
          <li>Cellular environment</li>
          <li>Thermodynamics and conformational entropy</li>
        </ul>
      </li>
      <li>Protein function
        <ul>
          <li>Diffusion</li>
          <li>Molecular dynamics and energy functions</li>
          <li>Enzymes</li>
          <li>Functional annotation</li>
        </ul>
      </li>
      <li>Functional regulation
        <ul>
          <li>Allosterism</li>
          <li>Transcriptional regulation</li>
          <li>Post translational modifications</li>
          <li>Proteostasis and host physiology</li>
        </ul>
      </li>
    </ol>
  </li>

  <li><strong><a href="https://www.youtube.com/watch?v=XXXX">Protein evolution</a></strong>: origin and diversification from simpler peptides.
    <ol type="a">
      <li>Levels of biological organization
        <ul>
          <li>Evolution across spatio-temporal scales</li>
          <li>Chemical evolution</li>
        </ul>
      </li>
      <li>Biological evolution
        <ul>
          <li>RNA world hypothesis and ribosome evolution</li>
          <li>Ancestral proteins</li>
          <li>Protein diversification</li>
        </ul>
      </li>
      <li>The sequence space
        <ul>
          <li>Mutations</li>
          <li>Robustness, evolvability and promiscuity</li>
          <li>Evolution of protein function</li>
        </ul>
      </li>
      <li>Epistasis: How interactions shape the evolution
        <ul>
          <li>Residue-residue and protein-protein interactions</li>
          <li>Randomness of mutations</li>
        </ul>
      </li>
    </ol>
  </li>

  <li><strong><a href="https://www.youtube.com/watch?v=XXXX">AlphaFold</a></strong>: overview of AF2 and AF3 architectures and impact.
    <ol type="a">
      <li>The impact of AlphaFold
        <ul>
          <li>AlphaFoldmania</li>
          <li>Protein structure prediction before AlphaFold</li>
        </ul>
      </li>
      <li>AlphaFold</li>
      <li>AlphaFold2
        <ul>
          <li>Protein language models</li>
          <li>Architecture</li>
          <li>Post-AlphaFold2 era</li>
        </ul>
      </li>
      <li>AlphaFold3
        <ul>
          <li>Diffusion models for macromolecular modeling</li>
          <li>Architecture</li>
          <li>Post-AlphaFold3 era</li>
        </ul>
      </li>
    </ol>
  </li>

  <li><strong><a href="https://www.youtube.com/watch?v=XXXX">AI-driven protein design</a></strong>: motivations and modern AI methods.
    <ol type="a">
      <li>Protein design
        <ul>
          <li>AI in the biotech market</li>
          <li>Advances from classical methods to AI-driven methods</li>
          <li>Basic considerations to increase the success of a design</li>
        </ul>
      </li>
      <li>Rational design
        <ul>
          <li>Classic experimental and bioinformatic approaches</li>
          <li>Macromolecular modeling and recombineering</li>
        </ul>
      </li>
      <li>Evolutionary design
        <ul>
          <li>Directed evolution, ancestral sequence reconstruction and consensus design</li>
        </ul>
      </li>
      <li>Representation learning
        <ul>
          <li>(Macro)Molecular representations</li>
          <li>Protein language models and ESMFold</li>
          <li>Explainability and interpretability of protein language models</li>
          <li>Scaling laws and multimodality in protein language models</li>
        </ul>
      </li>
      <li>Generative AI
        <ul>
          <li>Integration of multimodal data</li>
          <li>Sequence generation</li>
          <li>Generalization and fitness prediction with protein language models</li>
          <li>Inverse folding and ProteinMPNN</li>
          <li>Structure generation with diffusion models</li>
          <li>Model selection and computational scoring of candidates</li>
          <li>Model generalization and synthetic data</li>
        </ul>
      </li>
      <li>Summary</li>
    </ol>
  </li>

  <li><strong><a href="https://www.youtube.com/watch?v=XXXX">Data and biases</a></strong>: relevant databases and data processing.
    <ol type="a">
      <li>Big data is Omics
        <ul>
          <li>Properties of a good dataset</li>
        </ul>
      </li>
      <li>Main datasets
        <ul>
          <li>PDB</li>
          <li>UniProt</li>
          <li>NCBI datasets</li>
          <li>Other interesting datasets</li>
        </ul>
      </li>
      <li>Data processing
        <ul>
          <li>Data cleaning in biology</li>
          <li>Basic tools for biological data manipulation</li>
          <li>Data splitting</li>
        </ul>
      </li>
      <li>Generalization in (protein) biology
        <ul>
          <li>Data leakage and other inherent issues</li>
        </ul>
      </li>
      <li>Biases in the data</li>
      <li>A roadmap for AI-driven protein design</li>
    </ol>
  </li>
</ol>

# Access to the slides

This course includes +800 slides with image sources, citations, and recommended resources for deeper study in the notes section. I recommend reviewing the slides using PowerPoint. You can download the slides from Zenodo and Google Drive (to be published once all videos are available):

| Topic | Slides | YouTube |
|------|--------|---------|
| Basic computing concepts | Drive, Zenodo | Video |
| Machine learning | Drive, Zenodo | Video |
| Deep learning | Drive, Zenodo | Video |
| Transformers and language models | Drive, Zenodo | Video |
| Protein structure | Drive, Zenodo | Video |
| Protein function | Drive, Zenodo | Video |
| Protein evolution | Drive, Zenodo | Video |
| AlphaFold | Drive, Zenodo | Video |
| AI-based protein design | Drive, Zenodo | Video |
| Data and bias | Drive, Zenodo | Video |

By releasing these slides, my goal is to provide access to information for deeper learning. If you are a teacher and have adopted this material for your lectures, please let me know. I would love to learn how you improved the course and to know that more people have learned about protein science.

However, if you identify that someone has plagiarized this course in whole or in part **and is charging money for access**, I would appreciate being notified, as developing this material required a lot of time and effort, and plagiarism is a serious breach of professionalism and ethics.

# How to support this project

If you found this course useful and would like to support it financially, you can donate via PayPal. Donations can be of any amount, or USD 12, 30, or 45 (suggestions based on the economic reality of students in Latin America). Click the image below to donate.

[![paypal](https://github.com/miangoar/AI-driven-protein-design/raw/main/img/paypal.jpg)](https://www.paypal.com/donate/?hosted_button_id=AG42EZTZW9AJN)

If you do not have financial flexibility, but would like to express your gratitude, you can send me your comments by email: gamamiguelangel@gmail.com

Finally, I would appreciate it if you shared this course with interested colleagues or reposted the official course announcement on social media (to be published once all videos are available):

- [(X)Twitter](https://x.com/miangoar)
- [Instagram](https://www.instagram.com/miangoar/#)
- [BlueSky](https://bsky.app/profile/miangoar.bsky.social)
- [LinkedIn](https://www.linkedin.com/in/miguel-angel-gonzalez-arias-61b04b2ba/)

# About me

I'm Miguel Angel González Arias. I'm a Mexican biologist interested in proteins, microbes, and computation. For more details about me, my social networks, and other contact information, please visit the following page:

* [About me](https://miangoaren.github.io/talks/)


  











