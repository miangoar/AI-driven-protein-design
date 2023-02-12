# ciencia-de-proteinas-basada-en-IA

El objetivo de este repositorio es compartir a la comunidad hispanohablante una serie de recursos que seran de utilidad en el aprendisaje de la ciencia de proteinas basada en inteligencia artificial. Para ello se ofrece una serie de tutoriales que pueden ser ejecutados en la nube de Google Colab, una serie de videos (aun por grabar) con clases para comprender la evolucion de las proteinas y las  bases de como funcionan los algoritmos de inteligencia artificial que estan generando una revolucion en el area. Finalmente, se ofrecen un conjunto de herramientas utiles en la practica junto con otros recursos para continuar aprendiendo del tema.  



Nota: texto sin acentos


# Tutoriales
| Notebook | Descripcion | Link a Google Colab|
|-----------|-----------|-----------|  
| [Analisis_estructural_ESMAtlas](https://github.com/miangoar/ciencia-de-proteinas-basada-en-IA/blob/main/notebooks/analisis_estructural_ESMAtlas.ipynb) | En este cuaderno aprenderas como explorar la base de datos ESMAtlas, descargar los resultados, procesarlos y agrupar estructuras de proteinas | [![Open In Colab](https://github.com/miangoar/ciencia-de-proteinas-basada-en-IA/blob/main/img/colab_logo.svg)](https://colab.research.google.com/github/miangoar/ciencia-de-proteinas-basada-en-IA/blob/main/notebooks/analisis_estructural_ESMAtlas.ipynb) |
| [Fine tunning of a protein language model (By Matthew Carrigan @carrigmat en Twitter)]([https://github.com/miangoar/ciencia-de-proteinas-basada-en-IA/blob/main/notebooks/analisis_estructural_ESMAtlas.ipynb](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_language_modeling-tf.ipynb)) | En este cuaderno aprenderas como  re-entrenar (fine-tunning) un lenguaje de proteinas y usarlos para realizar tareas de prediccion a nivel de secuencia y residuo | [![Open In Colab](https://github.com/miangoar/ciencia-de-proteinas-basada-en-IA/blob/main/img/colab_logo.svg)]([https://colab.research.google.com/github/miangoar/ciencia-de-proteinas-basada-en-IA/blob/main/notebooks/analisis_estructural_ESMAtlas.ipynb](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_language_modeling-tf.ipynb)) |


# Repositorios utiles  
| Repo | Descripcion | 
|-----------|-----------| 
| [biolists/folding_tools](https://github.com/biolists/folding_tools) | Listado de herramientas para analisis de proteinas basado en IA |
| [hefeda/design_tools](https://github.com/hefeda/design_tools) | Listado de herramientas para analisis de proteinas basado en IA  |
| [duerrsimon/folding_tools](https://github.com/duerrsimon/folding_tools/) | Listado de herramientas para analisis de proteinas basado en IA  |
| [AlphaFold2 rapido](https://twitter.com/sokrypton/status/1535857255647690753) | Una version optimizada de AF2 para correr en segundos las predicciones a costa de perder un poco de calidad (se recomienda usarlo solo para fines de aprendisaje del plegamiento de las proteinas) |
| [ColabFold](https://github.com/sokrypton/ColabFold) | Implementacion de AF2 en la nube de Google colab  |


# Herramientas recomendadas
| Repo | Descripcion | 
|-----------|-----------| 
| [SeqKit](https://bioinf.shenwei.me/seqkit/) | Manipulacion de secuencias genomicas |
| [Diamond2](https://github.com/bbuchfink/diamond) | Blasteo de secuencias de proteinas a escala masiva |
| [MMseqs2](https://github.com/soedinglab/MMseqs2) | Blasteo de secuencias de proteinas a escala masiva |
| [FoldSeek](https://github.com/steineggerlab/foldseek) | Blasteo estructural de proteinas a escala masiva|
| [ProtLearn ](https://github.com/tadorfer/protlearn) | Codificacion de secuencias de proteinas en vectores para el entrenamiento de algoritmos de machine learning |
| [Pfeature](https://github.com/raghavagps/Pfeature) |Codificacion de secuencias de proteinas en vectores para el entrenamiento de algoritmos de machine learning  |
| [Graphein](https://github.com/a-r-j/graphein) | Codificacion de estructuras de proteinas en vectores usando  teoria de grafos |
| [PyUUL](https://pyuul.readthedocs.io/index.html) | Codificacion de estructuras de proteinas en vectores usando algoritmos de analisis 3D |
| [bio_embeddings](https://github.com/sacdallago/bio_embeddings) | Implementacion de varios lenguajes de proteinas  |
| [TRILL ](https://github.com/martinez-zacharya/TRILL) | Implementacion de varios lenguajes de proteinas  |
| [Graph-Part](https://github.com/graph-part/graph-part) |  Preparacion de datasets de secuencias de proteinas  |
| [LazyPredict](https://github.com/shankarpandala/lazypredict) | Comparacion automatica de varios algoritmos de clasificacion y regresion |


# Webservers basados en IA  
| Web | Descripcion | 
|-----------|-----------| 
| [ProteInfer](https://google-research.github.io/proteinfer/) | predicting the functional properties of a protein from its amino acid sequence using neural networks |
| [GoPredSim](https://embed.protein.properties/) | Predict protein properties from embeddings |
| [ProteinMPNN]([https://embed.protein.properties/](https://huggingface.co/spaces/simonduerr/ProteinMPNN)) | Fixed backbone design |
| [DeepFRI](https://beta.deepfri.flatironinstitute.org/) | structure-based protein function prediction (and functional residue identification) method using Graph Convolutional Networks with Language Model features |
| [Amber relaxation]([https://embed.protein.properties/](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/beta/relax_amber.ipynb)) | Usa AMBER para relajar tus estructruas de proteinas y evitar clashes estericos|



# Paginas web para aprender sobre las bases de machine learning   
| Web | Descripcion | 
|-----------|-----------| 
| [MLU-EXPLAIN](https://mlu-explain.github.io/) | Visual explanations of core machine learning concepts |
| [Seeing-Theory](https://seeing-theory.brown.edu/) | Una introduccion a probabilidad y estadistica con animaciones didacticas  |
| [Distill publications](https://distill.pub/) | Blogs interactivos sobre algoritmos de machine y deep learning |
| [Neural Network SandBox](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.05854&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false) | Blog para comprender las bases del funcionamiento de las redes neuronales |
| [Stats illustrations](https://github.com/allisonhorst/stats-illustrations) | Iluestraciones para comprender coneptos base de estadistica |


# Cursos recomendados
| [Ciencia de Datos con Python](https://github.com/GAL-Repository/EDA_Stuff) | Introduccion a python, programacion orientada a objetos y machine learning |



