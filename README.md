# ciencia-de-proteinas-basada-en-IA

El objetivo de este repositorio es compartir a la comunidad hispanohablante una serie de recursos que seran de utilidad en el aprendizaje de la ciencia de proteinas basada en inteligencia artificial. Para ello se ofrece una serie de tutoriales que pueden ser ejecutados en la nube de Google Colab, una serie de videos (aun por grabar) con clases para comprender la evolucion de las proteinas y las  bases de como funcionan los algoritmos de inteligencia artificial que estan generando una revolucion en el area. Finalmente, se ofrecen un conjunto de herramientas utiles en la practica junto con otros recursos para continuar aprendiendo del tema.  



Nota: texto sin acentos

# Videos recomendados
| Tema | Descripcion | Link a Youtube|
|-----------|-----------|-----------|  
|Introduccion a la metagenomica |Reconstruccion de genomas a partir de metagenomas|https://youtu.be/ckIbT93Qhjc| 
|AlphaFold y el Gran Desafío para resolver el plegamiento de proteínas |Breve revision de los algoritmos implementados en AlphaFold y AlphaFold2|https://youtu.be/nGVFbPKrRWQ| 
|Schrödinger y la biología: 75 años del libro ¿Qué es la vida? |Serie de conferencias para comprender el fenomeno d continguencia en evolucion asi como las contribuciones de varios cientificos a la consolidacion de la biologia molecular |Parte 1) https://www.youtube.com/live/XSWqcksA5vg?feature=share&t=711 <br> Parte 2) https://www.youtube.com/live/x35aQO8ifzM?feature=share&t=675 <br> Parte 3) https://www.youtube.com/live/PgbLyOYHEm4?feature=share&t=751| 
|Simbiosis y evolución | Serie de conferencias sobre los niveles de organizacion en biologia con especial enfasis en la simbiosis | Parte 1) https://www.youtube.com/watch?v=dF2GXGcTer8&ab_channel=elcolegionacionalmx <br> Parte 2) https://www.youtube.com/live/PfiZZaa_7BA?feature=share <br> Parte 3) https://www.youtube.com/watch?v=AZT7rf0KNeo&ab_channel=elcolegionacionalmx| 
|La evolución de las proteínas | Charla de introduccion respecto a la evolucion de proteinas|https://youtu.be/HFQqB27Uvbg| 
|Emile Zuckerkandl y el nacimiento de la evolución molecular | Charla de introduccion a la evolucion molecular |[https://youtu.be/ckIbT93Qhjc](https://www.youtube.com/watch?v=qCLgEnSUUmc&ab_channel=elcolegionacionalmx)| 
|Boston Protein Design and Modeling Club | Serie de seminarios sobre los trabajos mas actuales en ingenieria de proteinas |https://www.bpdmc.org/| 
|ML Protein Engineering Seminar Series |Serie de seminarios sobre los trabajos mas actuales en ingenieria de proteinas basado en machine learning |https://www.ml4proteinengineering.com/| 


# Tutoriales
| Notebook | Descripcion | Link a Google Colab|
|-----------|-----------|-----------|  
| [Analisis_estructural_ESMAtlas](https://github.com/miangoar/ciencia-de-proteinas-basada-en-IA/blob/main/notebooks/analisis_estructural_ESMAtlas.ipynb) | En este cuaderno aprenderas como explorar la base de datos ESMAtlas, descargar los resultados, procesarlos y agrupar estructuras de proteinas | [![Open In Colab](https://github.com/miangoar/ciencia-de-proteinas-basada-en-IA/blob/main/img/colab_logo.svg)](https://colab.research.google.com/github/miangoar/ciencia-de-proteinas-basada-en-IA/blob/main/notebooks/analisis_estructural_ESMAtlas.ipynb) |
| [sequentially_ESMFold](https://github.com/miangoar/ciencia-de-proteinas-basada-en-IA/blob/main/notebooks/sequentially_ESMFold.ipynb) | En este cuaderno aprenderas como realizar un video del proceso de plegamiento de una proteinas usando ESMFold | [![Open In Colab](https://github.com/miangoar/ciencia-de-proteinas-basada-en-IA/blob/main/img/colab_logo.svg)](https://colab.research.google.com/github/miangoar/ciencia-de-proteinas-basada-en-IA/blob/main/notebooks/sequentially_ESMFold.ipynb) |
| [Fine tunning of a protein language model. Notebook por Matthew Carrigan - @carrigmat en Twitter](https://github.com/miangoar/ciencia-de-proteinas-basada-en-IA/blob/main/notebooks/protein_language_modeling_by_Matthew_Carrigan.ipynb) | En este cuaderno aprenderas como  re-entrenar (fine-tunning) un lenguaje de proteinas y usarlos para realizar tareas de prediccion a nivel de secuencia y residuo | [![Open In Colab](https://github.com/miangoar/ciencia-de-proteinas-basada-en-IA/blob/main/img/colab_logo.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_language_modeling-tf.ipynb) |
| [ESMFold_batch](https://github.com/miangoar/ciencia-de-proteinas-basada-en-IA/blob/main/notebooks/ESMFold_batch.ipynb) | En este cuaderno aprenderas como predecir estructruas de proteinas por lotes usando ESMFold | [![Open In Colab](https://github.com/miangoar/ciencia-de-proteinas-basada-en-IA/blob/main/img/colab_logo.svg)](https://colab.research.google.com/github/miangoar/ciencia-de-proteinas-basada-en-IA/blob/main/notebooks/ESMFold_batch.ipynb) |


# Repositorios utiles  
| Repo | Descripcion | 
|-----------|-----------| 
| [biolists/folding_tools](https://github.com/biolists/folding_tools) | Listado de herramientas para analisis de proteinas basado en IA |
| [hefeda/design_tools](https://github.com/hefeda/design_tools) | Listado de herramientas para analisis de proteinas basado en IA  |
| [duerrsimon/folding_tools](https://github.com/duerrsimon/folding_tools/) | Listado de herramientas para analisis de proteinas basado en IA  |
| [AlphaFold2 mini](https://twitter.com/sokrypton/status/1535857255647690753) | Una version optimizada de AF2 para correr en segundos las predicciones a costa de perder un poco de calidad (se recomienda usarlo solo para fines de aprendizaje del plegamiento de las proteinas) |
| [Paper for protein desing with deep learning ](https://github.com/Peldom/papers_for_protein_design_using_DL) | Una lista curada de todos los trabajos de diseño de proteinas basado en aprendisaje profundo  |
| [ColabFold](https://github.com/sokrypton/ColabFold) | Implementacion de AF2 en la nube de Google colab  |
| [ColabFold Downloads](https://colabfold.mmseqs.com/) | Bases de datos de proteinas (siendo colabfold_envdb_202108 la mas grande, i.e. pesa 110GB)  |



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
| [MolecularNodes](https://github.com/BradyAJohnston/MolecularNodes) | Visualizacion de proteinas con calidad profesional usando Blender |
| [ProDy](http://prody.csb.pitt.edu/tutorials/) | Suite de herramientas de analisis de proteinas |
| [PyPDB](https://github.com/williamgilpin/pypdb) | API para interactuar con la PDB con python |
| [pdb-tools](https://github.com/haddocking/pdb-tools) | Herramientas para procesamiento y analisis de archivos .pdb |
| [seqlike](https://github.com/modernatx/seqlike) | Manipulacion y representacion de secuencias de proteinas |
| [BioNumpy](https://github.com/bionumpy/bionumpy/) | Manipulacion y representacion de secuencias usando numpy |

# Webservers basados en IA  
| Web | Descripcion | 
|-----------|-----------| 
| [ProteInfer](https://google-research.github.io/proteinfer/) | predicting the functional properties of a protein from its amino acid sequence using neural networks |
| [GoPredSim](https://embed.protein.properties/) | Predict protein properties from embeddings |
| [ProteinMPNN](https://huggingface.co/spaces/simonduerr/ProteinMPNN) | Fixed backbone design |
| [DeepFRI](https://beta.deepfri.flatironinstitute.org/) | structure-based protein function prediction (and functional residue identification) method using Graph Convolutional Networks with Language Model features |
| [Amber relaxation](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/beta/relax_amber.ipynb) | Usa AMBER para relajar tus estructruas de proteinas y evitar clashes estericos|



# Paginas web para aprender sobre las bases de machine learning   
| Web | Descripcion | 
|-----------|-----------| 
| [MLU-EXPLAIN](https://mlu-explain.github.io/) | Blogs interactivos de conceptos y algortimos base de machine learning |
| [Seeing-Theory](https://seeing-theory.brown.edu/) | Una introduccion a probabilidad y estadistica con animaciones didacticas  |
| [Distill publications](https://distill.pub/) | Blogs interactivos sobre algoritmos de machine y deep learning |
| [Neural Network SandBox](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.05854&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false) | Blog para comprender las bases del funcionamiento de las redes neuronales |
| [Stats illustrations](https://github.com/allisonhorst/stats-illustrations) | Ilustraciones para comprender conecptos base de estadistica |
| [3Blue1Brown ](https://www.youtube.com/@3blue1brown) | Videos con interpretaciones graficas sobre conceptos matematicos |
| [cpu.land](https://cpu.land/) | Una breve introduccion al computo en CPUs |


# Cursos recomendados
| Tema | Descripcion | 
|-----------|-----------| 
| [Algoritmos en Bioinformática Estructural](https://eead-csic-compbio.github.io/bioinformatica_estructural/) | Biologia estructural|
| [Cloud-based Tutorials on Structural Bioinformatics](https://github.com/pb3lab/ibm3202) | Biologia estructural |
| [Bash/Linux](https://vinuesa.github.io/intro2linux/index.html) | Introduccion al interprete de linux, Bash |
| [LLM](https://txt.cohere.com/llm-university/) | Introduccion a los grandes modelos de lenguaje (Large Language Models; LLM) |
| [Guide to undestanding PDB data](https://www.rcsb.org/news/feature/646e671b1d621d75127a7a52) | Una compilacion de informacion capturada por los creadores de la PDB para aprender sobre el formato de los datos |
| [regexlearn](https://regexlearn.com/es) | Breve curso para aprender el uso de expresiones regulares |


# Literatura de revision recomendada
| Tema | Descripcion | Link a Youtube|
|-----------|-----------|-----------|  
|Celebrating 50 Years of Journal of Molecular Evolution | Serie de pequeñas revisiones sobre evolucion molecular, su modelado e implementaciones  |https://link.springer.com/journal/239/topicalCollection/AC_ad1951b211df6035aed9ade2172865c4| 


## Agradecimientos 
- A [Lorenzo Segovia](https://www.ibt.unam.mx/perfil/3432/dr-lorenzo-patrick-segovia-forcella), [Alejandro Garciarrubio](https://www.ibt.unam.mx/perfil/1956/dr-alejandro-angel-garciarubio-granados) y [Jose A. Farias](https://www.fariaslab.org/)
- A [Xaira Rivera](https://github.com/xairigu) y Alejandro Alarcon 
