Protein Sequence Databases
uniprot
interpro
uniclust
lukprot
annotree
Dayhoff Atlas

Genome & Metagenome Databases
gtdb
AllTheBacteria
bakrep
Logan
OMG, Open MetaGenomic Dataset
gbrap
annoview

Viral-specific Databases
vire
bfvd
viro3d
Viral AlphaFold Database (VAD)
ViralZone
Unified Human Gut Virome Catalog

Protein Structure Databases (Experimental & Predicted)
PDB
alphafold DB
esmatlas
ModelArchive
CAZyme3D
afesm
alphasync
EcoFoldDB
MONDE⋅T

Domain & Fold Databases
Pfam
ECOD
cath DB
RepeatsDB
TED
TED: The Encyclopedia of Domains

Designed & Synthetic Proteins
PDA
revenant
fuzzle

Molecular Dynamics & Conformational Data
MD Repo
ATLAS
DynaRepo

Functional & Fitness Databases
M-CSA
Enzyme Engineering Database
protabank
mavedb
SAIR

Embedding & Representation Databases
Protein Dimension DB
Protein Embeddings
globdb
DPEB

ML Training Resources & Large-scale Sets
OpenProteinSet
ColabFold Downloads
AFDB foldseek clusters
Benchmarks & Evaluation Suites
PoseBench
plinder
pinder
Runs N' Poses
proteingym
MotifBench
posebusters
PDBench
RNAGym

Curated Lists, Tools & Ecosystem
ProteinDesignLabs
biolists/folding_tools
hefeda/design_tools
duerrsimon/folding_tools
Prtein language models
biodiffusion
Paper for protein desing with deep learning
Articulos curados por categoria
biomodes
List of papers about Proteins Design using Deep Learning
Awesome Bio-Foundation Models

Industry & Meta-resources
TechBio Company Database

General ML Datasets
pmlb
faker

# Sequence Databases
| Repo | Descripcion | 
|-----------|-----------| 
| [uniprot](https://www.uniprot.org/) | annotated proteins |
| [interpro](https://www.ebi.ac.uk/interpro/) |   functional analysis of proteins by classifying them into families and predicting domains and important sites |
| [uniclust](https://gwdu111.gwdg.de/%7Ecompbiol/uniclust/2023_02/) | clustered uniprot - [paper](https://academic.oup.com/nar/article/45/D1/D170/2605730) |
| [gtdb](https://gtdb.ecogenomic.org/) | microbial genomes |
| [Clustering NCBI's nr database](https://github.com/Arcadia-Science/2023-nr-clustering) | clustered nr at 90% length, 90% identity |
| [huggingface-task](https://huggingface.co/tasks) | a collection of differente ML task and their respective models on HuggingFace |
| [ProteinDesignLabs](https://github.com/Zuricho/ProteinDesignLabs) | List of computational protein design research labs |
| [lukprot](https://zenodo.org/records/13829058) |  eukaryotic predicted proteins based on EukProtDB |
| [AllTheBacteria](https://github.com/AllTheBacteria/AllTheBacteria) |  All WGS isolate bacterial INSDC data to August 2024 uniformly assembled, QC-ed, annotated, searchable |
| [bakrep](https://bakrep.computational.bio/) |  661,402 bacterial genomes consistently processed & characterized, enriched with metadata |
| [TechBio Company Database](https://harrisbio.substack.com/p/the-techbio-company-database) |  companies in the TechBio |
| [MD Repo](https://mdrepo.org/) | MD simulations for proteins, with or without ligands  |
| [OpenProteinSet](https://registry.opendata.aws/openfold/) |  >16 million MSAs with their associated structural homologs from the PDB, and AF2 protein structure predictions |
| [PlasmidScope](https://plasmid.deepomics.org/) |   852,600 plasmid sequences with annotations and structures |
| [Logan](https://github.com/IndexThePlanet/Logan) | a dataset of DNA and RNA sequences derived from NCBI-SRA which contains 50 petabases of public raw data  |
| [annotree](http://annotree.uwaterloo.ca/annotree/) | >280M proteins from the GTDB with functional annotations  |
| [annoview](http://annoview.uwaterloo.ca/annoview/) | genome visualization and exploration of gene neighborhoods  |
| [gbrap](http://tacclab.org/gbrap/) | carefully curated, high-quality genome statistics for all the organisms available in the RefSeq containing more than 200 columns of useful genomic information (Base counts, GC content, Shannon Entropy, Codon Usage etc.)  |
| [OMG, Open MetaGenomic Dataset](https://github.com/TattaBio/OMG) | 3.1T base pair metagenomic pretraining dataset, combining MGnify and IMG databases with translated amino acids for protein coding sequences, and nucleic acids for intergenic sequences |
| [Dayhoff Atlas](https://huggingface.co/datasets/microsoft/Dayhoff) | GigaRef = 3.34B natural protein sequences (1.7B clusters); BackboneRef = 46M synthetic sequences; OpenProteinSet =  16M MSAs unrolled |
| [vire](https://spire.embl.de/vire/) | 1.7M viral genomes recovered from >100k metagenomes from diverse ecosystems that contains >89M proteins |



# Structural Databases
| Repo | Descripcion | 
|-----------|-----------| 
| [alphafold DB](https://alphafold.ebi.ac.uk/) | predicted structures for 200M proteins from the uniprot  |
| [esmatlas](https://esmatlas.com/) | predicted structures for 600M proteins from MGnify  |
| [PDB](https://www.rcsb.org/stats/) | 200k protein structures |
| [PDA](https://pragmaticproteindesign.bio.ed.ac.uk/pda/) | ~1400 de novo designed proteins |
| [ModelArchive](https://modelarchive.org/) | ~620k structure models that are not based on experimental data  |
| [viro3d](https://viro3d.cvr.gla.ac.uk/) | 85k structural models from more >4.4k human and animal viruses  |
| [CAZyme3D  ](https://pro.unl.edu/CAZyme3D/) | 870k AlphaFold predicted 3D structures  |
| [ted](https://ted.cathdb.info/) |  The Encyclopedia of Domains [see also how was implemented in the AFDB](https://www.ebi.ac.uk/about/news/updates-from-data-resources/alphafold-database-ted/)|
| [ColabFold Downloads](https://colabfold.mmseqs.com/) | Bases de datos de proteinas (siendo colabfold_envdb_202108 la mas grande, i.e. pesa 110GB)  |
| [RepeatsDB](https://repeatsdb.org/home) |  annotation and classification of structural tandem repeat proteins  |
| [bfvd](https://bfvd.steineggerlab.workers.dev/) |  DB of protein structures from viruses |
| [revenant](https://revenant.inf.pucp.edu.pe/) | resurrected proteins structures |
| [pdbtm](https://pdbtm.unitmp.org/) | transmembrane protein selection of the PDB  |
| [afesm](https://afesm.steineggerlab.workers.dev/) |  ~820M structural predictions annotated with biome, taxonomy, domains, etc |
| [ATLAS](https://www.dsimb.inserm.fr/ATLAS) | >190 standardized molecular dynamics simulations of protein structures  |
| [Viral AlphaFold Database (VAD)](https://data-sharing.atkinson-lab.com/vad/) | ~27,000 representative viral proteins modeled with AlphaFold2  |
| [ViralZone](https://www.nature.com/articles/s41586-024-07809-y#data-availability) | 67,715 eukaryotic virus proteins modeled with AlphaFold2 |
| [fuzzle](https://fuzzle.uni-bayreuth.de:8443/) | evolutionary related protein fragments with ligand infromation |
| [TED: The Encyclopedia of Domains](https://ted.cathdb.info/data) | 365 million domains from AFDB  |
| [unisite](https://github.com/quanlin-wu/unisite) | Cross-Structure Dataset and Learning Framework for End-to-End Ligand Binding Site Detection  |
| [SAIR](https://pub.sandboxaq.com/data/ic50-dataset) | 5.2M protein-ligand structures with associated activity data  |
| [DynaRepo](https://dynarepo.inria.fr/) |  macromolecular conformational dynamics comprising ∼450 complexes and ∼270 single-chain proteins |
| [EcoFoldDB](https://github.com/timghaly/EcoFoldDB) | Database and pipeline for protein structure-guided annotations of ecologically relevant functions  |
| [Unified Human Gut Virome Catalog](https://uhgv.jgi.doe.gov/) | 870k genomes, 1M protein sequence clusters and 56k representative predictes structures of viruses |
| [alphasync](https://alphasync.stjude.org/) | residue-level features for predicted proteomes of model organisms with AlphaFold2  | 
| [MONDE⋅T](https://mondet.tuebingen.mpg.de/) | curated set of 23,149 structures from PDB that contains 1,895 unique non-canonical amino acids  |
| []() |   |



# Sequence classiication 
| Repo | Descripcion | 
|-----------|-----------|
| [Pfam](https://www.ebi.ac.uk/interpro/entry/pfam/#table) | protein families sequence calssification (see also the [FTP host](https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/) & [training resources](https://pfam-docs.readthedocs.io/en/latest/training.html)) |
| []() |   |

# Structural classiication 
| Repo | Descripcion | 
|-----------|-----------|
| [ECOD](http://prodata.swmed.edu/ecod/) | a hierarchical classification of protein domains according to their evolutionary relationship |
| [cath DB](https://www.cathdb.info/version/v4_3_0/superfamily/3.40.710.10) | Base de datos de plegamientos proteicos. Ejemplo con la superfamilia de serinbetalactamasas|
| []() |   |

# Functional Databases
| Repo | Descripcion | 
|-----------|-----------|
| [M-CSA](https://www.ebi.ac.uk/thornton-srv/m-csa/browse/) | Mechanism and Catalytic Site Atlas |
| [Enzyme Engineering Database](https://enzengdb.org/) | sequence-function data from enzyme engineering campaigns  |
| [protabank](https://www.protabank.org/) | protein-fitness datasets |
| [mavedb](https://www.mavedb.org/) | protein-fitness datasets |
| []() |   |

# Embedding Databases
| Repo | Descripcion | 
|-----------|-----------|
| [Protein Dimension DB](https://github.com/pentalpha/protein_dimension_db) | Datasets with PLM embeddings, GO annotations and taxonomy representations for all proteins in Uniprot/Swiss-Prot  |
| [Protein Embeddings](https://www.uniprot.org/help/embeddings) | per-protein and per-residue embeddings using the ProtT5 model for UniProtKB/Swiss-Prot |
| [globdb](https://globdb.org/) | Derreplecated and annotated genomes derived from 14 DBs that represents ~838M protein sequences. Clustering them resulted in ~83M non-singleton clusters with available ProtT5-XL-U50 embeddings |
| [DPEB](https://github.com/deepdrugai/DPEB) | AlphaFold2, ESM2, ProtTrans embeddings for 22,043 human proteins |



# Benchmarks
| Repo | Descripcion | 
|-----------|-----------| 
| [PoseBench](https://github.com/BioinfoMachineLearning/PoseBench) | protein-ligand structure prediction methods |
| [plinder](https://www.plinder.sh/) |  protein-ligand interactions |
| [pinder](https://www.pinder.sh/) | protein-protein interactions   |
| [Runs N' Poses](https://github.com/plinder-org/runs-n-poses) | protein-ligand co-folding prediction  |
| [proteingym](https://proteingym.org/) | comparing the ability of models to predict the effects of protein mutations  |
| [MotifBench](https://github.com/blt2114/MotifBench) | motif-scaffolding problems  |
| [posebusters](https://github.com/maabuu/posebusters) |  checks for generated molecule poses |
| [PDBench](https://github.com/wells-wood-research/PDBench) | evaluating fixed-backbone sequence design algorithms  |
| [RNAGym](https://github.com/MarksLab-DasLab/RNAGym) | suite for RNA fitness and structure prediction  |
| []() |   |

https://harrisbio.substack.com/p/the-techbio-company-database

# Machine learning datasets
| Repo | Descripcion | 
|-----------|-----------| 
| [pmlb](https://github.com/EpistasisLab/pmlb) |  datasets for evaluating supervised algorithms |
| [faker](https://github.com/joke2k/faker) | generates fake data |
| []() |   |


# Repositorios utiles  
| Repo | Descripcion | 
|-----------|-----------| 
| [biolists/folding_tools](https://github.com/biolists/folding_tools) | Listado de herramientas para analisis de proteinas basado en IA |
| [hefeda/design_tools](https://github.com/hefeda/design_tools) | Listado de herramientas para analisis de proteinas basado en IA  |
| [Protein language models](https://github.com/biolists/folding_tools/blob/main/pLM.md) | Listado de modelos de lenguaje de proteinas |
| [biodiffusion](https://github.com/biolists/biodiffusion) | Listado de modelos de difusion especificos para diseño de proteinas  |
| [duerrsimon/folding_tools](https://github.com/duerrsimon/folding_tools/) | Listado de herramientas para analisis de proteinas basado en IA  |
| [AlphaFold2 mini](https://twitter.com/sokrypton/status/1535857255647690753) | Una version optimizada de AF2 para correr en segundos las predicciones a costa de perder un poco de calidad (se recomienda usarlo solo para fines de aprendizaje del plegamiento de las proteinas) |
| [Paper for protein desing with deep learning ](https://github.com/Peldom/papers_for_protein_design_using_DL) | Una lista curada de todos los trabajos de diseño de proteinas basado en aprendisaje profundo  |
| [ColabFold](https://github.com/sokrypton/ColabFold) | Implementacion de AF2 en la nube de Google colab  |
| [Articulos curados por categoria](https://github.com/Peldom/papers_for_protein_design_using_DL) | Una lista de articulos de ciencia de proteinas basada en IA  |
| [biomodes](https://abeebyekeen.com/biomodes-biomolecular-design/) | Una lista de Modelos de IA curados por categoria  |
| [List of papers about Proteins Design using Deep Learning](https://github.com/Peldom/papers_for_protein_design_using_DL) | Literatura curada |
| [PDB documentation](https://www.rcsb.org/docs/general-help/organization-of-3d-structures-in-the-protein-data-bank) | Descripcion de todo lo relacionado a la PDB y como se analizan las proteinas |
| [AFDB foldseek clusters](https://afdb-cluster.steineggerlab.workers.dev) | Datos sobre los ~2.3M clusters de estructuras |
| [Awesome Bio-Foundation Models](https://github.com/apeterswu/Awesome-Bio-Foundation-Models) | collection of awesome bio-foundation models, including protein, RNA, DNA, gene, single-cell, and so on |
| []() |   |

