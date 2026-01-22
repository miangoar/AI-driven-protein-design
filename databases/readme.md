Data categories:
- [Protein sequence databases](#protein-sequence-databases)
- [Genome & metagenome databases](#genome--metagenome-databases)
- [Viral databases](#viral-databases)
- [Protein structure databases](#protein-structure-databases)
- [Domain & fold databases](#domain--fold-databases)
- [Designed & synthetic proteins](#designed--synthetic-proteins)
- [Molecular dynamics & conformational data](#molecular-dynamics--conformational-data)
- [Functional & fitness databases](#functional--fitness-databases)
- [Embedding databases](#embedding-databases)
- [Machine learning](#machine-learning)
- [Benchmarks](#benchmarks)
- [Interesting repositories](#interesting-repositories)

# Protein sequence databases
Databases of protein sequences with functional, taxonomic, or evolutionary annotations
| Name | Description | 
|-----------|-----------| 
| [uniprot](https://www.uniprot.org/) | high-quality, comprehensive and freely accessible protein sequences and functional information |
| [interpro](https://www.ebi.ac.uk/interpro/) | functional analysis of proteins by classifying them into families and predicting domains and important sites |
| [uniclust](https://uniclust.mmseqs.com/) | The Uniclust90, Uniclust50, Uniclust30 DBs cluster UniProtKB sequences at the level of 90%, 50% and 30% pairwise sequence identity. |
| [lukprot](https://zenodo.org/records/13829058) |  eukaryotic predicted proteins based on EukProtDB |
| [annotree](http://annotree.uwaterloo.ca/annotree/) | >280M proteins from the GTDB with functional annotations  |
| [Dayhoff Atlas](https://huggingface.co/datasets/microsoft/Dayhoff) | GigaRef = 3.34B natural protein sequences (1.7B clusters); BackboneRef = 46M synthetic sequences; OpenProteinSet =  16M unrolled MSAs |
| [Logan](https://github.com/IndexThePlanet/Logan) |  DNA and RNA sequences derived from NCBI-SRA which contains 50 petabases of public raw data  |
| [ColabFold Downloads](https://colabfold.mmseqs.com/) | databases to generate diverse MSAs to predict protein structures  |
| [Clustering NCBI's nr database](https://github.com/Arcadia-Science/2023-nr-clustering) | clustered nr at 90% length, 90% identity |


# Genome & metagenome databases
Databases of assembled genomes and metagenomic data from diverse organisms and environments
| Name | Description | 
|-----------|-----------| 
| [GTDB](https://gtdb.ecogenomic.org/) | microbial genomes |
| [AllTheBacteria](https://github.com/AllTheBacteria/AllTheBacteria) |  All WGS isolate bacterial INSDC data to August 2024 uniformly assembled, QC-ed, annotated, searchable |
| [bakrep](https://bakrep.computational.bio/) |  661,402 bacterial genomes consistently processed & characterized, enriched with metadata |
| [OMG, Open MetaGenomic Dataset](https://github.com/TattaBio/OMG) | 3.1T base pair metagenomic pretraining dataset, combining MGnify and IMG databases with translated amino acids for protein coding sequences, and nucleic acids for intergenic sequences |
| [gbrap](http://tacclab.org/gbrap/) | carefully curated, high-quality genome statistics for all the organisms available in the RefSeq containing more than 200 columns of useful genomic information (Base counts, GC content, Shannon Entropy, Codon Usage etc.)  |
| [annoview](http://annoview.uwaterloo.ca/annoview/) | genome visualization and exploration of gene neighborhoods  |
| [PlasmidScope](https://plasmid.deepomics.org/) |   852,600 plasmid sequences with annotations and structures |

# Viral databases
Databases for sequence and structures derived from viruses
| Name | Description | 
|-----------|-----------| 
| [bfvd](https://bfvd.steineggerlab.workers.dev/) |  DB of protein structures from viruses |
| [viro3d](https://viro3d.cvr.gla.ac.uk/) | 85k structural models from more >4.4k human and animal viruses  |
| [Viral AlphaFold Database (VAD)](https://data-sharing.atkinson-lab.com/vad/) | ~27,000 representative viral proteins modeled with AlphaFold2  |
| [ViralZone](https://www.nature.com/articles/s41586-024-07809-y#data-availability) | 67,715 eukaryotic virus proteins modeled with AlphaFold2 |
| [vire](https://spire.embl.de/vire/) | 1.7M viral genomes recovered from >100k metagenomes from diverse ecosystems that contains >89M proteins |
| [Unified Human Gut Virome Catalog](https://uhgv.jgi.doe.gov/) | 870k genomes, 1M protein sequence clusters and 56k representative predicted structures of viruses |

# Protein structure databases
Databases with experimental and computationally predicted protein structures
| Name | Description | 
|-----------|-----------| 
| [alphafold DB](https://alphafold.ebi.ac.uk/) | predicted structures for 200M proteins from the uniprot  |
| [esmatlas](https://esmatlas.com/) | predicted structures for 600M proteins from MGnify  |
| [PDB](https://www.rcsb.org/stats/) | 250k protein structures |
| [ModelArchive](https://modelarchive.org/) | ~620k structure models that are not based on experimental data  |
| [afesm](https://afesm.steineggerlab.workers.dev/) |  ~820M structural predictions annotated with biome, taxonomy, domains, etc |
| [CAZyme3D](https://pro.unl.edu/CAZyme3D/) | 870k AlphaFold predicted 3D structures  |
| [TED: The Encyclopedia of Domains](https://ted.cathdb.info/) | 365 million domains from AFDB ([see also how was implemented in the AFDB](https://www.ebi.ac.uk/about/news/updates-from-data-resources/alphafold-database-ted/))|
| [alphasync](https://alphasync.stjude.org/) | residue-level features for predicted proteomes of model organisms with AlphaFold2  | 
| [MONDE⋅T](https://mondet.tuebingen.mpg.de/) | curated set of 23,149 structures from PDB that contains 1,895 unique non-canonical amino acids  |
| [EcoFoldDB](https://github.com/timghaly/EcoFoldDB) | protein structure-guided annotations of ecologically relevant functions  |
| [AFDB foldseek clusters](https://afdb-cluster.steineggerlab.workers.dev) | 2.27M non-singleton structural clusters derived from AFDB |
| [pdbtm](https://pdbtm.unitmp.org/) | transmembrane protein selection of the PDB  |

# Domain & fold databases
Databases that classify protein domains and folds
| Name | Description | 
|-----------|-----------| 
| [Pfam](https://www.ebi.ac.uk/interpro/entry/pfam/#table) | protein families sequence classification (see also the [FTP host](https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/) & [training resources](https://pfam-docs.readthedocs.io/en/latest/training.html)) |
| [ECOD](http://prodata.swmed.edu/ecod/) | a hierarchical classification of protein domains according to their evolutionary relationship |
| [cath DB](https://www.cathdb.info/version/v4_3_0/superfamily/3.40.710.10) | Database of protein folds, illustrated with the serine beta-lactamase superfamily |
| [RepeatsDB](https://repeatsdb.org/home) |  annotation and classification of structural tandem repeat proteins  |

# Designed & synthetic proteins
Databases of engineered proteins
| Name | Description | 
|-----------|-----------| 
| [fuzzle](https://fuzzle.uni-bayreuth.de:8443/) | evolutionary related protein fragments with ligand information |
| [revenant](https://revenant.inf.pucp.edu.pe/) | resurrected proteins structures |
| [PDA](https://pragmaticproteindesign.bio.ed.ac.uk/pda/) | ~1400 de novo designed proteins |

# Molecular dynamics & conformational data
Molecular dynamics simulations and protein conformational ensembles
| Name | Description | 
|-----------|-----------| 
| [ATLAS](https://www.dsimb.inserm.fr/ATLAS) | >190 standardized molecular dynamics simulations of protein structures  |
| [MD Repo](https://mdrepo.org/) | MD simulations for proteins, with or without ligands  |
| [DynaRepo](https://dynarepo.inria.fr/) |  macromolecular conformational dynamics comprising ∼450 complexes and ∼270 single-chain proteins |

# Functional & fitness databases
Datasets linking protein sequence or structure to function, activity, or fitness measurements
| Name | Description | 
|-----------|-----------| 
| [M-CSA](https://www.ebi.ac.uk/thornton-srv/m-csa/browse/) | Mechanism and Catalytic Site Atlas |
| [Enzyme Engineering Database](https://enzengdb.org/) | sequence-function data from enzyme engineering campaigns  |
| [protabank](https://www.protabank.org/) | protein-fitness datasets |
| [mavedb](https://www.mavedb.org/) | protein-fitness datasets |
| [SAIR](https://pub.sandboxaq.com/data/ic50-dataset) | 5.2M protein-ligand structures with associated activity data  |
| [unisite](https://github.com/quanlin-wu/unisite) | Cross-Structure Dataset and Learning Framework for End-to-End Ligand Binding Site Detection  |


# Embedding databases
Protein embeddings from protein language models
| Name | Description | 
|-----------|-----------| 
| [Protein Dimension DB](https://github.com/pentalpha/protein_dimension_db) | Datasets with PLM embeddings, GO annotations and taxonomy representations for all proteins in Uniprot/Swiss-Prot  |
| [Protein Embeddings](https://www.uniprot.org/help/embeddings) | per-protein and per-residue embeddings using the ProtT5 model for UniProtKB/Swiss-Prot |
| [globdb](https://globdb.org/) | Derreplecated and annotated genomes derived from 14 DBs that represents ~838M protein sequences. Clustering them resulted in ~83M non-singleton clusters with available ProtT5-XL-U50 embeddings |
| [DPEB](https://github.com/deepdrugai/DPEB) | AlphaFold2, ESM2, ProtTrans embeddings for 22,043 human proteins |

# Machine learning
ML models
| Name | Description | 
|-----------|-----------| 
| [huggingface-task](https://huggingface.co/tasks) | ML tasks and their respective models on HuggingFace |

# Benchmarks
Standardized datasets for comparing ML methods
| Name | Description | 
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


# Interesting repositories
Tools, models, labs, companies, and literature in protein science and AI
| Name | Description | 
|-----------|-----------| 
| [ProteinDesignLabs](https://github.com/Zuricho/ProteinDesignLabs) | List of computational protein design research labs |
| [TechBio Company Database](https://harrisbio.substack.com/p/the-techbio-company-database) |  50+ TechBio companies |
| [biolists/folding_tools](https://github.com/biolists/folding_tools) | Curated list of AI-based tools for protein structure prediction and analysis |
| [hefeda/design_tools](https://github.com/hefeda/design_tools) | Curated list of AI-based tools for protein analysis and design |
| [Protein language models](https://github.com/biolists/folding_tools/blob/main/pLM.md) | Curated list of protein language models |
| [biodiffusion](https://github.com/biolists/biodiffusion) | Curated list of diffusion-based models for protein design |
| [duerrsimon/folding_tools](https://github.com/duerrsimon/folding_tools/) | Curated list of AI-based tools for protein structure analysis |
| [Papers for protein design](https://github.com/Peldom/papers_for_protein_design_using_DL) | Curated collection of research papers on deep learning–based protein design |
| [biomodes](https://abeebyekeen.com/biomodes-biomolecular-design/) | Curated list of biomolecular design models, including protein-focused AI methods |
| [Awesome Bio-Foundation Models](https://github.com/apeterswu/Awesome-Bio-Foundation-Models) | Curated collection of bio-foundation models covering proteins, RNA, DNA, genes, single-cell data, and related domains |

