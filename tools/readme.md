# Content list
1. [Protein sequences](#sequences)
2. [Protein structures](#structures)
3. [Structure prediction](#structure-prediction)
4. [Molecular dynamics](#molecular-dynamics)
5. [Representation learning](#representation-learning)
6. [Protein engineering](#protein-engineering)
7. [Generative AI with difussion models](#generative-AI-with-difussion-models)
8. [Generative AI with LLM](#generative-AI-with-LLM)
9. [Docking (ligands and proteins)](#docking-ligands-and-proteins)
10. [Molecules](#molecules)
11. [Machine learning](#machine-learning)
12. [Statistics](#statistics)
13. [Datavis for bio](#datavis-for-bio)
14. [Datavis](#datavis)
15. [Webservers](#webservers)
16. [Chatbots and agents](#chatbots-and-agents)

Note: resources marked with ⭐ are highly recommended



# Protein design & engineering
PyRosetta
GraphRelax
SCHEMA-RASPP
ProtLego
unicore
PAthreader
FoldPAthreader
ConservFold
consurf
AlphaPulldown
ColabDock
ColabDesign
LazyAF
CombFold
Cfold
Replacement Scan
protein_scoring
AF_unmasked
AncFlow
TRILL
AF2BIND
PyPEF
DeepProtein
FlexMol
ByProt
scikit-bio
BindCraft
FreeBindCraft
prosculpt
BinderFlow
proteindj
ovo
IPSAE
bagel
Protein Design Skills

# Representation learning & AI for proteins
FAESM
ESM-Efficient
ProtLearn
Pfeature
bio_embeddings
Graph-Part
ProteinFlow
docktgrid
Prop3D
SaProt
ProstT5
Graphein
PyUUL
colav
ProTrek
masif
peptidy
pypropel
atomworks
ZymCTRL
REXzyme_aa
ProGen2-finetuning
Pinal
Evolla
ProtRL

# Function & interaction modeling
CLEAN
DeepFRI
interproscan
HiQBind
InterfaceAnalyzerMover
PandaDock
ligysis
p2rank
PLACER
peppr


# Molecular simulation
making-it-rain
bioemu
orb
logMD
proprotein
packmol
mdanalysis


# Molecules & cheminformatics
rdkit
PDBe CCDUtils
PDBe Arpeggio
PDBe RelLig
MolPipeline
roshambo
molli
useful_rdkit_utils
deepchem
nvMolKit

Machine learning & statistics (general)
Colab forms
cuml
LazyPredict
TorchDR
Kerasify
pca
openTSNE
TabPFN
tabm
tabicl
setfit
skrub
cupy
Best-of Machine Learning
torchmetrics
DADApy
PySR
BERTopic
KeyBERT
PolyFuzz
hummingbird
skorch
Faiss
tmap
einops
pyod
autokeras
numba
langextract
cleanlab
dtype_diet
scikit-posthocs
statannotations
ggstatsplot
ggbetweenstats
statsmodels
pingouin
performance

11. Visualization & interfaces
MolecularNodes
CellScape
ChimeraX apps
chimerax_viridis
SSDraw
bioalphabet
ChatMol
plot_phylo
prettymol
VMD-2
gromacs_copilot
NIH bioart
bioicons
moldraw
Mol*
MolViewSpec
PoseEdit
FlatProt
quarto-molstar
alphabridge
weblogo
interprot
termal
py2Dmol
Nano Protein Viewer
Protein Viewer
molview
ProteinCHAOS
datamapplot
pypalettes
distinctipy
Visualize Architecture of Neural Network
tidyplots
pyCirclize
pycircular
great-table
plottable
d3blocks
morethemes
jsoncrack
torchvista
bivario

13. Data access, platforms & agents
RCSB API
ProteInfer
GoPredSim
protein structure relaxation
Model Memory Calculator
alphafind
DiffDock-Web
ESMFold
Foldseek clusters
damietta
easifa
MPI Bioinformatics Toolkit
moleculatio
AI in Biology Demos
ProteinsPlus
ChatGPT
Gemini
claude
Bing
HuggingChat
huggingface spaces
biologpt
consensus
typeset
mistral-chat
aistudio by Google
AI Python Libraries
paperfinder
AI Scientist agents
biomni

--------------------------------------
# Sequence-level analysis
Tools for manipulating biological sequences, building and processing MSAs, detecting homology, and performing phylogenetic analyses at the sequence level.
| Name | Description | 
|-----------|-----------| 
| [SeqKit](https://bioinf.shenwei.me/seqkit/) | FASTA/Q file manipulation (Check out this ([tutorial](https://sandbox.bio/tutorials/seqkit-intro))|
| [Diamond2](https://github.com/bbuchfink/diamond) | accelerated BLAST  |
| [MMseqs2](https://github.com/soedinglab/MMseqs2) | ultra fast and sensitive search and clustering suite (Check out the  [tutorial](https://github.com/soedinglab/MMseqs2/wiki/Tutorials) and the  [GPU implementation](https://github.com/soedinglab/MMseqs2/wiki#compile-from-source-for-linux-with-gpu-support)) |
| [seqlike](https://github.com/modernatx/seqlike) |  sequence manipulation |
| [BioNumpy](https://github.com/bionumpy/bionumpy/) | array programming on biological datasets |
| [pLM-BLAST](https://github.com/labstructbioinf/pLM-BLAST) | remote homology detection with protein language models |  
| [PLMSearch](https://github.com/maovshao/PLMSearch) | homologous protein search with protein language models |
| [LexicMap](https://github.com/shenwei356/LexicMap) | sequence alignment against millions of genomes |
| [pyfastx](https://github.com/lmdu/pyfastx) | fast random access to sequences from plain and gzipped FASTA/Q files |
| [any2fasta](https://github.com/tseemann/any2fasta) | Convert various sequence formats to FASTA |
| [Spacedust](https://github.com/soedinglab/Spacedust) | identification of conserved gene clusters among genomes based on homology and conservation of gene neighborhood |
| [ugene](https://ugene.net/) | genome analysis suite with graphic user interface |
| [BuddySuite](https://github.com/biologyguy/BuddySuite) | manipulating sequence, alignment, and phylogenetic tree files |

# Multiple sequence alignment
| Name | Description | 
|-----------|-----------| 
| [hh-suite](https://bioinf.shenwei.me/seqkit/) | remote homology detection  |
| [DeepMSA](https://zhanggroup.org/DeepMSA/) | create high-quality MSAs |
| [NEFFy](https://github.com/Maryam-Haghani/NEFFy) | calculating the Normalized Effective Number of Sequences (neff) for protein/nt MSAs. Also for format conversion |
| [ClipKIT](https://github.com/JLSteenwyk/ClipKIT) | a MSA-trimming algorithm for accurate phylogenomic inference |
| [PLMAlign](https://github.com/maovshao/PLMAlign) | create MSAs using per-residue embeddings from protein language models |
| [ClipKIT](https://github.com/JLSteenwyk/ClipKIT) | trimming algorithm for accurate phylogenomic inference and msa manipulation |
| [CIAlign](https://github.com/KatyBrown/CIAlign) | clean, interpret, visualise and edit MSAs |
| [TWILIGHT](https://github.com/TurakhiaLab/TWILIGHT) | ultrafast and ultralarge MSA |

# Structure-level analysis
Tools for parsing, cleaning, annotating, and extracting properties from 3D protein structures.
| Name | Description | 
|-----------|-----------| 
| [ProDy](http://prody.csb.pitt.edu/tutorials/) | protein structure, dynamics, and sequence analysis |
| [BioPandas](https://biopandas.github.io/biopandas/) | working with molecular structures in pandas |
| [foldcomp](https://github.com/steineggerlab/foldcomp) | compressing protein structures |
| [protestar](https://github.com/refresh-bio/protestar) | compress collections structures |
| [afpdb](https://github.com/data2code/afpdb) |  manipulation of protein structures in Python |
| [LocalPDB](https://github.com/labstructbioinf/localpdb) | manage structures and their annotations |
| [pdb-tools](https://github.com/haddocking/pdb-tools) | manipulating and editing PDB files |
| [pdbfixer](https://github.com/openmm/pdbfixer) | fixes problems in PDB files |
| [cath-tools](https://github.com/UCLOrengoGroup/cath-tools) | structure comparison tools |
| [Merizo-search](https://github.com/psipred/merizo_search) | domain structure embedding+search tool |
| [freesasa](https://github.com/mittinatten/freesasa) | calculating Solvent Accessible Surface Areas |
| [openstructure](https://openstructure.org/) | protein structure, complexes and docking comparison |
| [opendock](https://github.com/guyuehuo/opendock)| protein-Ligand Docking and Modeling |
| [pdb-redo](https://pdb-redo.eu/)| automated procedure to refine, rebuild and validate your models |
| [proteinshake](https://github.com/BorgwardtLab/proteinshake) | preprocessed and cleaned structure datasets |
| [GEMMI](https://github.com/project-gemmi/gemmi) | macromolecular crystallography library and utilities |
| [fpocket](https://github.com/Discngine/fpocket) | protein pocket detection based on Voronoi tessellation |
| [pocketeer](https://github.com/cch1999/pocketeer) | A lightweight, fast pocket finder |
| [dssp](https://github.com/PDB-REDO/dssp) | assign secondary structure to proteins (check out [the webserver](https://pdb-redo.eu/dssp)) |
| [PyDSSP](https://github.com/ShintaroMinami/PyDSSP) | PyTorch implementation of DSSP algorithm |
| [PDBCleanV2](https://github.com/fatipardo/PDBCleanV2) | create a curated ensemble of molecular structures |
| [ProLIF](https://github.com/chemosim-lab/ProLIF) | Interaction Fingerprints for protein-ligand complexes |
| [plip](https://github.com/pharmai/plip) | Analyze and visualize non-covalent protein-ligand and protein-protein interactions |
| [af_analysis](https://github.com/samuelmurail/af_analysis) | Analysis of alphafold and colabfold results |
| [InteracTor](https://github.com/Dias-Lab/InteracTor) |  structure analysis and conversion, allowing the extraction of molecular interactions (e.g.  Hbonds, van der Waals, hydrophobic contacts, and surface tension |
| [ProteinTools ](https://proteintools.uni-bayreuth.de/) | Analyze Hydrophobic Clusters, Hydrogen Bond Networks, Contact maps, Salt Bridges and Charge Segregation |
| [libraryPDB](https://github.com/CJ438837/libraryPDB) | searching, downloading, parsing, cleaning and analyzing protein structures |
| [lahuta](https://bisejdiu.github.io/lahuta/) | calculate atomomic interactions |

# Data access
| Name | Description | 
|-----------|-----------| 
| [iSeq](https://github.com/BioOmics/iSeq) | download data from sequence databases like GSA, SRA, ENA, and DDBJ |
| [NCBI datasets](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/) | download data from NCBI  |
| [ncbi-genome-download](https://github.com/kblin/ncbi-genome-download) | download data from the NCBI  |
| [ncbi-acc-download](https://github.com/kblin/ncbi-acc-download) | download data from NCBI Entrez by accession |
| [Pyrcsbsearchapi](https://github.com/rcsb/py-rcsbsearchapi) | Python interface for the RCSB PDB search API |
| [PyPDB](https://github.com/williamgilpin/pypdb) | python API for the PDB |
| [profet](https://github.com/alan-turing-institute/profet) | retrieves the cif or pdb files from either the PDB (using pypdb) or Alphafold using the Uniprot ID |
| [ActSeek](https://github.com/vttresearch/ActSeek) | mining in the AFDB based on the position of few amino acids |
| [Uniprot-PDB-mapper](https://github.com/iriziotis/Uniprot-PDB-mapper) | mapping of Uniprot sequences to PDB (see also this option to [map IDs between RefSeq and Uniprot](https://ncbiinsights.ncbi.nlm.nih.gov/2023/11/08/compare-ncbi-refseq-and-uniprot-datasets/) using the file `gene_refseq_uniprotkb_collab.gz`)|
| [PDBe API Training Notebooks](https://github.com/glevans/pdbe-api-training) | undestand how the PDBe REST API works |
| [RCSB API](https://github.com/rcsb/py-rcsb-api) | Python interface for RCSB PDB API services (check out the [guidelines and tutorials](https://pdb101.rcsb.org/news/684078fe300817f1b5de793a)) |
| [AFDB Structure Extractor](https://project.iith.ac.in/sharmaglab/alphafoldextractor/index.html) | download structures using AF IDs, Uniprot IDs, Locus tags, RefSeq Protein IDs and NCBI TaxIDs |

# Structure search & comparison
Tools for aligning, comparing, and searching protein structures to identify structural similarity, folds, or motifs.
| Name | Description | 
|-----------|-----------| 
| [Pairwise Structure Alignment Tool](https://www.rcsb.org/alignment) | webserver for structure alignment using PDB IDs, AFDB IDs or local files (check out the detailed description of [how to use the tool](https://www.rcsb.org/docs/tools/pairwise-structure-alignment#4-align-multiple-structures-to-an-alphafold-structure-))|
| [FoldSeek](https://github.com/steineggerlab/foldseek) | fast and sensitive comparisons of large structure sets|
| [folddisco](https://github.com/steineggerlab/folddisco) | indexing and search of discontinuous motifs |
| [USalign](https://github.com/pylelab/USalign) |structure alignment of monomeric and complex proteins and nucleic acids |
| [progres](https://github.com/greener-group/progres) | structure searching by structural embeddings (check out the [webserver](https://progres.mrc-lmb.cam.ac.uk/))|
| [pyScoMotif](https://github.com/3BioCompBio/pyScoMotif) | protein motif search |
| [pyRMSD](https://github.com/salilab/pyRMSD) | RMSD calculations of large sets of structures |
| [reseek](https://github.com/rcedgar/reseek) | structure alignment and search algorithm (check out the [webserver](https://reseek.online/))|
| [tmtools](https://github.com/jvkersch/tmtools) | Python bindings for the TM-align algorithm and code for protein structure comparison |
| [SoftAlign](https://github.com/jtrinquier/SoftAlign) | compare 3D protein structures |
| [foldmason](https://github.com/steineggerlab/foldmason) | Multiple Protein Structure Alignment at Scale |
| [pyjess](https://github.com/althonos/pyjess) | constraint-based structural template matching to identify catalytic residues from a known template |
| [gtalign](https://bioinformatics.lt/comer/gtalign/) | High-performance search and alignment for protein structures |
| [Muscle-3D](https://github.com/rcedgar/muscle) | multiple protein structure alignment |

# Structure prediction
Tools and pipelines for predicting protein structures from sequence, mainly based on AlphaFold and related methods.
| Name | Description | 
|-----------|-----------| 
| [Alphafold2](https://github.com/google-deepmind/alphafold)| protein structure prediction |
| [Alphafold3](https://github.com/google-deepmind/alphafold3) | predict biomolecular interactions using AlphaFold3 (check out the [webserver](https://alphafoldserver.com/) as well as this [solution to predict ~10k structures](https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/cloudnext25/examples/science/af3-slurm/README.md))|
| [ColabFold](https://github.com/sokrypton/ColabFold)| protein structure prediction on Google colab with a graphical user interface|
| [localcolabfold](https://github.com/YoshitakaMo/localcolabfold) | ColabFold on your local PC | 
| [flashfold](https://github.com/chayan7/flashfold) | command-line tool for faster protein structure prediction |
| [PAE Viewer](https://gitlab.gwdg.de/general-microbiology/pae-viewer) | view the predicted aligned error (PAE) of multimers, and integrates visualization of crosslink data (check out the [webserver](https://subtiwiki.uni-goettingen.de/v4/paeViewerDemo)) |
| [PyMOLfold](https://github.com/colbyford/PyMOLfold) | Plugin for folding sequences directly in PyMOL |
| [AFsample2](https://github.com/iamysk/AFsample2/) |  induce significant conformational diversity for a given protein |
| [alphafold3 tools](https://github.com/cddlab/alphafold3_tools) | Toolkit for input generation and output analysis |
| [af3cli](https://github.com/SLx64/af3cli) | generating AlphaFold3 input files |
| [RareFold](https://github.com/patrickbryant1/RareFold) | Structure prediction and design of proteins with 29 noncanonical amino acids |
| [AFusion](https://github.com/Hanziwww/AlphaFold3-GUI) | AlphaFold 3 GUI & Toolkit with Visualization |
| [Hackable AlphaFold 3](https://github.com/chaitjo/alphafold3/) | a lightweight, hackable way to run AF3 to experiment without the massive MSA databases or Docker overhead |
| [ABCFold](https://github.com/rigdenlab/ABCFold) | Scripts to run AlphaFold3, Boltz-1 and Chai-1 with MMseqs2 MSAs and custom templates  |


--------------------------------------------------------------------------------------

| [PyRosetta](https://github.com/RosettaCommons/PyRosetta.notebooks) | Rosetta suite for protein desing ported to python (See also these instructions for an [easy installation in Colab](https://x.com/miangoar/status/1835176497063030798) as well as the [documentation](https://graylab.jhu.edu/PyRosetta.documentation/index.html)) |
| [p2rank](https://github.com/rdk/p2rank) | Protein-ligand binding site prediction from protein structure |
| [PLACER](https://github.com/baker-laboratory/PLACER) |  local prediction of protein-ligand conformational ensembles |
| [peppr](https://github.com/aivant/peppr) | a package for evaluation of predicted poses like RMSD, TM-score, lDDT, lDDT-PLI, fnat, iRMSD, LRMSD, DockQ  |
| [unicore](https://github.com/steineggerlab/unicore) | core gene phylogeny with Foldseek and ProstT5 (i.e. 3Di alphabet) |
| [PAthreader and FoldPAthreader](https://github.com/iobio-zjut/PAthreader/tree/main/PAthreader_main) | PAthreader improve AF2 template selection by looking remote homologous in PDB/AFDB and FoldPAthreader predict the folding pathway  (see also [the webserver](http://zhanglab-bioinf.com/PAthreader/))  |
| [SCHEMA-RASPP](https://github.com/mattasmith/SCHEMA-RASPP) | structure-guided protein recombination (download and check the file [`schema-tools-doc.html`](https://github.com/mattasmith/SCHEMA-RASPP/blob/master/schema-tools-doc.html) for documentation)|
| [ProtLego](https://hoecker-lab.github.io/protlego/) | constructing protein chimeras and its structural analysis |
| [GraphRelax](https://github.com/delalamo/GraphRelax) | residue repacking and design   |
| []() |  |


# Phylogeny
| Name | Description | 
|-----------|-----------|
| [automlst2](https://automlst2.ziemertlab.com/index) | automatic generation of species phylogeny with reference organisms |
| [unicore](https://github.com/steineggerlab/unicore) | Universal and efficient core gene phylogeny with Foldseek and ProstT5  |
| [piqtree](https://github.com/iqtree/piqtree) | use IQ-TREE directly from Python |
| [torchtree](https://github.com/4ment/torchtree) | probabilistic framework in PyTorch for phylogenetic models |
| [fold_tree](https://github.com/DessimozLab/fold_tree) |  construct trees from protein structures |
| [3diphy](https://github.com/nmatzke/3diphy) | Maximum likelihood structural phylogenetics by including Foldseek 3Di characters |
| [PhyKIT](https://github.com/JLSteenwyk/PhyKIT) | toolkit for processing and analyzing MSAs and phylogenies |
| []() |  |



# Sequence generation
| Name | Description | 
|-----------|-----------|
| [ZymCTRL](https://huggingface.co/AI4PD/ZymCTRL) | conditional language model for the generation of artificial functional enzymes |
| [REXzyme_aa](https://huggingface.co/AI4PD/REXzyme_aa) | generate sequences that are predicted to perform their intended reactions |
| [ProGen2-finetuning](https://github.com/hugohrban/ProGen2-finetuning) | Finetuning ProGen2 for generation of sequences from selected families |
| [Pinal](https://github.com/westlake-repl/Denovo-Pinal) | Text-guided protein design |
| [Evolla](https://github.com/westlake-repl/Evolla) | chat about the function of a protein using its sequence and structure  (i.e. ChatGPT for proteins; see also the [webserver using the 10B param. version of the model](http://www.chat-protein.com/)) |
| [ProtRL](https://github.com/AI4PDLab/ProtRL) | Reinforcement Learning framework for autoregressive protein Language Models |
| []() |  |

# Structure generation
| Name | Description | 
|-----------|-----------|
| [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) |  structure generation, with or without conditional information (see also the [extended Documentation](https://sites.google.com/omsf.io/rfdiffusion) with a lot of descriptions and tutorials)|
| [RFdiffusion2](https://github.com/RosettaCommons/RFdiffusion2) | all-atom version at sub-angstrom resolution of RFdiffusion |
| [chroma](https://github.com/generatebio/chroma) | programmable protein design |
| [protein_generator](https://github.com/RosettaCommons/protein_generator) | Joint sequence and structure generation with RoseTTAFold sequence space diffusion |
| [RFdiffusion_all_atom](https://github.com/baker-laboratory/rf_diffusion_all_atom) | RFdiffusion with all atom modeling |
| [salad](https://github.com/mjendrusch/salad) | structure generation with sparse all-atom denoising models |
| [EnzymeFlow](https://github.com/WillHua127/EnzymeFlow) | generate catalytic pockets for specific substrates and catalytic reactions |
| [GENzyme](https://github.com/WillHua127/GENzyme) | design of catalytic pockets, enzymes, and enzyme-substrate complexes for any reaction |
| [BindCraft](https://github.com/martinpacesa/BindCraft) | binder design pipeline |
| [BoltzDesign1](https://github.com/yehlincho/BoltzDesign1) |  designing protein-protein interactions and biomolecular complexes |

# Inverse folding
| Name | Description | 
|-----------|-----------|
| [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) | Fixed backbone design ([see webserver](https://huggingface.co/spaces/simonduerr/ProteinMPNN))|
| [ProteinMPNN in JAX](https://github.com/sokrypton/ColabDesign/tree/main/mpnn) | Fast implementation of ProteinMPNN |
| [ligandMPNN](https://github.com/dauparas/LigandMPNN) | Fixed backbone design sensible to ligands ([see colab notebook](https://github.com/ullahsamee/ligandMPNN_Colab))|
| [SolubleMPNN](https://github.com/dauparas/ProteinMPNN/tree/main/soluble_model_weights) | Retrained version of ProteinMPNN by excluding transmembrane structures (see [Goverde et al. 2024](https://www.nature.com/articles/s41586-024-07601-y#Sec7) for more details)|
| [fampnn](https://github.com/richardshuai/fampnn) | full-atom version of ProteinMPNN |
| [LASErMPNN ](https://github.com/polizzilab/LASErMPNN) | All-Atom (Including Hydrogen!) Ligand-Conditioned Protein Sequence Design & Sidechain Packing Model |
| [HyperMPNN](https://github.com/meilerlab/HyperMPNN) | design thermostable proteins learned from hyperthermophiles |
| [ThermoMPNN](https://github.com/Kuhlman-Lab/ThermoMPNN) | predict changes in thermodynamic stability for protein point mutants |
| [ByProt](https://github.com/BytedProtein/ByProt) | Efficient non-autoregressive ProteinMPNN variant |
| [Caliby](https://github.com/ProteinDesignLab/caliby) | Potts model-based protein sequence design method that can condition on structural ensembles |
| []() |  |

# Function prediction/annotation
| Name | Description | 
|-----------|-----------|
| [CLEAN](https://github.com/tttianhao/CLEAN) | assign EC numbers to enzymes |
| [ProtNLM](https://colab.research.google.com/github/google-research/google-research/blob/master/protnlm/protnlm_use_model_for_inference_uniprot_2022_04.ipynb) | UniProt's Automatic Annotation pipeline  ([for mode details see](https://www.uniprot.org/help/ProtNLM)) |
| [DeepFRI](https://github.com/flatironinstitute/DeepFRI) | Deep functional residue identification |
| [interproscan](https://github.com/ebi-pf-team/interproscan) | interpro pipeline for functional annotation with multiple DBs |
| [ProtNLM](https://www.uniprot.org/help/ProtNLM) | UniProt's Automatic Annotation pipeline for protein sequences (see the [Colab notebook](https://colab.research.google.com/github/google-research/google-research/blob/master/protnlm/protnlm_evidencer_uniprot_2023_01.ipynb)) |
| []() |  |

# Sequence-structure co-generation
| Name | Description | 
|-----------|-----------|
| []() |  |


# molecular dynamics
| Name | Description | 
|-----------|-----------|
| [making-it-rain](https://github.com/pablo-arantes/making-it-rain) | Cloud-based molecular simulations for everyone |
| [bioemu](https://github.com/microsoft/bioemu) |  emulation of protein equilibrium ensembles  (see also this [this notebook](https://colab.research.google.com/github/pablo-arantes/making-it-rain/blob/main/BioEmu_HPACKER.ipynb) from ["Make it rain"](https://github.com/pablo-arantes/making-it-rain) that combines bioemu + [H-Packer](https://github.com/gvisani/hpacker) for side-chain reconstruction)| 
| [orb](https://github.com/orbital-materials/orb-models) | forcefield models from Orbital Materials |
| [logMD](https://github.com/log-md/logmd) | visualize MD trajectories in colab |
| [proprotein](https://proprotein.cs.put.poznan.pl/) | web server where, with a single click, the user can set up, configure, and run an MD simulation of the 3D structure of the peptide/protein |
| [packmol](https://github.com/m3g/packmol) | creates an initial point for MD simulations |
| [mdanalysis](https://github.com/MDAnalysis/mdanalysis) | analyze molecular dynamics simulations |
| []() |  |
| []() |  |


# representation learning
| Name | Description | 
|-----------|-----------|
| [FAESM](https://github.com/pengzhangzhi/faesm) | An Efficient Pytorch Implementation of ESM and ProGen PLM that can save up to 60% of memory usage and 70% of inference time | 
| [ESM-Efficient](https://github.com/uci-cbcl/esm-efficient) | Efficient implementatin of ESM family | 
| [ProtLearn](https://github.com/tadorfer/protlearn) | extracting protein sequence features |
| [Pfeature](https://github.com/raghavagps/Pfeature) | computing features of peptides and proteins |
| [bio_embeddings](https://github.com/sacdallago/bio_embeddings) | compute protein embeddings from sequences |
| [Graph-Part](https://github.com/graph-part/graph-part) | data partitioning method for ML |
| [ProteinFlow](https://github.com/adaptyvbio/ProteinFlow) | data processing pipeline fo ML |
| [docktgrid](https://github.com/gmmsb-lncc/docktgrid) | Create customized voxel representations of protein-ligand complexes |
| [Prop3D](https://github.com/bouralab/Prop3D) | toolkit for protein structure dataset creation and processing  |
| [SaProt](https://github.com/westlake-repl/SaProt) | Protein Language Model with Structural Alphabet (AA+3Di) (See also [ColabSaprot for structure-aware PLM](https://colab.research.google.com/github/westlake-repl/SaprotHub/blob/main/colab/SaprotHub_v2.ipynb) and [ColabSeprot for sequence-only PLM](https://colab.research.google.com/github/westlake-repl/SaprotHub/blob/main/colab/ColabSeprot.ipynb?hl=en)) |
| [ProstT5](https://github.com/mheinzinger/ProstT5) | Bilingual Language Model for Protein Sequence and Structure (see the [Foldseek adaptation](https://github.com/steineggerlab/foldseek?tab=readme-ov-file#structure-search-from-fasta-input)) |
| [Graphein](https://github.com/a-r-j/graphein) | geometric representations of biomolecules and interaction networks |
| [PyUUL](https://pyuul.readthedocs.io/index.html) | encode structures into differentiable data structures |
| [colav](https://github.com/Hekstra-Lab/colav) | feature extraction methods like dihedral angles, CA pairwise distances, and strain analysis |
| [ProTrek](https://github.com/westlake-repl/ProTrek) <br> [webserver](http://search-protrek.com/)| multimodal (sequence-structure-function) protein representations and annotations |
| [masif](https://github.com/LPDI-EPFL/masif) | molecular surface interaction fingerprints |
| [peptidy](https://github.com/molML/peptidy) | vectorize proteins for machine learning applications |
| [pypropel](https://github.com/2003100127/pypropel) | sequence and structural data preprocessing, feature generation, and post-processing for model performance evaluation and visualisation, |
| [atomworks](https://github.com/RosettaCommons/atomworks) | A generalized computational framework for biomolecular modeling |
| [mini3di](https://github.com/althonos/mini3di) | NumPy port of the foldseek code for encoding protein structures to 3di |
| []() |  |


# protein engineering
| Name | Description | 
|-----------|-----------|
| [biotite](https://www.biotite-python.org/latest/) | sequence and structure manipulation and analysis |
| [protkit](https://github.com/silicogenesis/protkit) | Unified Approach to Protein Engineering |
| [EvoProtGrad](https://github.com/NREL/EvoProtGrad) | directed evolution with MCMC and protein language models |
| [ConservFold](https://www.rodrigueslab.com/resources) | map amino acid conservation intro structures with the AF2 pipeline |
| [consurf](https://consurf.tau.ac.il/consurf_index.php) |identification of functionally important regions in proteins |
| [AlphaPulldown](https://www.embl-hamburg.de/AlphaPulldown/) | Complex moedeling with AF-Multimer |
| [ColabDock](https://github.com/JeffSHF/) | protein-protein docking |
| [ColabDesign](https://github.com/sokrypton/ColabDesign) | protein design pipelines |
| [LazyAF](https://github.com/ThomasCMcLean/LazyAF) | protein-protein interaction with AF2|
| [CombFold](https://github.com/dina-lab3D/CombFold) | structure predictions of large complexes |
| [Cfold](https://github.com/patrickbryant1/Cfold) | structure prediction of alternative protein conformations |
| [Replacement Scan](https://colab.research.google.com/github/sokrypton/ColabBio/blob/main/notebooks/replacement_scan.ipynb) | find how many amino acid replacements your protein can tolerate [see tw](https://x.com/sokrypton/status/1812769477228200086) |
| [protein_scoring](https://github.com/seanrjohnson/protein_scoring) | generating and scoring novel enzyme sequences  |
| [AF_unmasked](https://github.com/clami66/AF_unmasked) | structure prediction for huge protein complexes (~27 chains and ~8400aa) |
| [AncFlow](https://github.com/rrouz/AncFlow) | pipeline for the ancestral sequence reconstruction of clustered phylogenetic subtrees |
| [TRILL](https://github.com/martinez-zacharya/TRILL) | Sandbox for Deep-Learning based Computational Protein Design |
| [AF2BIND](https://github.com/sokrypton/af2bind) | Predicting ligand-binding sites based on AF2 |
| [PyPEF](https://github.com/Protein-Engineering-Framework/PyPEF) | sequence-based machine learning-assisted protein engineering |
| [DeepProtein](https://github.com/jiaqingxie/DeepProtein) | protein Property Prediction |
| [FlexMol](https://github.com/Steven51516/FlexMol) | construction and evaluation of diverse model architectures  |
| [Pinal](https://github.com/westlake-repl/Denovo-Pinal) | text-guided protein design |
| [ByProt (LM-Design)](https://github.com/bytedprotein/ByProt) | reprogramming pretrained protein LMs as generative models |
| [scikit-bio](https://github.com/scikit-bio/scikit-bio) | data structures, algorithms and educational resources for bioinformatics |
| [BindCraft](https://github.com/martinpacesa/BindCraft) | binder design pipeline (See also the [wiki-tutorial](https://github.com/martinpacesa/BindCraft/wiki/De-novo-binder-design-with-BindCraft)) |
| [FreeBindCraft](https://github.com/cytokineking/FreeBindCraft) | BindCraft modified to make PyRosetta use and installation optional, i.e. no license needed ([more details](https://www.ariax.bio/resources/freebindcraft-open-source-unleashed))|
| [prosculpt](https://github.com/ajasja/prosculpt) | encapsulates RFDiffusion, ProteinMPNN, AlphaFold2, and Rosetta into an easy-to-use workflow |
| [BinderFlow](https://github.com/cryoEM-CNIO/BinderFlow) | parallelised pipeline for protein binder design (i.e. RFD > ProteinMPNN > AF2 + Scoring) | 
| [proteindj](https://github.com/PapenfussLab/proteindj) | pipeline for de novo binder design (i.e. RFD > ProteinMPNN > AF2 + Scoring) |
| [ovo](https://github.com/MSDLLCpapers/ovo) | ecosystem for de novo protein design |
| [IPSAE](https://github.com/DunbrackLab/IPSAE) | Scoring function for interprotein interactions in AlphaFold2 and AlphaFold3 |
| [bagel](https://github.com/softnanolab/bagel) | model-agnostic and gradient-free exploration of an energy landscape in the sequence space |
| [Protein Design Skills](https://proteinbase.com/protein-design-skills) | Claude Code skills for protein design |



# docking (ligands and proteins)
| Name | Description | 
|-----------|-----------|
| [HiQBind](https://github.com/THGLab/HiQBind) | Workflow to clean up and fix structural problems in protein-ligand binding datasets |
| [InterfaceAnalyzerMover](https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Movers/movers_pages/analysis/InterfaceAnalyzerMover) | Calculate binding energies, buried interface surface areas, packing statistics, and other useful interface metrics for the evaluation of protein interfaces |
| [PandaDock](https://github.com/pritampanda15/PandaDock) | Physics-Based Molecular Docking |
| [ligysis](https://www.compbio.dundee.ac.uk/ligysis/) | analysis of biologically meaningful ligand binding sites |
| []() |  |

# molecules
| Name | Description | 
|-----------|-----------| 
| [rdkit](https://www.rdkit.org/docs/index.html) | cheminformatics and machine-learning software |
| [PDBe CCDUtils](https://pdbeurope.github.io/ccdutils/index.html)  | tools to deal with PDB chemical components and visualization ([see also](https://github.com/PDBeurope/pdbe-notebooks/tree/main/pdbe_ligands_tutorials))|
| [PDBe Arpeggio](https://github.com/PDBeurope/arpeggio) |  calculation of interatomic interactions in molecular structures|
| [PDBe RelLig](https://github.com/PDBeurope/rellig) | classifies ligands based on their functional role| 
| [MolPipeline](https://github.com/basf/MolPipeline) | processing molecules with RDKit in scikit-learn |
| [roshambo](https://github.com/molecularinformatics/roshambo) | molecular shape comparison |
| [FlexMol](https://github.com/Steven51516/FlexMol) | construction and evaluation of diverse model architectures |
| [molli](https://github.com/SEDenmarkLab/molli) | general purpose molecule library generation and handling  |
| [useful_rdkit_utils](https://github.com/PatWalters/useful_rdkit_utils) | A collection of useful RDKit and sci-kit learn functions |
| [deepchem](https://github.com/deepchem/deepchem) | toolchain that democratizes the use of deep-learning in drug discovery, materials science, quantum chemistry, and biology. |
| [nvMolKit](https://github.com/NVIDIA-Digital-Bio/nvMolKit) | GPU-accelerated library for key computational chemistry tasks, such as molecular similarity, conformer generation, and geometry relaxation |
| [deepchem](https://github.com/deepchem/deepchem) | toolkit for drug discovery, materials science, quantum chemistry, and biology |



# machine learning
| Name | Description | 
|-----------|-----------| 
| [csvtk](https://github.com/shenwei356/csvtk) | CSV/TSV manip MSA (3M protein sequences in 5min and 24GB of RAM) |
| [Colab forms](https://colab.research.google.com/notebooks/forms.ipynb) | how to convert a colab notebook to a user interface |
| [cuml](https://github.com/rapidsai/cuml) | GPU-based implementations of common machine learning algorithms ([more info for umap optimization](https://developer.nvidia.com/blog/even-faster-and-more-scalable-umap-on-the-gpu-with-rapids-cuml/) and [cuml.accel](https://developer.nvidia.com/blog/nvidia-cuml-brings-zero-code-change-acceleration-to-scikit-learn/) to boost scikit-learn and other libs in colab)|
| [LazyPredict](https://github.com/shankarpandala/lazypredict) | build a lot of basic models without much code |
| [TorchDR](https://github.com/TorchDR/TorchDR) | PyTorch Dimensionality Reduction |
| [Kerasify](https://github.com/moof2k/kerasify) | running trained Keras models from a C++ application |
| [pca](https://erdogant.github.io/pca/pages/html/index.html) | perform PCA and create insightful plots |
| [openTSNE](https://opentsne.readthedocs.io/en/stable/) | faster implementation of t-SNE that includes other optimizations |
| [TabPFN](https://github.com/PriorLabs/TabPFN) | model for tabular data that outperforms traditional methods while being dramatically faster |
| [tabm](https://github.com/yandex-research/tabm) | tabular DL architecture that efficiently imitates an ensemble of MLPs |
| [tabicl](https://github.com/soda-inria/tabicl) | tabular model for classification |
| [setfit](https://github.com/huggingface/setfit) | Efficient few-shot learning with Sentence Transformers |
| [skrub](https://github.com/skrub-data/skrub) | preprocessing and feature engineering for tabular machine learning |
| [cupy](https://github.com/cupy/cupy) |NumPy & SciPy for GPU|
| [Best-of Machine Learning](https://github.com/ml-tooling/best-of-ml-python) | list of awesome machine learning Python libraries |
| [torchmetrics](https://github.com/Lightning-AI/torchmetrics) | 100+ PyTorch metrics implementations |
| [DADApy](https://github.com/sissa-data-science/DADApy) | characterization of manifolds in high-dimensional spaces |
| [PySR](https://github.com/MilesCranmer/PySR) | High-Performance Symbolic Regression |
| [BERTopic](https://github.com/MaartenGr/BERTopic) | create clusters for easily interpretable topics |
| [KeyBERT](https://github.com/MaartenGr/KeyBERT) | Minimal keyword extraction with BERT |
| [PolyFuzz](https://github.com/MaartenGr/PolyFuzz) | Fuzzy string matching, grouping, and evaluation. |
| [hummingbird](https://github.com/microsoft/hummingbird) | compiles trained ML models into tensor computation for faster inference |
| [skorch](https://github.com/skorch-dev/skorch) | train PyTorch models in a way similar to Scikit-learn (eg. No need to manually write a training loop, just using fit(), predict(), score()) |
| [Faiss](https://github.com/facebookresearch/faiss) | efficient similarity search and clustering of dense vectors |
| [tmap](https://github.com/reymond-group/tmap) | tree-like and fast visualization library for large, high-dimensional data set |
| [einops](https://github.com/arogozhnikov/einops) | tensor operations for readable and reliable code |
| [skrub](https://github.com/skrub-data/skrub/) | doing machine learning with dataframes (see also the [learning materials](https://skrub-data.org/skrub-materials/index.html))|
| [pyod](https://github.com/yzhao062/pyod) | Outlier and Anomaly Detection, Integrating Classical and Deep Learning Techniques |
| [autokeras](https://autokeras.com/) | AutoML system based on Keras |
| [numba](https://github.com/numba/numba) | NumPy aware dynamic Python compiler  |
| [langextract](https://github.com/google/langextract) | extracting structured information from unstructured text using LLMs |
| [cleanlab](https://github.com/cleanlab/cleanlab) | clean data and labels by automatically |
| [dtype_diet](https://github.com/noklam/dtype_diet) | Optimize your memory consumption when using pandas by changing dtypes without data loss  |




# statistics
| Name | Description | 
|-----------|-----------| 
| [scikit-posthocs](https://scikit-posthocs.readthedocs.io/en/latest/) |  post hoc tests for pairwise multiple comparisons |
| [statannotations](https://github.com/trevismd/statannotations) | add statistical significance annotations on seaborn plots |
| [ggstatsplot](https://github.com/IndrajeetPatil/ggstatsplot) | creating graphics with details from statistical tests included in the information-rich plots themselves |
| [ggbetweenstats](https://indrajeetpatil.github.io/ggstatsplot/articles/web_only/ggbetweenstats.html) | making publication-ready plots with relevant statistical details |
| [statsmodels](https://www.statsmodels.org/stable/index.html) | classes and functions for the estimation of many different statistical models, as well as for conducting statistical tests, and statistical data exploration |
| [pingouin](https://pingouin-stats.org/build/html/index.html#) | Statistical package |
| [performance](https://github.com/easystats/performance) | computing indices of regression model quality and goodness of fit  |
| []() |  |


# datavis for bio
| Name | Description | 
|-----------|-----------| 
| [MolecularNodes](https://github.com/BradyAJohnston/MolecularNodes) | Toolbox for molecular animations in Blender |
| [CellScape](https://github.com/jordisr/cellscape) | Vector graphics cartoons from protein structure |
| [ChimeraX apps](https://cxtoolshed.rbvi.ucsf.edu/apps/all) | chimeraX  extensions |
| [chimerax_viridis](https://github.com/smsaladi/chimerax_viridis) | Colorblind-friendly, perceptually uniform palettes for chimerax (see [this tips](https://bsky.app/profile/jameslingford.bsky.social/post/3lmw5h3xxec2n) tore-create the RFDifussion style in ChimeraX or the [original pyMOL implementation](https://bsky.app/profile/spellock.bsky.social/post/3lnsup5m5ks22))|
| [SSDraw](https://github.com/ncbi/SSDraw) | generates secondary structure diagrams from 3D protein structures |
| [bioalphabet](https://github.com/davidhoksza/bioalphabet/) | convertor of texts to bio-domain alphabetss |
| [ChatMol](https://github.com/ChatMol/ChatMol) | a PyMOL ChatGPT Plugin that allow to interact with PyMOL using natural language  |
| [plot_phylo](https://github.com/KatyBrown/plot_phylo) | plot a phylogenetic tree on an existing matplotlib axis |
| [prettymol](https://github.com/zachcp/prettymol) | automatic protein structure plots with MolecularNodes  |
| [VMD-2](https://www.ks.uiuc.edu/Research/vmd/vmd2intro/index.html) | Visual Molecular Dynamics |
| [gromacs_copilot](https://github.com/ChatMol/gromacs_copilot) | AI-powered assistant for Molecular Dynamics  simulations |
| [NIH bioart](https://bioart.niaid.nih.gov/) | 2,000+ science and medical icons |
| [bioicons](https://bioicons.com/) | icons for science illustrations in biology and chemistry |
| [moldraw](https://moldraw.com/) | draw molecules |
| [Mol* at RCSB/PDB](https://onlinelibrary.wiley.com/doi/10.1002/pro.70093) |  web-based, 3D visualization software suite for examination and analyses of biostructures |
| [MolViewSpec](https://github.com/molstar/mol-view-spec/) | Python toolkit allows for describing views used in molecular visualizations |
| [PoseEdit](https://proteins.plus/) | interactive 2D ligand interaction diagrams (see this [tutorial](https://www.youtube.com/watch?v=8W1TvSvatSA&ab_channel=BioinformaticsInsights)) |
| [FlatProt](https://github.com/t03i/FlatProt) | 2D protein visualization aimed at improving the comparability of structures  |
| [quarto-molstar](https://github.com/jmbuhr/quarto-molstar) | embed proteins and trajectories with Mol* |
| [alphabridge](https://alpha-bridge.eu/) | summarise predicted interfaces and intermolecular interactions |
| [weblogo](https://weblogo.threeplusone.com/) |  generation of sequence logos |
| [interprot](https://interprot.com/#/) | inspect relevant features derived from protein language models in a particular protein |
| [Protein icons](https://bsky.app/profile/maxfus.bsky.social/post/3lobecnwdsc2w) |  How to create specific protein icons with ChatGPT |
| [termal](https://github.com/sib-swiss/termal) |  examining MSAs in a terminal |
| [py2Dmol](https://github.com/sokrypton/py2Dmol) | visualizing biomolecular structures in 2D in Google Colab and Jupyter environments ([check out the website](https://py2dmol.solab.org/))|
| [Nano Protein Viewer](https://marketplace.visualstudio.com/items?itemName=StevenYu.nano-protein-viewer) | protein structure viewer in VScode (try it using [web app](https://stevenyuyy.us/protein-viewer/) and also [check out this tutorial](https://youtu.be/srDyhfhoDm8))|
| [Protein Viewer](https://marketplace.visualstudio.com/items?itemName=ArianJamasb.protein-viewer) | visualisation of protein structures and molecular data in VScode |
| [molview](https://github.com/54yyyu/molview) | IPython/Jupyter widget for interactive molecular visualization, based on Molstar |
| [ProteinCHAOS](https://dzyla.github.io/ProteinCHAOS/) | an artistic tool inspired by molecular dynamics to capture protein flexibility over time |
| []() |  |


# datavis
| Name | Description | 
|-----------|-----------| 
| [datamapplot](https://github.com/TutteInstitute/datamapplot) | creating beautiful, interactive and massive scatterplots (e.g. [wikipedia articles](https://lmcinnes.github.io/datamapplot_examples/wikipedia/) and a [tutorial of how to reproduce it](https://x.com/leland_mcinnes/status/1937591460125090189)) |
| [pypalettes](https://github.com/JosephBARBIERDARNAL/pypalettes) | +2500 color maps  |
| [distinctipy ](https://github.com/alan-turing-institute/distinctipy) |  generating visually distinct colours |
| [Visualize Architecture of Neural Network](https://github.com/ashishpatel26/Tools-to-Design-or-Visualize-Architecture-of-Neural-Network) | set of tools (like [NN-SVG](https://alexlenail.me/NN-SVG/LeNet.html)) to plot neural nets |
| [tidyplots](https://jbengler.github.io/tidyplots/index.html) | creation of publication-ready plots for scientific papers |
| [pyCirclize](https://github.com/moshi4/pyCirclize) | Circular visualization in Python (Circos Plot, Chord Diagram, Radar Chart)  |
| [pycircular](https://github.com/albahnsen/pycircular) | circular data analysis |
| [great-table](https://github.com/posit-dev/great-tables) |  display tables |
| [plottable](https://github.com/znstrider/plottable) | plotting beautifully customized, presentation ready tables |
| [d3blocks](https://github.com/d3blocks/d3blocks) | create stand-alone and interactive d3 charts |
| [How to Vectorize Plots from R/Python in PowerPoint](https://nalinan.medium.com/how-to-vectorize-plots-from-r-in-powerpoint-bad7c238e86a) | import a vectorized image into ProwerPoint for easy manipulation ([see also this tutorial](https://www.youtube.com/watch?v=hoHkc7N6FZA&ab_channel=GenomicsBootCamp)) |
| [morethemes](https://github.com/JosephBARBIERDARNAL/morethemes) | More themes for matplotlib |
| [jsoncrack](https://github.com/AykutSarac/jsoncrack.com) | visualization application that transforms data formats such as JSON, YAML, XML, CSV and more, into interactive graphs |
| [torchvista](https://github.com/sachinhosmani/torchvista) | visualize the forward pass of a PyTorch model directly in the notebook |
| [bivario](https://github.com/RaczeQ/bivario) | plotting bivariate choropleth maps |
| []() |  |
| []() |  |


# webservers
| Web | Description | 
|-----------|-----------| 
| [ProteInfer](https://google-research.github.io/proteinfer/) | predicting functional properties from sequences |
| [GoPredSim](https://embed.protein.properties/) | Predict protein properties from embeddings |
| [DeepFRI](https://beta.deepfri.flatironinstitute.org/) | structure-based protein function prediction and functional residue identification |
| [protein structure relaxation](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/beta/relax_amber.ipynb) | Relax your structure using OpenMM/Amber |
| [Model Memory Calculator](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) | calculate how much vRAM is needed to train and perform inference on a model hosted on Hugging Face (see also [gpu_poor calculator](https://github.com/RahulSChand/gpu_poor) or [this ecuation](https://www.substratus.ai/blog/calculating-gpu-memory-for-llm)) |
| [alphafind](https://alphafind.fi.muni.cz/search) | structure-based search |
| [DiffDock-Web](https://huggingface.co/spaces/reginabarzilaygroup/DiffDock-Web) | molecular docking with ligands |
| [ESMFold](https://esmatlas.com/resources?action=fold) | structure prediction with ESMFold |
| [Foldseek clusters](https://cluster.foldseek.com/) | search for sctructural clusters in AFDB |
| [damietta](https://damietta.de/) | protein design toolkit |
| [easifa](http://easifa.iddd.group/) | active and binding site annotations for enzymes |
| [InterProt](https://interprot.com/#/) |  interpretability of features derived from protein language models using sparse autoencoders |
| [MPI Bioinformatics Toolkit!](https://toolkit.tuebingen.mpg.de/) | multiple bioinformatics tools |
| [moleculatio](https://moleculatio.yamlab.app) | chemoinformatics, quantum chemistry and molecular dynamics simulations or small molecules |
| [AI in Biology Demos]( https://huggingface.co/collections/hf4h/ai-in-biology-demos-65007d936a230e55a66cd31e) | applications of AI in biology and biochemistry |
| [ProteinsPlus](https://proteins.plus/) | structure mining and modeling, focussing on protein-ligand interactions |

| []() |  |


# chatbots and agents
| Name | 
|-----------|
| [ChatGPT](https://chat.openai.com/) |
| [Gemini](https://gemini.google.com/) |
| [claude](https://claude.ai/) |
| [Bing](https://www.bing.com/?cc=es) |
| [HuggingChat](https://huggingface.co/chat/) |
| [huggingface spaces](https://huggingface.co/spaces) | 
| [biologpt](https://biologpt.com/) |
| [consensus](https://consensus.app/) |  
| [typeset](https://typeset.io/) |  
| [mistral-chat](https://mistral.ai/news/mistral-chat/) |  
| [aistudio by Google](https://aistudio.google.com/) |
| [AI Python Libraries](https://www.aipythonlibraries.com/libraries/) | 
| [paperfinder](https://paperfinder.allen.ai/chat) | 
| [AI Scientist agents by futurehouse](https://platform.futurehouse.org/) |
| [OpenAI Deep Research Guide (by DAIR.AI)](https://docs.google.com/document/d/1vLaEMu5jirQT5RK0cW8RUXNFQyszMQ-xrjxUZF2wOg4/edit?tab=t.0#heading=h.2y9eo2rdwxv2) |  
| [biomni](https://github.com/snap-stanford/biomni) |  
| []() |  |
