##### Using python virtual environment with conda. Please create a Pytorch environment, install Pytorch and some other packages
```R
conda create -n pipeline
source activate pipeline

conda search r-base
conda install r-base=4.2.0
conda install python=3.8

conda install conda-forge::gmp
conda install conda-forge::r-seurat==4.4.0
conda install conda-forge::r-hdf5r
conda install bioconda::bioconductor-sc3

conda install conda-forge::pot
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda search -c conda-forge r-mclust
conda install -n pipeline -c conda-forge r-mclust=5.4.10

pip install scanpy
pip install anndata==0.8.0
pip install pandas==1.4.2

conda install -c conda-forge rpy2=3.5.1
pip install scikit-learn==1.1.1
pip install scipy==1.8.1
pip install tqdm==4.64.0
pip install scikit-misc

```

