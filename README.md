## SemReg: Semantics Constrained Point Cloud Registration


### Installation 

```
conda env create -f environment.yml
conda activate semreg
cd cpp_wrappers; sh compile_wrappers.sh; cd ..
```
### Download pretrained weights
```
wget https://github.com/NovaPhantomAnonymous/SemReg/releases/download/v1.0.0/weights.ckpt
```
### Run code

```
python3 demo.py 
```
