conda create -n tg python=3.7
conda activate tg

conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install "tensorflow-gpu<2.11"
pip install -r requirements.txt

python capsule_network1.py