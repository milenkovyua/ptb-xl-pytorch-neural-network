Make sure you have git installed. Refer to https://git-scm.com/downloads for more info

For easier testing and development, the repo has 2 main files that can be run: `train.py` and `load_and_test.py`. The first one will train the model and will save the trained model data into `ptb-xl-trained-seq100-age35-50.pt`. The second python script `load_and_test.py` will load that file and will perform some testing with dataset only for test.
The data loading and processing is done in `data_loader.py`. The model class itself is defined in `PTBXLModel.py`

Clone the repo to a local directory using
```
git clone https://github.com/milenkovyua/ptb-xl-pytorch-neural-network.git
```

Create virtual environment
( This step must be done only once, for creating the virtual environment, after the repo is cloned locally. Once the virtual environment is created, you don't need to run that command anymore )
```
python3 -m venv venv
```

Activate it with
```
source venv/bin/activate
```

Install all required packages with (this will couple of minutes)
```
pip3 install torch torchvision numpy pandas wfdb scikit-learn matplotlib jupyter
```

Before training the model, extract the file: `ptb-xl-data.zip` into `ptb-xl-data` directory

Training the model
```
python train.py
```
A file named `ptb-xl-trained-seq100-age35-50.pt` will appear in the folder or will be rewritten if already presented

Testing the model:
```
python3 load_and_test.py
```




