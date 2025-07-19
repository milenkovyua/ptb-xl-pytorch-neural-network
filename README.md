Make sure you have git installed. Refer to https://git-scm.com/downloads for more info

Clone the repo to a local directory using
```
git clone https://github.com/milenkovyua/ptb-xl-pytorch-neural-network.git
```

Create virtual environment 
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




