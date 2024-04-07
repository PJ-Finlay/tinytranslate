# tinytranslate
Machine translation demo built on [tinygrad](https://github.com/tinygrad/tinygrad)

Tinytranslate is a demo of using Transformer neural networks for machine translation. Tinytranslate is very lightweight and can train from scratch on a laptop CPU in a few minutes allowing for fast iteration and experimentation.

Tinytranslate is a lightweight tech demo and is not suitable for real world translations; use [Argos Translate](https://github.com/argosopentech/argos-translate) if you need a production translation system.

#### Quickstart
```
# Create Virtual Environment (optional)
virtualenv env
source env/bin/activate

# Install tinygrad
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
pip install -e .
cd ..

# Download TinyTranslate source code
git clone https://github.com/PJ-Finlay/tinytranslate
cd tinytranslate

# Download training data
./scripts/download_data.sh

# Run training
python translate.py
```

#### Example output:
```
loss 2.43 accuracy 0.37: 100%|█████████████████████████████████████████████████████████████████████████████| 100/100 [00:41<00:00,  2.41it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:03<00:00,  4.17it/s]
test set accuracy is 0.352500
reducing lr to 0.0017
Source: The provin
Target: Las provin
Pred:   Ea  proein

...

loss 0.51 accuracy 0.81: 100%|█████████████████████████████████████████████████████████████████████████████| 100/100 [00:36<00:00,  2.72it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:03<00:00,  4.30it/s]
test set accuracy is 0.799050
reducing lr to 0.0012
Source: 52+88
    
Target: 140
      
Pred:   110
      
loss 0.58 accuracy 0.80: 100%|█████████████████████████████████████████████████████████████████████████████| 100/100 [00:36<00:00,  2.73it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:03<00:00,  4.36it/s]
test set accuracy is 0.823600
reducing lr to 0.0010
Source: 51+43
    
Target: 94
       
Pred:   94
       
loss 0.46 accuracy 0.82: 100%|█████████████████████████████████████████████████████████████████████████████| 100/100 [00:36<00:00,  2.75it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:03<00:00,  4.32it/s]
test set accuracy is 0.836800
reducing lr to 0.0008
Source: 1+8
      
Target: 9
        
Pred:   50
       
```

#### Data Sources
- [Opus Parallel Corpora](https://opus.nlpl.eu/)
- [Argos Data](https://github.com/argosopentech/argos-data/blob/main/builddataset.go)


