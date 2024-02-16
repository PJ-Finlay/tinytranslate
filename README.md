# tinytranslate
Machine translation demo built on [tinygrad](https://github.com/tinygrad/tinygrad)

Tinytranslate is a demo of using Transformer neural networks for machine translation. Tinytranslate is very lightweight and can train from scratch on a laptop CPU in a few seconds allowing for fast iteration and experimentation.

Tinytranslate is a lightweight tech demo and is not suitable for real world translations; use [Argos Translate](https://github.com/argosopentech/argos-translate) if you need a production translation system.

#### Quickstart
```
# Download training data
./scripts/download_data.sh

# Run training
python translate.py
```


