mkdir run
cd run
wget https://object.pouta.csc.fi/OPUS-Wikipedia/v1.0/moses/en-es.txt.zip
unzip en-es.txt.zip
head -10000 Wikipedia.en-es.en > source
head -10000 Wikipedia.en-es.es > target
