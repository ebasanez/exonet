# Exonet
Planet detection using Convolutional Neural Networks

 ## Authors:

- Robert Austin Benn
- Enrique Basañez Mercader 
- Miguel Blanco Marcos
- Borja Sánchez Leirado

## Objective

Analyze state-of-the-art research about planet detection using transit method 
(see https://en.wikipedia.org/wiki/Methods_of_detecting_exoplanets) and use advanced machine learning techn	ics to improve 

## Pipeline

For this project pipeline, different modules are used:

### Raw data retrieval

Module _scr/com/saturdaysai/exonet/lightKurveApi/lightKurveApiCLient.py_

Raw data will be obtained from Kepler records stores at MAST(Mikulski Archive for Space Telescopes) using Python library LightKurve to access their api.
https://archive.stsci.edu/mast.html

https://docs.lightkurve.org/

### Data preprocessing

Raw data obtained have informatioon about transit period, duration and centroid.

Using that information, data will be normalized and re-sampled to fixed length to generate one-dimensional tensors.

### Model definition

Notebook _notebooks/model_definition.ipynb_

This notebook uses tensors generated (ans stored in /data folder) to define a and optimize a CNN based model. 
Hyper-parameters present in this notebook are the result of iterative training.

### Model validation

Notebook _notebooks/model_evaluation.ipynb_

Loads model defined in previous step and validates it using validation sub-dataset, obtaining cross entropy loss, accuracy and auc.

### Production stage

Notebook  _notebooks/model_predictions.ipynb_

Predictions will be made using Kepler Objects of Interest that are yet considered candidates of being exo-planets, ordering those by decreasing probability of having a light kurve that evidencnes an exo-planet.

### External links

Article in Medium: 
https://medium.com/@miguel.blanco.marcos/exonet-an-ai-saturdays-project-a1bda907bdef

Inspired in:
Shallue, Christopher J., and Andrew Vanderburg. “Identifying Exoplanets with Deep Learning: A Five-Planet Resonant Chain Around Kepler-80 and an Eighth Planet Around Kepler-90.” The Astronomical Journal 155.2 (2018): 94. Crossref. Web. (DOI: 10.3847/1538-3881/aa9e09)
