## Integrating Text and Image Modalities for Enhanced Image Classification
This work is the end-semester term project for MM-811. The overall pipeline is shown below:

<img width="819" alt="ModelArch" src="https://github.com/Tanmay98/Multi-Modal-Image-Classification/assets/31308303/c0feff25-e6ee-494a-b10a-c02004a5b047">

### Creating Virtual Environment
- using pip run
`pip install -r requirements.txt`

- using Conda run
`conda create --name <env_name> --file requirements.txt`

### Download the dataset
Please download CIFAR-10 dataset from torch. 

### Creating Textual Descriptions
Run all cells of `create_desc.ipynb`. Note: Please change paths to your respective directories.

### Creating Textual Embeddings
Run all cells of `create_text_embed.ipynb`. Note: Note: Please change paths to your respective directories.

### Train model
Run `python main.py` 

### Visualize results on Tensorboard
Run `tensorboard --logdir /runs --port 6006`
To visualize exiting tf events, check out runs directory
