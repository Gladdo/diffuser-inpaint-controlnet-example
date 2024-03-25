Setup environment:

- conda create -name diffuser-inpaint-controlnet python=3.10
- conda activate diffuser-inpaint-controlnet
- pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --index-url https://download.pytorch.org/whl/cu121
- pip install diffusers==0.21.4 transformers==4.34.1 accelerate==0.23.0
