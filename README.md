# To start the project for the first time

## download pytorch manually
`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`

## download requirements
`pip install -r requirements.txt`

## download datasets
open git bash in the AML_project folder and run:

`./download_data.sh`

# Run the yolo training changing the batch size and workers as needed (8, 0 for low end hardware)
`python train.py --batch 32 --workers 4 --epochs 10 --device 0`

