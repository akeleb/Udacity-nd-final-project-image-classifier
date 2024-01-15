
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse



def parse_command_line_arguments():
    """
    Parses command-line arguments.

    Returns:
        - args (argparse.Namespace): Parsed command-line arguments.
    """
    # Create the parser
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('image_path', type=str, help='Path to the image to be classified')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to return')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to the JSON file with category names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()
    return args

def load_checkpoint(filepath, device='cpu'):
    """
    Loads a PyTorch model checkpoint.

    Parameters:
        - filepath (str): Location of the checkpoint file.
        - device (str): Device to load the model onto ('cpu' or 'gpu').

    Returns:
        - model_arch (str): Model architecture.
        - input_units (int): Number of input units.
        - output_units (int): Number of output units.
        - hidden_units (int): Number of hidden units.
        - state_dict (dict): Model state dictionary.
        - class_to_idx (dict): Mapping of class labels to indices.
    """
    # Define map_location based on the specified device
    map_location = 'cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu'

    # Load the checkpoint
    try:
        checkpoint = torch.load(filepath, map_location=map_location)
    except FileNotFoundError:
        print(f"Checkpoint file '{filepath}' not found.")
        return None
    # Extract relevant information from the checkpoint
    model_arch = checkpoint['model_arch']
    input_units = checkpoint['clf_input']
    output_units = checkpoint['clf_output']
    hidden_units = checkpoint['clf_hidden']
    state_dict = checkpoint['state_dict']
    class_to_idx = checkpoint['model_class_to_index']
    
    print(f"Model architecture: {model_arch}")
    print(f"Input units: {input_units}")
    print(f"Output units: {output_units}")
    print(f"Hidden units: {hidden_units}")
    print(f"State dictionary: {state_dict}")
    print(f"Class to index mapping: {class_to_idx}")
    return model_arch,class_to_idx


def process_image(image_path):
    # Open the image
    processed_image = Image.open(image_path).convert('RGB')
    # Resize and center crop
    processed_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(processed_image)
    # Convert tensor to numpy array
    np_processed_image = processed_image.numpy()

    return np_processed_image

def class_to_label(file,classes):
        with open(file, 'r') as f:
            class_mapping =  json.load(f)
            labels = []
            for c in classes:
                labels.append(class_mapping[c])
            return labels
        

def predict(image_path, model,idx_mapping, topk=5, device="cpu"):
    pre_processed_image = torch.from_numpy(process_image(image_path))
    pre_processed_image = torch.unsqueeze(pre_processed_image, 0).to(device).float()
    model.to(device)
    model.eval()
    log_ps = model.forward(pre_processed_image)
    ps = torch.exp(log_ps)
    top_ps, top_idx = ps.topk(topk, dim=1)
    list_ps = top_ps.tolist()[0]
    list_idx = top_idx.tolist()[0]
    classes = []
    model.train()
    
    for x in list_idx:
        classes.append(idx_mapping[x])
    return list_ps, classes

def print_predictions(probabilities, classes, image, category_names=None):
    print(image)
    
    if category_names:
        labels = class_to_label(category_names, classes)
        for i, (ps, ls, cs) in enumerate(zip(probabilities, labels, classes), 1):
            print(f'{i}) {ps * 100:.2f}% {ls.title()} | Class No. {cs}')
    else:
        for i, (ps, cs) in enumerate(zip(probabilities, classes), 1):
            print(f'{i}) {ps * 100:.2f}% Class No. {cs} ')
    print('')
    
    


if __name__ == '__main__':
    args = parse_command_line_arguments()
    model_arch,class_to_idx = load_checkpoint(args.checkpoint)
    idx_mapping = {v: k for k, v in class_to_idx.items()}
    model = getattr(models, model_arch)(pretrained=True)
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.5)),
        ('fc2', nn.Linear(4096, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['state_dict'])
    probabilities, classes = predict(args.image_path, model,idx_mapping, topk=args.top_k, device="cpu")
    print_predictions(probabilities, classes, args.image_path, args.category_names)