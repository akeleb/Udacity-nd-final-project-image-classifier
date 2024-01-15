
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
import json
import os
import argparse
# define Arguments for the script


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description="Parser of training script")
    parser.add_argument('data_dir', help='Provide data directory.', type=str)
    parser.add_argument(
        '--save_dir', help='Provide saving directory. Default is current directory.', type=str, default=os.getcwd())
    parser.add_argument(
        '--arch', help='vgg16_bn can be used if this argument specified', type=str, default='vgg16_bn')
    parser.add_argument(
        '--lrn', help='Learning rate. Default value is 0.001.', type=float, default=0.001)
    parser.add_argument(
        '--hidden_units', help='Hidden units in Classifier. Default value is 2048.', type=int, default=4096)
    parser.add_argument(
        '--epochs', help='Number of epochs. Default value is 10.', type=int, default=10)
    # Parse the arguments
    args = parser.parse_args()
    return args


def load_data_and_define_tranformers(data_dir):
    # TODO: Load the datasets with ImageFolder
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Define common mean and std for normalization
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    # TODO: Load the datasets with ImageFolder
    train_data_set = datasets.ImageFolder(
        train_dir, transform=train_transforms)
    test_data_set = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data_set = datasets.ImageFolder(
        valid_dir, transform=valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_data_set_loader = torch.utils.data.DataLoader(
        train_data_set, batch_size=64, shuffle=True, pin_memory=True)
    test_data_set_loader = torch.utils.data.DataLoader(
        test_data_set, batch_size=32)
    valid_data_set_loader = torch.utils.data.DataLoader(
        valid_data_set, batch_size=32)

    return train_data_set_loader, test_data_set_loader, valid_data_set_loader, train_data_set.class_to_idx


# TODO: lable mapping
def map_category_lable_to_name():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    # Print the loaded mapping
    print("Category to Name Mapping:")
    print(cat_to_name)
    # Calculate the number of categories
    no_output_categories = len(cat_to_name)
    print("Number of Categories:", no_output_categories)
    return cat_to_name


# TODO: build and train model
def build_and_train_model(train_data_set_loader, valid_data_set_loader, hidden_units, epoches, arch, learning_rate):
    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # defines the classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout1', nn.Dropout(0.05)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    # replaces the pretrained classifier with the one created above
    model.classifier = classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # sets the training hyperparameters
    # makes use of momentum to avoid local minima
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()  # recommended when using Softmax
    print_every = 20  # the model trains on 20 batches of images at a time
    running_loss = running_accuracy = 0
    validation_losses, training_losses = [], []

    print(f'The device in use is {device}.\n')
    # defines the training process
    for e in range(epoches):
        batches = 0  # 1 batch = 64 images
        # turns on training mode
        model.train()

        for images, labels in train_data_set_loader:
            start = time.time()  # defines start time
            batches += 1

            # moves images and labels to the GPU
            images, labels = images.to(device), labels.to(device)

            # pushes batch through network
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            # calculates the metrics
            ps = torch.exp(log_ps)
            top_ps, top_class = ps.topk(1, dim=1)
            matches = (top_class == labels.view(
                *top_class.shape)).type(torch.FloatTensor)
            accuracy = matches.mean()

            # resets optimiser gradient and tracks metrics
            optimizer.zero_grad()
            running_loss += loss.item()
            running_accuracy += accuracy.item()

            # runs the model on the validation set every 5 loops
            if batches % print_every == 0:
                end = time.time()
                training_time = end-start
                start = time.time()
                # sets the metrics
                validation_loss = 0
                validation_accuracy = 0
                # turns on evaluation mode, turns off calculation of gradients
                model.eval()
                with torch.no_grad():
                    for images, labels in valid_data_set_loader:
                        images, labels = images.to(device), labels.to(device)
                        log_ps = model.forward(images)
                        loss = criterion(log_ps, labels)
                        ps = torch.exp(log_ps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        matches = (top_class ==
                                   labels.view(*top_class.shape)).type(torch.FloatTensor)
                        accuracy = matches.mean()

                        # tracks validation metrics (test of the model's progress)
                        validation_loss += loss.item()
                        validation_accuracy += accuracy.item()

                # tracks training metrics
                end = time.time()
                validation_time = end-start
                validation_losses.append(running_loss/print_every)
                training_losses.append(
                    validation_loss/len(valid_data_set_loader))

                # prints out metrics
                print(f'Epoch {e+1}/{epoches} | Batch {batches}')
                print(f'Running Training Loss: {running_loss/print_every:.3f}')
                print(
                    f'Running Training Accuracy: {running_accuracy/print_every*100:.2f}%')
                print(
                    f'Validation Loss: {validation_loss/len(valid_data_set_loader):.3f}%')
                print(
                    f'Validation Accuracy: {validation_accuracy/len(valid_data_set_loader)*100:.2f}%')

                # resets the metrics and turns on training mode
                running_loss = running_accuracy = 0
                model.train()
    return model


# Test teh rained model on the testing dataset
def test_model(model, test_data_set_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    test_accuracy = 0
    start_time = time.time()
    top_ps_list = []
    print('Validation started.')
    with torch.no_grad():
        for images, labels in test_data_set_loader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            ps = torch.exp(log_ps)
            top_ps, top_class = ps.topk(1, dim=1)
            matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
            accuracy = matches.mean()
            test_accuracy += accuracy
            top_ps_list.append(top_ps.item())
            average_top_ps = sum(top_ps_list) / len(top_ps_list)
    end_time = time.time()
    print('Validation ended.')
    validation_time = end_time - start_time
    print(f'Average Top Probability Value: {average_top_ps}')
    print('Validation time: {:.0f}m {:.0f}s'.format(validation_time / 60, validation_time % 60))
    print(f'Test Accuracy: {test_accuracy / len(test_data_set_loader) * 100:.2f}%')

    return test_accuracy / len(test_data_set_loader) * 100



# save checkpoints
def save_checkpoint(model, hidden_units, output_units, destination_directory, arch, class_to_idx):
    model_checkpoint = {
        'model_arch': arch,
        'clf_input': 25088,
        'clf_output': output_units,
        'clf_hidden': hidden_units,
        'state_dict': model.state_dict(),
        'model_class_to_index': class_to_idx,
    }

    # Determine the checkpoint filename based on the architecture
    checkpoint_filename = f"{arch}_checkpoint.pth"

    # Determine the full checkpoint path
    checkpoint_path = os.path.join(
        destination_directory, checkpoint_filename) if destination_directory else checkpoint_filename
    # If destination_directory is None or empty, use the current working directory
    if not destination_directory:
        checkpoint_path = os.path.join(os.getcwd(), checkpoint_path)
    # Save the checkpoint
    torch.save(model_checkpoint, checkpoint_path)
    if destination_directory:
        print(
            f"{arch} successfully saved to {destination_directory} as {checkpoint_filename}")
    else:
        print(
            f"{arch} successfully saved to current directory as {checkpoint_filename}")


if __name__ == '__main__':
    args = parse_command_line_arguments()
    data_dir = args.data_dir
    destination_directory = args.save_dir
    hidden_units = args.hidden_units
    epochs = args.epochs
    arch = args.arch
    learning_rate = args.lrn

    print('* Loading data and defining transformers ...')
    train_data_set_loader, test_data_set_loader, valid_data_set_loader, class_to_idx = load_data_and_define_tranformers(
        data_dir)
    print('* Data loaded successfully!\n')

    # map category lable to name
    print('* Mapping category lable to name ...')
    cat_to_name = map_category_lable_to_name()

    print('* Building and training model in progress ...')
    print('* Following are training loss, validation loss, and model accuracy:\n')
    model = build_and_train_model(
        train_data_set_loader, valid_data_set_loader, hidden_units, epochs, arch, learning_rate)
    # test model
    print('* Testing model ...')
    test_model(model, test_data_set_loader)

    # save checkpoint
    print('* Saving model checkpoint ...')
    save_checkpoint(model, hidden_units, 102,
                    destination_directory, arch, class_to_idx)
    print('* Saved checkpoint successfully!\n')

    print('* Done training successfully!\n')
