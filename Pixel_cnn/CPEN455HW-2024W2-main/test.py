'''
This code is used to evaluate the classification accuracy of the trained model.
You should at least guarantee this code can run without any error on validation set.
And whether this code can run is the most important factor for grading.
We provide the remaining code, all you should do are, and you can't modify other code:
1. Replace the random classifier with your trained model.(line 69-72)
2. modify the get_label function to get the predicted label.(line 23-29)(just like Leetcode solutions, the args of the function can't be changed)

REQUIREMENTS:
- You should save your model to the path 'models/conditional_pixelcnn.pth'
- You should Print the accuracy of the model on validation set, when we evaluate your code, we will use test set to evaluate the accuracy
'''
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import csv
NUM_CLASSES = len(my_bidict)

#TODO: Begin of your code
def get_label(model, model_input, device):
    # Write your code here, replace the random classifier with your trained model
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    answers=[]
    for image in model_input:
        lost_list=[]
        image_batch = image.unsqueeze(0)
        for i in range(NUM_CLASSES):
            label_tensor=torch.tensor([i],dtype=torch.long, device=device)
            model_output=model(image_batch,labels=label_tensor, sample=False)
            lost=loss_op(image_batch,model_output)
            lost_list.append(lost.item())
        answers.append(np.argmin(lost_list))        
    #model_output = model(model_input)
    # and return the predicted label, which is a tensor of shape (batch_size,)
    #answer = model(model_input, device)
    return torch.tensor(answers).to(device)
# End of your code

def classifier(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        original_label = [my_bidict[item] for item in categories]
        original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        answer = get_label(model, model_input, device)
        correct_num = torch.sum(answer == original_label)
        acc_tracker.update(correct_num.item(), model_input.shape[0])
    
    return acc_tracker.get_ratio()

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='test', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)

    #TODO:Begin of your code
    #You should replace the random classifier with your trained model
    #model = random_classifier(NUM_CLASSES)
    input_channels=3
    model = PixelCNN(nr_resnet=3, nr_filters=100, 
                input_channels=input_channels, nr_logistic_mix=20, embedding_dim=32)
    
    #End of your code
    
    model = model.to(device)
    #Attention: the path of the model is fixed to './models/conditional_pixelcnn.pth'
    #You should save your model to this path
    model_path = os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print('model parameters loaded')
        print(f"model we are evaluating on is {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.eval()
    
    acc = classifier(model = model, data_loader = dataloader, device = device)
    print(f"Accuracy: {acc}")

    # Create a new instance of the dataset in test (or validation) mode.
    # It is assumed that your CPEN455Dataset has an attribute 'files' that stores
    # the file path (name path) for each sample. If not, modify your dataset class accordingly.
    test_dataset = CPEN455Dataset(root_dir=args.data_dir, mode=args.mode, transform=ds_transforms)

    # Use a DataLoader with a batch size of 1 and no shuffling so that the order of
    # the samples corresponds to the order in test_dataset.files.
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            **kwargs)

    predictions = []

    # We don't need to compute gradients in evaluation.
    with torch.no_grad():
        # Iterate over the test data.
        for idx, (image, categories) in enumerate(tqdm(test_loader)):
            image = image.to(device)
            # Get the predicted label (a tensor of shape (1,)) for the single image.
            pred_tensor = get_label(model, image, device)
            pred_label = int(pred_tensor.item())
            
            # Retrieve the image path from the dataset. This example assumes that 
            # CPEN455Dataset stores the list of file paths in an attribute called "files".
            # (If your dataset returns file paths with each item, adjust accordingly.)
            image_path = test_dataset.files[idx]
            
            predictions.append([image_path, pred_label])

    # Write predictions into output.csv file.
    import csv
    with open("output.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header row.
        writer.writerow(["image_path", "class_label"])
        # Write each row: first column the image file path, second column the predicted label.
        writer.writerows(predictions)

    print("Predictions have been saved to output.csv")        