# System Modules
import os.path
import numpy as np
import matplotlib.pyplot as plt

# Deep Learning Modules
import torch

# User Defined Modules
from network.unet import get_network_prediction
from utils.metrics import calculate_confusion_matrix, calculate_metrics
from utils.tensor_board_helper import TB
from utils.experiment import save_model_config
from utils.visualization import get_overlay_image, display_output
from data.image_dataset import convert_tensor_to_numpy



from utils.visualization import get_overlay_image


def get_device(seed=1):
    """
    Checks if cuda is available and uses the gpu as device if available.
    The random seed is set for the device and returned.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.manual_seed(seed)
    return device


class Model:
    def __init__(self, exp, net, torch_seed=None):

        self.exp = exp
        self.model_info = exp.params['Network']

        # If seed is given by user, use that seed, else use the seed from the
        # config file.
        self.model_info['seed'] = torch_seed or self.model_info['seed']
        self.device = get_device(self.model_info['seed'])

        # Path for saving/loading model
        self.model_save_path = exp.params['network_output_path']

        # Network
        self.net = net().to(self.device)

    def setup_model(self, optimiser, optimiser_params, loss_function,
                    loss_func_params):
        """
        Setup the model by defining the model, optimiser,loss function ,
        learning rate,etc
        """

        # setup Tensor Board
        self.tensor_board = TB(self.exp.params['tf_logs_path'])


        ## Step 6a: Optimizer and loss function initialization
        # Initialize Training optimiser with given optimiser_params.
        # TO DO: Recall how the optimiser is initialized from the Pytorch
        # tutorial. Note that the optimiser and it's parameters
        # are passed to this function by the user dynamically while running the
        # program and is not known to us before.
        # Make use of the idea of  ** kwarg keyword arguments that you had used
        # in the road simulator task.
        # self.optimiser = ...
        self.optimiser = optimiser(self.net.parameters(), **optimiser_params)
        

        
        # Similarly use the ** kwargs for initializing the loss function with
        # the given loss_func_params
        # self.loss_function = ...
        self.loss_function = loss_function(**loss_func_params)


    def reset_train_stats(self):
        """ Reset all variables that are used to store training statistics such 
        as loss and confusion matrix, to their initial values.
        This function has to be run before the start of each training experiment.
        """

        # List for storing loss at each step of the epoch.
        self.loss_list = []

        # Tensor for storing the confusion matrix for all datasamples from
        # the epoch.
        self.epoch_confusion_matrix = torch.zeros((2, 2), dtype=torch.long)

        # List for storing the final f1 score on the validation data after
        # each epoch. Comparing on the f1 scores across all epochs, we will
        # get the best performing model at the end of the training.
        self.val_f1_history = []
       




    def train_step(self, images, labels):
        """Training step"""
        ## Step 6b: Training Step. Refer the PyTorch tutorial if required.
        # Move image and labels to self.device
        images = images.to(self.device)
        #print("images shape", images.shape)
        labels = labels.to(self.device)
        self.optimiser.zero_grad()
        # zero optimiser weights?
        

        # Forward pass
        net_out = self.net(images)
        





        # Calculate the loss and append the loss into self.loss_list
        loss = self.loss_function(net_out, labels)
        self.loss_list.append(loss)





        # Get network predictions using the function from unet.py.
        # Using the network predictions, calculate the mini batch confusion matrix.
        # Add the mini batch confusion matrix to the self.epoch_confusion_matrix
        predicted_labels, lane_probability = get_network_prediction(net_out)
        minibatch_confusion_matrix = calculate_confusion_matrix(labels, predicted_labels)
        self.epoch_confusion_matrix = self.epoch_confusion_matrix + minibatch_confusion_matrix



        # Backward pass and optimize
        loss.backward()
        self.optimiser.step()
        







    def validation_step(self, images, labels):
        """Test model after an epoch and calculate loss on test dataset"""


        with torch.no_grad():
            ## Step 6c: Validation Step
            # Apply same steps as training step, except for backward pass and
            # optimization
            images = images.to(self.device)
            labels = labels.to(self.device)
            net_out = self.net(images)
            loss = self.loss_function(net_out, labels)
            self.loss_list.append(loss)
            predicted_labels, lane_probability = get_network_prediction(net_out)
            minibatch_confusion_matrix = calculate_confusion_matrix(labels, predicted_labels)
            self.epoch_confusion_matrix = self.epoch_confusion_matrix + minibatch_confusion_matrix
            
            


            self.tensor_board.add_img(images, predicted_labels,
                                      lane_probability)

    def predict(self, images, video_writer=None):
        # This is to run our model once its trained to predict unseen data, for example
        # Similar to valid_step() but labels here.
        # Note: Since true labels is not available for predict, the evaluation metrics such as
        # loss, F1 score and accuracy can be calculated here. Instead we can evaluate visually
        # by generating an overlay image.
        self.net.eval()

        with torch.no_grad():
            # Batch operation: depending on batch size more than one
            # image can be processed.
            ## Step 6d : Prediction Step
            # Move images to self.device
            images = images.to(self.device)


            # Forward pass
            net_out = self.net(images)



            # Get network predictions
            predicted_labels, lane_probability = get_network_prediction(net_out)





            # For each datasample, generate the overlay image using network predictions.
            # Next convert the image, overlay_img and RGB_lane_probability
            # into numpy array using the function that
            # you have implemented in ImageDataset.py
            # Note: For lane probability, we need a RGB image. Create a 3D zero
            # tensor named as 'RGB_lane_probability' with same shape as input
            # image. Place lane_probability in the red channel (channel=0) of
            # the tensor.
            #print("images size", images.size)
            for i in range(images.size(0)):  #(outputs.size(0))?
                
                # Create a 3D tensor with lane probablities in the red channel.
                # RGB_lane_probability = ...
                image = images[i]
                #plt.imshow(convert_tensor_to_numpy(image))
                
                RGB_lane_probability = torch.zeros(image.shape)
                RGB_lane_probability[0] = lane_probability[i]
                
                ##RGB_lane_probability = torch.transpose(RGB_lane_probability, 1, 2)
                
                #plt.imshow(convert_tensor_to_numpy(image))



                # Get overlay image (as RGB image)
                # overlay_img = ...
                overlay_img = get_overlay_image(image.clone(), predicted_labels[i])
                #plt.imshow(convert_tensor_to_numpy(image))




                # Convert all 3 tensors to numpy arrays: input, RGB_lane_probability and overlay image
                image = convert_tensor_to_numpy(image)
                
                RGB_lane_probability = convert_tensor_to_numpy(RGB_lane_probability)
                overlay_img = convert_tensor_to_numpy(overlay_img)
                
                



                if video_writer:
                    # Generate video frame by concatenate input and
                    # overlay image horizontally.
                    video_writer.write_frame(image, overlay_img)
                else:
                    # Visualize results using display output function
                    display_output(image, RGB_lane_probability, overlay_img)





    def print_stats(self, mode):
        """Calculate metrics  for the epoch and print the result. The
        self.loss_list and self.epoch_confusion_matrix will be reset at the end.
        In validation mode, the f1 score of the epoch is stored in
        self.val_f1_history."""
        avg_loss, accuracy, lane_f1 = calculate_metrics(self.loss_list,
                                                        self.epoch_confusion_matrix)
        # Print result to tensor board and std. output
        self.tensor_board.push_results(mode, avg_loss)

        print("{} \t\t{:8.4f}\t {:8.2%}\t {:8.2%}".format(mode, avg_loss,
                                                          accuracy, lane_f1))
        if mode == "Test":
            self.val_f1_history.append(lane_f1)

        # Reset stats
        self.loss_list = []
        self.epoch_confusion_matrix = torch.zeros((2, 2), dtype=torch.long)



    def save_model(self, epoch):
        trained_model_name = 'Epoch_{}.pth'.format(epoch)
        model_full_path = os.path.join(self.model_save_path, trained_model_name)
        torch.save(self.net.state_dict(), model_full_path)



    def get_best_model(self):
        ## Step 6e: Delete the below two lines (i.e. best_F1=0, best_epoch=0) and complete the missing
        #  code.
        #best_F1 = 0
        #best_epoch = 0


        # Using max and arg max functions on the self.val_f1_history, obtain the
        # the details of the best model and store them in the variables given below

        # best_F1 = ...
        # best_epoch = ...
        best_F1 = np.max(self.val_f1_history)
        best_epoch = np.argmax(self.val_f1_history)







        # DO NOT delete below lines!
        #######################################################################
        # Save the best model
        best_model_name = 'Epoch_{}.pth'.format(best_epoch)
        best_model_full_path = os.path.join(self.model_save_path,
                                            best_model_name)
        self.exp.params["best_model_full_path"] = best_model_full_path
        save_model_config(self)
        print(
            "Best model at Epoch {} with F1 score {:.2%}".format(best_epoch + 1,
                                                                 best_F1))

    def load_trained_model(self):
        """
        Setup the model by defining the model, load the model from the pth
        file saved during training.
        """

        model_path = self.exp.params["best_model_full_path"]
        self.net.load_state_dict(torch.load(model_path))




if __name__ == '__main__':
    pass
