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
from data.image_dataset import convert_tensor_to_numpy
import os
from skimage.io import imsave



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


    def predict(self, images, video_writer=None, output_dir=None, show=True, idx_offset=0):
        """
        Predict on a batch of images. Optionally displays results (IDE-friendly)
        and saves outputs to `output_dir`.

        Args:
            images (Tensor): input batch (B, C, H, W)
            video_writer: (unused here; kept for compatibility)
            output_dir (str|None): if provided, saves per-sample images
            show (bool): if True, opens a matplotlib window to display results
            idx_offset (int): base index for naming saved files
        """

        self.net.eval()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            # Move to device and forward
            images = images.to(self.device)
            net_out = self.net(images)

            # Predictions
            predicted_labels, lane_probability = get_network_prediction(net_out)

            # For each item in the batch
            for i in range(images.size(0)):
                # Create RGB lane-probability tensor (prob into red channel)
                image_t = images[i].detach().clone()
                rgb_lane_prob = torch.zeros_like(image_t)
                rgb_lane_prob[0] = lane_probability[i]  # red channel

                # Overlay image
                overlay_t = get_overlay_image(image_t.clone(), predicted_labels[i])

                # Convert to numpy for viz/saving (expected H x W x 3 uint8)
                image_np = convert_tensor_to_numpy(image_t)
                rgb_lane_prob_np = convert_tensor_to_numpy(rgb_lane_prob)
                overlay_np = convert_tensor_to_numpy(overlay_t)

                # Optional: side-by-side composite for easier viewing
                # Make sure shapes match; if not, pad/crop as needed (usually they match).
                try:
                    composite = np.concatenate([image_np, rgb_lane_prob_np, overlay_np], axis=1)
                except ValueError:
                    # Fallback: resize lane prob/overlay to input width
                    h, w, _ = image_np.shape

                    def _safe_resize(x):
                        from skimage.transform import resize
                        x2 = resize(x, (h, w), preserve_range=True, anti_aliasing=True).astype(np.uint8)
                        return x2

                    composite = np.concatenate([image_np,
                                                _safe_resize(rgb_lane_prob_np),
                                                _safe_resize(overlay_np)], axis=1)

                # Show in an IDE-friendly window (blocking by default the first time)
                if show:
                    plt.figure("Prediction Results", figsize=(12, 4))
                    plt.imshow(composite)
                    plt.axis('off')
                    plt.tight_layout()
                    # Non-blocking show lets your loop continue but still pops a window.
                    # Close it manually or press 'x' depending on your IDE.
                    plt.show(block=False)
                    plt.pause(2)  # give GUI time to render

                # Save results
                if output_dir:
                    base = f"pred_{idx_offset + i:06d}"
                    in_path = os.path.join(output_dir, f"{base}_input.png")
                    prob_path = os.path.join(output_dir, f"{base}_laneprob.png")
                    ovl_path = os.path.join(output_dir, f"{base}_overlay.png")
                    cmp_path = os.path.join(output_dir, f"{base}_composite.png")

                    # Avoid skimage low-contrast warnings cluttering the console
                    try:
                        imsave(in_path, image_np)
                        imsave(prob_path, rgb_lane_prob_np)
                        imsave(ovl_path, overlay_np)
                        imsave(cmp_path, composite)
                    except Exception as e:
                        # If imsave fails (e.g., codec), fallback to matplotlib
                        import matplotlib.pyplot as _plt
                        _plt.imsave(in_path, image_np)
                        _plt.imsave(prob_path, rgb_lane_prob_np)
                        _plt.imsave(ovl_path, overlay_np)
                        _plt.imsave(cmp_path, composite)


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
