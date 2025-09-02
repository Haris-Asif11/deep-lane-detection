import torch


def calculate_confusion_matrix(true_label, predicted_labels):
    """Calculates the confusion matrix using the given true label tensor and
     the predicted label tensor. """
    # Refer: https://en.wikipedia.org/wiki/Confusion_matrix

    ## Step 4a: Delete the lines below (i.e. true_positive = 0, etc) and complete the missing code to obtain the number of
    # true positive, true negative, false positive and false negative pixels
    # in given the batch.
    # true_positive = ...


    #print(true_label)
    #print(predicted_labels)
    x = true_label+predicted_labels
   
    true_positive = torch.sum(x == 2)
    true_negative = torch.sum(x==0) #label shape is N x H x W
    false_positive = torch.sum(predicted_labels[x == 1] == 1)   #torch.sum(predicted_labels[true_label == 0] == 1)
    false_negative = torch.sum(predicted_labels[x == 1] == 0)



    confusion_matrix = torch.tensor([[true_positive, false_positive],#in lab false positive and false negative entries are changed 
                                     [false_negative, true_negative]])

    return confusion_matrix



def calculate_overall_accuracy(confusion_matrix):
    """Calculate the overall accuracy for the lane class using the
    confusion matrix """
    # Refer: https://en.wikipedia.org/wiki/Confusion_matrix
    # Important: Convert the datatype of numerator and denominator to float
    # before performing division. Otherwise, python will perform integer
    # division which will result in incorrect value.
    # Refer: https://en.wikipedia.org/wiki/Confusion_matrix

    ## Step 4b: Delete the below statement (i.e. accuracy = 0) and complete the
    # missing code.
    # you add confusion matrix after every batch so at end of epoch you have results for the entire dataset
    num = confusion_matrix[0][0]+confusion_matrix[1][1]
    num = num.float()
    den = confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[1][0]+confusion_matrix[1][1]
    den = den.float()
    accuracy = num/den 

    return accuracy





def calculate_lane_f1_score(confusion_matrix):
    """Calculate f1 score for the lane class using the confusion matrix"""
    # Refer: https://en.wikipedia.org/wiki/Confusion_matrix
    # Important: Convert the datatype of numerator and denominator to float
    # before performing division. Otherwise, python will perform integer
    # division which will result in incorrect value.

    ## Step 4c: Delete the below statement (i.e. lane_f1=0) and complete the
    # missing code.
    
    num = 2 * confusion_matrix[0][0]
    num = num.float()
    den = (2*confusion_matrix[0][0]) + confusion_matrix[0][1] + confusion_matrix[1][0]
    den = den.float()
    lane_f1 = num/den

    
    return lane_f1


def calculate_metrics(loss_list, confusion_matrix):
    """
    Return the average loss, overall accuracy and the lane f1 score for the
    epoch.
    """


    ## Step 4d: Fill in the missing code and uncomment the return statement in last line.
    # Calculate the average loss using the values in loss_list
    # Convert list to tensor inorder to torch functions
    t = torch.tensor(loss_list)
    avg_loss = torch.mean(t)
     

    # Calculate stats
    

    # Using the functions defined above, obtain the overall accuracy and
    # f1 score for the lane
    accuracy = calculate_overall_accuracy(confusion_matrix)
    lane_f1 = calculate_lane_f1_score(confusion_matrix)


    return avg_loss, accuracy, lane_f1




