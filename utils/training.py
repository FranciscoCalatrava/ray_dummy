import torch
import datetime
import os
from ray import tune


class Trainer():
    def __init__(self, g_function,classifier, train_dataloader, val_dataloader, device, optimizer, criterion, epochs, writer) -> None:
        self.g_function = g_function
        self.classifier = classifier
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.writer = writer
        self.epochs = epochs
    
    def model(self, encoder, classifier):
        return lambda x: classifier(encoder(x))
    
    def train_one_epoch(self):
        running_loss = 0.0
        samples_per_epoch = 0
        avg_loss = 0.0
        self.g_function.train(True)
        self.classifier.train(True)
        correct_predictions = 0
        accuracy = 0.0
        
        for i, data in enumerate(self.train_dataloader):
            inputs,targets = data[0].to(self.device).float(), data[1].to(self.device).long()
         
            self.optimizer.zero_grad()

            outputs = self.model(self.g_function, self.classifier)(inputs)

            loss = self.criterion(outputs, targets)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            predictions = torch.argmax(outputs, dim= 1)
            correct_predictions += (torch.squeeze(predictions) == targets).sum()

            samples_per_epoch += targets.size(0)
        accuracy =  correct_predictions/samples_per_epoch     
        avg_loss = running_loss/samples_per_epoch
        return avg_loss, accuracy
    
    def validation(self):
        self.g_function.eval()
        self.classifier.eval()
        running_validation_loss = 0.0
        samples = 0
        avg_validation_loss = 0
        correct_predictions = 0
        accuracy = 0.0

        with torch.no_grad():
            for i, vdata in enumerate(self.val_dataloader):
                inputs, targets = vdata[0].to(self.device).float(), vdata[1].to(self.device).long()
                outputs = self.model(self.g_function, self.classifier)(inputs)

                validation_loss = self.criterion(outputs, targets)
                running_validation_loss += validation_loss.item()

                predictions = torch.argmax(outputs, dim= 1)
                correct_predictions += (torch.squeeze(predictions) == targets).sum()
                samples += targets.size(0)
            accuracy = correct_predictions/samples
            avg_validation_loss = running_validation_loss/samples
        return avg_validation_loss, accuracy

    def train(self):
        for a in range(self.epochs):
            avg_loss, acc_train = self.train_one_epoch()
            avg_validation_loss, acc_val = self.validation()
            #tune.report(mean_accuracy=avg_validation_loss, mean_loss=acc_val)
            print(f"Loss {avg_loss} | {avg_validation_loss} ||| Acc {acc_train} | {acc_val}")
        return self.saveModel(self.g_function, self.classifier)

    
    def saveModel(self,encoder, classifier):
        ##*********************************************************************Inputs**********************************************************************##
        ## Description: It is save the model in an specific path:  yearmonthday-hourminutessencods                                                         ##
        ##.................................................................................................................................................##
        ## model -- The model that i am training                                                                                                           ##
        ##*********************************************************************Outputs*********************************************************************##
        ## Path -- The path in which I have save the model                                                                                                 ##
        ##*************************************************************************************************************************************************##
        path = "/home/calatrava/Documents/PhD/Papers/Cross_Subject_Transfer_Learning/experiments/other_works/pamap_supervised/saved_models"
        datetime_ = '_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"pid_"+str(os.getpid())
        torch.save(classifier, path+"classifier"+datetime_)
        torch.save(encoder,path+"encoder"+datetime_)
        path_encoder = path+"encoder"+datetime_
        path_classifier = path+"classifier"+datetime_
        return path_encoder, path_classifier
            

    


