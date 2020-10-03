import torch
import torchvision
from torchvision import transforms


#Path Variables
#Path to data
aachen_real_PATH = '../RoadDetection - SqueezeNet/CityScapes_RoadDetection-dtl/aachen/real_img/'
aachen_mask_PATH = '../RoadDetection - SqueezeNet/CityScapes_RoadDetection-dtl/aachen/masks_machine/'

dusseldorf_real_PATH = '../RoadDetection - SqueezeNet/CityScapes_RoadDetection-dtl/dusseldorf/real_img/'
dusseldorf_mask_PATH = '../RoadDetection - SqueezeNet/CityScapes_RoadDetection-dtl/dusseldorf/masks_machine/'

munster_real_PATH = '../RoadDetection - SqueezeNet/CityScapes_RoadDetection-dtl/munster/real_img/'
munster_mask_PATH = '../RoadDetection - SqueezeNet/CityScapes_RoadDetection-dtl/munster/masks_machine/'

ulm_real_PATH = '../RoadDetection - SqueezeNet/CityScapes_RoadDetection-dtl/ulm/real_img/'
ulm_mask_PATH = '../RoadDetection - SqueezeNet/CityScapes_RoadDetection-dtl/ulm/masks_machine/'

bremen_real_PATH = '../RoadDetection - SqueezeNet/CityScapes_RoadDetection-dtl/bremen/real_img/'
bremen_mask_PATH = '../RoadDetection - SqueezeNet/CityScapes_RoadDetection-dtl/bremen/masks_machine/'

lindau_real_PATH = '../RoadDetection - SqueezeNet/CityScapes_RoadDetection-dtl/lindau/real_img/'
lindau_mask_PATH = '../RoadDetection - SqueezeNet/CityScapes_RoadDetection-dtl/lindau/masks_machine/'

stuttgart_real_PATH = '../RoadDetection - SqueezeNet/CityScapes_RoadDetection-dtl/stuttgart/real_img/'
stuttgart_mask_PATH = '../RoadDetection - SqueezeNet/CityScapes_RoadDetection-dtl/stuttgart/masks_machine/'


class Roads():
    def __init__(self):
        pass
    

    
    #Load data method
    def load_data_from(self, path, grayscale):
        #The input images are RGB, while labels are grayscale
        if grayscale:
            #Creates a tensor for grayscale images. 
            temp_tensor = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
            
        else:
            #Just create a tensor
            temp_tensor = transforms.Compose([transforms.ToTensor()])
            
        #The ImageFolder method loads a dataset from the given path
        dataset = torchvision.datasets.ImageFolder(root = path, transform = temp_tensor)
        
        return dataset
    
    
    def load_data(self):
        dataset_aachen_real = self.load_data_from(aachen_real_PATH, False)
        dataset_aachen_mask = self.load_data_from(aachen_mask_PATH, True)
        # print("Quantidade de imagens - aachen: ",len(dataset_aachen_real))
        
        dataset_dusseldorf_real = self.load_data_from(dusseldorf_real_PATH, False)
        dataset_dusseldorf_mask = self.load_data_from(dusseldorf_mask_PATH, True)
        # print("Quantidade de imagens - dusseldorf: ",len(dataset_dusseldorf_real))
        
        dataset_munster_real = self.load_data_from(munster_real_PATH, False)
        dataset_munster_mask = self.load_data_from(munster_mask_PATH, True)
        # print("Quantidade de imagens - munster: ",len(dataset_munster_real))
        
        dataset_ulm_real = self.load_data_from(ulm_real_PATH, False)
        dataset_ulm_mask = self.load_data_from(ulm_mask_PATH, True)
        # print("Quantidade de imagens - ulm: ",len(dataset_ulm_real))
        
        dataset_bremen_real = self.load_data_from(bremen_real_PATH, False)
        dataset_bremen_mask = self.load_data_from(bremen_mask_PATH, True)
        # print("Quantidade de imagens - bremen: ",len(dataset_bremen_real))
        
        dataset_lindau_real = self.load_data_from(lindau_real_PATH, False)
        dataset_lindau_mask = self.load_data_from(lindau_mask_PATH, True)
        # print("Quantidade de imagens - lindau: ",len(dataset_lindau_real))
        
        dataset_stuttgart_real = self.load_data_from(stuttgart_real_PATH, False)
        dataset_stuttgart_mask = self.load_data_from(stuttgart_mask_PATH, True)
        # print("Quantidade de imagens - stuttgart: ",len(dataset_stuttgart_real))
        
    
        #Concatenate the pre-datasets
        training_set_x = torch.utils.data.ConcatDataset([dataset_aachen_real,
                                                dataset_bremen_real,
                                                dataset_dusseldorf_real,
                                                dataset_stuttgart_real,
                                                dataset_ulm_real])
        
        training_set_y = torch.utils.data.ConcatDataset([dataset_aachen_mask,
                                                dataset_bremen_mask,
                                                dataset_dusseldorf_mask,
                                                dataset_stuttgart_mask,
                                                dataset_ulm_mask])
        
        
        #############TEMPORARIOOO##################
        # training_set_x = torch.utils.data.ConcatDataset([
                                               
        #                                         dataset_ulm_real])
        
        
        # training_set_y = torch.utils.data.ConcatDataset([
                                                
        #                                         dataset_ulm_mask])
        ###########################################
        
        #Loads all the training data into a single dataset
        training_set = [training_set_x, training_set_y]
        
        #Concatenate the pre-datasets 
        test_set_x = torch.utils.data.ConcatDataset([dataset_lindau_real,
                                               dataset_munster_real])

        test_set_y = torch.utils.data.ConcatDataset([dataset_lindau_mask,
                                                       dataset_munster_mask])
        
        test_set = [test_set_x, test_set_y]
        

        # print(len(training_set_x) )
        # print(len(training_set_y))
        #print(len(training_set))                         

        return training_set, test_set


