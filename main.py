from SignLanguageHacettepe import SignLanguageRecognizer
import torch
"""Use this callable parameter to take img input
and return a prediction class"""




save_dir='saved_images'

#take letters
letters = open("letters.txt", "r")
human_letter = letters.read().split("\n")



vgg16=torch.load("trained_vggModel")
Recognizer=SignLanguageRecognizer(torch_model=vgg16,prediction_interval=0.1,save_key='s',quit_key='q',kernel_size=3)
Recognizer.Track(save_dir,human_letter)

