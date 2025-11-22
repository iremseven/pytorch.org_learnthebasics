#save_and_load_the_model

import torch
import torchvision.models as models


#%%

#saving_and_loading_model_weights

model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')


#%%

#load_state_dict()

model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()


#%%

#saving_and_loading_models_with_shapes

torch.save(model, 'model.pth')


#%%

#torch.load()

model = torch.load('model.pth', weights_only=False)


#%%

