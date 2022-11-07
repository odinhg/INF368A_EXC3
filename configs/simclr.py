# Settings SimCLR self-supervised learning 
batch_size = 128 
epochs = 35 
lr = 0.0014
val = 0.05 #Use 5% for validation data 
test = 0.2 #Use 20% for test data
image_size = (128, 128)

temperature = 0.5 # NTXent loss temperature parameter

model_type = "simclr"
