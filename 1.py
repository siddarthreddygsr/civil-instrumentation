
import numpy as np
import os
import time
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
import tensorflow as tf

from keras.applications.vgg16 import preprocess_input
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

# Loading the training data
PATH = os.getcwd()
# Define data path
data_path = PATH + '/data'
data_dir_list = ['Crack','NoCrack']

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+"/"+ dataset)
	#print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = data_path + '/'+ dataset + '/'+ img 
		imge = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(imge)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		#print('Input image shape:', x.shape)
		img_data_list.append(x)

img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
#print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
#print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)


# Define the number of classes
num_classes = 2
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:int(num_of_samples/2)]=0

labels[3*int(num_of_samples/2):num_of_samples]=1


names = ['Crack','NoCrack']
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=2)

###########################################################################################################################
# Custom_resnet_model_1
#Training the classifier alone
image_input = Input(shape=(224,224, 3))

model = tf.keras.applications.ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')
#model.summary()
x= Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='output_layer')(x)
custom_resnet_model = Model(inputs=image_input,outputs= out)
#custom_resnet_model.summary()

for layer in custom_resnet_model.layers[:-1]:
	layer.trainable = False

custom_resnet_model.layers[-1].trainable

custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
Noepochs=10
t=time.time()
hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=Noepochs, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

###########################################################################################################################


aaa = custom_resnet_model.predict(X_test)
i=0
while(i<len(aaa)):
    if(aaa[i][0]<=0.5):
          aaa[i][0]=0
    else:
          aaa[i][0]=1
    if(aaa[i][1]<=0.5):
          aaa[i][1]=0
    else:
          aaa[i][1]=1
    
    
          
    i+=1

#####################################################################################
#predicting and dividing Crack and NoCrack into different folders
import shutil
_, _, predict_images = next(os.walk(PATH+'/data/predict/data'))
pred_data_list = []
for img in predict_images:
		img_path = PATH+'/data/predict/data' + '/'+ img 
		imge = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(imge)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		#print('Input image shape:', x.shape)
		pred_data_list.append(x)
        
pred_data = np.array(pred_data_list)
#img_data = img_data.astype('float32')
pred_data=np.rollaxis(pred_data,1,0)
pred_data=pred_data[0]
print (img_data.shape)
data_class = custom_resnet_model.predict(pred_data)
i=0
while(i<len(data_class)):
    if(data_class[i][0]<=0.5):
          data_class[i][0]=0
    else:
          data_class[i][0]=1
    if(data_class[i][1]<=0.5):
          data_class[i][1]=0
    else:
          data_class[i][1]=1
    
    i+=1

shutil.rmtree(PATH+'/data/predict/Cracks/',ignore_errors=True) 
shutil.rmtree(PATH+'/data/predict/NoCracks/',ignore_errors=True) 
shutil.rmtree(PATH+'/data/predict/Spallingg/',ignore_errors=True)
shutil.rmtree(PATH+'/data/predict/Corrisionn/',ignore_errors=True)   
os.makedirs(PATH+'/data/predict/Cracks/') 
os.makedirs(PATH+'/data/predict/NoCracks/') 

i=0
for j in data_class:
    if j[0]==1:
        shutil.copyfile(PATH+'/data/predict/data/'+predict_images[i], PATH+'/data/predict/Cracks/'+predict_images[i])
    elif j[1]==1:
        shutil.copyfile(PATH+'/data/predict/data/'+predict_images[i], PATH+'/data/predict/NoCracks/'+predict_images[i])  
    i+=1
shutil.rmtree(PATH+'/Cracks/',ignore_errors=True) 
os.makedirs(PATH+'/Cracks/')
from distutils.dir_util import copy_tree
copy_tree(PATH+'/data/predict/Cracks/', PATH+'/Cracks/')

#####################################################################################
import matplotlib.pyplot as plt
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
xc=range(Noepochs)

plt.figure(1,figsize=(8,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(8,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show(block=False)




cm = confusion_matrix(y_test.argmax(axis=1), aaa.argmax(axis=1))
ac = y_test.argmax(axis=1)
pc = aaa.argmax(axis=1)
tp =0
for i,j in zip(ac,pc):
    if(i==j==0):
        tp+=1
tn =0
for i,j in zip(ac,pc):
    if(i==j==1):
        tn+=1
fn =0;
for i,j in zip(ac,pc):
    if(i!=j==1):
        fn+=1
fp =0;
for i,j in zip(ac,pc):
    if(i!=j==0):
        fp+=1
pres = tp/(tp+fp)
recall = tp/(tp+fn)
fmeasure = (2*pres*recall)/(pres+recall)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Crack','NoCrack'])

disp.plot(cmap=plt.cm.Blues)
plt.show()
print("Precision: ","{:.3f}".format(pres))
print("Recall: ","{:.3f}".format(recall))
print("F-measure","{:.3f}".format(fmeasure))


