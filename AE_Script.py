# %%
from load_modules import *
# %%
np.random.seed(42)
#load csv into 
# data_path = r"C:\Users\joeli\OneDrive\Documents\GitHub\EncoderDecoder\JJ_LUT.csv"
headers = "Cm,Ch,Bm,Bh,T,sR,sG,sB,L,A,B"

lut_path = r"C:\Users\joeli\OneDrive\Documents\GitHub\PhysicalModels\LargeMCLUT.csv"
df = pd.read_csv(lut_path, sep=",", header=None, names=headers.split(","))
df.drop(df.index[0], inplace=True)
# Filter out non-numeric rows
def is_numeric(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

x = df[['sR', 'sG', 'sB']].applymap(is_numeric)
x = df[x.all(axis=1)][['sR', 'sG', 'sB']].to_numpy(dtype='float32')
y = df[['Cm', 'Ch', 'Bm', 'Bh', 'T']].applymap(is_numeric)
y = df[y.all(axis=1)][['Cm', 'Ch', 'Bm', 'Bh', 'T']].to_numpy(dtype='float32')

df.head()

#train nn on x,y
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)
#remove any header values
x_train = x_train[1:]
x_test = x_test[1:]
y_train = y_train[1:]
y_test = y_test[1:]

#numpy arrays
x_train = np.asarray(x_train).reshape(-1,3).astype('float32')
x_test = np.asarray(x_test).reshape(-1,3).astype('float32')

print(f"bef norm x_train[0] {x_train[0]}")

#normalize
x_train = x_train/255.0
x_test = x_test/255.0
print(f"aft norm x_train[0] {x_train[0]}")

print(f"length of x_train {len(x_train)}")
print(f"length of x_test {len(x_test)}")
print(f"length of y_train {len(y_train)}")
print(f"length of y_test {len(y_test)}")
df.head()

# %%
np.random.seed(42)

def decoder():
    input = Input(shape=(5,))
    x = Dense(70, activation='relu')(input)
    x = Dense(70, activation='relu')(x)
    out = Dense(3)(x)
    model = Model(inputs=input, outputs=out, name='decoder')
    return model
def encoder():
    input = Input(shape=(3,))
    x = Dense(70, activation='relu')(input)
    x = Dense(70, activation='relu')(x)
    out = Dense(5)(x)
    model = Model(inputs=input, outputs=out, name = 'encoder')
    return model
def autoencoder(encoder, decoder):
    input_end_to_end = Input(shape=(3,))
    l1 = encoder(input_end_to_end)
    l2 = decoder(l1)
    input_list = [encoder.input, decoder.input, input_end_to_end]
    output_list = [encoder.output, decoder.output, l2]
    model = Model(inputs=input_list, outputs=output_list, name = 'autoencoder')
    return model
encoder = encoder()
decoder = decoder()
autoencoder = autoencoder(encoder, decoder)
print(encoder.summary())
print(decoder.summary())
print(autoencoder.summary())
albedo_pred_values = []
albedo_true_values = []
def albedo_loss(y_true, y_pred):
    #l1 norm
    l1_norm = K.sum(K.abs(y_pred - y_true), axis=-1)
    albedo_pred_values.append(y_pred)
    albedo_true_values.append(y_true)
    return l1_norm

def parameter_loss(y_true, y_pred):
    #l2 norm
    l2_norm = K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
    return l2_norm
def end_to_end_loss(y_true, y_pred):
    #l1 norm
    l1_norm = K.sum(K.abs(y_pred - y_true), axis=-1)
    return l1_norm

# # Compile the autoencoder with the custom loss function and optimizer
autoencoder.compile(optimizer='adam', loss=[parameter_loss, albedo_loss, end_to_end_loss], loss_weights=[.3,.1, .6])


# %%
checkpoint = ModelCheckpoint(r"C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\autoencoder_best.h5py", monitor='loss', verbose=0,
    save_best_only=True, mode='auto', period=200)
adjust_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-7)
# Define the Keras TensorBoard callback.

logdir=r"C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\tensorboard_log_dir"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(filepath=logdir, save_weights_only = True,save_freq = 100,verbose = 1)

callbacks = [
    checkpoint,
    adjust_lr
]
with tf.device('/device:GPU:0'):
    #show device name
    print(tf.test.gpu_device_name())
    #ae_in: enc_in, dec_in, end_to_end_in
    x = [x_train, y_train,x_train]
     #ae_out: enc_out, dec_out, end_to_end_out
    x_val = [x_test, y_test,x_test]
    #outputs: encoder, decoder, autoencoder
    y = [y_train,x_train,x_train]
    y_val = [y_test,x_test,x_test]
    # print(tf.config.list_physical_devices('GPU'))
    autoencoder.fit(x,y, epochs=400, batch_size=64, shuffle=True, validation_data=(x_val, y_val), callbacks=callbacks)

       

# %%
plt.rcParams['grid.linestyle'] = ''
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
# plt.style.use('tableau-colorblind10')

#set font to segoi ui
plt.rcParams['font.family'] = 'Segoe UI'
plt.plot(autoencoder.history.history['loss'])
plt.plot(autoencoder.history.history['val_loss'])
plt.title('model loss \n = encoder loss (L1 norm) + \n decoder loss (L2 norm) +\n end to end loss (L1 norm) ')
plt.ylabel('loss')
plt.yscale('log')
plt.xlabel('epoch')
plt.yticks
# plt.xlim(-1,50)
plt.legend(['train', 'test'], loc='upper right')
plt.show()



# %%
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

CMAP_SPECULAR = plt.get_cmap('viridis')

def encode(img):
    image = np.asarray(img).reshape(-1,3).astype('float32')/255.0
    pred_maps = encoder.predict_on_batch(image)
    return pred_maps

def decode(encoded):
    recovered = decoder.predict_on_batch(encoded)
    recovered = np.clip(recovered, 0, 1)
    return recovered

def process_single_albedo_image(image_path):
    if not os.path.exists(image_path):
        print(f"Path {image_path} does not exist.")
        return

    top_path = "LargeLUT_2k"
    if not os.path.exists(top_path):
        os.mkdir(top_path)
        print(f"created {top_path}")

    albedo_path = os.path.dirname(image_path)
    recovered_path = os.path.join(albedo_path, "recovered")
    original_path = os.path.join(albedo_path, "original")
    parameter_path = os.path.join(albedo_path, "parameters")
    result_path = os.path.join(albedo_path, "results")
    top_path = os.path.join(albedo_path, top_path)

    for path in [recovered_path, original_path, parameter_path, result_path]:
        if not os.path.exists(path):
            os.mkdir(path)
            print(f"created {path}")

    mpl.rcParams['axes.grid'] = False

    file_name = os.path.basename(image_path)
    file_name = file_name.split("_")[1]
    print(file_name)
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(image_path)
    image = image.resize((4096,4096))
    image = image.convert('RGB')
    height, width = image.size
    print(width, height)

    image = np.asarray(image)
    print(image.shape)
    dim = (width, height)
    encode_start = time.time()
    pred_maps = encode(image)
    #melanin
    melanin = pred_maps[:,0].reshape((width, height))
    # #increase melanin 
    # melanin = melanin * 1.1+0.1
    # pred_maps[:,0] = melanin.reshape((width*height))
    # #hemoglobin
    hemoglobin = pred_maps[:,1].reshape((width, height))
    # #Bm
    # blend_mel = pred_maps[:,2].reshape((width, height))
    # blend_mel = blend_mel * 1.2+0.1
    encode_end = time.time()
    encode_time = encode_end - encode_start
    print(f"encode time: {encode_time}")
    decode_start = time.time()
    recovered = decode(pred_maps)
    decode_end = time.time()
    decode_time = decode_end - decode_start
    print(f"decode time: {decode_time}")
    recovered = np.reshape(recovered, (width, height, 3))
    print(recovered.shape)
    return image, pred_maps, recovered, melanin, hemoglobin


if __name__ == "__main__":
    image_path = r"C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\Results\txyz01\Original_Albedos_2048by2048\albedo_m32.png"
    image, pred_maps, recovered,melanin, hemoglobin = process_single_albedo_image(image_path)
    #plot original and recovered side by side set fig size bigger than default
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Original")
    plt.subplot(1,2,2)
    plt.title("Recovered")
    plt.imshow(recovered)
    plt.show()
    #plot the maps
    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.imshow(melanin, cmap='viridis')
    plt.title("Melanin")
    plt.subplot(2,2,2)
    plt.imshow(hemoglobin, cmap='viridis')
    plt.title("Hemoglobin")
    #save recovered image
    plt.imsave(r"C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\Results\txyz01\Original_Albedos_2048by2048\Increased_Melanin_Bm_recovered_m32.png", recovered, dpi=2000)


# %%
def encode(img):
    image = np.asarray(img).reshape(-1,3).astype('float32')/255.0
    # pred_maps = encoder.predict(image)
    pred_maps = encoder.predict_on_batch(image)
    print(f"shape of pred_maps {pred_maps.shape}")
    return pred_maps
 
def decode(encoded):
    # recovered = decoder.predict(encoded)
    recovered = decoder.predict_on_batch(encoded)
    #clamp to 0-1
    recovered = np.clip(recovered, 0, 1)
    return recovered


test_path = r"C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\AlbedoTextures\text_xyz\original\original"
# C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\Results\txyz01\Original_Albedos_2048by2048
top_path = "LargeLUT_2k"
if not os.path.exists(top_path):
    os.mkdir(top_path)
    print(f"created {top_path}")
recovered_path = os.path.join(test_path, "recovered")
original_path = os.path.join(test_path, "original")
parameter_path = os.path.join(test_path, "parameters")
result_path = os.path.join(test_path, "results")
top_path = os.path.join(test_path, top_path)
if not os.path.exists(recovered_path):
    os.mkdir(recovered_path)
    print(f"created {recovered_path}")
if not os.path.exists(original_path):
    os.mkdir(original_path)
    print(f"created {original_path}")
if not os.path.exists(parameter_path):
    os.mkdir(parameter_path)
    print(f"created {parameter_path}")
if not os.path.exists(result_path):
    os.mkdir(result_path)
    print(f"created {result_path}")
test_images = os.listdir(test_path)
length = len(test_images)
#for each image in the test set
for i in range(len(test_images)):
    if not test_images[i].endswith(".png"):
        continue
    length -= 1
    if length <= 2:
        break
    mpl.rcParams['axes.grid'] = False

    image_path = os.path.join(test_path, test_images[i])
    file_name = os.path.basename(image_path)
    print(file_name)
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(image_path)
    image = image.resize((2048,2048))
    image = image.convert('RGB')
    height, width = image.size
    print(width, height)
    image = np.asarray(image)
    print(image.shape)
    dim = (width, height)
    encode_start = time.time()
    pred_maps = encode(image)


    encode_end = time.time()
    encode_time = encode_end - encode_start
    print(f"encode time: {encode_time}")
    decode_start = time.time()
    recovered = decode(pred_maps)
    decode_end = time.time()
    decode_time = decode_end - decode_start
    print(f"decode time: {decode_time}")
    recovered = np.reshape(recovered, (width, height, 3))
    print(recovered.shape)

    #plot original and recovered side by side set fig size bigger than default
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Original")
    plt.subplot(1,2,2)
    plt.title("Recovered")
    plt.imshow(recovered)
    plt.tight_layout(h_pad=0, w_pad=0, pad=0)
    for ax in plt.gcf().axes:
        ax.axis('off')
    plt.show()
    # pred_maps = np.reshape(pred_maps, (width,height, 5))

    c_m = np.asarray(pred_maps[:,0]).reshape(width, height)
    #increase melanin amount
    c_m = c_m * 10+0.2
    c_h = np.asarray(pred_maps[:,1]).reshape(width, height)
    b_m = np.asarray(pred_maps[:,2]).reshape(width, height)
    b_h = np.asarray(pred_maps[:,3]).reshape(width, height)
    t = np.asarray(pred_maps[:,4]).reshape(width, height)

    #save images
    print(f"file {file_name}")
    dir_file = f"{file_name}"
    path = os.path.join(original_path, dir_file)
    # plt.imsave(path, image, dpi=1000)
    dir_file = f"Cm_{file_name}"
    path = os.path.join(parameter_path, dir_file)
    plt.imsave(path, c_m, cmap=CMAP_SPECULAR, dpi=1000)
    dir_file = f"Ch_{file_name}"
    path = os.path.join(parameter_path, dir_file)
    plt.imsave(path, c_h, cmap=CMAP_SPECULAR, dpi=1000)
    dir_file = f"Bm_{file_name}"
    path = os.path.join(parameter_path, dir_file)
    plt.imsave(path, b_m, cmap=CMAP_SPECULAR, dpi=1000)
    dir_file = f"Bh_{file_name}"
    path = os.path.join(parameter_path, dir_file)
    plt.imsave(path, b_h, cmap=CMAP_SPECULAR, dpi=1000)
    dir_file = f"T_{file_name}"
    path = os.path.join(parameter_path, dir_file)
    plt.imsave(path, t, cmap=CMAP_SPECULAR, dpi=1000)
    dir_file = f"recovered_{file_name}"
    path = os.path.join(recovered_path, dir_file)
    recovered = recovered.reshape(width, height, 3)
    plt.imsave(path, recovered, dpi=1000)
    dir_file = f"original_{file_name}"
    path = os.path.join(original_path, dir_file)
    plt.imsave(path, image, dpi=1000)
    #plot image, Cm,Ch,Bm,t, reconstructed image
    fig, axs = plt.subplots(1, 5, figsize=(40, 7) )
    #set wspace and hspace to 0
    fig.subplots_adjust(wspace=0, hspace=0)
    axs[0].imshow(c_m, cmap=CMAP_SPECULAR)
    # plt.colorbar(axs[1].imshow(c_m, cmap=CMAP_SPECULAR), ax=axs[1],fraction=0.046, pad=0.04)
    axs[0].set_title('Cm', fontsize=22)
    axs[1].imshow(c_h, cmap=CMAP_SPECULAR)
    # plt.colorbar(axs[2].imshow(c_h, cmap=CMAP_SPECULAR), ax=axs[2],fraction=0.046, pad=0.04)
    axs[1].set_title('Ch', fontsize=22)
    axs[2].imshow(b_m, cmap=CMAP_SPECULAR)
    # plt.colorbar(axs[3].imshow(b_m, cmap=CMAP_SPECULAR), ax=axs[3],fraction=0.046, pad=0.04)
    axs[2].set_title('Bm', fontsize=22)
    axs[3].imshow(b_h, cmap=CMAP_SPECULAR)
    # plt.colorbar(axs[4].imshow(b_h, cmap=CMAP_SPECULAR), ax=axs[4],fraction=0.046, pad=0.04)
    axs[3].set_title('Bh', fontsize=22)
    axs[4].imshow(t, cmap=CMAP_SPECULAR)
    plt.colorbar(axs[4].imshow(t, cmap=CMAP_SPECULAR), ax=axs[4],fraction=0.046)
    axs[4].set_title('T', fontsize=22)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    #save figure
    dir_file = f"parameter_maps_{file_name}"
    path = os.path.join(result_path, dir_file)
    plt.savefig(path, dpi=1000)

    plt.show()


# %%

mpl.rcParams['axes.grid'] = False
plt.rcParams['figure.dpi'] = 1000
# Set the directories for the original and recovered images
original_directory = r'C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\AlbedoTextures\text_xyz\original\original\original'
recovered_directory = r'C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\AlbedoTextures\text_xyz\recovered'

# Get a list of file names in the original and recovered image directories
original_files = os.listdir(original_directory)
recovered_files = os.listdir(recovered_directory)

for i, (original_file, recovered_file) in enumerate(zip(original_files, recovered_files)):
    original_img = Image.open(os.path.join(original_directory, original_file)).convert('RGB')
    recovered_img = Image.open(os.path.join(recovered_directory, recovered_file)).convert('RGB')
    print(original_img.size)
    print(recovered_img.size)

    # Convert the images to the CAM02-UCS color space and calculate the delta E loss
    original_lab = colorspacious.cspace_convert(np.array(original_img), 'sRGB1', 'CAM02-UCS')
    recovered_lab = colorspacious.cspace_convert(np.array(recovered_img), 'sRGB1', 'CAM02-UCS')
    delta_e = np.sqrt(np.sum((original_lab - recovered_lab)**2, axis=2))



    plt.imshow(delta_e, cmap='viridis', vmin=0, vmax=20)

    #title
    title = original_file.split('\\')[-1].split('.')[0].split('_')[2]
    # title = "ms180lll90k009"
    #get number from string
    title = ''.join(filter(lambda x: x.isdigit(), title))
    title = f"txyz_model_{title}"
    print(title)
    plt.title(f'{title} \n Delta E Loss, mean: {np.mean(delta_e):.2f} \n min: {np.min(delta_e)} max: {np.max(delta_e)} \n Delta E Loss std: {np.std(delta_e):.2f}')
    #add space between subplots
    plt.tight_layout(w_pad=0.02, h_pad=.4)
    #set text size
    plt.rcParams.update({'font.size': 8})
    # Add a colorbar to the figure
    cbar = plt.colorbar()
    # plt.suptitle('Delta E Loss between: \nOriginal and Recovered Images \n', fontsize=8)
    #set padding to minimal
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.show()

# %%
import os
import numpy as np
from PIL import Image
import colorspacious
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False

# Set the directories for the original and recovered images
original_directory = r'C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\AlbedoTextures\text_xyz\original\original\original'
recovered_directory = r'C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\AlbedoTextures\text_xyz\recovered'

# Get a list of file names in the original and recovered image directories
original_files = os.listdir(original_directory)
recovered_files = os.listdir(recovered_directory)

# Create a single figure for all heat maps
fig, axes = plt.subplots(5, 2, figsize=(10,25), dpi=1000)

# Loop over the original files and match them to the recovered files
for i, (original_file, recovered_file) in enumerate(zip(original_files, recovered_files)):
    original_img = Image.open(os.path.join(original_directory, original_file)).convert('RGB')
    recovered_img = Image.open(os.path.join(recovered_directory, recovered_file)).convert('RGB')
    print(original_img.size)
    print(recovered_img.size)

    # Convert the images to the CAM02-UCS color space and calculate the delta E loss
    original_lab = colorspacious.cspace_convert(np.array(original_img), 'sRGB1', 'CAM02-UCS')
    recovered_lab = colorspacious.cspace_convert(np.array(recovered_img), 'sRGB1', 'CAM02-UCS')
    delta_e = np.sqrt(np.sum((original_lab - recovered_lab)**2, axis=2))

    # Add a subplot for the current pair of images
    ax = axes[i // 2, i % 2]
    im = ax.imshow(delta_e, cmap='viridis', vmin=0, vmax=20)
    ax.set_title(f'{original_file} and {recovered_file} \n Delta E Loss, mean: {np.mean(delta_e):.2f} \n min: {np.min(delta_e)} max: {np.max(delta_e)} \n Delta E Loss std: {np.std(delta_e):.2f}')
#add space between subplots
plt.tight_layout(w_pad=0.02, h_pad=.4)
#set text size
plt.rcParams.update({'font.size': 8})
# Add a colorbar to the figure
fig.colorbar(im, ax=fig.axes, orientation='vertical', fraction=0.025)
plt.suptitle('Delta E Loss between: \nOriginal and Recovered Images \n', fontsize=8)
plt.show()


# %%
image_path = r"C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\AlbedoTextures\text_xyz\albedo_m64.png"
image = Image.open(image_path).convert('RGB')
# image = np.asarray(image)
# image = image/255.0


height, width = image.size
print(width, height)
#resize image to 4096
image = image.resize((4096,4096))
width = 4096
height = 4096
#numpy array
image = np.asarray(image)
print(image.shape)
dim = (width, height)
encode_start = time.time()
pred_maps = encode(image)


encode_end = time.time()
encode_time = encode_end - encode_start
print(f"encode time: {encode_time}")
decode_start = time.time()
recovered = decode(pred_maps)
decode_end = time.time()
decode_time = decode_end - decode_start
print(f"decode time: {decode_time}")
recovered = np.reshape(recovered, (width, height, 3))
print(recovered.shape)

print(f"image shape {image.shape}")

#plot image, parametric maps, reconstructed image
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
for ax in axs:
    ax.axis('off')
axs[0].imshow(image)
axs[0].set_title("Original Image")
axs[1].imshow(recovered)
axs[1].set_title("Recovered Image")
plt.suptitle(f"AE Results: LUT Size {17500} Image Size {height}x{width}' \n Encode Time: {encode_time}s, Decode Time: {decode_time}s",y=1.2, fontsize=20)
plt.tight_layout(pad=0.25)
# plt.suptitle('Original Image, recovered Image \n  Encode Time: {}s, Decode Time: {}s'.format(encode_time, decode_time),y=1.2, fontsize=20)
plt.show()

fig, axs = plt.subplots(1, 5, figsize=(15, 5))
axs[0].imshow(c_m, cmap=CMAP_SPECULAR)
axs[0].set_title("Cm")
plt.colorbar(axs[0].imshow(c_m, cmap=CMAP_SPECULAR), ax=axs[0],fraction=0.046, pad=0.04)
axs[1].imshow(c_h, cmap=CMAP_SPECULAR)
axs[1].set_title("Ch")
plt.colorbar(axs[1].imshow(c_h, cmap=CMAP_SPECULAR), ax=axs[1],fraction=0.046, pad=0.04)
axs[2].imshow(b_m, cmap=CMAP_SPECULAR)
axs[2].set_title("Bm")
plt.colorbar(axs[2].imshow(b_m, cmap=CMAP_SPECULAR), ax=axs[2],fraction=0.046, pad=0.04)
axs[3].imshow(b_h, cmap=CMAP_SPECULAR)
axs[3].set_title("Bh")
plt.colorbar(axs[3].imshow(b_h, cmap=CMAP_SPECULAR), ax=axs[3],fraction=0.046, pad=0.04)
axs[4].imshow(t, cmap=CMAP_SPECULAR)
axs[4].set_title("T")
plt.colorbar(axs[4].imshow(t, cmap=CMAP_SPECULAR), ax=axs[4],fraction=0.046, pad=0.04)
# plt.tight_layout(pad=0.25, w_pad=0.44, h_pad=.85)
#remove axis ticks
for ax in axs:
    ax.axis('off')

plt.suptitle(f'Biophysical Parameter Maps \n Image Size {height}x{width}',y=.85, fontsize=12)
    
plt.show()

# %%
import os

# input_name = input("enter name of model to save:")
input_name = "March25"

#save model
save_path = r"C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\TrainedModels"
encoder_path = os.path.join(save_path, f"{input_name}_encoder")
decoder_path = os.path.join(save_path, f"{input_name}_decoder")
autoencoder_path = os.path.join(save_path, f"{input_name}_autoencoder")
print(encoder_path)
print(decoder_path)
print(autoencoder_path)
#save model weights and architecture to single file as h5
encoder.save(encoder_path, save_format="h5py")
decoder.save(decoder_path, save_format="h5py")

# %%
saved_encoder_h5 = r"C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\TrainedModels\March25_encoder"
saved_decoder_h5 = r"C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\TrainedModels\March25_decoder"
reload_encoder = tf.keras.models.load_model(saved_encoder_h5)
reload_decoder = tf.keras.models.load_model(saved_decoder_h5)



# %%
# input_name = input("enter name of model to save:")
input_name = "March25"

#save model
save_path = r"C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\json_models"
encoder_path = os.path.join(save_path, f"{input_name}_encoder")
decoder_path = os.path.join(save_path, f"{input_name}_decoder")
autoencoder_path = os.path.join(save_path, f"{input_name}_autoencoder")
#save as json
encoder_path 
encoder_json = encoder.to_json()
decoder_json = decoder.to_json()
with open(f"{encoder_path}.json", "w") as json_file:
    json_file.write(encoder_json)
with open(f"{decoder_path}.json", "w") as json_file:
    json_file.write(decoder_json)


# %%
#save model weights and architecture to single file as h5
encoder.save_weights(encoder_path)
decoder.save_weights(decoder_path)

# %%
import os
import tensorflow as tf
from tensorflow.python.framework import graph_io
# from tensorflow.python.compiler.tensorrt import trt_convert as trt

def export_model(model, input_node_names, output_node_name, model_name):
    if not os.path.exists('out'):
        os.mkdir('out')

    # Save the model in the TensorFlow SavedModel format
    model.save(r'C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\json_models\{model_name}', save_format='h5py')

    # Convert the Keras model to a frozen graph
    frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess=tf.compat.v1.keras.backend.get_session(),
        input_graph_def=tf.compat.v1.keras.backend.get_session().graph.as_graph_def(),
        output_node_names=[output_node_name]
    )

    # Save the frozen graph to a binary protobuf file
    graph_io.write_graph(frozen_graph, 'out', f'frozen_{model_name}.pb', as_text=False)

    # Optimize the frozen graph for inference
    input_graph_def = frozen_graph
    output_graph_def = tf.compat.v1.graph_util.remove_training_nodes(input_graph_def)

    # Save the optimized graph to a binary protobuf file
    with tf.io.gfile.GFile(f'out/opt_{model_name}.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())

    print("Graph saved!")

# Example usage:
encoder_model = encoder
input_node_names = [encoder_model.input.name.split(':')[0]]
output_node_name = encoder_model.output.name.split(':')[0]
export_model(encoder_model, input_node_names, output_node_name, "encoder")




