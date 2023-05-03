#%%
from load_modules import *
tf.compat.v1.disable_eager_execution()
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
tf.compat.v1.GPUOptions(allow_growth=True)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.list_physical_devices('GPU')
#gpu name
tf.test.gpu_device_name()
#check if gpu is 4090
name = tf.test.gpu_device_name()
if name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(name))
from tensorflow.python.client import device_lib

device_lib.list_local_devices()
class Autoencoder:

    def __init__(self):
        np.random.seed(42)
        self.encoder_model = self.encoder()
        self.decoder_model = self.decoder()
        self.model = self.autoencoder(self.encoder_model, self.decoder_model)

        self.lut_path = r"C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\data\JJ_LUT_v8.csv"
        self.save_path = r"C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\TrainedModels"
    def decoder(self):
        input = Input(shape=(5,))
        x = Dense(70, activation='relu')(input)
        x = Dense(70, activation='relu')(x)
        out = Dense(3)(x)
        model = Model(inputs=input, outputs=out, name='decoder')
        return model
    def encoder(self):
        input = Input(shape=(3,))
        x = Dense(70, activation='relu')(input)
        x = Dense(70, activation='relu')(x)
        out = Dense(5)(x)
        model = Model(inputs=input, outputs=out, name = 'encoder')
        return model
    def autoencoder(self, encoder, decoder):
        input_end_to_end = Input(shape=(3,))
        l1 = encoder(input_end_to_end)
        l2 = decoder(l1)
        input_list = [encoder.input, decoder.input, input_end_to_end]
        output_list = [encoder.output, decoder.output, l2]
        self.model = Model(inputs=input_list, outputs=output_list, name = 'autoencoder')
        opt = "adam"
        self.model.compile(optimizer=opt, loss=[self.custom_loss, self.custom_loss,self.custom_loss], loss_weights=[.4,.1, .5])
        return self.model
    
    def custom_loss(self, y_true, y_pred):
        return K.mean(K.square(y_true - y_pred))
    
    def train(self):
        checkpoint = ModelCheckpoint(r"autoencoder_best.h5py", monitor='loss', verbose=0,
        save_best_only=True, mode='auto', period=100)
        adjust_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=0, mode='auto', min_delta=0.0001, cooldown=1, min_lr=1e-5)
        callbacks = [
            checkpoint,
            adjust_lr
        ]
        x_train, x_test, y_train, y_test = self.load_data()
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
            self.model.fit(x,y, epochs=400, batch_size=512, shuffle=True, validation_data=(x_val, y_val), callbacks=callbacks)
    def load_data(self):
        np.random.seed(42)
        #load csv into 
        headers = "Cm,Ch,Bm,Bh,T,sR,sG,sB"
        df = pd.read_csv(self.lut_path, sep=",", header=None, names=headers.split(","))
        df.head()
        #remove header
        df = df.iloc[1:]
        #inputs = Cm,Ch,Bm,epi_thick
        y = df[['Cm','Ch','Bm','Bh','T']]
        print(y.head())
        #outputs = sR,sG,sB
        x = df[['sR','sG','sB']]
        print(x.head())
        df.head()
        #remove headers and convert to numpy array
        x = df[['sR','sG','sB']].iloc[1:].to_numpy()
        y = df[['Cm','Ch','Bm','Bh','T']].iloc[1:].to_numpy()
        #train nn on x,y
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)
        #numpy arrays
        self.x_train = np.asarray(self.x_train).reshape(-1,3).astype('float32')
        self.x_test = np.asarray(self.x_test).reshape(-1,3).astype('float32')

        print(f"bef norm self.x_train[0] {self.x_train[0]}")
        #normalize
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0
        return self.x_train, self.x_test, self.y_train, self.y_test
    def encode(self, img):
        image = np.asarray(img).reshape(-1,3).astype('float32') / 255.0
        # pred_maps = encoder.predict(image)
        pred_maps = self.encoder_model.predict_on_batch(image)
        return pred_maps  
    def decode(self, encoded):
        # recovered = decoder.predict(encoded)
        recovered = self.decoder_model.predict_on_batch(encoded)
        return recovered
    
class Load_Save:
    def __init__(self, save_path_encoder=None, save_path_decoder=None):
        self.save_path_encoder = save_path_encoder or r"C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\TrainedModels\lutv8_encoder"
        self.save_path_decoder = save_path_decoder or r"C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\TrainedModels\lutv8_decoder"
        
    def save_models(self, encoder, decoder):
        # Save the models
        encoder.save(self.save_path_encoder)
        decoder.save(self.save_path_decoder)
    
    def load_models(self):
        # Load the models
        loaded_encoder = load_model(self.save_path_encoder, compile=False)
        loaded_decoder = load_model(self.save_path_decoder, compile=False)
        return loaded_encoder, loaded_decoder

    def make_prediction(self, input_image, loaded_encoder, loaded_decoder):
        # Normalize the input image
        input_image = np.asarray(input_image).reshape(-1, 3).astype('float32') / 255.0

        # Encode the input image
        encoded_data = loaded_encoder.predict_on_batch(input_image)

        # Decode the encoded data
        decoded_data = loaded_decoder.predict_on_batch(encoded_data)

        return decoded_data
    def encode(self, img, loaded_encoder):
        image = np.asarray(img).reshape(-1,3).astype('float32') / 255.0
        # pred_maps = encoder.predict(image)
        pred_maps = loaded_encoder.predict_on_batch(image)
        return pred_maps
    def decode(self, encoded, loaded_decoder):
        # recovered = decoder.predict(encoded)
        recovered = loaded_decoder.predict_on_batch(encoded)
        return recovered
    
if __name__ == "__main__":
    """ 
    ae = Autoencoder()
    autoencoder = ae.model
    encoder = ae.encoder_model
    decoder = ae.decoder_model
    print(autoencoder.summary())
    print(encoder.summary())
    print(decoder.summary())
    ae.train()
    # Plot the loss
    #set text size
    plt.rcParams.update({'font.size': 8})
    #set font to segoi ui
    plt.plot(autoencoder.history.history['loss'])
    plt.plot(autoencoder.history.history['val_loss'])
    plt.title('model loss \n = mse(encoded_pred - encoded_true) + \n mse(decoded_pred - decoded_true) +\n mse(input_image - decoded_pred)')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    """
    #%%
    image_path = r"C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\AlbedoTextures\XYZ_albedo_lin_srgb.png"
    save_path = r"C:\Users\joeli\OneDrive\Documents\GitHub\AutoEncoder_Integrated\Results\Metina"

    image = Image.open(image_path).convert('RGB')
    height, width = image.size
    print(width, height)
    image = np.asarray(image)
    print(image.shape)
    dim = (width, height)
    #%%
    ls = Load_Save()
    encoder, decoder = ls.load_models()
    encode_start = time.time()
    with tf.device('/device:GPU:0'):
        pred_maps = ls.encode(image, encoder)
    encode_end = time.time()
    encode_time = encode_end - encode_start
    print(f"encode time: {encode_end - encode_start}")
   
    #%%
    decode_start = time.time()
    with tf.device('/device:GPU:0'):
        recovered = ls.decode(pred_maps, decoder)
    decode_end = time.time()
    decode_time = decode_end - decode_start
    print(f"decode time: {decode_end - decode_start}")
    #%%
    recovered = np.reshape(recovered, (width, height, 3))
    print(recovered.shape)
    #plot original and recovered side by side set fig size bigger than default
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Original")
    plt.subplot(1,2,2)
    plt.title("Recovered")
    plt.imshow(recovered)

    plt.show()
    # pred_maps = np.reshape(pred_maps, (width,height, 5))

    c_m = np.asarray(pred_maps[:,0]).reshape(width, height)
    c_h = np.asarray(pred_maps[:,1]).reshape(width, height)
    b_m = np.asarray(pred_maps[:,2]).reshape(width, height)
    b_h = np.asarray(pred_maps[:,3]).reshape(width, height)
    t = np.asarray(pred_maps[:,4]).reshape(width, height)
    """ 
    #save images
    dir_file = f"Cm_{width}x{height}.png"
    Cm_path = os.path.join(save_path, dir_file)
    plt.imsave(Cm_path, c_m, cmap=CMAP_SPECULAR)
    dir_file = f"Ch_{width}x{height}.png"
    Ch_path = os.path.join(save_path, dir_file)
    plt.imsave(Ch_path, c_h, cmap=CMAP_SPECULAR)
    dir_file = f"Bm_{width}x{height}.png"
    Bm_path = os.path.join(save_path, dir_file)
    plt.imsave(Bm_path, b_m, cmap=CMAP_SPECULAR)
    dir_file = f"Bh_{width}x{height}.png"
    Bh_path = os.path.join(save_path, dir_file)
    plt.imsave(Bh_path, b_h, cmap=CMAP_SPECULAR)
    dir_file = f"Epi_{width}x{height}.png"
    Epi_path = os.path.join(save_path, dir_file)
    plt.imsave(Epi_path, t, cmap=CMAP_SPECULAR)
    dir_file = f"recovered_{width}x{height}.png"
    recovered_path = os.path.join(save_path, dir_file)
    print(f"recovered shape {recovered.shape}")
    print(f"max value {np.max(recovered)}")
    recovered = recovered.reshape(width, height, 3)
    # recovered = Image.fromarray((recovered*255).astype(np.uint8))
    # plt.imsave(recovered_path, recovered)
    """

    #plot image, Cm,Ch,Bm,t, reconstructed image
    fig, axs = plt.subplots(1, 7, figsize=(35, 7))
    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    axs[1].imshow(c_m)
    plt.colorbar(axs[1].imshow(c_m, cmap=CMAP_SPECULAR), ax=axs[1],fraction=0.046, pad=0.04)
    axs[1].set_title('Cm')
    axs[2].imshow(c_h)
    plt.colorbar(axs[2].imshow(c_h, cmap=CMAP_SPECULAR), ax=axs[2],fraction=0.046, pad=0.04)
    axs[2].set_title('Ch')
    axs[3].imshow(b_m)
    plt.colorbar(axs[3].imshow(b_m, cmap=CMAP_SPECULAR), ax=axs[3],fraction=0.046, pad=0.04)
    axs[3].set_title('Bm')
    axs[4].imshow(b_h)
    plt.colorbar(axs[4].imshow(b_h, cmap=CMAP_SPECULAR), ax=axs[4],fraction=0.046, pad=0.04)
    axs[4].set_title('Bh')
    axs[5].imshow(t)
    plt.colorbar(axs[5].imshow(t, cmap=CMAP_SPECULAR), ax=axs[5],fraction=0.046, pad=0.04)
    axs[5].set_title('Epi Thickness')
    axs[6].imshow(recovered)
    axs[6].set_title('recovered Image')
    plt.suptitle('Original Image, Biophysical Parameter Maps, recovered Image \n  Encode Time: {}s, Decode Time: {}s'.format(encode_time, decode_time),y=.85, fontsize=20)
    #tight layout
    plt.tight_layout(pad=0.25, w_pad=0.3, h_pad=.75)
    #save figure
    dir_file = f"Original_Cm_Ch_Bm_Epi_recovered_{width}x{height}.png"
    save_path = os.path.join(save_path, dir_file)
    # plt.savefig(save_path)
    # plt.show()

    

# %%