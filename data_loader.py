
import os
import regex as re
import spectral as spy
import numpy as np
import utils
import cv2 as cv
import pandas as pd
import feather
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import params_update

spy.settings.envi_support_nonlowercase_params = True

class Sample_loader:
    def __init__(self,
                img_fold = None,
                data_folder =None,
                load_img_file = False,
                load_fourier_file=False,
                avocado_num = None,
                saving_segmentation = params_update.saving_segmentation,
                saving_fourier = params_update.saving_fourier
                    ):
        self.saving_segmentation = saving_segmentation
        self.saving_fourier = saving_fourier
        if load_img_file:
            self.how_many = 1
        else:
            self.how_many = 4
        if not load_img_file:
            print('Loading hyper spectral image: '+str(img_fold))
            files_found = 0
            self.img_fold =img_fold
            img_fold = os.path.join(data_folder,img_fold+'/capture')
            for file in os.listdir(img_fold):
                if file.endswith('.raw'):
                    raw_file = os.path.join(img_fold,file)
                    files_found+=1

                if file.endswith('.hdr'):
                    hdr_file = os.path.join(img_fold,file)
                    files_found+=1

            
            if files_found<2:
                self.working=False
            # else:
            #     self.working=True
            try:
                self.open_image(hdr_file,raw_file)
                self.working=True
            
            except:
                self.working=False

            self.tot_bands = self.spec_img.shape[2]
            # self.irtype = (re.search(r'.19\\(.*?)\\', img_fold).group(1)).split('-')
            self.samples = (re.search(r'avocado_avocado_(.*?)_2019', img_fold).group(1)).split('-')
            self.ndvi_bands = [680.06,770.64]

        if load_img_file:
            self.avocado_num = avocado_num
            self.load_img_data()


    def open_image(self,hdr_file,raw_file):
        spec_img = spy.io.envi.open(hdr_file, raw_file)
        self.wavelength_range = np.array(spec_img.metadata['Wavelength']).astype(float)
        self.spec_img = np.transpose(spec_img._memmap,(0,2,1))

    def segmentation(self,load_file =False, draw=False):

        if load_file:
            load = feather.read_dataframe(self.saving_segmentation+'segmentation_'+str(self.img_fold)+'.feather')
            self.avocado_num = list(list(load['avocado_num'])[0])
            self.bounding_boxes = list(list(load['bounding_boxes'])[0])
            self.avocado_num = list(list(load['avocado_num'])[0])
            self.ndvi_img = load['ndvi_img'][0][0].reshape((3900,1600))
            self.centers = np.array(load['centers'][0])
            self.avocado_pixs = load['avocodo_pixs'][0][0].reshape((3900,1600))
            self.countours = []
            countours = load['countours'][0]
            shapes = load['cont_shapes'][0]
            for contour,sh in zip(countours,shapes):
                self.countours.append(contour.reshape(np.array(sh)))

        else:
            print('Avocado segmentation is calculated...')
            imgs = []
            for band in self.ndvi_bands:
                ind = np.where(self.wavelength_range==band)[0][0]
                imgs.append(self.spec_img[:,:,ind])
            self.bounding_boxes, self.countours, self.ndvi_img = utils.segmentation(imgs)
            

            centers = []
            for (bounding_box) in self.bounding_boxes:
                center = (bounding_box[0]+0.5*bounding_box[2],bounding_box[1]+0.5*bounding_box[3])
                centers.append(center)
            self.centers = np.array(centers)
            c_centers = (np.mean(self.centers[:,0]),np.mean(self.centers[:,1]) )
            vecs = self.centers-c_centers
            loc = [3,1,2,0]
            self.avocado_num = []
            for vec in vecs:
                s0 = np.sign(vec[0])
                s1 = np.sign(vec[1])
                ind = np.argmax(np.array([s0<0 , s0<0, s0>0,s0>0]).astype(int)+np.array([s1>0 , s1<0, s1>0,s1<0]).astype(int))
                self.avocado_num.append(loc[ind]+int(self.samples[0]))

            img = np.zeros_like(self.ndvi_img)
            self.avocado_pixs = utils.get_avocado_pix(self.avocado_num,self.countours,img)

            new_countours = []
            cont_shapes = []
            for contour in self.countours:
                new_countours.append(np.array(contour).flatten())
                cont_shapes.append(contour.shape)
            print('Saving segmentation')
            df = {'avocado_num': [self.avocado_num],'bounding_boxes':[self.bounding_boxes],'ndvi_img': [[self.ndvi_img.flatten()]*4],
                    'countours': [new_countours],'cont_shapes': [cont_shapes], 'centers': [list(self.centers)], 'avocodo_pixs': [[self.avocado_pixs.flatten()]*4]}
            df = pd.DataFrame(df)
            df.columns = df.columns.astype(str)
            df.reset_index(drop=True)
            df.to_feather(self.saving_segmentation+'segmentation_'+str(self.img_fold)+'.feather')

        if draw:
            self.draw_segmentation()

    def draw_segmentation(self):

        direc = "segmentation//"
        name = direc+'segmentation_'+self.samples[0]+'-'+self.samples[1]
        if not os.path.exists(direc):
            os.makedirs(direc)

        utils.draw_segmentation(self.centers,self.avocado_num,self.countours,self.ndvi_img,name = name)
    
    def split_avocados(self,draw=False):

        direc = "all_avocados//"
        
        if not os.path.exists(direc):
            os.makedirs(direc)
        
        img_direc = "segmentation//exampls//"
        
        if not os.path.exists(img_direc):
            os.makedirs(img_direc)

        print('Spiltting image for every Avocado...')

        for avocado_num,box in zip(self.avocado_num,self.bounding_boxes):
            print('Extracting Avocado num: ' +str(avocado_num))
            all_bands = {'band': [], 'image': [], 'shape': []}
            imgs = []
            
            shapes = []
            for band in range(self.tot_bands):
                k = np.zeros_like(self.avocado_pixs)
                k[self.avocado_pixs==avocado_num] = self.spec_img[:,:,band][self.avocado_pixs==avocado_num]
                k = k[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
                imgs.append(k.flatten())
                shapes.append((box[3],box[2]))

            name = direc+'avocado_'+str(avocado_num)+'.feather'
            print('saving file: '+name)
            all_bands['image'] = imgs
            all_bands['shape'] = shapes
            all_bands['band'] = list(self.wavelength_range)
            all_bands = pd.DataFrame(all_bands)
            all_bands.columns = all_bands.columns.astype(str)
            all_bands.reset_index(drop=True)
            all_bands.to_feather(name)
            


            if draw:
                cv.imwrite(img_direc+'Avocado_'+str(avocado_num)+'.png',np.array(imgs[700]).reshape((box[3],box[2])))
    
    def get_fourier(self,selected_img):

        dft = cv.dft(np.float32(selected_img),flags = cv.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        return magnitude_spectrum

    def get_img(self,b,avocado_num,box,band_inds):

        
        # mask image by specific avocado pixels
        selected_img = self.spec_img[:,:,band_inds]
        mask = np.zeros(self.spec_img.shape[:2], dtype="uint8")
        mask[self.avocado_pixs==avocado_num] = 255
        masked = cv.bitwise_and(selected_img, selected_img, mask=mask)
        selected_img = masked[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
        return selected_img
    
    def load_img_data(self):
        self.all_imgs = feather.read_dataframe(self.saving_fourier+'img_avocado_'+str(self.avocado_num)+'.feather')


    def Process_by_band(self,bands,other_nums = [], save=True,fourier=True):

        all_data = []
        all_fourier_data = []
        self.all_imgs = []
        self.all_shapes = []
        self.all_bands = []
    
        if len(other_nums)>0:
            nums = other_nums
        else:
            nums = self.avocado_num
        
        selectes_avocados_inds = np.isin(self.avocado_num,nums)


        for avocado_num,box in zip(np.array(self.avocado_num)[selectes_avocados_inds],np.array(self.bounding_boxes)[selectes_avocados_inds]):
            fourier_data = {'Avocado_num': [],'band': [], 'image': [], 'img_shape': []}
            img_data = {'Avocado_num': [],'band': [], 'image': [], 'img_shape': []}
            imgs = []
            fft_imgs = []
            shapes = []
            samps = []
            actual_bands = []


            print('Extracting image for avocado : ' +str(avocado_num))
            for b in bands:

                band_inds = abs(self.wavelength_range-b).argmin()
                band = self.wavelength_range[band_inds]
                actual_bands.append(band)

                selected_img = self.get_img(b,avocado_num,box,band_inds)
                imgs.append(np.array(selected_img).flatten())
                # cv.imwrite(saving_fourier+'before_fourier_'+str(avocado_num)+'_'+str(b)+'.png',utils.scale_img(selected_img))
                shapes.append((box[3],box[2]))
                samps.append(avocado_num)

                if fourier:
                    print('Calculation fourier transform for avocado : ' +str(avocado_num))
                    magnitude_spectrum = self.get_fourier(selected_img)
                    # cv.imwrite(saving_fourier+'fourier_'+str(avocado_num)+'_'+str(b)+'.png',utils.scale_img(magnitude_spectrum))
                    fft_imgs.append(np.array(magnitude_spectrum).flatten())
                

            if fourier:
                fourier_data['image'] = fft_imgs
                fourier_data['img_shape'] = shapes
                fourier_data['band'] = list(actual_bands)
                fourier_data['Avocado_num'] = samps

            img_data['image'] = imgs
            img_data['img_shape'] = shapes
            img_data['band'] = list(actual_bands)
            img_data['Avocado_num'] = samps
            self.all_imgs.append(imgs)
            self.all_shapes.append(shapes)
            self.all_bands.append(actual_bands)

            if save:
                if fourier:
                    fourier_data = pd.DataFrame(fourier_data)
                    fourier_data.columns = fourier_data.columns.astype(str)
                    fourier_data.reset_index(drop=True)
                    name = self.saving_fourier+'fourier_avocado_'+str(avocado_num)+'.feather'
                    print('saving file: '+name)
                    fourier_data.to_feather(name)

                img_data = pd.DataFrame(img_data)
                img_data.columns = img_data.columns.astype(str)
                img_data.reset_index(drop=True)
                name = self.saving_fourier+'img_avocado_'+str(avocado_num)+'.feather'
                print('saving file: '+name)
                img_data.to_feather(name)

            if fourier:
                all_fourier_data.append(fourier_data)

            all_data.append(img_data)

        return all_fourier_data,all_data
    
    def get_pca(self):
        pca_data = {'first_component_loading': [], 'nums': []}
        fcl = []
        av_nums = []
        pca = PCA(n_components=5)
        if self.how_many>1:
            for i in range(self.how_many):

                all_bands = []
                for img in self.all_imgs[i]:
                    all_bands.append(img)
                all_bands  = np.array(all_bands)
                all_bands = all_bands.transpose((1,0))

                X_transformed = pca.fit_transform(all_bands)
                first_component_loading = pca.components_[0]
                fcl.append(first_component_loading)
                av_nums.append(self.avocado_num)

        pca_data['first_component_loading'] = fcl
        pca_data['nums'] = av_nums
        pca_data = pd.DataFrame(pca_data)
        pca_data.columns = pca_data.columns.astype(str)
        pca_data.reset_index(drop=True)
        name = 'pca/pca_data_'+str(min(self.avocado_num))+'.feather'
        print('saving file: '+name)
        pca_data.to_feather(name)


        return fcl,self.avocado_num
    
    def feature_extraction(self):
        
        refs = []
        if self.how_many>1:
            data = {'bands': [],'ref': [], 'partial_ref': [] , 'size_ratio': [] }
            
            for i in range(self.how_many): # for every avocado
                
                norm_band = self.all_imgs[i][20]
                # norm_band = self.all_imgs[i][2]
                max_val = np.max(norm_band)
                min_val = np.min(norm_band)
                ref = []
                bands = []
                partial_ref = []
                size_ratios = []
                ((centx,centy), (width,height), angle) = cv.fitEllipse(self.countours[i])

                size_ratio = width/height


                band_counter = 0
                for img in self.all_imgs[i]: # loop over all bands

                    # features of every image in every band:

                    # Avg refleactnce of the avocado in this band
                    new_im = img[img!=0]
                    new_im = (new_im-min_val)/(max_val-min_val)
                    ref.append(np.mean(new_im))

                    # save this wavelength
                    bands.append(self.all_bands[i][band_counter])
                    band_counter+=1

                    #update size ratio:
                    size_ratios.append(size_ratio)

                    # calculate refleactnce for parts of the avocado
                    shape = self.all_shapes[i][0]
                    shaped_img = img.reshape(shape)
                    rot_img = utils.rotate_image(shaped_img,angle)
                    x_split = 3
                    y_split = 1
                    x_inc = int(rot_img.shape[0]/x_split)
                    y_inc = int(rot_img.shape[1]/y_split)
                    sub_imgs_ref = []
                    for row in range(x_split):
                        for col in range(y_split):
                            sub_img = rot_img[row*x_inc:(row+1)*x_inc,col*y_inc:(col+1)*y_inc]
                            sub_img_new = sub_img[sub_img!=0]
                            # normaliztion
                            sub_img_new = (sub_img_new-min_val)/(max_val-min_val)
                            sub_imgs_ref.append(np.mean(sub_img_new))
                            # print(i*x_inc,(i+1)*x_inc,j*y_inc,(j+1)*y_inc)
                    partial_ref.append(sub_imgs_ref)
                

                data['ref'] = ref
                data['bands'] = bands
                data['partial_ref'] = partial_ref
                data['size_ratio'] = size_ratios
                
                data = pd.DataFrame(data)
                data.columns = data.columns.astype(str)
                data.reset_index(drop=True)
                name = 'ref/ref_data_'+str(self.avocado_num[i])+'.feather'
                print('saving file: '+name)
                data.to_feather(name)

        


