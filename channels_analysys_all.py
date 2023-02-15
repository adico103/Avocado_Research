import sys, os
import feather
import numpy as np
import matplotlib.pyplot as plt
from data_loader import Sample_loader
import regex as re
import pandas as pd
import params_update



# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class Channels_Analyzer:
    def __init__(self,
                hpc = False,
                model_type='random_forest',
                method = 'drop_one',
                plantations = False,
                test_plantation = 1
                    ):

        self.plantations = plantations
        self.test_plantation = test_plantation
        self.model_type = model_type
        self.method = method
        self.model_name = self.model_type+'_'+str(self.method)+'_method'
        

        if self.plantations:
            subfolderslist = ['train23_test1','train13_test2','train12_test3']
            self.channels_folder = 'channels_plantations//'+str(subfolderslist[self.test_plantation-1])
            self.analyze_folder = 'analyze/plantations//'+str(subfolderslist[self.test_plantation-1])
            self.best_models_folder = 'best_models_plantations//'+str(subfolderslist[self.test_plantation-1])
            self.graphs_folder = 'graphs_plantations//'+str(subfolderslist[self.test_plantation-1])
            self.data_type = 'Plantation '+str(self.test_plantation)
            



        else:
            self.channels_folder = 'channels'
            self.analyze_folder = 'analyze/random'
            self.best_models_folder = 'best_models//'
            self.graphs_folder = 'graphs'
            self.data_type = 'Random'

        # make dirs if nor exist
        if not os.path.exists(self.analyze_folder):
                    os.makedirs(self.analyze_folder)
        if not os.path.exists(self.graphs_folder):
                    os.makedirs(self.graphs_folder)
        if not os.path.exists(self.graphs_folder+'//best_channel_number'):
                    os.makedirs(self.graphs_folder+'//best_channel_number')
        if not os.path.exists(self.graphs_folder+'//total_best_wl'):
                    os.makedirs(self.graphs_folder+'//total_best_wl')

        if hpc:
            self.ref_folder = 'ref//'
            self.labels_file = 'labels//days_to_stage_4.feather'
            data_folder ='data_vnir'
        else:
            self.ref_folder = 'ref\\'
            self.labels_file = 'labels\days_to_stage_4.feather'
            data_folder = params_update.data_folder
        
        
        img_folder = 'vnir_avocado_avocado_25-28_2019-12-03_11-26'
        blockPrint()
        sample = Sample_loader(img_folder,data_folder)
        enablePrint()
        self.wl_range = sample.wavelength_range

        print('Analyzing: |Model = '+self.model_type+'| |Method = '+self.method+'| |Test Data Split = '+self.data_type+'|')
    
    def get_filename(self,bands_num,i):
        filename = str(self.model_type)+'_'+str(bands_num)+'bands_0.6per_train_'+str(i)+'iter.feather'
        return filename
            

    def save_data(self,all_lengths,all_best_bands,train_errs,test_errs,cvs,upper_bounds):
        model_data = {}
        model_data['num_channels'] = all_lengths
        model_data['best_bands'] = all_best_bands
        model_data['train_score'] = train_errs
        model_data['test_score'] = test_errs
        model_data['cv_score'] = cvs
        model_data['upper_bound'] = upper_bounds

        model_data = pd.DataFrame(model_data)
        model_data.columns = model_data.columns.astype(str)
        model_data.reset_index(drop=True)

        
        model_data.to_feather(self.analyze_folder+'//'+self.model_name+'_'+str(len(all_best_bands))+'top_models_Analysis.feather')

    def sort_best_models(self,n):
        # sort models by "worst case" - the upper bound (worst error) of the estimation
        self.sorted_models = np.unravel_index(np.argsort(np.array(self.upper_stds), axis=None), np.array(self.upper_stds).shape)
        all_lengths = []
        all_best_bands = []
        train_errs = []
        test_errs = []
        cvs = []
        upper_bounds = []
        for i in range(n):
            ind = self.sorted_models[0][i]
            all_lengths.append(self.all_lengths[ind])
            all_best_bands.append(self.all_best_bands[ind])
            train_errs.append(self.train_errs[ind])
            test_errs.append(self.test_errs[ind])
            cvs.append(self.cvs[ind])
            upper_bounds.append(self.upper_stds[ind])

        self.save_data(all_lengths,all_best_bands,train_errs,test_errs,cvs,upper_bounds)


    def all_best_models(self,n):
        self.all_lengths = []
        self.all_best_bands = []
        self.train_errs = []
        self.test_errs = []
        self.cvs = []

        
        for filename in os.listdir(self.best_models_folder):
            if filename.startswith(self.model_type):
                if (self.method in filename):
                    params_file = os.path.join(self.channels_folder,self.method)+'\\'+filename[:-7]+'.feather'
                    params = feather.read_dataframe(params_file)
                    num_bands = len(params.bands_selection[0])

                    self.all_lengths.append(num_bands)
                    self.all_best_bands.append(params.bands[0])
                    self.train_errs.append(params.train_score[0])
                    self.test_errs.append(params.test_score[0])
                    self.cvs.append(params.cv_score[0])

        self.upper_stds = []
        self.lower_stds = []
        for cv,test_err in zip(self.cvs,self.test_errs):
            std = cv*test_err
            self.upper_stds.append(test_err+np.sqrt(std))
            self.lower_stds.append(test_err-np.sqrt(std))

        self.sort_best_models(n)

        self.save_data(self.all_lengths,self.all_best_bands,self.train_errs,self.test_errs,self.cvs,self.upper_stds)

    def best_wavelengths(self,n,plot =True):
        all_best_bands = []
        test_errs = []
        
        for i in range(n):
            ind = self.sorted_models[0][i]
            all_best_bands.append(self.all_best_bands[ind])
            test_errs.append(self.test_errs[ind])

        flatlist=[]
        for sublist in all_best_bands:
            for element in sublist:
                flatlist.append(element)
        self.unique_bands = np.unique(np.array(flatlist))
        # print(len(self.unique_bands))

        self.scores_list = np.zeros_like(self.unique_bands).astype(float)
        max_test_err = max(test_errs)

        for i in range(len(all_best_bands)):
            bands = all_best_bands[i]
            bands_score = (max_test_err - test_errs[i]).astype(float)+1
            self.scores_list[np.isin(self.unique_bands,bands)]+=bands_score

        self.wls = self.wl_range[self.unique_bands]
        if plot:
            self.plot_best_wls(n)


    def plot_best_wls(self,n):
        x = self.wls
        y = 100*(self.scores_list/max(self.scores_list))
        
        plt.figure()
        plt.plot(x,y)
        extraticks = []
        old_ticks = list(plt.xticks()[0])
        inds = np.unravel_index(np.argsort(np.array(y), axis=None), np.array(y).shape)
        for counter in range(min(5,len(inds[0]))):
            ind = inds[0][len(inds[0])-counter-1]
            x_sel = x[ind]
            y_sel = y[ind]
            plt.plot((x_sel, x_sel), (0, y_sel), 'r',linewidth=0.5)
            extraticks.append(x_sel)
            old_ticks = np.delete(old_ticks,np.where(old_ticks==np.round(x_sel/100)*100))

            
        plt.xticks(list(old_ticks) + extraticks)

        plt.grid()
        plt.title('Wavelength Importance: '+self.model_name)
        plt.ylim([min(y)-2,max(y)+2])
        # plt.xticks(range(1, 20))
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Importance')
        plt.savefig(self.graphs_folder+'//total_best_wl//'+self.model_name+'tot_best_wl'+str(n)+'models.png')
        plt.close()


    def plot_best_ch_num_train_test(self):

        X = self.all_lengths
        Y = self.train_errs
        y1 = [x for _,x in sorted(zip(X,Y))]
        Y = self.test_errs
        y2 = [x for _,x in sorted(zip(X,Y))]
        x = np.sort(X)
        # self.cvs
        
        plt.figure()
        plt.plot(x,y1,label = 'train')
        plt.plot(x,y2,label = 'test')
        

        plt.grid()
        plt.title('Channel Number Importance: '+self.model_name)
        plt.legend()
        plt.xlim([0,max(x)+5])
        plt.xlabel('Channels No.')
        plt.ylabel('Error [Day]')
        plt.savefig(self.graphs_folder+'//best_channel_number//'+self.model_name+'ch_num_train_test.png')
        plt.close()


    def plot_best_ch_num_only_test(self):

        X = self.all_lengths
        Y = self.train_errs
        y1 = [x for _,x in sorted(zip(X,Y))]
        Y = self.test_errs
        y2 = [x for _,x in sorted(zip(X,Y))]
        x = np.sort(X)


        fig,ax =plt.subplots()
        # plt.plot(x,y1,label = 'train')
        ax.plot(x,y2,label = 'Test')
        extraticksx = []
        extraticksy = []
        ax.fill_between(x, self.lower_stds, self.upper_stds, color='b', alpha=.15,label='Standard Deviation')
        old_ticksx = list(plt.xticks()[0])
        old_ticksy = list(plt.yticks()[0])
        
        
        for counter in range(5):
            ind = self.sorted_models[0][counter]
            x_sel = x[ind]
            y_sel = self.upper_stds[ind]
            if ((counter ==0) or (counter == 4)) :
                plt.plot((x_sel, 0), (y_sel, y_sel), 'r--',linewidth=0.5)
                extraticksy.append(y_sel)

                
            plt.plot((x_sel, x_sel), (0, y_sel), 'r--',linewidth=0.5)
            extraticksx.append(x_sel)
            
            old_ticksx = np.delete(old_ticksx,np.where(old_ticksx==np.round(x_sel/10)*10))
            old_ticksy = np.delete(old_ticksy,np.where(old_ticksy==np.round(y_sel)))


        
        plt.xticks(list(old_ticksx) + extraticksx)
        plt.yticks(list(old_ticksy) + extraticksy)
        
        plt.grid()
        plt.title('Channel Number Importance: '+self.model_name)
        plt.legend()
        plt.xlim([0,max(x)])
        # plt.ylim([min(y2)-0.05,max(y2)+0.05])
        plt.xlabel('Channels No.')
        plt.ylabel('Error [Day]')
        plt.savefig(self.graphs_folder+'//best_channel_number//'+self.model_name+'ch_num_test.png')
        plt.close()


    def plot_everything(self):
        self.plot_best_ch_num_train_test()
        self.plot_best_ch_num_only_test()



hpc=params_update.hpc

model_types = ['random_forest','xgboost','svm']
methods = ['drop_one', 'drop_more', 'random_select','add_one']



for model_type in model_types:
    for method in methods:
        channels_tester = Channels_Analyzer(hpc=hpc, 
                                            model_type=model_type,
                                            method=method,
                                            )
        
        channels_tester.all_best_models(n=5)
        channels_tester.best_wavelengths(n=len(channels_tester.all_best_bands))
        channels_tester.best_wavelengths(n=5)
        channels_tester.best_wavelengths(n=1)
        channels_tester.plot_everything()
        print('Complete')


# Plantations data split
for plns in range(3):
# plns = 2
    model_types = ['random_forest','xgboost','svm']
    methods = ['drop_one', 'drop_more', 'random_select','add_one']
    for model_type in model_types:
        for method in methods:
            channels_tester = Channels_Analyzer(hpc=hpc, 
                                                model_type=model_type,
                                                method=method,
                                                plantations = True,
                                                test_plantation=plns+1)
                                                
            
            channels_tester.all_best_models(n=5)
            channels_tester.best_wavelengths(n=len(channels_tester.all_best_bands))
            channels_tester.best_wavelengths(n=5)
            channels_tester.best_wavelengths(n=1)
            channels_tester.plot_everything()
            print('Complete')


