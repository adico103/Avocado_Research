import feather
import numpy as np
import random
import pandas as pd
from all_models import random_forest,my_xgboost,LinearRegression_model,SVM_model
import itertools
import statistics
import pickle
import joblib
import params_update

class Channels_Tester:
    def __init__(self,
                hpc = False,
                model_type='random_forest',
                train_per = 0.6,
                max_bands = 84,
                max_drop = 1,
                num_trials = 100,
                method = 'drop_one',
                plantations = False,
                test_plantation = 1

                    ):

        self.plantations = plantations
        self.test_plantation = test_plantation
        self.max_drop = max_drop
        self.max_bands = max_bands
        self.model_type=model_type
        self.train_per = train_per
        self.num_trials = num_trials
        self.method = method
        self.drop_func = lambda mxbnds,bnds : ((max_drop-1)*bnds)/(mxbnds-2) - ((max_drop-1)*mxbnds)/(mxbnds-2) + max_drop

        if self.plantations:
            subfolderslist = ['train23_test1','train13_test2','train12_test3']
            self.channels_folder = 'channels_plantations//'+str(subfolderslist[self.test_plantation-1])
            self.best_models_folder = 'best_models_plantations//'+str(subfolderslist[self.test_plantation-1])+'//'
        else:
            self.channels_folder = 'channels'
            self.best_models_folder = 'best_models//'

        if hpc:
            self.ref_folder = 'ref//'
            self.labels_file = 'labels//days_to_stage_4.feather'
        else:
            self.ref_folder = 'ref\\'
            self.labels_file = 'labels\days_to_stage_4.feather'
            
        print('Testing '+str(max_bands)+' channels on '+model_type+' model. Method: '+str(method))
        if model_type=='random_forest':
            self.model = random_forest
        elif model_type == 'xgboost':
            self.model = my_xgboost
        elif model_type == 'linreg':
            self.model = LinearRegression_model
        elif model_type == 'svm':
            self.model = SVM_model
        else:
            print('Model Type Is Not Recognized')
            return

    def split_data(self,train_per,labels_file):
        if not self.plantations:
            self.labels = feather.read_dataframe(labels_file)
            self.unique_labels = np.unique(self.labels['5'].values)

            self.labels_inds = np.where(self.labels['sample_num']<105)
            self.avocado_nums = np.array(self.labels['sample_num'].iloc[self.labels_inds])

            self.train_num_samples = int(train_per*len(self.labels_inds[0]))
            self.test_num_samples = len(self.labels_inds[0]) - self.train_num_samples

            self.train_indices = random.sample(list(self.labels_inds[0]),self.train_num_samples)
            self.train_avocado = np.array(self.labels['sample_num'].iloc[self.train_indices])

            self.test_indices = set(self.labels_inds[0])-set(self.train_indices)
            self.test_indices = list(self.test_indices)
            self.test_avocado = np.array(self.labels['sample_num'].iloc[self.test_indices])

            self.train_labels = np.array(self.labels['5'])[self.train_indices]
            self.test_labels = np.array(self.labels['5'])[self.test_indices]
        
        if self.plantations: # split the data by plantations

            self.labels = feather.read_dataframe(labels_file)
            self.unique_labels = np.unique(self.labels['5'].values)

            self.labels_inds = np.where(self.labels['sample_num']<105)
            self.avocado_nums = np.array(self.labels['sample_num'].iloc[self.labels_inds])

            test_data_ims = ((self.test_plantation-1)*40+1,self.test_plantation*40)
            self.test_indices = np.array(self.labels_inds[0])[np.where(self.labels_inds[0]+1>=test_data_ims[0])]
            self.test_indices = np.array(self.test_indices)[np.where(self.test_indices+1<=test_data_ims[1])]
            self.test_avocado = np.array(self.labels['sample_num'].iloc[self.test_indices])
            self.train_indices = list(set(self.labels_inds[0])-set(self.test_indices))
            self.train_avocado = np.array(self.labels['sample_num'].iloc[self.train_indices])
            self.train_labels = np.array(self.labels['5'])[self.train_indices]
            self.test_labels = np.array(self.labels['5'])[self.test_indices]



    def get_data_by_bands(self,bands,avocado_nums,ref_folder):
        # if isrand:
        bands = bands*840
        bands = bands.astype(int)
        bands[np.where(bands==840)]=839
        self.bands = bands
        
        all_data = []
        for avocado_num in avocado_nums:
            file = 'ref_data_'+str(int(avocado_num))+'.feather'
            load = feather.read_dataframe(ref_folder+file)
            ref_data = np.array(load['ref'][bands])
            partial_ref_data =  np.array(load['partial_ref'][bands])
            partial_ref_data = np.array(list(partial_ref_data)).flatten()
            size_ratio = np.array(load['size_ratio'][0])
            data = np.append(ref_data,partial_ref_data)
            data = np.append(data,size_ratio)

            all_data.append(data)

        all_data = np.array(all_data)
        return all_data

    def estimate_model_by_bands(self,orig_bands):

        # train model
        self.train_data = self.get_data_by_bands(orig_bands,self.train_avocado,self.ref_folder)
        self.new_model = self.model(self.train_data,self.train_labels,max_depth = 7,random_state=0)

        # estimate scores
        self.test_data = self.get_data_by_bands(orig_bands,self.test_avocado,self.ref_folder)


        # temp save
        joblib.dump(self.new_model,'temp.joblib')
        self.new_model = joblib.load('temp.joblib')


        test_pred = self.new_model.predict(self.test_data)
        train_pred = self.new_model.predict(self.train_data)
        std = statistics.stdev(np.sqrt(np.square(np.subtract(test_pred, self.test_labels))))
        self.test_score = np.sqrt(np.square(np.subtract(test_pred, self.test_labels))).mean()
        self.cv_score = std/self.test_score
        self.train_score = np.sqrt(np.square(np.subtract(train_pred, self.train_labels))).mean()


    def save_params(self,iterartion,orig_bands):

        model_data = {}
        model_data['bands'] = [self.bands]
        model_data['bands_selection'] = [orig_bands]
        model_data['train_score'] = [self.train_score]
        model_data['test_score'] = [self.test_score]
        model_data['train_samples'] = [self.train_avocado]
        model_data['test_samples'] = [self.test_avocado]
        model_data['train_labels'] = [self.train_labels]
        model_data['test_labels'] = [self.test_labels]
        model_data['cv_score'] = [self.cv_score]

        model_data = pd.DataFrame(model_data)
        model_data.columns = model_data.columns.astype(str)
        model_data.reset_index(drop=True)

        self.model_name = self.model_type+'_'+str(len(self.bands))+'bands_'+str(self.train_per)+'per_train_'+str(iterartion)+'iter_'+str(self.method)+'_method.feather'
        # print('saving file: '+model_name)

        if self.method=='drop_more':
            model_data.to_feather(self.channels_folder+'//drop_more//'+self.model_name)
        if self.method=='drop_one':
                model_data.to_feather(self.channels_folder+'//drop_one//'+self.model_name)
        if self.method=='random_select':
            model_data.to_feather(self.channels_folder+'//random_select//'+self.model_name)
        if self.method=='add_one':
            model_data.to_feather(self.channels_folder+'//add_one//'+self.model_name)

    def save_best_model(self):
        joblib.dump(self.best_model,self.best_models_folder+self.model_name[:-8]+'.joblib')



    def add_one_every_time(self):
        # base function
        if (len(self.selected_bands)==(self.max_bands)):
            return False

        best_cv_score = 1000
        print('Testing '+str(len(self.selected_bands)+1)+' bands')
        x = (1-np.isin(self.orig_bands,self.selected_bands)*1)
            
        i=0
        for band in (self.orig_bands[x.astype(bool)]):
            temp_bands = np.append(self.selected_bands,band)
            self.estimate_model_by_bands(temp_bands)
            self.save_params(i,temp_bands)

            if self.cv_score<best_cv_score:
                best_cv_score = self.cv_score
                best_bands = temp_bands
                self.best_model = self.new_model
            i+=1
        self.selected_bands = best_bands
        self.save_best_model()

        return True



    def drop_one_every_time(self):

        best_cv_score = 1000
        # base function
        if len(self.orig_bands)<2:
            return False

        print('Testing '+str(len(self.orig_bands))+' bands')
        for i in range(len(self.orig_bands)):
            temp_bands = np.delete(self.orig_bands,i)
            self.estimate_model_by_bands(temp_bands)
            self.save_params(i,temp_bands)

            if self.cv_score<best_cv_score:
                best_cv_score = self.cv_score
                best_bands = temp_bands
                self.best_model = self.new_model

        self.orig_bands = best_bands
        self.save_best_model()

        return True

    def drop_n_every_time(self):
        
        self.num_trials = min(self.num_trials,len(self.orig_bands)**2)
        best_cv_score = 1000
        # base function
        if len(self.orig_bands)<2:
            return False
        self.drop = int(self.drop_func(self.max_bands,len(self.orig_bands)))

        print('Testing '+str(len(self.orig_bands))+' bands')
        for i in range(self.num_trials):
            temp_bands = np.array(random.sample(list(self.orig_bands),(len(self.orig_bands)-self.drop)))
            self.estimate_model_by_bands(temp_bands)
            self.save_params(i,temp_bands)

            if self.cv_score<best_cv_score:
                best_cv_score = self.cv_score
                best_bands = temp_bands
                self.best_model = self.new_model

        self.orig_bands = best_bands

        self.save_best_model()

        return True

    def random_select(self):

        best_cv_score = 1000
        # base function
        if self.bands_to_test<2:
            return False
        
        
        print('Testing '+str(self.bands_to_test)+' bands')
        for i in range(self.num_trials):
            temp_bands = np.random.random(size=self.bands_to_test)
            self.estimate_model_by_bands(temp_bands)
            self.save_params(i,temp_bands)

            if self.cv_score<best_cv_score:
                best_cv_score = self.cv_score
                best_bands = temp_bands
                self.best_model = self.new_model


        self.drop = int(self.drop_func(self.max_bands,self.bands_to_test))
        self.bands_to_test-=self.drop
        self.save_best_model()

        return True

    def run(self):

        self.split_data(train_per=self.train_per,labels_file=self.labels_file)
        keep_going = True

        if self.method=='drop_one':
            self.orig_bands = np.linspace(0,1,self.max_bands)
            self.estimate_model_by_bands(self.orig_bands)
            self.save_params(0,self.orig_bands)
            while keep_going:
                keep_going = self.drop_one_every_time()

        if self.method=='drop_more':
            self.orig_bands = np.linspace(0,1,self.max_bands)
            self.estimate_model_by_bands(self.orig_bands)
            self.save_params(0,self.orig_bands)
            while keep_going:
                keep_going = self.drop_n_every_time()
        
        if self.method=='random_select':
            self.bands_to_test = self.max_bands
            while keep_going:
                    keep_going = self.random_select()
        
        if self.method=='add_one':
            self.orig_bands = np.linspace(0,1,self.max_bands)
            # find 2 best bands
            i = 0
            best_test_score = 10000
            for temp_bands in itertools.combinations(self.orig_bands, 2):
                temp_bands = np.array(temp_bands)
                self.estimate_model_by_bands(temp_bands)
                if self.test_score<best_test_score:
                    best_test_score = self.test_score
                    best_bands = temp_bands
                    self.save_params(i,temp_bands) # save only the ones that improves results
                    i+=1
            self.selected_bands = best_bands
            while keep_going:
                    keep_going = self.add_one_every_time()


        
        print('Testing Complete')


hpc=params_update.hpc

if hpc:
    max_bands = 84
    max_drop = 10

else:
    max_bands = 8
    max_drop = 4

# # Random data split
model_types = ['random_forest','xgboost','svm']
methods = ['drop_one', 'drop_more', 'random_select','add_one']

for model_type in model_types:
    for method in methods:
        channels_tester = Channels_Tester(hpc=hpc, 
                                            model_type=model_type,
                                            max_bands=max_bands,
                                            max_drop=max_drop,
                                            method=method)
        channels_tester.run()

# Plantations data split
for plns in range(3):
    model_types = ['random_forest','xgboost','svm']
    methods = ['drop_one', 'drop_more', 'random_select','add_one']
    for model_type in model_types:
        for method in methods:
            channels_tester = Channels_Tester(hpc=hpc, 
                                                model_type=model_type,
                                                max_bands=max_bands,
                                                max_drop=max_drop,
                                                method=method,
                                                plantations = True,
                                                test_plantation=plns+1)
            channels_tester.run()

