import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from pathlib import Path
import os.path
import sys, glob, shutil

import numpy as np
from copy import deepcopy

import astropy.units as u

from gammapy.datasets import MapDatasetEventSampler
from gammapy.data import DataStore

from base_simbmvtool_creator import BaseSimBMVtoolCreator
from toolbox import (get_empty_obs_simu,
                     get_empty_dataset_and_obs_simu,
                     stack_with_meta)   

class SimulatorCreator(BaseSimBMVtoolCreator):

    def __init__(self, config_path:str):
        super().__init__(simulator=True)
        if config_path is not None: self.init_config(config_path)
        else: ValueError("Config file needed")

    def do_simulation(self,config_path=None):
            if config_path is not None: self.init_config(config_path)
            Path(self.save_path).mkdir(parents=True, exist_ok=True)
            files = glob.glob(str(self.save_path)+'/*fit*')
            for f in files:
                Path(f).unlink()
            
            bkg_irf_arr =[]
            for i in np.arange(self.n_wobbles): bkg_irf_arr.append(self.bkg_true_irf) # In case you want different IRFs you can change it here

            for wobble,bkg_irf,run_info,random_state in zip(np.arange(self.n_wobbles)+1,bkg_irf_arr,self.wobble_run_info,self.wobble_seeds):
                i = 0
                if self.single_pointing and (wobble ==  2): break

                # Loop pour résolution temporelle. Paramètre "oversampling"
                for iobs in range(1 if self.one_obs_per_wobble else self.n_run):
                    print(iobs)
                    events_all = None
                    oversampling = self.time_oversampling
                    obs_id = iobs + 1 + (1 if self.one_obs_per_wobble else self.n_run*(wobble-1))*(wobble > 1)
                    verbose = iobs == 0
                    sampler = MapDatasetEventSampler(random_state=random_state+iobs)
                    if self.true_collection: obs = get_empty_obs_simu(self.bkg_true_collection[iobs],None,run_info,self.source,self.path_data,self.flux_to_0,self.t_ref,i*self.delay,verbose)
                    else: obs = get_empty_obs_simu(bkg_irf,None,run_info,self.source,self.path_data,self.flux_to_0,self.t_ref,i*self.delay,verbose)
                    n = int(run_info[3]/oversampling.to_value("s")) + 1
                    oversampling = (run_info[3]/n) * u.s
                    run_info_over = deepcopy(run_info)
                    run_info_over[3] = oversampling.to_value("s")
                    for j in range(n):
                        print(j)
                        if self.true_collection: tdataset, tobs = get_empty_dataset_and_obs_simu(self.bkg_true_collection[iobs],None,run_info_over,self.source,self.path_data,self.flux_to_0,self.t_ref,i*self.delay+j*oversampling.to_value("s"),verbose=False)
                        else: tdataset, tobs = get_empty_dataset_and_obs_simu(bkg_irf,None,run_info_over,self.source,self.path_data,self.flux_to_0,self.t_ref,i*self.delay+j*oversampling.to_value("s"),verbose=False)
                        tdataset.fake(random_state=random_state+iobs)

                        events = sampler.run(tdataset, tobs)
                        events.table.meta['ALT_PNT']=None
                        events.table.meta['AZ_PNT']=None
                        if events_all is None:
                            events_all = events
                        else:
                            events_all = stack_with_meta(events_all, events)
                    
                    events_all.table.meta['OBS_ID'] = obs_id
                    obs._events = events_all
                    obs.write(f"{self.save_path}/obs_W{wobble}_{'0'*(obs_id<10)+'0'*(obs_id<100)}{obs_id}.fits",overwrite=True)
                    i+=1
            del(obs)
            shutil.copyfile(self.config_path, f'{self.save_path}/config_simu.yaml')
            print(f"Simulation dir: {self.save_path}")
            path = Path(self.save_path) 
            paths = sorted(list(path.rglob("obs_*.fits")))
            data_store = DataStore.from_events_files(paths)
            data_store.obs_table.write(f"{self.save_path}/obs-index.fits.gz", format='fits',overwrite=True)
            data_store.hdu_table.write(f"{self.save_path}/hdu-index.fits.gz", format='fits',overwrite=True)
            self.load_observation_collection()
            