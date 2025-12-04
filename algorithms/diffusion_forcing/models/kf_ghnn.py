import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json

from ghnn.nets.ghnn import GHNN
from video_vae_project.video_vae.GHNN_solver import PhysicsSolver
from video_vae_project.video_vae.utils.Encoder_Time_Embedding import Encoder_TE
from video_vae_project.video_vae.utils.Decoder_Time_Embedding import Decoder_TE


class KF_GHNN_Model(nn.Module):
    def __init__(
        self, 
        ghnn_model_specs, 
        cfg, 
        physics_model="GHNN", 
        canon_type="cartesian"
    ):
        super().__init__()
        self.cfg = cfg



        ####################
        #### GHNN Model ####
        ####################
        
        # initialize parameters for GHNN model
        self.physics_model = physics_model
        self.canon_type = canon_type

        # load GHNN settings
        self.ghnn_model_specs = ghnn_model_specs
        ghnn_root_dir = self.ghnn_model_specs["ghnn_dir"]
        with open(os.path.join(
            ghnn_root_dir, 'ghnn', 'training', 
            'default_GHNN.json' if self.physics_model == "GHNN" 
                                else f'default_MLP_wsymp.json')
            ) as file_:
            settings = json.load(file_)
        
        # augment settings for both GHNN and MLP_wsymp
        settings['step_size'] = None
        del(settings['bodies'])
        del(settings['dims'])
        settings['data_path'] = None
        if self.ghnn_model_specs["type"] == "pendulum":
            print("Using pendulum settings...")
            print("Number of objects:", self.ghnn_model_specs["num_of_objs"])
            if self.ghnn_model_specs["num_of_objs"] == 1:
                settings['feature_names'] = ['q_A','p_A']
                settings['label_names'] = ['q_A','p_A']
            elif self.ghnn_model_specs["num_of_objs"] == 2:
                settings['feature_names'] = ['q_A','p_A','q_B','p_B']
                settings['label_names'] = ['q_A','p_A','q_B','p_B']
            else:
                raise NotImplementedError("Pendulum dataset with more than 2 objects is not implemented.")
            print("Feature names:", settings['feature_names'])
            settings['batch_size'] = self.ghnn_model_specs["batch_size"]
            settings['t_in_T'] = False
            settings['period_q'] = None
            settings['seed'] = 0    
        else:
            if self.ghnn_model_specs["num_of_objs"] == 2:
                if self.canon_type == "cartesian":
                    settings["feature_names"] = ['r_x', 'r_y', 'p_x', 'p_y']
                    settings["label_names"] = ['r_x', 'r_y', 'p_x', 'p_y']
                elif self.canon_type == "polar":
                    raise NotImplementedError("Polar canonicalization is not implemented yet.")
                else:
                    settings["feature_names"] = ['q_A_x', 'q_A_y', 'q_B_x', 'q_B_y', 'p_A_x', 'p_A_y', 'p_B_x', 'p_B_y']
                    settings["label_names"] = ['q_A_x', 'q_A_y', 'q_B_x', 'q_B_y', 'p_A_x', 'p_A_y', 'p_B_x', 'p_B_y']
            elif self.ghnn_model_specs["num_of_objs"] == 3:
                settings["feature_names"] = ['q_A_x', 'q_A_y', 'q_B_x', 'q_B_y', 'q_C_x', 'q_C_y', 
                                                'p_A_x', 'p_A_y', 'p_B_x', 'p_B_y', 'p_C_x', 'p_C_y']
                settings["label_names"] = ['q_A_x', 'q_A_y', 'q_B_x', 'q_B_y', 'q_C_x', 'q_C_y', 
                                            'p_A_x', 'p_A_y', 'p_B_x', 'p_B_y', 'p_C_x', 'p_C_y']
            else:
                raise NotImplementedError("Multi-body dataset with more than 3 objects is not implemented.")

        # save settings to json file
        run_dir = self.ghnn_model_specs["run_dir"]
        path_to_settings = os.path.join(
            run_dir, 'ghnn_settings.json' \
                if self.physics_model == "GHNN"
                else 'mlp_wsymp_settings.json')
        with open(path_to_settings, 'w') as file_:
            json.dump(settings, file_, indent=4, separators=(',', ': '))

        # create GHNN model
        self.raw_physics_model = GHNN(
            path_to_settings,
            device=f"gpu{self.cfg.model.device}"
        )

        # create physics solver
        self.ghnn_solver = PhysicsSolver(
            physics_model=self.raw_physics_model
        )


        
        #######################
        ## Encoder & Decoder ##
        #######################

        # Initialize encoder
        self.encoder = Encoder_TE(
            image_size=cfg.data.image_size,
            image_channels=cfg.data.image_channels,
            stochastic=False
        )
        
        # Initialize decoder
        self.decoder = Decoder_TE(
            image_size=cfg.data.image_size,
            image_channels=cfg.data.image_channels,
            stochastic=False
        )
        

        
        ####################
        ### Linear Gain ####
        ####################

        # initialize linear gain matrix K
        
        


    def forward(self, x_noised_next, t_next, z):
        # 1) get physics prediction by performing z_next_pred = GHNN(z)
        # 2) predict x_t^k_t (i.e. x_noised_next from methods parameters) by decoding z_next_pred
        # 3) calculate difference between x_noised_next and x_t^k_t
        # 4) correct z_next_pred by linear gain using the difference calculated in step 3
        #    i.e z_next_corrected = z_next_pred + K * (encoder(x_noised_next - x_t^k_t))
        #
        # 5) finally, return z_next_corrected

        pass
