import argparse
from tqdm import tqdm
from core.basic_math import Noisy_MH
from core.data_model import SynData


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True, default=None, help="dataset save dir path")
    parser.add_argument("--max_delta_time", type=int, required=False, default=12, help="sets the max delta time")
    args = parser.parse_args()

    correction_factors: list = [0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

    dataset: list = []
    for mass in tqdm(range(50, 122)): # mass
        for cf in correction_factors: # each cf
            for ambient_T in range(-10, 37+1): # ambient temp
                for tod in range(0, args.max_delta_time+1):
                    dataset.append(
                        SynData(
                            mass=mass,
                            cf=cf,
                            static_ambient_temp=ambient_T,
                            time_of_death=tod,
                            rectal_temp=Noisy_MH(
                                t=tod,
                                T_ambient=ambient_T,
                                cf=cf,
                                m=mass
                            )
                        )
                    )
                    
    [item.save_as_npy(args.out_dir) for item in dataset]
