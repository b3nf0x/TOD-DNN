import argparse
from tqdm import tqdm
import random
from core.basic_math import Noisy_MH
from core.data_model import SynData


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--desired_dataset_length", type=int, required=True, default=None, help="number of rows to generate")
    parser.add_argument("--out_dir", type=str, required=True, default=None, help="dataset save dir path")
    parser.add_argument("--max_time_delta", type=int, required=False, default=12, help="sets the max delta time")
    parser.add_argument("--min_ambient_temp", type=int, required=False, default=-10, help="")
    parser.add_argument("--max_ambient_temp", type=int, required=False, default=36, help="sets the max delta time")
    parser.add_argument("--min_mass", type=int, required=False, default=50, help="min mass to use, kg")
    parser.add_argument("--max_mass", type=int, required=False, default=121, help="max mass to use, kg")
    parser.add_argument("--dry_run", type=bool, required=False, default=False, help="if true, no file will be saved, but displayed")
    args = parser.parse_args()

    correction_factors: list = [0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

    dataset: list = []

    for i in tqdm(range(0, args.desired_dataset_length)):
        # base profile parameters
        random_mass = random.randint(args.min_mass, args.max_mass) # 71 
        random_cf = random.choice(correction_factors) # 7
        random_static_ambient_temp = random.randint(args.min_ambient_temp, args.max_ambient_temp) # 46 
        random_time_of_death = random.randint(0, args.max_time_delta) # 12

        temp_dataset: list = []

        # 10 distortion sample for each profile
        for _ in range(0, 100):
            temp_dataset.append(
                SynData(
                    mass=random_mass,
                    cf=random_cf,
                    static_ambient_temp=random_static_ambient_temp,
                    time_of_death=random_time_of_death,
                    rectal_temp=Noisy_MH(
                        t=random_time_of_death,
                        T_ambient=random_static_ambient_temp,
                        cf=random_cf,
                        m=random_mass
                    )
                )
            )

        dataset.append(random.choice(temp_dataset))


    if not args.dry_run:
        [item.save_as_npy(args.out_dir) for item in dataset]
    else:
        [print(item.to_json()) for item in dataset]