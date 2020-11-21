import EvD_setup as setup
from models import DenseNet
from utils.profile import *


# depths = [40, 64, 88, 112]
# growth_rates = [12, 20, 28, 36]

depths = [40]
growth_rates = [12]

ensemble_size = 4
dataset = "cifar_10"

directory = "./runs/profiles/test/"

for depth in depths:
    for growth_rate in growth_rates:

        filename = directory  + "d_" + str(depth) + "_gr_" + str(growth_rate) + "_e_" + str(ensemble_size) + "_" + dataset
        print(filename)
        
        block_config = [(depth - 4) // 6 for _ in range(3)]
        ensemble_depth, ratio_param = setup.get_ensemble_depth(depth=depth, growth_rate=growth_rate, ensemble_size=ensemble_size)
        ensemble_growth_rate, ratio_param = setup.get_ensemble_growth_rate(depth=depth, growth_rate=growth_rate, ensemble_size=ensemble_size)
        ensemble_block_config = [(ensemble_depth - 4) // 6 for _ in range(3)]

        
        model = DenseNet(
            growth_rate=growth_rate,
            block_config=block_config,
            num_classes=10,
            small_inputs=True,
            efficient=False,
        )
        profile_and_save(model, filename=filename+"_s.csv")

        # # vertical 
        model = DenseNet(
            growth_rate=growth_rate,
            block_config=ensemble_block_config,
            num_classes=10,
            small_inputs=True,
            efficient=False,
        )
        profile_and_save(model, filename=filename+"_v.csv")

        # horizontal
        ensemble_block_config = [(ensemble_depth - 4) // 6 for _ in range(3)]
        model = DenseNet(
                growth_rate=ensemble_growth_rate,
                block_config=block_config,
                num_classes=10,
                small_inputs=True,
                efficient=False,
            )
        profile_and_save(model, filename=filename+"_h.csv")

