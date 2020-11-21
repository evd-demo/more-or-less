import torchvision.datasets as datasets
from torchvision import transforms
from utils.bv import *
from utils.analysis import *
from models import DenseNet
from utils.experiments import *

def get_test_loader(e, batch_size = 256):
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]

    test_transforms = transforms.Compose([
    transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    if e['dataset'] == "cifar10":
        data_function = datasets.CIFAR10
        num_classes = 10
    elif  e['dataset'] == "cifar100":
        data_function = datasets.CIFAR100
        num_classes = 100

    test_set = data_function("./data/", train=False, transform=test_transforms, download=True)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=0)
    return test_loader


def run(e, criterion="ce"):
   
    result = load_pretrained_models(e["directory"], e["d_list"], e["gr_list"], ensemble_size=e['ens_size'], dataset=e['dataset'])
    
    testloader = get_test_loader(e)

    if criterion == "mse":
        bv_function = compute_bias_variance_mse
    else:
        bv_function = compute_bias_variance_ce

    for exp_type in result.keys():
        with open(os.path.join(e["directory"],'bias_var_'+exp_type+'_'+criterion+'.csv'), 'w') as f:
                f.write('experiment, bias2, variance \n')
       
        for exp in result[exp_type].keys():
            print(exp_type, exp)
            
            if exp_type == "vertical" or exp_type == "horizontal":
                bias2, variance = bv_function(result[exp_type][exp], testloader, ensemble=True, ensemble_size=e['ens_size'])
            else:
                bias2, variance = bv_function(result[exp_type][exp], testloader)
            
            with open(os.path.join(e["directory"],'bias_var_' + exp_type + '_' + criterion + '.csv'), 'a') as f:
                f.write('%s,%0.6f,%0.6f,\n' % (exp, bias2, variance))
    
################################################
# Run both types of bias variance decompositions
################################################

run(exp["vary_both_e_4_cifar10_v_trials_2"], criterion="mse")
run(exp["vary_both_e_4_cifar10_v_trials_2"], criterion="ce")






