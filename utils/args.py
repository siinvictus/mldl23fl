import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--federated', action='store_true', default=False,
                        help='Run the federated learning pipeline instead of the centralized version')
    parser.add_argument('--d_clients', type=int, default=0,
                        help="The d subsect of clients in power of choices algorithm")
    parser.add_argument('--prune', type=bool, default=False,
                        help='Activate the pruning criteria')
    parser.add_argument('--conv', type=bool, default=False,
                        help='Prune convolutional layer')
    parser.add_argument('--linear', type=bool, default=False,
                        help='Prune linear layer')
    parser.add_argument('--structured', type=bool, default=False,
                        help='Structured pruning, False unstructured')
    parser.add_argument('--amount_prune', type=float, default=0.2, help='Amount to prune')
    
    
    parser.add_argument('--power_of_choice_m', type=int, default=1,
                        help="The m=c*K in power of choices algorithm which will be half of d")
    parser.add_argument('--client_select', type=int, default=0, 
                        help='0:uniform distribution, 1:10% with 0.5 probability, 2:30% with 0.0001 probability')
    parser.add_argument('--view_summary', type=bool, default=False, 
                        help='View the model summary')
    parser.add_argument('--rotation', action='store_true', default=False,
                        help='Rotate images for domain generalization')
    parser.add_argument('--loo', action='store_true', default=False,
                        help='apply leave one out for domain generalization')
    
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dataset', type=str, default='femnist', choices=['idda', 'femnist'], help='dataset name')
    parser.add_argument('--niid', action='store_true', default=False,
                        help='Run the experiment with the non-IID partition (IID by default). Only on FEMNIST dataset.')
    parser.add_argument('--model', type=str, default='cnn', choices=['deeplabv3_mobilenetv2', 'resnet18', 'cnn'], help='model name')
    parser.add_argument('--num_rounds', type=int, default = 5, help='number of rounds')
    # keep the local epoc = 1 bcs of nnid setting risk overfitting 
    parser.add_argument('--num_epochs', type=int, default = 1, help='number of local epochs')
    parser.add_argument('--clients_per_round', type=int, default = 5, help='number of clients trained per round')
    parser.add_argument('--clients_test', type=int, default = 2, help='number of clients tested per round')
    parser.add_argument('--hnm', action='store_true', default=False, help='Use hard negative mining reduction or not')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--bs', type=int, default= 32, help='batch size')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--m', type=float, default=0.9, help='momentum')
    parser.add_argument('--print_train_interval', type=int, default=10, help='client print train interval')
    parser.add_argument('--print_test_interval', type=int, default=10, help='client print test interval')
    parser.add_argument('--eval_interval', type=int, default=10, help='eval interval')
    parser.add_argument('--test_interval', type=int, default=10, help='test interval')
    return parser
