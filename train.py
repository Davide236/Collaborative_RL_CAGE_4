import argparse
from trainers.ippo_trainer import PPOTrainer
from trainers.maddpg_trainer import MADDPGTrainer
from trainers.mappo_trainer import MAPPOTrainer
from trainers.qmix_trainer import QMIXTrainer
from trainers.r_ippo_trainer import RecurrentIPPOTrainer
from trainers.r_mappo_trainer import RecurrentMAPPOTrainer
def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some input parameters.")

    # Add arguments
    parser.add_argument(
        '--Method',
        type=str,
        default='IPPO',
        help='The method to be used (default: IPPO)'
    )
    
    parser.add_argument(
        '--Messages',
        type=bool,
        default=False,
        help='Boolean flag for messages (default: False)'
    )

    parser.add_argument(
        '--Load_last',
        type=bool,
        default=False,
        help='Boolean flag for loading the last saved network (default: False)'
    )

    parser.add_argument(
        '--Load_best',
        type=bool,
        default=False,
        help='Boolean flag for loading the best saved network (default: False)'
    )
    # Parse the arguments
    args = parser.parse_args()
    method = args.Method
    print(f'Using method: {method}, loading network: {args.Load_best | args.Load_last}, Using messages: {args.Messages}')
    if method == 'R_IPPO':
        trainer = RecurrentIPPOTrainer(args)
        trainer.run()
    elif method == 'R_MAPPO':
        trainer = RecurrentMAPPOTrainer(args)
        trainer.run()
    elif method == 'MAPPO':
        trainer = MAPPOTrainer(args)
        trainer.run()
    elif method == 'MADDPG':
        trainer = MADDPGTrainer(args)
        trainer.run()
    elif method == 'QMIX':
        trainer = QMIXTrainer(args)
        trainer.run()
    elif method == 'VDN':
        print('Not implemented yet')
    else:
        trainer = PPOTrainer(args)
        trainer.run()

if __name__ == '__main__':
    main()
