import argparse
from evaluators.ippo_eval import IPPOEvaluator
from evaluators.mappo_eval import MAPPOEvaluator
from evaluators.r_ippo_eval import R_IPPOEvaluator
from evaluators.r_mappo_eval import R_MAPPOEvaluator
from evaluators.qmix_eval import QMIXEvaluator
from evaluators.r_qmix_eval import R_QMIXEvaluator
from evaluators.maddpg_eval import MADDPGEvaluator

# File made to evaluate different trained RL algorithms
def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process input parameters for method evaluation")

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
        trainer = R_IPPOEvaluator(args)
        trainer.run()
    elif method == 'R_MAPPO':
        trainer = R_MAPPOEvaluator(args)
        trainer.run()
    elif method == 'MAPPO':
        trainer = MAPPOEvaluator(args)
        trainer.run()
    elif method == 'MADDPG':
        trainer = MADDPGEvaluator(args)
        trainer.run()
    elif method == 'QMIX':
        trainer = QMIXEvaluator(args)
        trainer.run()
    elif method == 'R_QMIX':
        trainer = R_QMIXEvaluator(args)
        trainer.run()
    else:
        trainer = IPPOEvaluator(args)
        trainer.run()

if __name__ == '__main__':
    main()
