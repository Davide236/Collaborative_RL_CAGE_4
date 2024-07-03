import argparse
from evaluators.ippo_eval import IPPOEvaluator
from evaluators.mappo_eval import MAPPOEvaluator

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
        default=True,
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
        trainer = IPPOEvaluator(args)
        trainer.run()
    elif method == 'R_MAPPO':
        trainer = IPPOEvaluator(args)
        trainer.run()
    elif method == 'MAPPO':
        trainer = MAPPOEvaluator(args)
        trainer.run()
    elif method == 'MADDPG':
        trainer = IPPOEvaluator(args)
        trainer.run()
    elif method == 'QMIX':
        trainer = IPPOEvaluator(args)
        trainer.run()
    elif method == 'VDN':
        trainer = IPPOEvaluator(args)
        trainer.run()
    else:
        trainer = IPPOEvaluator(args)
        trainer.run()

if __name__ == '__main__':
    main()
