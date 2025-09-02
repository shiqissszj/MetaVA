maml = {
    'device': 'cuda:0',
    'innerlr': 1e-3,
    'outerlr': 1e-3,
    'shots': 5,  # shots, choice = [1,5]
    'tasks': 25,
    'update': 5,
    'ways': 2,
    'first_order': True,
    'batch_size': 4,
    'valid_step': 5
}

pretrain = {
    'lr': 1e-3,
    'shots': 5,
    'update': 20,
    'batch_size': 4,
    'ways': 2
}

# MTL = {
#     'prelr': 1e-3,
#     'inlr': 1e-3,
#     'outlr': 1e-3,
#     'shots': 5,
#     'tasks': 30,
#     'preupdate': 10,
#     'update': 20,
#     'ways': 2,
#     'batch_size': 1
# }

# Network = {
#     'in_channels': 1,
#     'base_filters': 128,
#     'ratio': 1.0,
#     'filter_list': [128, 64, 64, 32, 32, 16, 16],
#     'm_blocks_list': [2, 2, 2, 2, 2, 2, 2],
#     'kernel_size': 16,
#     'stride': 2,
#     'groups_width': 16,
#     'verbose': False,
#     'n_classes': 2
# }
