NORMALIZATION = {
    'imagenet': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)], # [(0.4814, 0.4542, 0.4033), (0.2726, 0.2643, 0.2774)]
    'cifar10': [(0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)],
    'cifar10_36': [(0.4917, 0.4824, 0.4468), (0.2411, 0.2376, 0.2563)],
    'cifar10_224': [(0.4914, 0.4822, 0.4465), (0.2413, 0.2378, 0.2564)],
    'cifar100': [(0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2761)],
    'cifar100_224': [(0.5070, 0.4865, 0.4409), (0.2622, 0.2513, 0.2714)],
    'tiny_imagenet': [(0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)],
    'tiny_imagenet_224': [(0.4805, 0.4484, 0.3978), (0.2667, 0.2584, 0.2722)],
    'caltech256': [(0.5415, 0.5187, 0.4880), (0.3081, 0.3040, 0.3169)],
    'flower102': [(0.5115, 0.4159, 0.3407), (0.2957, 0.2493, 0.2889)],
    'oxford_pet': [(0.4830, 0.4448, 0.3956), (0.2591, 0.2531, 0.2596)],
    'svhn': [(0.4378, 0.4439, 0.4729), (0.1981, 0.2011, 0.1970)],
}
NUM_CLASSES = {
    'imagenet': 1000,
    'cifar10': 10,
    'cifar100': 100,
    'tiny_imagenet': 200,
    'flower102': 102,
    'oxford_pet': 37,
    'svhn': 10,
}
IMG_SIZE = {
    'imagenet': 224,
    'cifar10': 32,
    'cifar100': 32,
    'tiny_imagenet': 64,
    'flower102': 224,
    'oxford_pet': 224,
    'svhn': 32,
}
