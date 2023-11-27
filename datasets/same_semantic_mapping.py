import numpy as np


same_semantic_labels = {
    "wall": {
        0: 'wall',
    },
    "ceiling": {
        5: 'ceiling',
    },
    "cabinet": {
        10: 'cabinet',
        24: 'shelf',
        44: 'chest;of;drawers;chest;bureau;dresser',
        62: 'bookcase',
        35: 'wardrobe;closet;press',
        45: 'counter',
        55: 'case;display;case;showcase;vitrine',
    },
    "floor": {
        3: 'floor;flooring',
        28: 'rug;carpet;carpeting',
        6: 'road;route',
    },
    "seat": {
        19: 'chair',
        75: 'swivel;chair',
        30: 'armchair',
        31: 'seat',
        23: 'sofa;couch;lounge',
        69: 'bench',
        110: 'stool',
        97: 'ottoman;pouf;pouffe;puff;hassock',
    },
    "bed": {
        7: 'bed',
        57: 'pillow',
        131: 'blanket;cover',
    },
    "table": {
        15: 'table',
        56: 'pool;table;billiard;table;snooker;table',
        64: 'coffee;table;cocktail;table',
        70: 'countertop',
        99: 'buffet;counter;sideboard',
        77: 'bar',
        33: 'desk',
    },
    "door": {
        14: 'door;double;door',
        58: 'screen;door;screen',
                
    },
    "screen": {
        130: 'screen;silver;screen;projection;screen',
        141: 'crt;screen',
        89: 'television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box',
        143: 'monitor;monitoring;device',
        74: 'computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system',
    },
    "window": {
        63: 'blind;screen',
        8: 'windowpane;window',
        18: 'curtain;drape;drapery;mantle;pall',
    },
    "car": {
        20: 'car;auto;automobile;machine;motorcar',
        127: 'bicycle;bike;wheel;cycle',
        
    },
    "microwave": {
        124: 'microwave;microwave;oven',
    },
    "washer": {
        129: 'dishwasher;dish;washer;dishwashing;machine',
        107: 'washer;automatic;washer;washing;machine',
    },
    "refrig": {
        50: 'refrigerator;icebox',
        
    },
    "stove": {
        71: 'stove;kitchen;stove;range;kitchen;range;cooking;stove',
        118: 'oven',
        47: 'sink',
    },
    "hood": {
        133: 'hood;exhaust;hood',
    },
    "toilet": {
        65: 'toilet;can;commode;crapper;pot;potty;stool;throne',
    },
    "bath": {
        37: 'bathtub;bathing;tub;bath;tub',
        145: 'shower',
    },
    "radiator": {
        146: 'radiator',
        
    },
    "objects": {
        36: 'lamp',
        41: 'box',
        22: 'painting;picture',
        39: 'cushion',
        66: 'flower',
        81: 'towel',
        67: 'book',
        78: 'arcade;machine',
        42: 'column;pillar',
        98: 'bottle',
        108: 'plaything;toy',
        115: 'bag',
        119: 'ball',
        120: 'food;solid;food',
        137: 'tray',
        135: 'vase',
        139: 'fan',
        148: 'clock',
        82: 'light;light;source',
        4: 'tree',
        111: 'barrel;cask',
        100: 'poster;posting;placard;notice;bill;card',
        27: 'mirror',
        43: 'signboard;sign',
        17: 'plant;flora;plant;life',
        122: 'tank;storage;tank',
        132: 'sculpture',
        134: 'sconce',
        138: 'ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin',
        142: 'plate',
        125: 'pot;flowerpot',
        147: 'glass;drinking;glass',
        149: 'flag',
        112: 'basket;handbasket',
        144: 'bulletin;board;notice;board',
        92: 'apparel;wearing;apparel;dress;clothes',
        
    },
    "stand": {
        40: 'base;pedestal;stand',
        93: 'pole',
        
    }, 
    "stair": {
        59: 'stairway;staircase',
        121: 'step;stair',
        38: 'railing;rail',
        53: 'stairs;steps',
        96: 'escalator;moving;staircase;moving;stairway',
        95: 'bannister;banister;balustrade;balusters;handrail',
    },
    "person": {
        12: 'person;individual;someone;somebody;mortal;soul',
        
    },
    "building": {
        1: 'building;edifice',
        
    },
    "booth": {
        88: 'booth;cubicle;stall;kiosk',
    },
    "fireplace": {
        49: 'fireplace;hearth;open;fireplace',
    }
}

class MaskLabelMapping:
    
    def __init__(self):
        
        labels = [x for i in same_semantic_labels for x in same_semantic_labels[i]]
        self.mapping = np.zeros((max(labels)+1), dtype=int) - 1
        self.mapping_name = []
        for index, name in enumerate(same_semantic_labels):
            self.mapping_name.append(name)
            for label in same_semantic_labels[name]:
                self.mapping[label] = index
                
    def __len__(self):
        return len(self.mapping_name)
                
    def __call__(self, labels):
        return self.mapping[labels]