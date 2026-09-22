import logging
import random
from torchvision import datasets, transforms
import utils.dists as dists
import numpy as np

class Generator(object):
    """Generate federated learning training and testing data."""

                            
    def read(self, path):
                                                          
        raise NotImplementedError

                             
    def group(self):
                                     
        grouped_data = {label: []
                        for label in self.labels}                             

                                    
        for datapoint in self.trainset:                       
            _, label = datapoint                 
            label = self.labels[label]

            grouped_data[label].append(                             
                datapoint)

        self.trainset = grouped_data                                        

                         
    def generate(self, path):
        self.read(path)
        self.trainset_size = len(self.trainset)                         
        self.group()

        return self.trainset


class Loader(object):
    """Load and pass IID data partitions."""

    def __init__(self, config, generator):
                                 
        self.config = config
        self.trainset = generator.trainset
        self.testset = generator.testset
        self.labels = generator.labels
        self.trainset_size = generator.trainset_size

                                    
        self.used = {label: [] for label in self.labels}
        self.used['testset'] = []

    def extract(self, label, n):
        if len(self.trainset[label]) > n:
            extracted = self.trainset[label][:n]                
            self.used[label].extend(extracted)                     
            del self.trainset[label][:n]                        
            return extracted
        else:
            logging.warning('Insufficient data in label: {}'.format(label))
            logging.warning('Dumping used data for reuse')

                                 
            for label in self.labels:
                self.trainset[label].extend(self.used[label])
                self.used[label] = []

                                      
            return self.extract(label, n)

    def get_partition(self, partition_size):
                                                    

                                  
        dist = dists.uniform(partition_size, len(self.labels))

        partition = []                                          
        for i, label in enumerate(self.labels):
            partition.extend(self.extract(label, dist[i]))

                                
        random.shuffle(partition)

        return partition

    def get_testset(self):
                                   
        return self.testset


class BiasLoader(Loader):
    """Load and pass 'preference bias' data partitions."""

    def get_partition(self, partition_size, pref):
                                                            

                                                
        bias = self.config.data.bias['primary']
        secondary = self.config.data.bias['secondary']

                                                         
        majority = int(partition_size * bias)
        minority = partition_size - majority

                                          
        len_minor_labels = len(self.labels) - 1

        if secondary:
                                                  
            dist = [0] * len_minor_labels
            dist[random.randint(0, len_minor_labels - 1)] = minority
        else:
                                                  
            dist = dists.uniform(minority, len_minor_labels)

                                           
        dist.insert(self.labels.index(pref), majority)

        partition = []                                          
        for i, label in enumerate(self.labels):
            partition.extend(self.extract(label, dist[i]))

                                
        random.shuffle(partition)

        return partition


class ShardLoader(Loader):
    """Load and pass 'shard' data partitions."""

    def create_shards(self):
                                                 
        per_client = self.config.data.shard['per_client']

                                                    
        total = self.config.clients.total * per_client
        shard_size = int(self.trainset_size / total)

        data = []                
        for _, items in self.trainset.items():
            data.extend(items)

        shards = [data[(i * shard_size):((i + 1) * shard_size)]
                  for i in range(total)]
        random.shuffle(shards)

        self.shards = shards
        self.used = []

        logging.info('Created {} shards of size {}'.format(
            len(shards), shard_size))

    def extract_shard(self):
        shard = self.shards[0]
        self.used.append(shard)
        del self.shards[0]
        return shard

    def get_partition(self):
                               

                                             
        per_client = self.config.data.shard['per_client']

                               
        partition = []
        for i in range(per_client):
            partition.extend(self.extract_shard())

                                
        random.shuffle(partition)

        return partition

class NonIIDLoader(Loader):
    """Load and pass 'shard' data partitions."""

    def get_partition(self, partition_size, cls_num):
                                                                                     

                                                 
        cls_list = np.random.choice(np.arange(len(self.labels)), cls_num,
                                              replace=False)

        dist = [0] * len(self.labels)
        avg = partition_size / cls_num
        for i, c in enumerate(cls_list):
            dist[c] = int((i + 1) * avg) - int(i * avg)

        partition = []                                          
        for i, label in enumerate(self.labels):
            partition.extend(self.extract(label, dist[i]))

                                
        random.shuffle(partition)

        return partition


class LEAFLoader(object):
    """Load and pass IID data partitions."""

    def __init__(self, config, generator):
                                 
        self.config = config
        self.trainset = generator.trainset
        self.valset = generator.valset
        self.testset = generator.testset
        self.labels = generator.labels
        if config.loader == 'leaf':
            self.num_clients = len(generator.trainset['users'])

    def extract(self, client_id):
                                                                 
                                                                               
        user_name = self.trainset['users'][client_id]
        return self.trainset['user_data'][user_name], self.testset['user_data'][user_name]

    def get_testset(self):
        """Return only the final independent test set."""
        testset = {'x': [], 'y': []}
        for user in self.testset['users']:
            testset['x'] += self.testset['user_data'][user]['x']
            testset['y'] += self.testset['user_data'][user]['y']
                                                       
        return testset

    def get_valset(self):
        """Return only the global validation set; never fall back to test."""
        valset = {'x': [], 'y': []}
        for user in self.valset['users']:
            valset['x'] += self.valset['user_data'][user]['x']
            valset['y'] += self.valset['user_data'][user]['y']
        return valset
