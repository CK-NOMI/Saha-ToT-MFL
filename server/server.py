import client
import load_data
import logging
import numpy as np
import pickle
import random
import sys
from threading import Thread
import torch
import utils.dists as dists                                     
from .record import Profile

class Server(object):
    """Basic federated learning server."""

    def __init__(self, config):
        self.config = config

                   
    def boot(self):
        logging.info('Booting {} server...'.format(self.config.server.mode))

        model_path = self.config.paths.model

                                     
        sys.path.append(model_path)

                                 
        self.load_data()
        self.load_model()
        if self.config.loader != 'leaf':
            num_clients = self.config.clients.total
            self.make_clients(num_clients)
        else:
            num_clients = min(self.config.clients.total, self.loader.num_clients)
            self.make_clients_leaf(num_clients)

    def load_data(self):
        import fl_model                                

                                    
        config = self.config

                               
        generator = fl_model.Generator()

                       
        data_path = self.config.paths.data
        data = generator.generate(data_path)
        self.labels = generator.labels

        if self.config.loader != 'leaf':
            logging.info('Dataset size: {}'.format(
                sum([len(x) for x in [data[label] for label in self.labels]])))
            logging.debug('Labels ({}): {}'.format(
                len(self.labels), self.labels))

                            
        self.loader = {
            'basic': load_data.Loader(config, generator),
            'bias': load_data.BiasLoader(config, generator),
            'shard': load_data.ShardLoader(config, generator),
            'noniid': load_data.NonIIDLoader(config, generator),
            'leaf': load_data.LEAFLoader(config, generator)
        }[self.config.loader]

        logging.info('Loader: {}'.format(self.config.loader))


    def load_model(self):
        import fl_model                                

        saved_model_path = self.config.paths.saved_model
        model_type = self.config.model

        logging.info('Model: {}'.format(model_type))

                             
        self.model = fl_model.Net()
        self.save_model(self.model, saved_model_path)

                                                   
        if self.config.paths.reports:
            self.saved_reports = {}
            self.save_reports(0, [])                      

    def make_clients(self, num_clients):
        IID = self.config.data.IID
        labels = self.loader.labels
        loader = self.config.loader
        loading = self.config.data.loading

        if not IID:                                                        
            dist = {
                "uniform": dists.uniform(num_clients, len(labels)),
                "normal": dists.normal(num_clients, len(labels))
            }[self.config.clients.label_distribution]
            random.shuffle(dist)                        

                                
        clients = []
        for client_id in range(num_clients):

                               
            new_client = client.Client(client_id)

            if not IID:                                      
                if self.config.data.bias:
                                          
                    bias = self.config.data.bias
                                                       
                    pref = random.choices(labels, dist)[0]

                                                    
                    new_client.set_bias(pref, bias)
                elif self.config.data.shard:
                                           
                    shard = self.config.data.shard

                                         
                    new_client.set_shard(shard)
                elif self.config.data.noniid:
                                                        
                    min_cls = self.config.data.noniid["min_cls"]
                    max_cls = self.config.data.noniid["max_cls"]
                    cls_num = random.randint(min_cls, max_cls)

                                         
                    new_client.set_cls_num(cls_num)

            clients.append(new_client)

        logging.info('Total clients: {}'.format(len(clients)))

        if loader == 'bias':
            logging.info('Label distribution: {}'.format(
                [[client.pref for client in clients].count(label) for label in labels]))

        if loader == 'noniid':
            logging.info('Class distribution: {}'.format(
                [client.cls_num for client in clients]))

        if loading == 'static':
            if loader == 'shard':                      
                self.loader.create_shards()

                                                
            [self.set_client_data(client) for client in clients]

        self.clients = clients

                                                   
        self.profile = Profile(num_clients, self.loader.labels)
        if bool(self.config.data.bias):
            self.profile.set_primary_label(
                [client.pref for client in self.clients])

    def make_clients_leaf(self, num_clients):
                                            
        clients = []
        self.select_loader_client = np.random.choice(np.arange(self.loader.num_clients),
                                                     num_clients, replace=False)
        for client_id in range(num_clients):
                               
            new_client = client.Client(client_id)

                                            
            train_data, test_data = self.loader.extract(self.select_loader_client[client_id])
            new_client.set_data_leaf(
                train_data,
                test_data,
                self.config
            )

            clients.append(new_client)

        self.clients = clients

        logging.info('Total clients: {} Total samples: {}'.format(num_clients,
            sum([len(client.data['x']) for client in self.clients])))
        logging.info('LEAF clients: {} LEAF samples: {}'.format(self.loader.num_clients,
            sum([len(self.loader.trainset['user_data'][user]['x']) for user in self.loader.trainset['users']])))

        logging.info('Number of train samples on clients: {}'.format(
            [self.loader.trainset['num_samples'][i] for i in self.select_loader_client]))
        logging.info('Number of test samples on clients: {}'.format(
            [self.loader.testset['num_samples'][i] for i in self.select_loader_client]))

                                                   
        self.profile = Profile(num_clients, self.loader.labels)
        if self.config.loader != 'leaf' and not self.config.data.IID:
            self.profile.set_primary_label(
                [client.pref for client in self.clients])

                            
    def run(self):
        rounds = self.config.fl.rounds
        target_accuracy = self.config.fl.target_accuracy
        reports_path = self.config.paths.reports

        if target_accuracy:
            logging.info('Training: {} rounds or {}% accuracy\n'.format(
                rounds, 100 * target_accuracy))
        else:
            logging.info('Training: {} rounds\n'.format(rounds))

                                              
        for round in range(1, rounds + 1):
            logging.info('**** Round {}/{} ****'.format(round, rounds))

                                              
            accuracy = self.round()

                                                    
            if target_accuracy and (accuracy >= target_accuracy):
                logging.info('Target accuracy reached.')
                break

        if self.config.model == 'AVE':
            import fl_model
            testset = self.loader.get_testset()
            testloader = fl_model.get_evalloader(testset, self.config.fl.batch_size)
            final_test_loss, final_test_accuracy = fl_model.evaluate(self.model, testloader)
            logging.info(
                'Final test loss: {} Final test accuracy: {:.2f}%'.format(
                    final_test_loss, 100 * final_test_accuracy
                )
            )

        if reports_path:
            with open(reports_path, 'wb') as f:
                pickle.dump(self.saved_reports, f)
            logging.info('Saved reports: {}'.format(reports_path))

    def round(self):
        import fl_model                                

                                                    
        sample_clients = self.selection()

                                  
        self.configuration(sample_clients)

                                                                 
        threads = [Thread(target=client.run) for client in sample_clients]
        [t.start() for t in threads]
        [t.join() for t in threads]

                                
        reports = self.reporting(sample_clients)

                                    
        logging.info('Aggregating updates')
        updated_weights = self.aggregation(reports)

                              
        fl_model.load_weights(self.model, updated_weights)

                                                   
        if self.config.paths.reports:
            self.save_reports(round, reports)

                                   
        self.save_model(self.model, self.config.paths.model)

                                    
        if self.config.clients.do_test and self.config.model != 'AVE':
            accuracy = self.accuracy_averaging(reports)
        elif self.config.model == 'AVE':
            valset = self.loader.get_valset()
            batch_size = self.config.fl.batch_size
            valloader = fl_model.get_evalloader(valset, batch_size)
            _, accuracy = fl_model.evaluate(self.model, valloader)
        else:
            testset = self.loader.get_testset()
            batch_size = self.config.fl.batch_size
            testloader = fl_model.get_testloader(testset, batch_size)
            accuracy = fl_model.test(self.model, testloader)

        logging.info('Average accuracy: {:.2f}%\n'.format(100 * accuracy))
        return accuracy

    def configuration(self, sample_clients):
        loader_type = self.config.loader
        loading = self.config.data.loading

        if loading == 'dynamic':
                                         
            if loader_type == 'shard':
                self.loader.create_shards()

                                                                
        for client in sample_clients:
            if loading == 'dynamic':
                self.set_client_data(client)                                 

                                       
            config = self.config

                                             
            client.configure(config)

    def reporting(self, sample_clients):
                                             
        reports = [client.get_report() for client in sample_clients]

        logging.info('Reports received: {}'.format(len(reports)))
        assert len(reports) == len(sample_clients)

        return reports

    def aggregation(self, reports):
        return self.federated_averaging(reports)

                        
    def extract_client_updates(self, reports):
        import fl_model                                

                                        
        baseline_weights = fl_model.extract_weights(self.model)

                                      
        weights = [report.weights for report in reports]

                                        
        updates = []
        for weight in weights:
            update = []
            for i, (name, w) in enumerate(weight):
                bl_name, baseline = baseline_weights[i]

                                                        
                assert name == bl_name

                                  
                delta = w - baseline
                update.append((name, delta))
            updates.append(update)

        return updates

    def federated_averaging(self, reports):
        import fl_model                                

        valid_reports = []
        for report in reports:
            try:
                num_samples = float(getattr(report, "num_samples", 0.0))
            except (TypeError, ValueError):
                num_samples = 0.0
            if np.isfinite(num_samples) and num_samples > 0.0:
                valid_reports.append((report, num_samples))

        if len(valid_reports) == 0:
            logging.warning(
                "Skip federated averaging: no valid positive-sample reports."
            )
            return fl_model.extract_weights(self.model)

        safe_reports = [item[0] for item in valid_reports]

                                      
        updates = self.extract_client_updates(safe_reports)

                                         
        total_samples = float(sum([item[1] for item in valid_reports]))
        if total_samples <= 0.0:
            logging.warning(
                "Skip federated averaging: total_samples=%.6f", total_samples
            )
            return fl_model.extract_weights(self.model)

                                    
        avg_update = [torch.zeros(x.size())                             
                      for _, x in updates[0]]
        for i, update in enumerate(updates):
            num_samples = valid_reports[i][1]
            for j, (_, delta) in enumerate(update):
                                                           
                avg_update[j] += delta * (num_samples / total_samples)

                                        
        baseline_weights = fl_model.extract_weights(self.model)

                                         
        updated_weights = []
        for i, (name, weight) in enumerate(baseline_weights):
            updated_weights.append((name, weight + avg_update[i]))

        return updated_weights

    def accuracy_averaging(self, reports):
        if len(reports) == 0:
            logging.warning("No reports for accuracy averaging; return 0.")
            return 0.0, 0.0

        valid_reports = []
        for report in reports:
            try:
                num_samples = float(getattr(report, "num_samples", 0.0))
            except (TypeError, ValueError):
                num_samples = 0.0
            if np.isfinite(num_samples) and num_samples > 0.0:
                valid_reports.append((report, num_samples))

                                     
        total_samples = float(sum([item[1] for item in valid_reports]))
        if total_samples <= 0.0:
            test_loss = float(np.mean([float(getattr(r, "test_loss", 0.0)) for r in reports]))
            accuracy = float(np.mean([float(getattr(r, "accuracy", 0.0)) for r in reports]))
            logging.warning(
                "Fallback to unweighted averaging: total_samples=%.6f", total_samples
            )
            return test_loss, accuracy

                                    
        test_loss = 0
        accuracy = 0
        for report, num_samples in valid_reports:
            test_loss += report.test_loss * (num_samples / total_samples)
            accuracy += report.accuracy * (num_samples / total_samples)

        return test_loss, accuracy

                       
    @staticmethod
    def flatten_weights(weights):
                                      
        weight_vecs = []
        for _, weight in weights:
            weight_vecs.extend(weight.flatten().tolist())

        return np.array(weight_vecs)

    def set_client_data(self, client):
        loader = self.config.loader

                                 
        if loader != 'shard':
            if self.config.data.partition.get('size'):
                partition_size = self.config.data.partition.get('size')
            elif self.config.data.partition.get('range'):
                start, stop = self.config.data.partition.get('range')
                partition_size = random.randint(start, stop)

                                           
        if loader == 'basic':
            data = self.loader.get_partition(partition_size)
        elif loader == 'bias':
            data = self.loader.get_partition(partition_size, client.pref)
        elif loader == 'shard':
            data = self.loader.get_partition()
        elif loader == 'noniid':
            data = self.loader.get_partition(partition_size, client.cls_num)
            labels = np.arange(10)
            logging.info('Client {} Label distribution: {}'.format(
                client.client_id,
                [[d[1] for d in data].count(label) for label in labels]))
        else:
            logging.critical('Unknown data loader type')

                             
        client.set_data(data, self.config)

    def save_model(self, model, path):
        path += '/global'
        torch.save(model.state_dict(), path)
        logging.info('Saved global model: {}'.format(path))

    def save_reports(self, round, reports):
        import fl_model                                

        if reports:
            self.saved_reports['round{}'.format(round)] = [(report.client_id, self.flatten_weights(
                report.weights)) for report in reports]

                                
        self.saved_reports['w{}'.format(round)] = self.flatten_weights(
            fl_model.extract_weights(self.model))
