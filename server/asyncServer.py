import logging
import pickle
import numpy as np
from threading import Thread
import torch
from queue import PriorityQueue
import os
import sys
import time
from .syncServer import SyncServer
from .record import Record
from .asyncEvent import asyncEvent

class AsyncServer(SyncServer):
    """Asynchronous federated learning server."""

    def load_model(self):
        import fl_model                                

        saved_model_path = self.config.paths.saved_model
        model_type = self.config.model

        logging.info('Model: {}'.format(model_type))

                             
        self.model = fl_model.Net()
        self.async_save_model(self.model, saved_model_path, 0.0)

                                                   
        if self.config.paths.reports:
            self.saved_reports = {}
            self.save_reports(0, [])                      

                                                     
    def server_warmup(self):
        logging.info('Server warmup ...')

                                                           
        _ = [client.async_global_configure(self.config, 0.0) for client in
             self.clients]

        if self.config.model not in ['CIFAR-10', 'FEMNIST', 'Shakespeare']:
            threads = [Thread(target=client.run(reg=True, rho=self.config.async_params.rho))
                       for client in self.clients]
            [t.start() for t in threads]
            [t.join() for t in threads]
        else:                                                      
            _ = [client.run(reg=True, rho=self.config.async_params.rho) for client in self.clients]

                                
        reports = self.reporting(self.clients)

                                                         
        self.grads = [report.grads for report in reports]
        self.grads = np.array(self.grads)                                        
        self.client_samples = [report.num_samples for report in reports]
        self.client_samples = np.array(self.client_samples)

                                                                  
        if self.config.pca_dim > 0:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=self.config.pca_dim)
            self.grads = self.pca.fit_transform(self.grads)                             

                                                     
            for client in self.clients:
                client.pca = self.pca

                                         
    def run(self, logger=None):
        import fl_model                                
        rounds = self.config.server.rounds
        target_accuracy = self.config.fl.target_accuracy
                                                  
        model = self.config.model
        run_wall_start = time.time()

                               
        self.alpha_0 = self.config.async_params.alpha_0
        self.staleness_func = self.config.async_params.staleness_func
        self.mu = float(getattr(self.config.async_params, 'mu', 0.5))

        if target_accuracy:
            logging.info('Training: {} rounds or {}% accuracy\n'.format(
                rounds, 100 * target_accuracy))
        else:
            logging.info('Training: {} rounds\n'.format(rounds))

                                                 
                                                                                   
                                                                           
                                                              
        qEvents = PriorityQueue()
        next_gw_agg = []
        for gateway in self.gateways:
                                                     
            gateway.update(self.grads, self.client_samples)

                                                                        
                                 
            gateway.async_gateway_configure(self.config, 0.0, 0)

                                                    
            T_new = gateway.async_run(0.0, self.loader, logger)

            next_gw_agg.append(T_new)

                                                          
            new_event = asyncEvent(gateway, 0, 0.0, T_new)
            qEvents.put(new_event)

                                                    
        for round in range(1, rounds + 1):
            logging.info('\n**** Round {}/{} ****'.format(round, rounds))

            event = qEvents.get()
            select_gateway = event.client
            T_cur = event.aggregate_time                       
            display_time = self._display_time(T_cur, run_wall_start)
            print(next_gw_agg)

                                    
            report = select_gateway.get_report()
            logging.info('Select gateway {}, time {} s, regional validation loss: {}, accuracy: {}'.format(
                select_gateway.gateway_id, display_time, report.val_loss, report.val_accuracy))

                                                        
            self.grads[report.conn_ind] = report.grads[report.conn_ind]
            self.client_samples[report.conn_ind] = report.client_samples[report.conn_ind]
            self._accumulate_comm_from_report(report)
            self.gateway_cs_time[report.gateway_id] += report.gateway_cs_time
            self.gateway_round_time[report.gateway_id] += report.gateway_round_time

                                        
                                                       
            staleness = round - event.download_round
            updated_weights = self.aggregation(report, staleness)

                                  
            fl_model.load_weights(self.model, updated_weights)

                                                       
                                          
                                                   

                                       
            saved_model_path = self.config.paths.saved_model
            self.async_save_model(self.model, saved_model_path, T_cur)

                                        
            val_loss, val_accuracy = self._evaluate_global_split('val')

            display_time = self._display_time(T_cur, run_wall_start)
            logging.info(
                'time: {} Validation loss: {} Validation accuracy: {:.2f}%\n'.format(
                    display_time, val_loss, 100 * val_accuracy
                ))

                                
            if logger is not None:
                logger.log_value('val_loss', val_loss, int(display_time * 1000))
                logger.log_value('val_accuracy', val_accuracy, int(display_time * 1000))
                logger.log_value('cs_gamma', self.gateways[0].cs_gamma, int(display_time * 1000))

                           
            wall_clock_s = float(time.time() - run_wall_start)
            self.records.append_record(t=display_time, val_loss=val_loss,
                                       val_acc=val_accuracy,
                                       cloud_ca_time=self.ca.asso_time,
                                       wall_clock_s=wall_clock_s,
                                       **self._comm_record_kwargs())
            for gateway_id in range(len(self.gateways)):
                self.records.append_to_key("gw_cs_time_{}".format(gateway_id),
                                           self.gateway_cs_time[gateway_id])
                self.records.append_to_key("gw_round_time_{}".format(gateway_id),
                                           self.gateway_round_time[gateway_id])
            self._save_latest_record_if_enabled()

                                                    
            if model != 'HPWREN' and target_accuracy and\
                    (self.records.get_latest_acc() >= target_accuracy):
                logging.info('Target validation accuracy reached.')
                break
            elif model == 'HPWREN' and target_accuracy and\
                    (self.records.get_latest_acc() <= target_accuracy):
                logging.info('Target validation MSE reached.')
                break

                                               
            if round % self.config.server.adjust_round == 0:
                self.conn = self.ca.solve(self.conn_ub, self.grads, self.client_samples,
                                          self.R, self.R_ub, self.config.ca_phi)
                for i in range(len(self.clients)):
                                                 
                    gateway_id_old = self.clients[i].gateway_id
                    gateway_id = np.where(self.conn[i])[0][0]
                                       
                    if gateway_id_old != gateway_id:
                        self.gateways[gateway_id_old].remove_client(self.clients[i].client_id)
                        self.gateways[gateway_id].add_client(self.clients[i].client_id)
                        self.clients[i].set_gateway(gateway_id)

                                                                        
                                 
            select_gateway.async_gateway_configure(self.config, T_cur, round)

                                                    
            T_new = select_gateway.async_run(T_cur, self.loader, logger)

            next_gw_agg[select_gateway.gateway_id] = T_new

                                                          
            new_event = asyncEvent(select_gateway, round, T_cur, T_new-T_cur)
            qEvents.put(new_event)

                         
                                                
                                                   
                                                                   

                                           
        self._log_final_test(logger, self._display_time(T_cur, run_wall_start) * 1000)
        saved_model_path = self.config.paths.saved_model
        self.rm_old_models(saved_model_path, T_cur + 1.0)

    def aggregation(self, reports, staleness=None):
        return self.federated_async(reports, staleness)

    def extract_client_weights(self, reports):
                                      
        weights = [report.weights for report in reports]

        return weights

    def federated_async(self, report, staleness):
        import fl_model                                

                                      
        weights = self.extract_client_weights([report])[0]

                                                       
        baseline_weights = fl_model.extract_weights(self.model)

                                                                            
        alpha_0 = float(self.alpha_0)
        f_tau = float(self.staleness(staleness))
        alpha_t = alpha_0 * f_tau
        logging.info(
            '{} alpha_0: {} staleness(tau): {} f(tau): {} alpha_t: {}'.format(
                self.staleness_func, alpha_0, staleness, f_tau, alpha_t
            )
        )

                                         
        updated_weights = []
        for i, (t1, t2) in enumerate(zip(baseline_weights, weights)):
            assert t1[0] == t2[0], "weights names do not match!"
            name, old_weight = t1[0], t1[1]
            new_weight = t2[1]
            updated_weights.append(
                (name, (1 - alpha_t) * old_weight + alpha_t * new_weight)
            )

        return updated_weights

    def staleness(self, staleness):
        if self.staleness_func == "constant":
            return 1
        elif self.staleness_func == "polynomial":
            mu = max(0.0, float(getattr(self, 'mu', 0.5)))
            return pow(staleness+1, -mu)
        elif self.staleness_func == "hinge":
            a, b = 10, 4
            if staleness <= b:
                return 1
            else:
                return 1 / (a * (staleness - b) + 1)

    def async_save_model(self, model, path, download_time):
        path += '/global_' + '{}'.format(download_time)
        torch.save(model.state_dict(), path)
        logging.info('Saved global model: {}'.format(path))

    def rm_old_models(self, path, cur_time):
        for filename in os.listdir(path):
            try:
                model_time = float(filename.split('_')[1])
                if model_time < cur_time:
                    os.remove(os.path.join(path, filename))
                    logging.info('Remove model {}'.format(filename))
            except Exception as e:
                logging.debug(e)
                continue
