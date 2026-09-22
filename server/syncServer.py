import sys
import numpy as np
import logging
import pickle
from threading import Thread
import time
import os
from .server import Server
from .gateway import Gateway
from .record import Record
from .clientAssociation import ClientAssociation

class SyncServer(Server):
    """Synchronous federated learning server."""

    def boot(self):
        logging.info('Booting {} server...'.format(self.config.server.mode))

        model_path = self.config.paths.model
        total_clients = self.config.clients.total
        total_gateways = self.config.gateways.total

                                     
        sys.path.append(model_path)

                     
        self.load_data()
        self.load_model()
        if self.config.loader != 'leaf':
            num_clients = self.config.clients.total
            self.make_clients(num_clients)
        else:
            num_clients = min(self.config.clients.total, self.loader.num_clients)
            self.make_clients_leaf(num_clients)
        self.make_gateways(total_gateways)

                                                    
        self.server_warmup()

                         
        self.set_link()

        for gateway in self.gateways:
            logging.info('Gw {}: {} clients from '
                         'total feasible clients of {}'.format(
                gateway.gateway_id,
                self.conn.sum(axis=0)[gateway.gateway_id],
                self.conn_ub.sum(axis=0)[gateway.gateway_id]
            ))

                                    
        self.records = Record("t", "val_loss", "val_acc", "total_comm_size",
                              "cloud_ca_time")
        for gateway_id in range(total_gateways):
            self.records.insert_key("gw_cs_time_{}".format(gateway_id),
                                    "gw_round_time_{}".format(gateway_id))
        self.terminal_only = self._terminal_only_enabled()
        self.metrics_csv_path = None
        if not self.terminal_only:
            self.metrics_csv_path = self._resolve_metrics_csv_path()
            self._prepare_metrics_csv_for_new_run(self.metrics_csv_path)
        self.total_comm_size = 0.0
        self.total_comm_client_up = 0.0
        self.total_comm_gw_to_cloud_up = 0.0
        self.total_comm_cloud_to_gw_down = 0.0
        self.total_comm_prefetch = 0.0
        self.total_comm_size_v2 = 0.0
        self.gateway_cs_time = [0.0 for _ in range(total_gateways)]
        self.gateway_round_time = [0.0 for _ in range(total_gateways)]
        self.sim_comm_profile = self._detect_sim_comm_profile()
        self.use_legacy_ave_logic = (self.sim_comm_profile == "legacy_ave")
        logging.info(
            'Server sim/comm profile=%s data=%s',
            self.sim_comm_profile,
            str(getattr(getattr(self.config, 'paths', None), 'data', ''))
        )
        logging.info('Terminal-only=%s', int(self.terminal_only))

    def _detect_sim_comm_profile(self):
        raw_profile = str(os.environ.get('MMQS_SIM_COMM_PROFILE', '')).strip().lower()
        if raw_profile in ('legacy_ave', 'legacy', 'ave_legacy'):
            return 'legacy_ave'
        if raw_profile in ('modern', 'new', 'v2'):
            return 'modern'

        data_path = str(getattr(getattr(self.config, 'paths', None), 'data', ''))
        data_norm = data_path.replace('\\', '/').lower()
        if '/ave_fed/' in data_norm or data_norm.endswith('/ave_fed'):
            return 'legacy_ave'
        return 'modern'

    @staticmethod
    def _terminal_only_enabled():
                                                                 
                                                         
        raw = str(os.environ.get('MMQS_TERMINAL_ONLY', '1')).strip().lower()
        if raw not in ('1', 'true', 'yes', 'on'):
            logging.info('MMQS_TERMINAL_ONLY=%s ignored; forced terminal-only mode enabled.', raw)
        return True

    def _display_time(self, sim_time, run_wall_start):
        mode = str(getattr(getattr(self.config, 'server', None), 'mode', '')).strip().lower()
        if mode == 'sync':
            return float(sim_time)
        return float(max(0.0, time.time() - float(run_wall_start)))

    def _save_latest_record_if_enabled(self):
        if self.terminal_only:
            return
        if self.metrics_csv_path is None:
            return
        self.records.save_latest_record(self.metrics_csv_path)

    def _evaluate_global_split(self, split_name):
        """Evaluate the global model on one explicit AVE split only."""
        import fl_model
        if split_name == 'val':
            evalset = self.loader.get_valset()
        elif split_name == 'test':
            evalset = self.loader.get_testset()
        else:
            raise ValueError('unsupported evaluation split: {}'.format(split_name))
        evalloader = fl_model.get_evalloader(evalset, self.config.fl.batch_size)
        return fl_model.evaluate(self.model, evalloader)

    def _log_final_test(self, logger, timestamp_ms):
        """Final-test reporting only; never feed this result into training control."""
        final_test_loss, final_test_accuracy = self._evaluate_global_split('test')
        logging.info(
            'Final test loss: {} Final test accuracy: {:.2f}%'.format(
                final_test_loss, 100 * final_test_accuracy
            )
        )
        if logger is not None:
            logger.log_value('final_test_loss', final_test_loss, int(timestamp_ms))
            logger.log_value('final_test_accuracy', final_test_accuracy, int(timestamp_ms))
        return final_test_loss, final_test_accuracy

    def _resolve_metrics_csv_path(self):
        raw = getattr(self.config.paths, "metrics_csv", None)
        if raw is None or len(str(raw).strip()) <= 0:
            raw = self.config.model_name + ".csv"
        path = str(raw).strip()
        if not path.lower().endswith(".csv"):
            path += ".csv"
        return os.path.abspath(path)

    @staticmethod
    def _prepare_metrics_csv_for_new_run(path):
        folder = os.path.dirname(path)
        if folder and (not os.path.isdir(folder)):
            os.makedirs(folder, exist_ok=True)
        if os.path.isfile(path):
            os.remove(path)
            logging.info("Removed stale metrics csv: %s", path)

    def _accumulate_comm_from_report(self, report):
        model_size = float(getattr(self.config.fl, "model_size", 0.0))
        legacy_gateway_comm = float(getattr(report, "gateway_comm_size", 0.0))
        self.total_comm_size += legacy_gateway_comm + model_size

        if self.use_legacy_ave_logic:
            self.total_comm_client_up += legacy_gateway_comm
            self.total_comm_gw_to_cloud_up += model_size
            self.total_comm_size_v2 = self.total_comm_size
            return

        comm_client_up = float(getattr(report, "gateway_comm_client_up", legacy_gateway_comm))
        comm_prefetch = float(getattr(report, "gateway_comm_prefetch", 0.0))
        comm_gw_to_cloud_up = model_size
                                                                                                
        comm_cloud_to_gw_down = model_size

        self.total_comm_client_up += comm_client_up
        self.total_comm_prefetch += comm_prefetch
        self.total_comm_gw_to_cloud_up += comm_gw_to_cloud_up
        self.total_comm_cloud_to_gw_down += comm_cloud_to_gw_down
        self.total_comm_size_v2 = (
            self.total_comm_client_up +
            self.total_comm_gw_to_cloud_up +
            self.total_comm_cloud_to_gw_down +
            self.total_comm_prefetch
        )

    def _comm_record_kwargs(self):
        if self.use_legacy_ave_logic:
            return {
                "total_comm_size": self.total_comm_size,
            }
        return {
            "total_comm_size": self.total_comm_size,
            "total_comm_size_v2": self.total_comm_size_v2,
            "comm_client_up": self.total_comm_client_up,
            "comm_gw_to_cloud_up": self.total_comm_gw_to_cloud_up,
            "comm_cloud_to_gw_down": self.total_comm_cloud_to_gw_down,
            "comm_prefetch": self.total_comm_prefetch,
        }

    def make_gateways(self, num_gws):
        gateways = []
        for gateway_id in range(num_gws):
                                
            new_gw = Gateway(gateway_id, self.clients, self.config)
            gateways.append(new_gw)

        self.gateways = gateways

    def set_link(self):
        model_size = self.config.fl.model_size

        if self.config.delay_mode == 'nycmesh':
                                                         
            delay_sr_to_gw = np.loadtxt(self.config.delays.gateway_client, delimiter=',')[:, :self.config.gateways.total]
            delay_sr = np.loadtxt(self.config.delays.comp_time, delimiter=',')

                                                                                
            valid_sr_id = ~np.all(delay_sr_to_gw == 0, axis=1)
            delay_sr_to_gw = delay_sr_to_gw[valid_sr_id][:self.config.clients.total]
            delay_sr = delay_sr[valid_sr_id][:self.config.clients.total]
            self.conn_ub = (delay_sr_to_gw > 0).astype(np.int)

            assert len(self.clients) <= delay_sr_to_gw.shape[0],\
                "More clients than the rows in the provided delay matrix!"
            assert len(self.gateways) <= delay_sr_to_gw.shape[1],\
                "More gateways than the columns in the provided delay matrix!"

        elif self.config.delay_mode == 'uniform':
                                                                    
                                                                                                
                                                 
            self.conn_ub = (np.random.uniform(size=(len(self.clients), len(self.gateways))) <
                            self.config.link.sparse_ratio).astype(np.int)

                                                               
            zero_rows_flag = (self.conn_ub.sum(axis=1) < .1)
            print(zero_rows_flag)
            if np.sum(zero_rows_flag) > 0:                      
                self.conn_ub[zero_rows_flag, 0] = 1

            speed_sr_to_gw = np.random.uniform(low=self.config.link.min,
                                               high=self.config.link.max,
                                               size=(len(self.clients), len(self.gateways)))
            delay_sr_to_gw = self.conn_ub * (model_size / speed_sr_to_gw)

            delay_sr = np.random.uniform(low=self.config.comp_time.min,
                                         high=self.config.comp_time.max,
                                         size=len(self.clients))
        else:
            raise ValueError(
                    "delay mode not implemented: {}".format(self.config.delay_mode))

                                                  
        if self.config.loader == 'bias':
            pref = [client.pref for client in self.clients]
            self.ca = ClientAssociation(self.config.association,
                                        self.config.model_name,
                                        pref=pref, labels=self.labels)
        elif self.config.loader == 'noniid':
            cls_num = [client.cls_num for client in self.clients]
            self.ca = ClientAssociation(self.config.association,
                                        self.config.model_name,
                                        cls_num=cls_num, labels=self.labels)
        else:
            self.ca = ClientAssociation(self.config.association,
                                        self.config.model_name)

                                                                               
        self.est_total_delay = delay_sr_to_gw + delay_sr.reshape((-1, 1)) + 1e-10
        self.est_total_delay = self.est_total_delay
        self.R = np.divide(model_size, self.est_total_delay)
        self.R_ub = np.array([self.config.gateways.throughput_ub] * self.config.gateways.total)
        self.conn = self.ca.solve(self.conn_ub, self.grads, self.client_samples,
                                  self.R, self.R_ub, self.config.ca_phi)

        for i in range(len(self.clients)):
                                                  
            gateway_id = np.where(self.conn[i])[0][0]
                               
            self.gateways[gateway_id].add_client(self.clients[i].client_id)
            self.clients[i].set_gateway(gateway_id)

            comm_delay = delay_sr_to_gw[i, gateway_id]
            comp_delay = delay_sr[i]

            if self.config.delay_mode == 'nycmesh':
                self.clients[i].set_delay_to_gateway(comm_delay, comp_delay,
                                                     model_size)
            elif self.config.delay_mode == 'uniform':
                speed_mean = speed_sr_to_gw[i, gateway_id]
                self.clients[i].set_delay_to_gateway(comm_delay, comp_delay,
                                                     model_size, speed_mean,
                                                     self.config.link.std,
                                                     self.config.comp_time.std)
            else:
                raise ValueError(
                    "delay mode not implemented: {}".format(self.config.delay_mode))

        print('Client to gateway delay distribution: {}'.format(
            [client.delay for client in self.clients]))

                                 
        delay_gw_to_cl = np.loadtxt(self.config.delays.cloud_gateway, delimiter=',')
        for i in range(len(self.gateways)):
            self.gateways[i].set_delay_to_cloud(delay_gw_to_cl[i])

        print('Gateway to cloud delay distribution: {}'.format(
            [gateway.delay for gateway in self.gateways]))

                                                     
    def server_warmup(self):
        logging.info('Server warmup ...')
                                                           
        _ = [client.sync_global_configure(self.config) for client in self.clients]
        if self.config.model not in ['CIFAR-10', 'FEMNIST']:
            threads = [Thread(target=client.run) for client in self.clients]
            [t.start() for t in threads]
            [t.join() for t in threads]
        else:                                                      
            _ = [client.run() for client in self.clients]

                                
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

        if target_accuracy:
            logging.info('Training: {} rounds or {}% accuracy\n'.format(
                rounds, 100 * target_accuracy))
        else:
            logging.info('Training: {} rounds\n'.format(rounds))

                                              
        T_old = 0.0
        for round in range(1, rounds + 1):
            logging.info('**** Round {}/{} ****'.format(round, rounds))

                                                   
            T_new = self.sync_round(T_old, logger)
            display_time = self._display_time(T_new, run_wall_start)
            logging.info('Round finished at time {} s\n'.format(display_time))

                         
            T_old = T_new

                                        
            val_loss, val_accuracy = self._evaluate_global_split('val')

            logging.info(
                'Validation loss: {} Validation accuracy: {:.2f}%'.format(
                    val_loss, 100 * val_accuracy
                ))

                                                                           
            for gateway in self.gateways:
                updater = getattr(gateway, 'update_global_loss_feedback', None)
                if updater is None:
                    continue
                try:
                    updater(val_loss, round)
                except Exception as exc:                                
                    logging.warning(
                        'Push global loss feedback failed: gw=%s round=%s err=%s',
                        getattr(gateway, 'gateway_id', -1), round, str(exc)
                    )

                                
            if logger is not None:
                logger.log_value('val_loss', val_loss, int(display_time * 1000))
                logger.log_value('val_accuracy', val_accuracy, int(display_time * 1000))

                           
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

        self._log_final_test(logger, self._display_time(T_old, run_wall_start) * 1000)

                         
                                                
                                                   
                                                                   

    def sync_round(self, T_old, logger):
        import fl_model                                

                                                 
        for gateway in self.gateways:
            gateway.update(self.grads, self.client_samples)

                                        
        _ = [gateway.sync_gateway_configure(self.config) for gateway in self.gateways]

                                                                 
                                                                               
                                                 
                                     
                                    
        for gateway in self.gateways:
            gateway.sync_run(T_old, self.loader, logger)

                                
        reports = self.reporting(self.gateways)

                                 
        T_cur = max([report.finish_time for report in reports])

                                                                        
                            
        for report in reports:
            self.grads[report.conn_ind] = report.grads[report.conn_ind]
            self.client_samples[report.conn_ind] = report.client_samples[report.conn_ind]
            self._accumulate_comm_from_report(report)
            self.gateway_cs_time[report.gateway_id] += report.gateway_cs_time
            self.gateway_round_time[report.gateway_id] += report.gateway_round_time

                                    
        logging.info('Cloud aggregating updates')
        updated_weights = self.aggregation(reports)

                              
        fl_model.load_weights(self.model, updated_weights)

                                                   
                                      
                                              

                                   
        saved_model_path = self.config.paths.saved_model
        self.save_model(self.model, saved_model_path)

        return T_cur
