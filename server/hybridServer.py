import logging
import pickle
import numpy as np
from threading import Thread
import torch
from queue import PriorityQueue
import os
import sys
import json
import time
from datetime import datetime
from .syncServer import SyncServer
from .record import Record
from .asyncEvent import asyncEvent

class HybridServer(SyncServer):
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

                                      
        if self.config.model not in ['CIFAR-10', 'FEMNIST']:
            threads = [Thread(target=client.run)
                       for client in self.clients]
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

                               
        self.alpha_0 = self.config.async_params.alpha_0
        self.staleness_func = self.config.async_params.staleness_func
        self.mu = float(getattr(self.config.async_params, 'mu', 0.5))
        self.hybrid_v1_enabled = bool(
            getattr(getattr(self.config, 'hybrid', None), 'enabled', False)
        )
                                                                           
        self.hybrid_eta_blend = 0.30
        self.hybrid_eta_floor = 0.85
        self.hybrid_eta_ceil = 1.15
        self.latest_cloud_staleness = 0
        self.latest_cloud_alpha_t = 0.0
        self.latest_cloud_eta_r = 1.0
        self.latest_cloud_eta_raw = 1.0
        self.latest_cloud_eta_scaled = 1.0
                                                                      
        best_acc_so_far = -float('inf')
        best_val_loss_so_far = float('inf')
        best_t_so_far = 0.0
        best_total_comm_size_so_far = 0.0
        best_round_so_far = 0
        is_hpwren = (model == 'HPWREN')

        if target_accuracy:
            logging.info('Training: {} rounds or {}% accuracy\n'.format(
                rounds, 100 * target_accuracy))
        else:
            logging.info('Training: {} rounds\n'.format(rounds))

                                                    
        self.server_warmup()

                                                 
                                                                                   
                                                                           
                                                              
        qEvents = PriorityQueue()
        next_gw_agg = []
        for gateway in self.gateways:
                                                     
            gateway.update(self.grads, self.client_samples)

                                          
                                                                        
                                
            gateway.hybrid_gateway_configure(self.config, 0.0)

                                                   
            T_new = gateway.sync_run(0.0, self.loader, logger)

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
            logging.info('Select gateway {}, time {} s'.format(select_gateway.gateway_id, display_time))

                                    
            report = select_gateway.get_report()

                                                        
            self.grads[report.conn_ind] = report.grads[report.conn_ind]
            self.client_samples[report.conn_ind] = report.client_samples[report.conn_ind]
            self._accumulate_comm_from_report(report)
            self.gateway_cs_time[report.gateway_id] += report.gateway_cs_time
            self.gateway_round_time[report.gateway_id] += report.gateway_round_time

                                        
            logging.info('Cloud aggregating updates')
            staleness = round - event.download_round
            eta_r = self._compute_region_eta(report) if self.hybrid_v1_enabled else 1.0
            updated_weights = self.aggregation([report], staleness, eta_r=eta_r)

                                  
            fl_model.load_weights(self.model, updated_weights)

                                                       
                                          
                                                   

                                       
            saved_model_path = self.config.paths.saved_model
            self.async_save_model(self.model, saved_model_path, T_cur)

                                        
            val_loss, val_accuracy = self._evaluate_global_split('val')

            logging.info(
                'Validation loss: {} Validation accuracy: {:.2f}%'.format(
                    val_loss, 100 * val_accuracy
                ))
            if float(val_accuracy) >= 0.9:
                logging.info(
                    'comm={:.3f}'.format(float(self.total_comm_size) / 1000.0)
                )

                                                                           
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

                                                                                        
            improve_eps = 1e-12
            if is_hpwren:
                is_best_checkpoint = bool(float(val_loss) < (best_val_loss_so_far - improve_eps))
            else:
                is_best_checkpoint = bool(float(val_accuracy) > (best_acc_so_far + improve_eps))
            if is_best_checkpoint:
                best_acc_so_far = float(val_accuracy)
                best_val_loss_so_far = float(val_loss)
                best_t_so_far = float(T_cur)
                best_total_comm_size_so_far = float(self.total_comm_size)
                best_round_so_far = int(round)
                self.async_save_best_model(self.model, saved_model_path)
                self.async_save_best_meta(
                    saved_model_path,
                    {
                        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "trial": int(getattr(self.config, "trial", -1)),
                        "model_name": str(getattr(self.config, "model_name", "")),
                        "best_round": int(best_round_so_far),
                        "best_t": float(best_t_so_far),
                        "best_acc": float(best_acc_so_far),
                        "best_val_loss": float(best_val_loss_so_far),
                        "best_total_comm_size": float(best_total_comm_size_so_far),
                        "target_accuracy": float(target_accuracy) if target_accuracy is not None else None
                    }
                )
                if float(best_acc_so_far) >= 0.9:
                    logging.info(
                        'Best checkpoint updated: round=%s t=%.6f acc=%.6f loss=%.6f comm=%.3f',
                        best_round_so_far,
                        best_t_so_far,
                        best_acc_so_far,
                        best_val_loss_so_far,
                        best_total_comm_size_so_far / 1000.0
                    )
                else:
                    logging.info(
                        'Best checkpoint updated: round=%s t=%.6f acc=%.6f loss=%.6f',
                        best_round_so_far,
                        best_t_so_far,
                        best_acc_so_far,
                        best_val_loss_so_far
                    )

            display_time = self._display_time(T_cur, run_wall_start)

                                
            if logger is not None:
                logger.log_value('val_loss', val_loss, int(display_time * 1000))
                logger.log_value('val_accuracy', val_accuracy, int(display_time * 1000))

                           
            prefetch_total = int(np.sum([gateway.prefetch_total for gateway in self.gateways]))
            prefetch_hit = int(np.sum([gateway.prefetch_hit for gateway in self.gateways]))
            prefetch_miss = int(np.sum([gateway.prefetch_miss for gateway in self.gateways]))
            prefetch_hit_rate = float(prefetch_hit) / float(max(1, prefetch_total))
            k_cap = int(getattr(select_gateway, 'k_cap_placeholder', 0) or 0)
            k_selected = int(getattr(select_gateway, 'k_selected_placeholder', 0) or 0)
            selected_client_ids_hash = str(
                getattr(select_gateway, 'selected_client_ids_hash_placeholder', '')
            )
            quality_guard_active = int(getattr(select_gateway, 'quality_guard_active_placeholder', 0) or 0)
            quality_guard_hold_until_round = int(
                getattr(select_gateway, 'quality_guard_hold_until_round', 0) or 0
            )
            quality_guard_trigger_count = int(
                getattr(select_gateway, 'quality_guard_trigger_count', 0) or 0
            )
            self.records.append_record(t=display_time, val_loss=val_loss,
                                       val_acc=val_accuracy,
                                       cloud_ca_time=self.ca.asso_time,
                                       wall_clock_s=float(time.time() - run_wall_start),
                                       cloud_staleness=self.latest_cloud_staleness,
                                       cloud_alpha_t=self.latest_cloud_alpha_t,
                                       cloud_eta_r=self.latest_cloud_eta_r,
                                       cloud_eta_raw=self.latest_cloud_eta_raw,
                                       cloud_eta_scaled=self.latest_cloud_eta_scaled,
                                       **self._comm_record_kwargs(),
                                       k_cap=k_cap,
                                       k_selected=k_selected,
                                       selected_client_ids_hash=selected_client_ids_hash,
                                       quality_guard_active=quality_guard_active,
                                       quality_guard_hold_until_round=quality_guard_hold_until_round,
                                       quality_guard_trigger_count=quality_guard_trigger_count,
                                       is_best_checkpoint=int(is_best_checkpoint),
                                       best_acc_so_far=float(best_acc_so_far),
                                       best_val_loss_so_far=float(best_val_loss_so_far),
                                       best_t_so_far=float(best_t_so_far),
                                       best_total_comm_size_so_far=float(best_total_comm_size_so_far),
                                       best_round_so_far=int(best_round_so_far),
                                       prefetch_total=prefetch_total,
                                       prefetch_hit=prefetch_hit,
                                       prefetch_miss=prefetch_miss,
                                       prefetch_hit_rate=prefetch_hit_rate)
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

                                          
                                                                        
                                
            select_gateway.hybrid_gateway_configure(self.config, T_cur)

                                                   
            T_new = select_gateway.sync_run(T_cur, self.loader, logger)

            next_gw_agg[select_gateway.gateway_id] = T_new

                                                          
            new_event = asyncEvent(select_gateway, round, T_cur, T_new-T_cur)
            qEvents.put(new_event)

                         
                                                
                                                   
                                                                   

                                           
        best_path = os.path.join(self.config.paths.saved_model, 'global_best')
        if os.path.isfile(best_path):
            self.model.load_state_dict(torch.load(best_path))
            logging.info('Loaded validation-selected checkpoint for final test: %s', best_path)
        self._log_final_test(logger, self._display_time(T_cur, run_wall_start) * 1000)
        saved_model_path = self.config.paths.saved_model
        self.rm_old_models(saved_model_path, T_cur + 1.0)

    def aggregation(self, reports, staleness=None, eta_r=1.0):
        return self.federated_async(reports, staleness, eta_r=eta_r)

    def extract_client_weights(self, reports):
                                      
        weights = [report.weights for report in reports]

        return weights

    def federated_async(self, reports, staleness, eta_r=1.0):
        import fl_model                                

                                                       
        baseline_weights = fl_model.extract_weights(self.model)

                                                                            
        alpha_0 = float(self.alpha_0)
        f_tau = float(self.staleness(staleness))
        alpha_t = alpha_0 * f_tau
        eta_raw = float(np.clip(float(eta_r), 0.0, 1.0))
        gw_count = max(1, int(len(self.gateways)))
        eta_scaled = float(np.clip(
            eta_raw * float(gw_count),
            float(self.hybrid_eta_floor),
            float(self.hybrid_eta_ceil)
        ))
        blend = float(np.clip(float(self.hybrid_eta_blend), 0.0, 1.0))
        eta_effective = (1.0 - blend) + blend * eta_scaled
        alpha_eff = alpha_t * eta_effective
        self.latest_cloud_staleness = int(staleness) if staleness is not None else 0
        self.latest_cloud_alpha_t = float(alpha_t)
        self.latest_cloud_eta_r = float(eta_effective)
        self.latest_cloud_eta_raw = float(eta_raw)
        self.latest_cloud_eta_scaled = float(eta_scaled)
        logging.info(
            '{} alpha_0: {} staleness(tau): {} f(tau): {} alpha_t: {} eta_raw: {} eta_scaled: {} eta_effective: {} alpha_eff: {}'.format(
                self.staleness_func, alpha_0, staleness, f_tau, alpha_t,
                eta_raw, eta_scaled, eta_effective, alpha_eff
            )
        )

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
                "Cloud aggregation skipped: no valid positive-sample reports (staleness=%s).",
                staleness
            )
            return baseline_weights

        safe_reports = [item[0] for item in valid_reports]

                                      
        weights = self.extract_client_weights(safe_reports)

                                         
        total_samples = float(sum([item[1] for item in valid_reports]))
        if total_samples <= 0.0:
            logging.warning(
                "Cloud aggregation skipped: total_samples=%.6f (staleness=%s).",
                total_samples, staleness
            )
            return baseline_weights

                                    
        new_weights = [torch.zeros(x.size())                             
                      for _, x in weights[0]]
        for i, update in enumerate(weights):
            num_samples = valid_reports[i][1]
            for j, (_, weight) in enumerate(update):
                                                           
                new_weights[j] += weight * (num_samples / total_samples)

                                         
        updated_weights = []
        for i, (name, weight) in enumerate(baseline_weights):
            updated_weights.append(
                (name, (1 - alpha_eff) * weight + alpha_eff * new_weights[i])
            )

        return updated_weights

    def _compute_region_eta(self, report):
        """Compute eta_r = Nbar_r / sum_q Nbar_q with safe fallbacks."""
        numer = float(getattr(report, 'num_samples', 0.0))
        if numer <= 0.0:
            return 1.0

        totals = []
        for gateway in self.gateways:
            try:
                gw_samples = float(np.sum(gateway.client_samples[gateway.conn_ind]))
            except Exception:
                gw_samples = float('nan')
            if np.isfinite(gw_samples) and gw_samples > 0.0:
                totals.append(gw_samples)

        denom = float(np.sum(totals)) if len(totals) > 0 else 0.0
        if denom <= 0.0:
            return 1.0

        eta_r = numer / denom
        return float(np.clip(eta_r, 0.0, 1.0))

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

    def async_save_best_model(self, model, path):
        best_path = os.path.join(path, 'global_best')
        torch.save(model.state_dict(), best_path)
        logging.info('Saved best global model: {}'.format(best_path))

    @staticmethod
    def async_save_best_meta(path, payload):
        meta_path = os.path.join(path, 'global_best_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        logging.info('Saved best global meta: {}'.format(meta_path))

    def rm_old_models(self, path, cur_time):
        for filename in os.listdir(path):
            if filename in ('global_best', 'global_best_meta.json'):
                continue
            try:
                model_time = float(filename.split('_')[1])
                if model_time < cur_time:
                    os.remove(os.path.join(path, filename))
                    logging.info('Remove model {}'.format(filename))
            except Exception as e:
                logging.debug(e)
                continue
