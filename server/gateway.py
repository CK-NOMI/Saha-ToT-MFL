import logging
import numpy as np
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import PriorityQueue
import torch
import copy
import os
from .asyncEvent import asyncEvent
from .clientSelection import ClientSelection
import time

class Gateway(object):
    """Gateway on the middle level."""

    def __init__(self, gateway_id, all_clients, config):
        self.gateway_id = gateway_id
        self.all_clients = all_clients                       
        self.conn_ind = np.zeros(len(all_clients), dtype=np.bool)
        self.config = config
        self.throughput_ub = config.gateways.throughput_ub
        self.selection = config.selection
        self.cs_gamma = config.cs_gamma_from
        self.cs_alpha = config.cs_alpha
        self.global_rounds = config.server.rounds
        self.mu = float(getattr(config.async_params, 'mu', 0.5))
                                                                           
        mmqs_cfg = getattr(config, 'mmqs', None)
        self.mmqs_enabled = getattr(mmqs_cfg, 'enabled', False)
        self.prefetch_enabled = getattr(config, 'prefetch').enabled if hasattr(config, 'prefetch') else False
        self.hybrid_v1_enabled = getattr(config, 'hybrid').enabled if hasattr(config, 'hybrid') else False
        self.mmqs_loss_aware_topk_enabled = bool(getattr(mmqs_cfg, 'loss_aware_topk_enabled', False))
        self.mmqs_weight_mode = getattr(mmqs_cfg, 'weight_mode', 'static')
        self.mmqs_W = getattr(mmqs_cfg, 'W', 0)
        self.mmqs_tot_api = getattr(config, 'mmqs_tot_api', None)
        self.quality_guard_enabled = getattr(config, 'quality_guard').enabled if hasattr(config, 'quality_guard') else False
        self.quality_guard_warmup_rounds = getattr(config, 'quality_guard').warmup_rounds if hasattr(config, 'quality_guard') else 1
        self.quality_guard_hold_rounds = getattr(config, 'quality_guard').hold_rounds if hasattr(config, 'quality_guard') else 2
        self.quality_guard_acc_drop_threshold = getattr(config, 'quality_guard').acc_drop_threshold if hasattr(config, 'quality_guard') else 0.015
        self.quality_guard_loss_rise_threshold = getattr(config, 'quality_guard').loss_rise_threshold if hasattr(config, 'quality_guard') else 0.03
        self.quality_guard_ema_factor = getattr(config, 'quality_guard').ema_factor if hasattr(config, 'quality_guard') else 0.3
        self.quality_guard_prefetch_ratio_scale = 0.5
        self.quality_guard_min_trigger_round = 3
        self.prefetch_strict_cache_match = getattr(config, 'prefetch').strict_cache_match if hasattr(config, 'prefetch') else False
        self.prefetch_top_m = getattr(config, 'prefetch').top_m if hasattr(config, 'prefetch') else 0
        self.prefetch_delay_reduction_ratio = getattr(config, 'prefetch').delay_reduction_ratio if hasattr(config, 'prefetch') else 0.0
                                                         
        self.k_cap_placeholder = None
        self.k_selected_placeholder = 0
        self.selected_client_ids_hash_placeholder = ""
                                                    
        self.loss_aware_topk_loss_ema = None
        self.loss_aware_topk_delta_ema = None
        self.loss_aware_topk_global_loss_prev = np.nan
        self.loss_aware_topk_global_loss_latest = np.nan
        self.loss_aware_topk_global_round = -1
        self.loss_aware_topk_last_target = None
        self.mmqs_ema_factor = 0.2
        self.mmqs_beta = max(0.0, self._safe_float(
            getattr(mmqs_cfg, 'beta', 4.0), 4.0
        ))
        self.mmqs_rho_min = float(np.clip(
            self._safe_float(getattr(mmqs_cfg, 'rho_min', 0.75), 0.75),
            0.1, 1.0
        ))
        self.mmqs_rho_max = float(np.clip(
            self._safe_float(getattr(mmqs_cfg, 'rho_max', 1.0), 1.0),
            self.mmqs_rho_min, 1.0
        ))
        self.mmqs_delta_ref = 0.08
        self.loss_aware_topk_delta_max = 2.0
        self.mmqs_eps = 1e-12
        self.loss_aware_topk_enabled_runtime = bool(self.mmqs_loss_aware_topk_enabled)
        self.prefetch_delay_reduction_ratio_runtime = float(self.prefetch_delay_reduction_ratio)
        self.quality_guard_round = 0
        self.quality_guard_hold_until_round = 0
        self.quality_guard_acc_ema = None
        self.quality_guard_loss_ema = None
        self.quality_guard_trigger_count = 0
        self.quality_guard_active_placeholder = 0
        self.prefetch_total = 0
        self.prefetch_hit = 0
        self.prefetch_miss = 0
        self.prefetch_hit_rate = 0.0
        self.prefetch_global_version = 'global_0.0'
        self.prefetch_round_seq = 0
                                                                                                   
        self.prefetch_shadow_cache = {}

                                                                            
        self.client_samples = None
        self.grads = None
        self.cs = None
        self.sim_comm_profile = self._detect_sim_comm_profile()
        self.use_legacy_ave_logic = (self.sim_comm_profile == "legacy_ave")
        logging.info(
            'Gw %s sim/comm profile=%s data=%s',
            self.gateway_id,
            self.sim_comm_profile,
            str(getattr(getattr(self.config, 'paths', None), 'data', ''))
        )

    def add_client(self, client_id):
        self.conn_ind[client_id] = True
        self.clients = [self.all_clients[i] for i in
                        range(len(self.all_clients))
                        if self.conn_ind[i]]

    def remove_client(self, client_id):
        self.conn_ind[client_id] = False
        self.clients = [self.all_clients[i] for i in
                        range(len(self.all_clients))
                        if self.conn_ind[i]]

                         
    def download(self, argv):
                                   
        try:
            return copy.deepcopy(argv)
        except:
            return argv

    def upload(self, argv):
                              
        try:
            return copy.deepcopy(argv)
        except:
            return argv

    def get_report(self):
                                   
        return self.upload(self.report)

    def update(self, grads, client_samples):
        self.grads = self.download(grads)               
        self.client_samples = self.download(client_samples)               

    def set_delay_to_cloud(self, delay):
                                                           
        self.delay = delay

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

    def _get_or_create_client_selector(self, rounds, semi_period):
        """Build selector once and reuse it to preserve MMQS/ToT runtime memory."""
        tot_api_cfg = self.mmqs_tot_api._asdict() if self.mmqs_tot_api is not None else None
        if self.cs is None:
            self.cs = ClientSelection(self.clients, self.selection, self.config.model_name,
                                      self.throughput_ub, rounds, self.cs_gamma,
                                      self.cs_alpha, semi_period,
                                      mmqs_weight_mode=self.mmqs_weight_mode,
                                      mmqs_tot_api_cfg=tot_api_cfg,
                                      mmqs_W=self.mmqs_W)
        else:
                                                                                        
            self.cs.clients = self.clients
            self.cs.n_clients = len(self.clients)
            self.cs.select_type = self.selection
            self.cs.thpt_ub = self.throughput_ub
            self.cs.rounds = rounds
            self.cs.gamma = self.cs_gamma
            self.cs.delay_alpha = self.cs_alpha
            self.cs.semi_period = semi_period
            self.cs.mmqs_weight_mode = self.cs._mmqs_normalize_weight_mode(self.mmqs_weight_mode)
            self.cs.mmqs_tot_api_cfg = tot_api_cfg or {}
            self.cs.mmqs_W = int(max(0, self._safe_float(self.mmqs_W, 0)))

                                                                                         
        self.mmqs_eps = float(getattr(self.cs, 'mmqs_eps', self.mmqs_eps))

        if self.selection in ['divfl', 'oort'] or 'coreset' in self.selection:
            self.cs.update_grads(self.grads[self.conn_ind],
                                 self.client_samples[self.conn_ind])
        return self.cs

    def sync_gateway_configure(self, config):
        import fl_model                                

                                           
        saved_model_path = config.paths.saved_model
        path = saved_model_path + '/global'
        self.model = fl_model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logging.info('Gateway {} load global model: {}'.format(
            self.gateway_id, path))

        self.sync_save_gateway_model(self.model, saved_model_path)
        self.prefetch_global_version = 'global'
        self.prefetch_round_seq = 0
        self.prefetch_shadow_cache = {}

                                                                            
                                                           
        self.semi_qEvents = PriorityQueue()

    def async_gateway_configure(self, config, download_time, cur_round):
        import fl_model                                

                                           
        saved_model_path = config.paths.saved_model
        path = saved_model_path + '/global_' + '{}'.format(download_time)
        self.model = fl_model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logging.info('Gateway {} load global model: {}'.format(
            self.gateway_id, path))

        self.async_save_gateway_model(self.model, saved_model_path,
                                      download_time)
        self.prefetch_global_version = 'global_{}'.format(download_time)
        self.prefetch_round_seq = 0
        self.prefetch_shadow_cache = {}

        self.global_download_time = download_time                                    

                                                       
        self.adjust_client_selection_gamma(cur_round)

    def adjust_client_selection_gamma(self, cur_round):
                                                                        
                                         
                                       
        self.cs_gamma = self.config.cs_gamma_from -\
                        (self.config.cs_gamma_from - self.config.cs_gamma_to) *\
                        cur_round / self.global_rounds

    def hybrid_gateway_configure(self, config, download_time):
        import fl_model                                

                                           
        saved_model_path = config.paths.saved_model
        path = saved_model_path + '/global_' + '{}'.format(download_time)
        self.model = fl_model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logging.info('Gateway {} load global model: {}'.format(
            self.gateway_id, path))

        self.sync_save_gateway_model(self.model, saved_model_path)
        self.prefetch_global_version = 'global_{}'.format(download_time)
        self.prefetch_round_seq = 0
        self.prefetch_shadow_cache = {}

    def sync_client_configure(self, sample_clients):
        loader_type = self.config.loader
        loading = self.config.data.loading

        if loading == 'dynamic':
                                         
            if loader_type == 'shard':
                self.loader.create_shards()

                                                                
        for client in sample_clients:
            if loading == 'dynamic':
                self.set_client_data(client)                                 

                                       
            config = self.config

                                              
            client.sync_client_configure(config)

    def async_client_configure(self, sample_clients, gateway_download_time):
        loader_type = self.config.loader
        loading = self.config.data.loading

        if loading == 'dynamic':
                                         
            if loader_type == 'shard':
                self.loader.create_shards()

                                                                
        for client in sample_clients:
            if loading == 'dynamic':
                self.set_client_data(client)                                 

                                       
            config = self.config

                                              
            client.async_client_configure(config, gateway_download_time)

                                        
    def sync_run(self, T_old, loader, logger):
        import fl_model                                

        rounds = self.config.gateways.rounds
        self.loader = loader

        self.cs = self._get_or_create_client_selector(
            rounds=rounds,
            semi_period=self.config.semi_period
        )
        self.cs.sel_time = 0.0

                                              
        gateway_comm_size = 0.0
        gateway_comm_client_up = 0.0
        gateway_comm_prefetch = 0.0
        T_start = T_old
        for round in range(1, rounds + 1):
            logging.info('**** Gw {} Round {}/{} ****'.format(self.gateway_id,
                                                              round, rounds))
            self._quality_guard_prepare_round(round)

                                                   
            T_new, comm_size, comm_client_up, comm_prefetch = self.sync_round(T_start)
            gateway_comm_size += comm_size
            gateway_comm_client_up += comm_client_up
            gateway_comm_prefetch += comm_prefetch
            logging.info('Gw {} Round finished at time {} s\n'.format(self.gateway_id,
                                                                      T_new))

                  
            valset = loader.get_valset()
            batch_size = self.config.fl.batch_size
            valloader = fl_model.get_evalloader(valset, batch_size)
            val_loss, val_accuracy = fl_model.evaluate(self.model, valloader)

            self._push_mmqs_regional_feedback(round, val_accuracy)
            logging.info('val loss: {} val acc: {}'.format(val_loss, val_accuracy))
            self._quality_guard_update_after_eval(round, val_loss, val_accuracy)
            logger.log_value('gw{}_val_accuracy'.format(self.gateway_id),
                             val_accuracy, int(T_new * 1000))

                         
            T_start = T_new

                                    
        gateway_weights = fl_model.extract_weights(self.model)
        total_samples = np.sum(self.client_samples[self.conn_ind])

        self.report = Report(self, gateway_weights, self.grads,
                             self.client_samples,
                             total_samples, T_new + self.delay,
                             T_new - T_old + self.delay,
                             self.cs.sel_time, gateway_comm_size, val_loss,
                             val_accuracy,
                             gateway_comm_client_up=gateway_comm_client_up,
                             gateway_comm_prefetch=gateway_comm_prefetch)
        return T_new + self.delay

    def sync_round(self, T_old):
        import fl_model                                

                                                    
        self.throughput = 0
        sel_time_before = float(getattr(self.cs, 'sel_time', 0.0) or 0.0)
        sample_clients = self.cs.select(cur_thpt=self.throughput)
        sel_time_after = float(getattr(self.cs, 'sel_time', 0.0) or 0.0)
        round_sel_time = max(0.0, sel_time_after - sel_time_before)
        branch_plan = None
        branch_scores = None
        branch_winner_clients = None
        branch_winner_name = ''
        if self.selection == 'mmqs':
            getter = getattr(self.cs, 'get_mmqs_tot_branch_plan', None)
            if getter is not None:
                try:
                    branch_plan = getter()
                except Exception as exc:                                
                    logging.warning('Gw %s get ToT branch plan failed: %s', self.gateway_id, str(exc))
            if isinstance(branch_plan, dict):
                try:
                    branch_scores, branch_winner_clients, branch_winner_name = self._evaluate_tot_branch_plan(branch_plan)
                except Exception as exc:                                
                    logging.warning('Gw %s ToT branch strict eval failed, fallback selected branch: %s', self.gateway_id, str(exc))
                if isinstance(branch_scores, dict) and len(branch_scores) > 0:
                    pending = getattr(self.cs, 'mmqs_tot_api_pending_cycle', None)
                    if isinstance(pending, dict) and branch_winner_name:
                        pending['selected_branch'] = str(branch_winner_name)
                        branch_items = branch_plan.get('branches', [])
                        if isinstance(branch_items, list):
                            for b in branch_items:
                                if not isinstance(b, dict):
                                    continue
                                if str(b.get('name', '')) != str(branch_winner_name):
                                    continue
                                weight_vals = b.get('weights', [])
                                pending['selected_weights'] = [float(v) for v in np.array(weight_vals, dtype=float).tolist()]
                                pending['selected_weights_key'] = str(self.cs._mmqs_weights_key(pending['selected_weights']))
                                pending['selected_parse_source'] = 'strict_regional_eval'
                                pending['selected_blend_api'] = 1.0
                                                                                                 
                                self.cs.mmqs_tot_api_last_weights = np.array(
                                    pending['selected_weights'], dtype=float
                                )
                                self.cs.mmqs_tot_api_last_profile = 'tot_api_{}'.format(str(branch_winner_name))
                                self.cs.mmqs_tot_api_last_resolve_round = int(getattr(self.cs, 'current_select_step', 0))
                                break
                    applier = getattr(self.cs, 'apply_mmqs_tot_branch_scores', None)
                    if applier is not None:
                        try:
                            applier(branch_scores, int(getattr(self.cs, 'current_select_step', 0)))
                        except Exception as exc:                                
                            logging.warning('Gw %s apply ToT branch scores failed: %s', self.gateway_id, str(exc))
                    self.cs._mmqs_apply_tot_api_regional_feedback(
                        self.cs._mmqs_collect_state_metrics(
                            candidates=[c for c in self.clients if c.available and c.throughput < (self.throughput_ub - 0)],
                            participations=np.array([
                                float(max(0, getattr(c, 'participation_count', 0)))
                                for c in [c for c in self.clients if c.available and c.throughput < (self.throughput_ub - 0)]
                            ], dtype=float),
                            cur_round=int(getattr(self.cs, 'current_select_step', 0))
                        ),
                        allow_prune_api=(int(getattr(self.cs, 'current_select_step', 0)) % int(self.cs.mmqs_tot_api_prune_interval) == 0)
                    )
                if isinstance(branch_winner_clients, list) and len(branch_winner_clients) > 0:
                    sample_clients = list(branch_winner_clients)
        sample_clients = self._apply_mmqs_loss_aware_topk(sample_clients)
        self._update_mmqs_participation_history(sample_clients)
        current_required_version = self._prefetch_required_version_token()
        prefetch_hit_mask = None
        if self.prefetch_enabled:
            prefetch_hit_mask = self._prefetch_shadow_count_hits(
                sample_clients, current_required_version
            )

        self.sync_client_configure(sample_clients)

        _ = [client.set_unavailable() for client in sample_clients]
        self.throughput = sum([client.throughput for client in sample_clients])

        logging.info('Gw {} throughput {} kB/s'.format(self.gateway_id,
                                                       self.throughput))

                                                                            
        max_delay = 0
        prefetch_ratio_runtime = float(np.clip(
            getattr(self, 'prefetch_delay_reduction_ratio_runtime', self.prefetch_delay_reduction_ratio),
            0.0, 0.95
        ))
        if len(sample_clients) > 0:
            if self.prefetch_enabled and prefetch_ratio_runtime > 0.0:
                max_delay = self._prefetch_effective_max_delay(
                    sample_clients, prefetch_hit_mask, prefetch_ratio=prefetch_ratio_runtime
                )
            else:
                max_delay = max([c.delay for c in sample_clients])

                                                                 
        if self.config.model not in ['CIFAR-10', 'FEMNIST']:
            threads = [Thread(target=client.run) for client in
                       sample_clients]
            [t.start() for t in threads]
            [t.join() for t in threads]
        else:                                                      
            _ = [client.run() for client in sample_clients]

        agg_tx_time = self._estimate_agg_tx_time(sample_clients)
        if self.use_legacy_ave_logic:
            round_duration = max(0.0, float(max_delay))
            logging.info(
                'Gw %s timing breakdown(profile=legacy_ave): max_delay=%.6f total=%.6f',
                self.gateway_id, float(max_delay), float(round_duration)
            )
        else:
            round_duration = max(0.0, float(max_delay)) + float(agg_tx_time) + float(round_sel_time)
            logging.info(
                'Gw %s timing breakdown(profile=modern): max_delay=%.6f agg_tx=%.6f sel=%.6f total=%.6f',
                self.gateway_id, float(max_delay), float(agg_tx_time),
                float(round_sel_time), float(round_duration)
            )
        T_cur = T_old + round_duration                                 

                                
        reports = self.reporting(sample_clients)

                                         
        for report in reports:
            self.grads[report.client_id] = report.grads
            self.client_samples[report.client_id] = report.num_samples

                                    
        logging.debug('Gw {} aggregating updates'.format(self.gateway_id))
        updated_weights = fl_model.extract_weights(self.model)
        if len(sample_clients) > 0:
            updated_weights = self.federated_averaging(reports)

                              
        fl_model.load_weights(self.model, updated_weights)

                                   
        saved_model_path = self.config.paths.saved_model
        self.sync_save_gateway_model(self.model, saved_model_path)
        prefetch_push_count = 0
        if self.prefetch_enabled:
            self.prefetch_round_seq += 1
            next_required_version = self._prefetch_required_version_token()
            prefetch_push_count = int(
                self._prefetch_shadow_publish(sample_clients, next_required_version)
            )

                                       
        _ = [client.set_available() for client in sample_clients]

                                      
        if self.selection in ['divfl', 'oort'] or 'coreset' in self.selection:
            self.cs.update_grads(self.grads[self.conn_ind],
                                 self.client_samples[self.conn_ind])
        elif self.selection == 'tier' and len(sample_clients) > 0:
            self.cs.tier_change_prob()

        model_size = float(self.config.fl.model_size)
        comm_client_up = model_size * len(sample_clients)
        comm_prefetch = model_size * max(0, int(prefetch_push_count))
        comm_total = comm_client_up + comm_prefetch
        return T_cur, comm_total, comm_client_up, comm_prefetch

    def _evaluate_tot_branch_plan(self, branch_plan):
        import fl_model                                
        branches = branch_plan.get('branches', [])
        if not isinstance(branches, list) or len(branches) <= 0:
            return {}, None, ''
        id2client = {}
        for c in self.clients:
            cid = getattr(c, 'client_id', None)
            try:
                id2client[int(cid)] = c
            except (TypeError, ValueError):
                continue
        if len(id2client) <= 0:
            return {}, None, ''

        saved_model_path = self.config.paths.saved_model
        base_path = saved_model_path + '/gateway{}'.format(self.gateway_id)
        if not os.path.isfile(base_path):
            return {}, None, ''
        base_state = torch.load(base_path)
        best_name = ''
        best_acc = -1.0
        best_clients = None
        branch_scores = {}
        eval_jobs = []
        for item in branches:
            if not isinstance(item, dict):
                continue
            bname = str(item.get('name', '')).strip()
            ids = item.get('client_ids', [])
            if (not bname) or (not isinstance(ids, list)):
                continue
            clients_b = []
            for cid in ids:
                try:
                    c = id2client[int(cid)]
                except Exception:
                    continue
                if getattr(c, 'available', False):
                    clients_b.append(c)
            if len(clients_b) <= 0:
                continue
            eval_jobs.append((bname, clients_b))

        if len(eval_jobs) <= 0:
            self.model.load_state_dict(copy.deepcopy(base_state))
            self.sync_save_gateway_model(self.model, saved_model_path)
            return {}, None, ''

        client_owner = {}
        overlap_exists = False
        for bname, clients_b in eval_jobs:
            for c in clients_b:
                cid = getattr(c, 'client_id', None)
                try:
                    cid = int(cid)
                except (TypeError, ValueError):
                    continue
                prev = client_owner.get(cid, None)
                if prev is None:
                    client_owner[cid] = bname
                elif prev != bname:
                    overlap_exists = True

        def _eval_branch_job(branch_name, clients_branch):
            local_model = fl_model.Net()
            local_model.load_state_dict(copy.deepcopy(base_state))
            for client in clients_branch:
                client.sync_client_configure_from_state(
                    self.config,
                    model_state=base_state,
                    gateway_id=self.gateway_id
                )
            _ = [client.set_unavailable() for client in clients_branch]
            try:
                if self.config.model not in ['CIFAR-10', 'FEMNIST']:
                    threads = [Thread(target=client.run) for client in clients_branch]
                    [t.start() for t in threads]
                    [t.join() for t in threads]
                else:
                    _ = [client.run() for client in clients_branch]
                reports_branch = self.reporting(clients_branch)
                updated_weights_branch = fl_model.extract_weights(local_model)
                if len(clients_branch) > 0:
                    updated_weights_branch = self._federated_averaging_for_model(local_model, reports_branch)
                fl_model.load_weights(local_model, updated_weights_branch)
                valset = self.loader.get_valset()
                batch_size = self.config.fl.batch_size
                valloader = fl_model.get_evalloader(valset, batch_size)
                _, acc_branch = fl_model.evaluate(local_model, valloader)
                return branch_name, float(acc_branch), list(clients_branch)
            finally:
                _ = [client.set_available() for client in clients_branch]

        if overlap_exists or len(eval_jobs) <= 1:
            for bname, clients_b in eval_jobs:
                name, acc_b, clients_ret = _eval_branch_job(bname, clients_b)
                branch_scores[name] = float(acc_b)
                if float(acc_b) >= float(best_acc):
                    best_acc = float(acc_b)
                    best_name = name
                    best_clients = list(clients_ret)
        else:
            max_workers = max(1, len(eval_jobs))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(_eval_branch_job, bname, clients_b): bname
                    for bname, clients_b in eval_jobs
                }
                for future in as_completed(future_map):
                    bname = future_map[future]
                    try:
                        name, acc_b, clients_ret = future.result()
                    except Exception as exc:                                
                        logging.warning(
                            'Gw %s ToT branch eval job failed on branch=%s: %s',
                            self.gateway_id, bname, str(exc)
                        )
                        continue
                    branch_scores[name] = float(acc_b)
                    if float(acc_b) >= float(best_acc):
                        best_acc = float(acc_b)
                        best_name = name
                        best_clients = list(clients_ret)

        if len(branch_scores) > 0:
            logging.info(
                'Gw %s ToT strict branch eval: winner=%s acc=%.6f scores=%s',
                self.gateway_id, best_name, float(best_acc), branch_scores
            )

        self.model.load_state_dict(copy.deepcopy(base_state))
        self.sync_save_gateway_model(self.model, saved_model_path)
        return branch_scores, best_clients, best_name

    def async_run(self, T_old, loader, logger):
        """Run one async round until the slowest client finish one round"""
        import fl_model

        rounds = self.config.gateways.rounds

                               
        self.alpha_0 = self.config.async_params.alpha_0
        self.rho = self.config.async_params.rho
        self.staleness_func = self.config.async_params.staleness_func
        self.mu = float(getattr(self.config.async_params, 'mu', self.mu))

        self.cs = self._get_or_create_client_selector(
            rounds=rounds,
            semi_period=self.config.semi_period
        )


                        
                                                    
        self.throughput = 0
        sample_clients = self.cs.select(self.throughput)
        self._update_mmqs_participation_history(sample_clients)
        if len(sample_clients) == 0:
            available_cnt = len([c for c in self.clients if c.available])
            logging.warning(
                'Gw {} async init selected 0 clients (available={}, thpt_ub={}, cur_thpt={}); fallback no-op round.'.format(
                    self.gateway_id, available_cnt, self.throughput_ub, self.throughput
                )
            )
            valset = loader.get_valset()
            batch_size = self.config.fl.batch_size
            valloader = fl_model.get_evalloader(valset, batch_size)
            test_loss, accuracy = fl_model.evaluate(self.model, valloader)
            gateway_weights = fl_model.extract_weights(self.model)
            if self.client_samples is None:
                total_samples = 0
            else:
                total_samples = np.sum(self.client_samples[self.conn_ind])
            self.report = Report(
                self, gateway_weights, self.grads, self.client_samples,
                total_samples, T_old + self.delay, self.delay, 0.0, 0,
                test_loss, accuracy
            )
            _ = [client.set_available() for client in self.clients]
            return T_old + self.delay

                                                                     
        qEvents = PriorityQueue()

                                    
        logging.debug('Gw {} T_old: {}'.format(self.gateway_id, T_old))
        self.async_client_configure(sample_clients, T_old)

        for client in sample_clients:
                                                                   
                                           
            client.set_unavailable()

                                                          
            new_event = asyncEvent(client, 0, T_old, client.delay)
            qEvents.put(new_event)

        self.throughput = sum([client.throughput for client in sample_clients])

                                                    
        gateway_comm_size = 0.0
        gateway_comm_client_up = 0.0
        gateway_comm_prefetch = 0.0
        self.cs.sel_time = 0
        T_cur = T_old
        test_loss = None
        accuracy = None
        for round in range(1, rounds + 1):
            logging.info('**** Gw {} Round {}/{} ****'.format(self.gateway_id,
                                                              round, rounds))

            if qEvents.empty():
                logging.warning(
                    'Gw {} async queue drained before round {} (throughput={}); stop this gateway round gracefully.'.format(
                        self.gateway_id, round, self.throughput
                    )
                )
                break
            event = qEvents.get()
            select_client = event.client

                               
            select_client.run(reg=True, rho=self.rho)
            event_delay = max(0.0, float(getattr(select_client, 'delay', 0.0)))
            agg_tx_time = self._estimate_agg_tx_time([select_client])
            sel_time_before = float(getattr(self.cs, 'sel_time', 0.0) or 0.0)
            model_size = float(self.config.fl.model_size)
            gateway_comm_size += model_size
            gateway_comm_client_up += model_size

                                                                
            report = select_client.get_report()

                          
            self.grads[report.client_id] = report.grads
            self.client_samples[report.client_id] = report.num_samples

                                           
            select_client.set_available()
            self.throughput -= select_client.throughput

                                                          
            if self.selection in ['divfl', 'oort'] or 'coreset' in self.selection:
                self.cs.update_grads(self.grads[self.conn_ind],
                                     self.client_samples[self.conn_ind])

            new_clients = self.cs.select(self.throughput)
            sel_time_after = float(getattr(self.cs, 'sel_time', 0.0) or 0.0)
            round_sel_time = max(0.0, sel_time_after - sel_time_before)
            if self.use_legacy_ave_logic:
                event_duration = float(event_delay)
                logging.info(
                    'Gw %s async timing breakdown(profile=legacy_ave): max_delay=%.6f total=%.6f',
                    self.gateway_id, float(event_delay), float(event_duration)
                )
            else:
                event_duration = float(event_delay) + float(agg_tx_time) + float(round_sel_time)
                logging.info(
                    'Gw %s async timing breakdown(profile=modern): max_delay=%.6f agg_tx=%.6f sel=%.6f total=%.6f',
                    self.gateway_id, float(event_delay), float(agg_tx_time),
                    float(round_sel_time), float(event_duration)
                )
            T_cur = event.download_time + event_duration
            logging.info('Gw {} Round finished at time {} s'.format(self.gateway_id, T_cur))

                                        
            logging.info('Aggregating updates on gateway {} from clients {} at {}'.format(
                self.gateway_id, select_client.client_id, T_cur))
            staleness = round - event.download_round
            updated_weights = self.federated_async(report, staleness)

                                                               
            fl_model.load_weights(self.model, updated_weights)
            saved_model_path = self.config.paths.saved_model
            self.async_save_gateway_model(self.model, saved_model_path, T_cur)

                  
            valset = loader.get_valset()
            batch_size = self.config.fl.batch_size
            valloader = fl_model.get_evalloader(valset, batch_size)
            test_loss, accuracy = fl_model.evaluate(self.model, valloader)

            self._push_mmqs_regional_feedback(round, accuracy)
            logging.info('val loss: {} val acc: {}\n'.format(test_loss, accuracy))
            logger.log_value('gw{}_val_accuracy'.format(self.gateway_id),
                             accuracy, int(T_cur * 1000))

            self._update_mmqs_participation_history(new_clients)
            if len(new_clients) == 0:
                if qEvents.empty():
                    logging.warning(
                        'Gw {} async selected 0 replacement clients and queue is empty at round {} (throughput={}); stop this gateway round gracefully.'.format(
                            self.gateway_id, round, self.throughput
                        )
                    )
                    break
                logging.debug(
                    'Gw {} async selected 0 replacement clients at round {} (pending_events={}).'.format(
                        self.gateway_id, round, qEvents.qsize()
                    )
                )
                continue

                                        
            self.async_client_configure(new_clients, T_cur)

            for client in new_clients:
                                              
                                               
                client.set_unavailable()

                                                              
                new_event = asyncEvent(client, round, T_cur, client.delay)
                qEvents.put(new_event)

                self.throughput += client.throughput

        if test_loss is None or accuracy is None:
            valset = loader.get_valset()
            batch_size = self.config.fl.batch_size
            valloader = fl_model.get_evalloader(valset, batch_size)
            test_loss, accuracy = fl_model.evaluate(self.model, valloader)

        gateway_weights = fl_model.extract_weights(self.model)
        total_samples = np.sum(self.client_samples[self.conn_ind])

        self.report = Report(self, gateway_weights, self.grads,
                             self.client_samples,
                             total_samples, T_cur + self.delay,
                             T_cur - T_old + self.delay,
                             self.cs.sel_time, gateway_comm_size, test_loss,
                             accuracy,
                             gateway_comm_client_up=gateway_comm_client_up,
                             gateway_comm_prefetch=gateway_comm_prefetch)

                               
        _ = [client.set_available() for client in self.clients]

        return T_cur + self.delay

    def semi_async_run(self, T_old, loader, logger):
        """Run one async round until the slowest client finish one round"""
        import fl_model

                                    
        rounds = self.config.gateways.rounds
        semi_period = self.config.semi_period
        self.lambda_ = self.config.async_params.lambda_

        self.cs = self._get_or_create_client_selector(
            rounds=rounds,
            semi_period=semi_period
        )

                        
                                                    
        self.throughput = 0
        sample_clients = self.cs.select(self.throughput)
        self._update_mmqs_participation_history(sample_clients)
        print('Select client delays: {}'.format([client.delay for client in sample_clients]))

        for client in sample_clients:
                                                                   
                                           
            client.set_unavailable()

                        
            self.sync_client_configure([client])
            client.run()

                                                          
            new_event = asyncEvent(client, 0, T_old, client.delay)
            self.semi_qEvents.put(new_event)

        self.throughput = sum([client.throughput for client in sample_clients])

                                                        
        gateway_comm_size = 0.0
        gateway_comm_client_up = 0.0
        gateway_comm_prefetch = 0.0
        self.cs.sel_time = 0
        for round in range(1, rounds + 1):
            logging.info('**** Gw {} Round {}/{} ****'.format(self.gateway_id,
                                                              round, rounds))

            normal_reports, straggler_reports = [], []
            max_staleness = 0

            while not self.semi_qEvents.empty():                                          
                event = self.semi_qEvents.get()
                select_client = event.client
                T_cur = event.download_time + select_client.delay

                                                                                  
                if T_cur > T_old + round * semi_period:
                                                                                        
                                                                     
                    self.semi_qEvents.put(event)
                    break

                                                                                       
                model_size = float(self.config.fl.model_size)
                gateway_comm_size += model_size
                gateway_comm_client_up += model_size

                                                                    
                                                                  
                report = select_client.get_report()
                max_staleness = max(max_staleness, round - event.download_round)
                if event.download_round == round - 1:                  
                    normal_reports.append(report)
                else:
                    straggler_reports.append(report)

                              
                self.grads[report.client_id] = report.grads
                self.client_samples[report.client_id] = report.num_samples

                                               
                select_client.set_available()
                self.throughput -= select_client.throughput

                                        
            logging.info('Aggregating updates on gateway {} from normal clients {} '
                         'and straggler clients {}'.format(self.gateway_id,
                                                           [report.client_id for report in normal_reports],
                                                           [report.client_id for report in straggler_reports]))
            updated_weights = self.federated_semi_async(normal_reports, straggler_reports,
                                                        max_staleness)

                                                               
            fl_model.load_weights(self.model, updated_weights)
            saved_model_path = self.config.paths.saved_model
            self.sync_save_gateway_model(self.model, saved_model_path)

                  
            valset = loader.get_valset()
            batch_size = self.config.fl.batch_size
            valloader = fl_model.get_evalloader(valset, batch_size)
            test_loss, accuracy = fl_model.evaluate(self.model, valloader)

            self._push_mmqs_regional_feedback(round, accuracy)
            logging.info('val loss: {} val acc: {}\n'.format(test_loss, accuracy))
            logger.log_value('gw{}_val_accuracy'.format(self.gateway_id),
                             accuracy, int(T_cur * 1000))

                                                          
            if self.selection in ['divfl', 'oort'] or 'coreset' in self.selection:
                self.cs.update_grads(self.grads[self.conn_ind],
                                     self.client_samples[self.conn_ind])

            new_clients = self.cs.select(self.throughput)
            self._update_mmqs_participation_history(new_clients)

            for client in new_clients:
                                              
                                               
                client.set_unavailable()

                                                              
                new_event = asyncEvent(client, round, T_cur, client.delay)
                self.semi_qEvents.put(new_event)

                self.throughput += client.throughput

        gateway_weights = fl_model.extract_weights(self.model)
        total_samples = np.sum(self.client_samples[self.conn_ind])
        T_cur = T_old + rounds * semi_period

        self.report = Report(self, gateway_weights, self.grads,
                             self.client_samples,
                             total_samples, T_cur + self.delay,
                             T_cur - T_old + self.delay,
                             self.cs.sel_time, gateway_comm_size, test_loss,
                             accuracy,
                             gateway_comm_client_up=gateway_comm_client_up,
                             gateway_comm_prefetch=gateway_comm_prefetch)

                               
        _ = [client.set_available() for client in self.clients]

        return T_cur + self.delay

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
                "Gw %s skip federated averaging: no valid positive-sample reports.",
                self.gateway_id
            )
            return fl_model.extract_weights(self.model)

        safe_reports = [item[0] for item in valid_reports]

                                      
        updates = self.extract_client_updates(safe_reports)

                                         
        total_samples = float(sum([item[1] for item in valid_reports]))
        if total_samples <= 0.0:
            logging.warning(
                "Gw %s skip federated averaging: total_samples=%.6f",
                self.gateway_id, total_samples
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

    @staticmethod
    def _federated_averaging_for_model(model, reports):
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
            return fl_model.extract_weights(model)

        safe_reports = [item[0] for item in valid_reports]
        baseline_weights = fl_model.extract_weights(model)
        weights = [report.weights for report in safe_reports]
        updates = []
        for weight in weights:
            update = []
            for i, (name, w) in enumerate(weight):
                bl_name, baseline = baseline_weights[i]
                assert name == bl_name
                delta = w - baseline
                update.append((name, delta))
            updates.append(update)

        total_samples = float(sum([item[1] for item in valid_reports]))
        if total_samples <= 0.0:
            return fl_model.extract_weights(model)

        avg_update = [torch.zeros(x.size()) for _, x in updates[0]]                             
        for i, update in enumerate(updates):
            num_samples = valid_reports[i][1]
            for j, (_, delta) in enumerate(update):
                avg_update[j] += delta * (num_samples / total_samples)

        updated_weights = []
        for i, (name, weight) in enumerate(baseline_weights):
            updated_weights.append((name, weight + avg_update[i]))
        return updated_weights

    def federated_async(self, report, staleness):
        import fl_model                                

                                      
        weights = self.extract_client_weights([report])[0]

                                                       
        baseline_weights = fl_model.extract_weights(self.model)

                                                                            
        alpha_0 = float(self.alpha_0)
        f_tau = float(self.staleness(staleness))
        alpha_t = alpha_0 * f_tau
        if self.config.server.mode == 'pureasync':
            alpha_t = 1.0                                                
        logging.info(
            '{} alpha_0: {} staleness(tau): {} f(tau): {} alpha_t: {}'.format(
                self.staleness_func, alpha_0, staleness, f_tau, alpha_t
            )
        )

                                         
        updated_weights = []
        for i, (t1, t2) in enumerate(zip(baseline_weights, weights)):
            assert t1[0] == t2[0], "weights names do not match!"
            name, old_weight, new_weight = t1[0], t1[1], t2[1]
            updated_weights.append(
                (name, (1 - alpha_t) * old_weight + alpha_t * new_weight)
            )

        return updated_weights

    def federated_semi_async(self, normal_reports, straggler_reports, max_staleness):
        import fl_model                                

                                                       
        baseline_weights = fl_model.extract_weights(self.model)

        valid_normal_reports = []
        for report in normal_reports:
            try:
                num_samples = float(getattr(report, "num_samples", 0.0))
            except (TypeError, ValueError):
                num_samples = 0.0
            if np.isfinite(num_samples) and num_samples > 0.0:
                valid_normal_reports.append((report, num_samples))

        valid_straggler_reports = []
        for report in straggler_reports:
            try:
                num_samples = float(getattr(report, "num_samples", 0.0))
            except (TypeError, ValueError):
                num_samples = 0.0
            if np.isfinite(num_samples) and num_samples > 0.0:
                valid_straggler_reports.append((report, num_samples))

                                                                            
        normal_avg_weights = None
        if len(valid_normal_reports) > 0:
                                                          
            safe_normal_reports = [item[0] for item in valid_normal_reports]
            weights = self.extract_client_weights(safe_normal_reports)
            total_samples = float(sum([item[1] for item in valid_normal_reports]))
            if total_samples > 0.0:
                                            
                normal_avg_weights = [torch.zeros(x.size())                             
                                      for _, x in weights[0]]
                for i, weight in enumerate(weights):
                    num_samples = valid_normal_reports[i][1]
                    for j, (_, w) in enumerate(weight):
                                                                   
                        normal_avg_weights[j] += w * (num_samples / total_samples)
            else:
                logging.warning(
                    "Gw %s skip normal semi-async averaging: total_samples=%.6f",
                    self.gateway_id, total_samples
                )

                                                             
        straggler_avg_weights = None
        if len(valid_straggler_reports) > 0:
                                                      
            safe_straggler_reports = [item[0] for item in valid_straggler_reports]
            weights = self.extract_client_weights(safe_straggler_reports)
            total_samples = float(sum([item[1] for item in valid_straggler_reports]))
            if total_samples > 0.0:
                                                       
                straggler_avg_weights = [torch.zeros(x.size())                             
                                         for _, x in weights[0]]
                for i, weight in enumerate(weights):
                    num_samples = valid_straggler_reports[i][1]
                    for j, (n, w) in enumerate(weight):
                                                                   
                        straggler_avg_weights[j] += w * (num_samples / total_samples)
            else:
                logging.warning(
                    "Gw %s skip straggler semi-async averaging: total_samples=%.6f",
                    self.gateway_id, total_samples
                )

                                                                            
                                                        
        lambda_t = self.lambda_ * np.exp(- max_staleness)

        updated_weights = []
        if normal_avg_weights is not None and straggler_avg_weights is not None:
            for i, (name, weight) in enumerate(baseline_weights):
                updated_weights.append(
                    (name, (1 - lambda_t) * normal_avg_weights[i] +
                     lambda_t * straggler_avg_weights[i])
                )
        elif normal_avg_weights is not None:
            for i, (name, weight) in enumerate(baseline_weights):
                updated_weights.append((name, normal_avg_weights[i]))
        elif straggler_avg_weights is not None:
            for i, (name, weight) in enumerate(baseline_weights):
                updated_weights.append((name, straggler_avg_weights[i]))
        else:
            logging.warning(
                "Gw %s semi-async fallback to baseline weights: no valid reports.",
                self.gateway_id
            )
            updated_weights = baseline_weights                        

        return updated_weights

    def accuracy_averaging(self, reports):
        if len(reports) == 0:
            logging.warning("Gw %s no reports for accuracy averaging.", self.gateway_id)
            return 0.0

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
            accuracy = float(np.mean([float(getattr(r, "accuracy", 0.0)) for r in reports]))
            logging.warning(
                "Gw %s fallback to unweighted accuracy averaging: total_samples=%.6f",
                self.gateway_id, total_samples
            )
            return accuracy

                                    
        accuracy = 0.0
        for report, num_samples in valid_reports:
            accuracy += report.accuracy * (num_samples / total_samples)

        return accuracy

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

    def extract_client_weights(self, reports):
                                      
        weights = [report.weights for report in reports]

        return weights

    def extract_client_grads(self, reports):
                                      
        grads = [report.grads for report in reports]

        return grads

    def _update_mmqs_participation_history(self, sample_clients):
        """Update MMQS history only on the final selected client list."""
        if self.selection != 'mmqs' or len(sample_clients) == 0:
            return

        cur_step = getattr(self.cs, 'current_select_step', 0)
        try:
            cur_step = max(0, int(cur_step))
        except (TypeError, ValueError):
            cur_step = 0

        for client in sample_clients:
            try:
                prev_count = int(getattr(client, 'participation_count', 0))
            except (TypeError, ValueError):
                prev_count = 0
            client.participation_count = max(0, prev_count) + 1
            client.last_participation_round = cur_step

    def _push_mmqs_regional_feedback(self, round_id, accuracy):
        """Push gateway-side regional validation accuracy into MMQS ToT controller."""
        if self.selection != 'mmqs':
            return
        if not hasattr(self, 'cs') or self.cs is None:
            return
        updater = getattr(self.cs, 'update_mmqs_regional_accuracy', None)
        if updater is None:
            return
        try:
            updater(accuracy, round_id)
        except Exception as exc:                                
            logging.warning(
                'MMQS regional feedback push failed on gw%s round=%s: %s',
                self.gateway_id, round_id, str(exc)
            )

    @staticmethod
    def _selected_client_ids_string(sample_clients):
        ids = []
        for client in sample_clients:
            client_id = getattr(client, 'client_id', None)
            if client_id is None:
                continue
            try:
                ids.append(int(client_id))
            except (TypeError, ValueError):
                continue
        ids.sort()
        return ','.join(str(client_id) for client_id in ids)

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            val = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(val):
            return float(default)
        return float(val)

    def _quality_guard_prepare_round(self, round_id):
        """Prepare runtime knobs before client selection for the current round."""
        self.quality_guard_round = max(0, int(round_id))
        guard_active = False
        if self.quality_guard_enabled:
            warmup = max(0, int(self._safe_float(self.quality_guard_warmup_rounds, 1)))
            hold_until = max(0, int(self._safe_float(self.quality_guard_hold_until_round, 0)))
            guard_active = (self.quality_guard_round > warmup) and (self.quality_guard_round <= hold_until)

        self.loss_aware_topk_enabled_runtime = bool(self.mmqs_loss_aware_topk_enabled)
        base_ratio = float(np.clip(self._safe_float(self.prefetch_delay_reduction_ratio, 0.0), 0.0, 0.95))
        if guard_active:
            ratio_scale = float(np.clip(
                self._safe_float(self.quality_guard_prefetch_ratio_scale, 0.5),
                0.0, 1.0
            ))
            self.prefetch_delay_reduction_ratio_runtime = base_ratio * ratio_scale
        else:
            self.prefetch_delay_reduction_ratio_runtime = base_ratio
        self.quality_guard_active_placeholder = 1 if guard_active else 0

        if self.quality_guard_enabled:
            logging.info(
                'MMQS quality-guard gw%s round=%s active=%s hold_until=%s topk_runtime=%s prefetch_ratio_runtime=%.4f',
                self.gateway_id,
                self.quality_guard_round,
                guard_active,
                self.quality_guard_hold_until_round,
                self.loss_aware_topk_enabled_runtime,
                self.prefetch_delay_reduction_ratio_runtime
            )

    def update_global_loss_feedback(self, val_loss, round_id):
        """Receive global validation loss for one-step-lag Dynamic Top-k control."""
        loss = self._safe_float(val_loss, np.nan)
        if not np.isfinite(loss):
            return
        if np.isfinite(self.loss_aware_topk_global_loss_latest):
            self.loss_aware_topk_global_loss_prev = float(self.loss_aware_topk_global_loss_latest)
        self.loss_aware_topk_global_loss_latest = float(loss)
        try:
            self.loss_aware_topk_global_round = int(round_id)
        except (TypeError, ValueError):
            pass

    def _alpha_tr_from_global_loss(self):
        """Compute alpha(t_r) from global loss relative-change EMA if feedback is available."""
        latest = self._safe_float(self.loss_aware_topk_global_loss_latest, np.nan)
        prev = self._safe_float(self.loss_aware_topk_global_loss_prev, np.nan)
        if (not np.isfinite(latest)) or (not np.isfinite(prev)):
            return None

        denom = max(self.mmqs_eps, abs(prev))
        delta_raw = abs(latest - prev) / denom
        delta_max = max(0.0, self._safe_float(self.loss_aware_topk_delta_max, 2.0))
        delta_clip = float(np.clip(delta_raw, 0.0, delta_max))

        ema_factor = float(np.clip(self.mmqs_ema_factor, 0.0, 1.0))
        if self.loss_aware_topk_delta_ema is None:
            delta_ema = delta_clip
        else:
            delta_ema = (1.0 - ema_factor) * float(self.loss_aware_topk_delta_ema) + ema_factor * delta_clip
        self.loss_aware_topk_delta_ema = float(delta_ema)

        beta = max(0.0, float(self.mmqs_beta))
        rho_min = float(np.clip(self.mmqs_rho_min, 0.1, 1.0))
        rho_max = float(np.clip(self.mmqs_rho_max, rho_min, 1.0))
        delta_ref = max(0.0, self._safe_float(self.mmqs_delta_ref, 0.05))
        sigmoid = 1.0 / (1.0 + np.exp(-beta * (delta_ema - delta_ref)))
        alpha_tr = rho_min + (rho_max - rho_min) * sigmoid

        return {
            'source': 'global_loss',
            'alpha_tr': float(alpha_tr),
            'delta_raw': float(delta_raw),
            'delta_clip': float(delta_clip),
            'delta_ema': float(delta_ema),
            'latest_loss': float(latest),
            'prev_loss': float(prev),
        }

    def _alpha_tr_from_local_proxy(self, sample_clients):
        """Fallback alpha(t_r) from selected-client local loss EMA delta."""
        losses = []
        for client in sample_clients:
            try:
                loss = float(getattr(client, 'loss', np.nan))
            except (TypeError, ValueError):
                loss = np.nan
            if np.isfinite(loss):
                losses.append(loss)
        if len(losses) <= 0:
            return None

        current_loss = float(np.mean(losses))
        if self.loss_aware_topk_loss_ema is None:
            prev_ema = current_loss
        else:
            prev_ema = float(self.loss_aware_topk_loss_ema)
        ema_factor = float(np.clip(self.mmqs_ema_factor, 0.0, 1.0))
        ema_now = (1.0 - ema_factor) * prev_ema + ema_factor * current_loss
        self.loss_aware_topk_loss_ema = float(ema_now)

        delta_l = float(ema_now - prev_ema)
        beta = max(0.0, float(self.mmqs_beta))
        rho_min = float(np.clip(self.mmqs_rho_min, 0.1, 1.0))
        rho_max = float(np.clip(self.mmqs_rho_max, rho_min, 1.0))
        sigmoid = 1.0 / (1.0 + np.exp(-beta * delta_l))
        alpha_tr = rho_min + (rho_max - rho_min) * sigmoid

        return {
            'source': 'local_proxy',
            'alpha_tr': float(alpha_tr),
            'delta_raw': float(delta_l),
            'delta_clip': float(delta_l),
            'delta_ema': float(delta_l),
            'latest_loss': float(current_loss),
            'prev_loss': float(prev_ema),
        }

    def _quality_guard_update_after_eval(self, round_id, val_loss, val_accuracy):
        """Update guard status after current-round validation metrics are available."""
        if not self.quality_guard_enabled:
            return

        loss_now = self._safe_float(val_loss, np.nan)
        acc_now = self._safe_float(val_accuracy, np.nan)
        if (not np.isfinite(loss_now)) or (not np.isfinite(acc_now)):
            return

        ema_factor = float(np.clip(self._safe_float(self.quality_guard_ema_factor, 0.3), 0.0, 1.0))
        if self.quality_guard_acc_ema is None or self.quality_guard_loss_ema is None:
            self.quality_guard_acc_ema = float(acc_now)
            self.quality_guard_loss_ema = float(loss_now)
            return

        prev_acc_ema = float(self.quality_guard_acc_ema)
        prev_loss_ema = float(self.quality_guard_loss_ema)
        acc_ema = (1.0 - ema_factor) * prev_acc_ema + ema_factor * acc_now
        loss_ema = (1.0 - ema_factor) * prev_loss_ema + ema_factor * loss_now
        self.quality_guard_acc_ema = float(acc_ema)
        self.quality_guard_loss_ema = float(loss_ema)

        acc_drop = float(max(0.0, prev_acc_ema - acc_ema))
        loss_rise = float(max(0.0, loss_ema - prev_loss_ema))
        acc_th = max(0.0, self._safe_float(self.quality_guard_acc_drop_threshold, 0.015))
        loss_th = max(0.0, self._safe_float(self.quality_guard_loss_rise_threshold, 0.03))
        warmup = max(0, int(self._safe_float(self.quality_guard_warmup_rounds, 1)))
        hold_rounds = max(1, int(self._safe_float(self.quality_guard_hold_rounds, 2)))
        round_now = max(0, int(round_id))
        min_trigger_round = max(
            int(self._safe_float(self.quality_guard_min_trigger_round, 3)),
            warmup + 1
        )
        in_hold_now = (round_now > warmup) and (
            round_now <= max(0, int(self._safe_float(self.quality_guard_hold_until_round, 0)))
        )
        trigger_hit = (acc_drop >= acc_th) and (loss_rise >= loss_th)

        if (round_now >= min_trigger_round) and (not in_hold_now) and trigger_hit:
            hold_until = round_now + hold_rounds
            self.quality_guard_hold_until_round = max(
                int(self.quality_guard_hold_until_round),
                int(hold_until)
            )
            self.quality_guard_trigger_count += 1
            logging.warning(
                'MMQS quality-guard trigger gw%s round=%s hold_until=%s acc_drop=%.6f(th=%.6f) loss_rise=%.6f(th=%.6f)',
                self.gateway_id, round_now, self.quality_guard_hold_until_round,
                acc_drop, acc_th, loss_rise, loss_th
            )

    def _apply_mmqs_loss_aware_topk(self, sample_clients):
        """Apply loss-aware Top-K budget cap on final MMQS list (optional)."""
        k_base = len(sample_clients)
        if k_base <= 0:
            self.k_cap_placeholder = 0
            self.k_selected_placeholder = 0
            self.selected_client_ids_hash_placeholder = ""
            return sample_clients

                                                                                 
        self.k_cap_placeholder = int(k_base)
        self.k_selected_placeholder = int(k_base)
        self.selected_client_ids_hash_placeholder = self._selected_client_ids_string(sample_clients)

        loss_aware_topk_enabled = bool(
            getattr(self, 'loss_aware_topk_enabled_runtime', self.mmqs_loss_aware_topk_enabled)
        )
        if (not loss_aware_topk_enabled) or (self.selection != 'mmqs') or (k_base <= 2):
            return sample_clients

        dynk = self._alpha_tr_from_global_loss()
        if dynk is None:
            dynk = self._alpha_tr_from_local_proxy(sample_clients)
        if dynk is None:
            return sample_clients

        alpha_tr = float(dynk.get('alpha_tr', 1.0))
        k_target = int(np.floor(float(k_base) * alpha_tr))
        k_target = max(2, min(k_base, k_target))

                                                                     
        if self.loss_aware_topk_last_target is not None:
            prev_k = int(self.loss_aware_topk_last_target)
            lower = max(2, prev_k - 1)
            upper = min(k_base, prev_k + 1)
            if lower > upper:
                lower = upper
            k_target = max(lower, min(upper, k_target))
        k_target = max(2, min(k_base, k_target))
        self.loss_aware_topk_last_target = int(k_target)

        selected = sample_clients[:k_target]
        self.k_cap_placeholder = int(k_target)
        self.k_selected_placeholder = int(len(selected))
        self.selected_client_ids_hash_placeholder = self._selected_client_ids_string(selected)

        logging.info(
            'MMQS loss-aware Top-K gw%s enabled: source=%s |S_r|=%s K_r(t_r)=%s prev_loss=%.6f latest_loss=%.6f delta_raw=%.6f delta_clip=%.6f delta_ema=%.6f alpha(t_r)=%.6f',
            self.gateway_id, str(dynk.get('source', 'unknown')), k_base, k_target,
            float(dynk.get('prev_loss', np.nan)),
            float(dynk.get('latest_loss', np.nan)),
            float(dynk.get('delta_raw', 0.0)),
            float(dynk.get('delta_clip', 0.0)),
            float(dynk.get('delta_ema', 0.0)),
            alpha_tr
        )
        return selected

    def _prefetch_required_version_token(self):
        """Build strict prefetch token for current gateway model version."""
        base_version = getattr(self, 'prefetch_global_version', 'global_0.0')
        seq = getattr(self, 'prefetch_round_seq', 0)
        return '{}|gw{}|seq{}'.format(base_version, self.gateway_id, seq)

    @staticmethod
    def _prefetch_client_key(client):
        """Build stable key for gateway-local prefetch shadow cache."""
        client_id = getattr(client, 'client_id', None)
        if client_id is not None:
            return ('cid', int(client_id))
        return ('obj', id(client))

    def _prefetch_shadow_count_hits(self, sample_clients, required_version):
        """Prefetch v0: strict hit/miss accounting only, no behavior change."""
        total = len(sample_clients)
        if total <= 0:
            return []

        strict = bool(self.prefetch_strict_cache_match)
        hit = 0
        hit_mask = []
        for client in sample_clients:
            cache_entry = self.prefetch_shadow_cache.get(
                self._prefetch_client_key(client), {}
            )
            cache_ready = bool(cache_entry.get('ready', False))
            cache_version = cache_entry.get('version', None)
            cache_hash = cache_entry.get('hash', None)
            cache_source = cache_entry.get('source_gateway_id', -1)
            cache_version_match = (cache_version == required_version)
            cache_hash_match = (cache_hash == str(required_version))
            cache_source_match = (int(cache_source) == int(self.gateway_id))

            if strict:
                is_hit = (
                    cache_ready and cache_version_match and
                    cache_hash_match and cache_source_match
                )
            else:
                is_hit = cache_ready

            if is_hit:
                hit += 1
            hit_mask.append(bool(is_hit))

        miss = total - hit
        self.prefetch_total += total
        self.prefetch_hit += hit
        self.prefetch_miss += miss
        self.prefetch_hit_rate = float(self.prefetch_hit) / float(max(1, self.prefetch_total))

        logging.info(
            'Prefetch v0 gw%s strict=%s required_version=%s total=%s hit=%s miss=%s hit_rate=%.4f',
            self.gateway_id, strict, required_version, total, hit, miss, self.prefetch_hit_rate
        )
        return hit_mask

    def _prefetch_effective_max_delay(self, sample_clients, hit_mask, prefetch_ratio=None):
        """Prefetch v1: apply delay reduction only on selected hit clients."""
        if len(sample_clients) <= 0:
            return 0.0
        if hit_mask is None or len(hit_mask) != len(sample_clients):
            hit_mask = [False for _ in sample_clients]

        if prefetch_ratio is None:
            ratio = float(np.clip(
                getattr(self, 'prefetch_delay_reduction_ratio_runtime', self.prefetch_delay_reduction_ratio),
                0.0, 0.95
            ))
        else:
            ratio = float(np.clip(prefetch_ratio, 0.0, 0.95))
        if ratio <= 0.0:
            return max([c.delay for c in sample_clients])

        effective_delays = []
        hit_count = 0
        for client, is_hit in zip(sample_clients, hit_mask):
            try:
                raw_delay = float(getattr(client, 'delay', 0.0))
            except (TypeError, ValueError):
                raw_delay = 0.0
            raw_delay = max(0.0, raw_delay)
            if is_hit:
                hit_count += 1
                effective_delays.append(raw_delay * (1.0 - ratio))
            else:
                effective_delays.append(raw_delay)

        max_delay = float(max(effective_delays)) if len(effective_delays) > 0 else 0.0
        logging.info(
            'Prefetch v1 gw%s delay ratio=%.4f selected=%s hit=%s max_delay_eff=%.6f',
            self.gateway_id, ratio, len(sample_clients), hit_count, max_delay
        )
        return max_delay

    def _estimate_agg_tx_time(self, sample_clients):
        if len(sample_clients) <= 0:
            return 0.0
        try:
            model_size = float(getattr(self.config.fl, 'model_size', 0.0))
        except (TypeError, ValueError):
            model_size = 0.0
        agg_throughput = float(sum([
            max(0.0, float(getattr(client, 'throughput', 0.0)))
            for client in sample_clients
        ]))
        if model_size > 0.0 and agg_throughput > 0.0:
            return model_size / agg_throughput
        return 0.0

    def _prefetch_shadow_publish(self, sample_clients, next_required_version):
        """Prefetch v0: publish cache metadata only, no delay/comm shortcut."""
        if len(sample_clients) <= 0:
            return 0

        try:
            top_m = int(self.prefetch_top_m)
        except (TypeError, ValueError):
            top_m = 0
        if top_m <= 0:
            top_m = len(sample_clients)
        top_m = min(top_m, len(sample_clients))

        target_clients = sample_clients[:top_m]
        ready_time = time.time()
        req_hash = str(next_required_version)
        req_source = int(self.gateway_id)
        pushed_client_ids = []
        skipped_hit = 0
        skipped_old_or_equal = 0

        for client in target_clients:
            cache_key = self._prefetch_client_key(client)
            cache_entry = self.prefetch_shadow_cache.get(cache_key, {})
            cache_ready = bool(cache_entry.get('ready', False))
            cache_version = cache_entry.get('version', None)
            cache_hash = cache_entry.get('hash', None)
            cache_source = cache_entry.get('source_gateway_id', -1)
            cache_source_match = (int(cache_source) == req_source)

                                                                                   
            is_hit = (
                cache_ready and
                (cache_version == next_required_version) and
                (cache_hash == req_hash) and
                cache_source_match
            )
            if is_hit:
                skipped_hit += 1
                continue

            same_version = (cache_version == next_required_version)
            same_version_mismatch = same_version and (
                (cache_hash != req_hash) or (not cache_source_match)
            )
            req_newer = self._prefetch_version_is_newer(
                next_required_version, cache_version
            )

                                               
                                                                       
                                                                
            should_push = bool(req_newer or same_version_mismatch or (not cache_ready))
            if not should_push:
                skipped_old_or_equal += 1
                continue

            self.prefetch_shadow_cache[cache_key] = {
                'version': next_required_version,
                'hash': req_hash,
                'source_gateway_id': req_source,
                'ready': True,
                'ready_time': ready_time
            }
            pushed_client_ids.append(getattr(client, 'client_id', -1))

        logging.info(
            'Prefetch v0 gw%s publish next_version=%s target_n=%s push_n=%s skip_hit=%s skip_old_or_equal=%s pushed_ids=%s',
            self.gateway_id, next_required_version, len(target_clients),
            len(pushed_client_ids), skipped_hit, skipped_old_or_equal, pushed_client_ids
        )
        return len(pushed_client_ids)

    @staticmethod
    def _prefetch_parse_version_token(version_token):
        """Parse version token like 'global_12.3|gw0|seq5' into comparable fields."""
        raw = '' if version_token is None else str(version_token).strip()
        if raw == '':
            return {'raw': '', 'base': '', 'base_num': np.nan, 'seq': -1}

        parts = raw.split('|')
        base = parts[0].strip() if len(parts) > 0 else raw
        seq = -1
        for part in parts[1:]:
            item = part.strip().lower()
            if item.startswith('seq'):
                try:
                    seq = int(item[3:])
                except (TypeError, ValueError):
                    seq = -1

        base_num = np.nan
        if base.startswith('global_'):
            try:
                base_num = float(base[len('global_'):])
            except (TypeError, ValueError):
                base_num = np.nan

        return {'raw': raw, 'base': base, 'base_num': base_num, 'seq': seq}

    def _prefetch_version_is_newer(self, req_version, cache_version):
        """Return True only when req_version can be proven newer than cache_version."""
        cache_raw = '' if cache_version is None else str(cache_version).strip()
        if cache_raw == '':
            return True

        req_info = self._prefetch_parse_version_token(req_version)
        cache_info = self._prefetch_parse_version_token(cache_version)

        if req_info['raw'] == cache_info['raw']:
            return False

        req_base_num = req_info['base_num']
        cache_base_num = cache_info['base_num']
        if np.isfinite(req_base_num) and np.isfinite(cache_base_num):
            if req_base_num > cache_base_num:
                return True
            if req_base_num < cache_base_num:
                return False
            return int(req_info['seq']) > int(cache_info['seq'])

        if req_info['base'] == cache_info['base']:
            return int(req_info['seq']) > int(cache_info['seq'])

                                                                       
        return False

                       
    @staticmethod
    def flatten_weights(weights):
                                      
        weight_vecs = []
        for _, weight in weights:
            weight_vecs.extend(weight.flatten().tolist())

        return np.array(weight_vecs)

    def reporting(self, sample_clients):
                                             
        reports = [client.get_report() for client in sample_clients]

        logging.info('Reports received: {}'.format(len(reports)))
        assert len(reports) == len(sample_clients)

        return reports

    def sync_save_gateway_model(self, model, path):
        path += '/gateway{}'.format(self.gateway_id)
        torch.save(model.state_dict(), path)
        logging.debug('Saved global model: {}'.format(path))

    def async_save_gateway_model(self, model, path, download_time):
        path += '/gateway{}_{}'.format(self.gateway_id, download_time)
        torch.save(model.state_dict(), path)
        logging.debug('Saved gateway {} model: {}'.format(self.gateway_id, path))


class Report(object):
    """Federated learning client report."""

    def __init__(self, gateway, weights, grads, client_samples,
                 total_samples, finish_time, gateway_round_time,
                 gateway_cs_time, total_comm_size,
                 val_loss, val_accuracy,
                 gateway_comm_client_up=0.0,
                 gateway_comm_prefetch=0.0):
        self.gateway_id = gateway.gateway_id
        self.conn_ind = gateway.conn_ind
        self.weights = weights
        self.grads = grads                
        self.client_samples = client_samples               
        self.num_samples = total_samples
        self.finish_time = finish_time
        self.gateway_round_time = gateway_round_time
        self.gateway_cs_time = gateway_cs_time
        self.gateway_comm_size = total_comm_size
        self.gateway_comm_client_up = gateway_comm_client_up
        self.gateway_comm_prefetch = gateway_comm_prefetch
        self.val_loss = val_loss
        self.val_accuracy = val_accuracy
        # Compatibility aliases for legacy report consumers. New code must use
        # val_loss/val_accuracy so these values are not mistaken for final test metrics.
        self.test_loss = val_loss
        self.accuracy = val_accuracy
