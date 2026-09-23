from collections import namedtuple
import json
import logging

class Config(object):
    """Configuration module."""

    def __init__(self, args):
        self.paths = ""
                          
        with open(args.config, 'r', encoding='utf-8-sig') as config:
            self.config = json.load(config)
        self.selection = args.selection
        self.cs_gamma_from = args.cs_gamma_from
        self.cs_gamma_to = args.cs_gamma_to
        self.cs_alpha = args.cs_alpha
        self.association = args.association
        self.ca_phi = args.ca_phi
        self.delay_mode = args.delay_mode
        self.semi_period = args.semi_period
        self.pca_dim = args.pca_dim
        self.trial = args.trial
        self.mmqs_enabled = args.mmqs_enabled
        self.mmqs_loss_aware_topk_enabled = args.mmqs_loss_aware_topk_enabled
        self.mmqs_beta = args.mmqs_beta
        self.mmqs_rho_min = args.mmqs_rho_min
        self.mmqs_rho_max = args.mmqs_rho_max
        self.mmqs_weight_mode = args.mmqs_weight_mode
        self.mmqs_W = args.mmqs_W
        self.mmqs_tot_api_enabled = args.mmqs_tot_api_enabled
        self.mmqs_tot_api_url = args.mmqs_tot_api_url
        self.mmqs_tot_api_model = args.mmqs_tot_api_model
        self.mmqs_tot_api_timeout = args.mmqs_tot_api_timeout
        self.mmqs_tot_api_max_tokens = args.mmqs_tot_api_max_tokens
        self.mmqs_tot_api_temperature = args.mmqs_tot_api_temperature
        self.mmqs_tot_api_top_p = args.mmqs_tot_api_top_p
        self.mmqs_tot_api_retry = args.mmqs_tot_api_retry
        self.mmqs_tot_api_key_env = args.mmqs_tot_api_key_env
        self.mmqs_tot_q = args.mmqs_tot_q
        self.mmqs_tot_api_reward_ema_factor = args.mmqs_tot_api_reward_ema_factor
        self.mmqs_tot_api_memory_bonus = args.mmqs_tot_api_memory_bonus
        self.mmqs_tot_api_call_interval = args.mmqs_tot_api_call_interval
        self.mmqs_tot_api_prune_interval = args.mmqs_tot_api_prune_interval
        self.mmqs_tot_api_force_non_stream = args.mmqs_tot_api_force_non_stream
        self.mmqs_quality_guard_enabled = args.mmqs_quality_guard_enabled
        self.mmqs_quality_guard_warmup_rounds = args.mmqs_quality_guard_warmup_rounds
        self.mmqs_quality_guard_hold_rounds = args.mmqs_quality_guard_hold_rounds
        self.mmqs_quality_guard_acc_drop_threshold = args.mmqs_quality_guard_acc_drop_threshold
        self.mmqs_quality_guard_loss_rise_threshold = args.mmqs_quality_guard_loss_rise_threshold
        self.mmqs_quality_guard_ema_factor = args.mmqs_quality_guard_ema_factor
        self.prefetch_enabled = args.prefetch_enabled
        self.prefetch_top_m = args.prefetch_top_m
        self.prefetch_strict_cache_match = args.prefetch_strict_cache_match
        self.prefetch_delay_reduction_ratio = args.prefetch_delay_reduction_ratio
        self.hybrid_v1_enabled = args.hybrid_v1_enabled

                               
        self.extract()

    @staticmethod
    def _resolve_enabled_from_cli(module_name, cli_enabled, module_cfg):
        """Step A.1: enabled only comes from CLI; JSON enabled is ignored."""
        effective_enabled = bool(cli_enabled)
        if 'enabled' in module_cfg:
            json_enabled = bool(module_cfg.get('enabled'))
            if json_enabled != effective_enabled:
                logging.warning(
                    '%s.enabled in config is ignored (%s). Effective value comes from CLI: %s',
                    module_name, json_enabled, effective_enabled
                )
        return effective_enabled

    def extract(self):
        config = self.config

                     
        self.model = config['model']

                       
        fields = ['total', 'per_round', 'label_distribution',
                  'do_test', 'test_partition']
        defaults = (0, 0, 'uniform', False, 0.2)
        params = [config['clients'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.clients = namedtuple('clients', fields)(*params)

        assert self.clients.per_round <= self.clients.total

                    
        fields = ['loading', 'partition', 'IID', 'bias', 'shard', 'noniid']
        defaults = ('static', 0, False, None, None, None)
        params = [config['data'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.data = namedtuple('data', fields)(*params)

                                       
        if self.model in ['MNIST', 'FashionMNIST', 'CIFAR-10']:
            assert self.data.IID ^ bool(self.data.bias) ^\
                   bool(self.data.shard) ^ bool(self.data.noniid)
            if self.data.IID:
                self.loader = 'basic'
            elif self.data.bias:
                self.loader = 'bias'
            elif self.data.shard:
                self.loader = 'shard'
            elif self.data.noniid:
                self.loader = 'noniid'
        else:
            self.loader = 'leaf'

                                  
        fields = ['target_accuracy', 'task', 'epochs', 'batch_size',
                  'model_size']
        defaults = (None, 'train', 0, 0, 0)
        params = [config['federated_learning'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.fl = namedtuple('fl', fields)(*params)

                      
        fields = ['mode', 'rounds', 'adjust_round']
        defaults = ('sync', 400, 20)
        params = [config['server'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.server = namedtuple('server', fields)(*params)

                        
        fields = ['mode', 'rounds', 'total', 'throughput_ub']
        defaults = ('sync', 5, 1, 1000)
        params = [config['gateways'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.gateways = namedtuple('gateways', fields)(*params)

                     
        fields = ['data', 'model', 'saved_model', 'reports', 'plot', 'metrics_csv']
        defaults = ('./data', './models', None, None, './plots', None)
        params = [config['paths'].get(field, defaults[i])
                  for i, field in enumerate(fields)]

                                                                              
        mmqs_cfg = config.get('mmqs', {})
        quality_guard_cfg = config.get('quality_guard', {})
        prefetch_cfg = config.get('prefetch', {})
        hybrid_cfg = config.get('hybrid', {})
        self.mmqs_enabled_effective = self._resolve_enabled_from_cli(
            'mmqs', self.mmqs_enabled, mmqs_cfg
        )
        self.quality_guard_enabled_effective = self._resolve_enabled_from_cli(
            'quality_guard', self.mmqs_quality_guard_enabled, quality_guard_cfg
        )
        self.prefetch_enabled_effective = self._resolve_enabled_from_cli(
            'prefetch', self.prefetch_enabled, prefetch_cfg
        )
        self.prefetch_enabled_effective = bool(
            self.prefetch_enabled_effective or
            self.mmqs_enabled_effective or
            str(self.selection).strip().lower() == 'mmqs'
        )
        self.hybrid_enabled_effective = self._resolve_enabled_from_cli(
            'hybrid', self.hybrid_v1_enabled, hybrid_cfg
        )

                                 
        if self.loader == 'leaf' or self.data.IID:       
            distrib = 'na'
        elif self.data.bias:               
            distrib = 'p{}s{}'.format(
                float(self.data.bias['primary']), float(self.data.bias['secondary'])
            )
        elif self.data.noniid:                 
            distrib = 'min{}max{}'.format(
                int(self.data.noniid['min_cls']), int(self.data.noniid['max_cls'])
            )
        else:
            raise ValueError("data distribution type not implemented")

        self.model_name = '{}_{}_{}_iid{}_{}_c{}_th{}_{}_{}_{}_{}_{}_{}_{}'.format(
            self.model, self.server.mode, self.delay_mode, int(self.data.IID),
            distrib, self.clients.total, self.gateways.throughput_ub,
            self.selection, self.cs_alpha,
            self.association, self.ca_phi,
            self.semi_period, self.pca_dim, self.trial
        )

                                                                     
        if self.mmqs_enabled_effective:
            self.model_name += '_mmqs1'
        if self.prefetch_enabled_effective:
            self.model_name += '_prefetch1'
        if self.hybrid_enabled_effective:
            self.model_name += '_hybrid1'

        params[fields.index('model')] += '/' + self.model
        params[fields.index('saved_model')] = params[fields.index('model')] +\
                                              '/' + self.model_name
        if params[fields.index('metrics_csv')] is None:
            params[fields.index('metrics_csv')] = self.model_name + ".csv"

        self.paths = namedtuple('paths', fields)(*params)

                     
        async_cfg = config.get('async', {})
        fields = ['alpha_0', 'rho', 'staleness_func', 'lambda_', 'mu']
        defaults = (0.4, 1.0, 'constant', 0.5, 0.5)
        params = [
            async_cfg.get('alpha_0', defaults[0]),
            async_cfg.get('rho', defaults[1]),
            async_cfg.get('staleness_func', defaults[2]),
            async_cfg.get('lambda_', defaults[3]),
            async_cfg.get('mu', defaults[4]),
        ]
        self.async_params = namedtuple('async_params', fields)(*params)

                          
        fields = ['min', 'max', 'std', 'sparse_ratio']
        defaults = (200, 5000, 100, 0.5)
        params = [config['link_speed'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.link = namedtuple('link_speed', fields)(*params)

                                  
        fields = ['min', 'max', 'std']
        defaults = (15, 100, 10)
        params = [config['comp_time'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.comp_time = namedtuple('comp_time', fields)(*params)

                      
        fields = ['cloud_gateway', 'gateway_client', 'comp_time']
        defaults = (0, 0, 0)
        params = [config['delays'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.delays = namedtuple('delays', fields)(*params)

                                           
        fields = ['enabled', 'loss_aware_topk_enabled', 'weight_mode',
                  'beta', 'rho_min', 'rho_max', 'W']
        defaults = (
            self.mmqs_enabled_effective,
            self.mmqs_loss_aware_topk_enabled,
            self.mmqs_weight_mode,
            self.mmqs_beta,
            self.mmqs_rho_min,
            self.mmqs_rho_max,
            self.mmqs_W
        )
        params = [
            defaults[0],                     
            mmqs_cfg.get('loss_aware_topk_enabled', defaults[1]),
            mmqs_cfg.get('weight_mode', defaults[2]),
            mmqs_cfg.get('beta', defaults[3]),
            mmqs_cfg.get('rho_min', defaults[4]),
            mmqs_cfg.get('rho_max', defaults[5]),
            mmqs_cfg.get('W', defaults[6])
        ]
        _mmqs_tuple = namedtuple('mmqs', fields)
        self.mmqs = _mmqs_tuple(*params)
                                                        
        self.mmqs = self.mmqs._replace(
            **{
                'loss_aware_topk_enabled': bool(self.mmqs.loss_aware_topk_enabled),
                'beta': float(self.mmqs.beta),
                'rho_min': float(self.mmqs.rho_min),
                'rho_max': float(self.mmqs.rho_max),
                'W': int(self.mmqs.W),
            }
        )

                                                                       
        fields = ['enabled', 'api_url', 'model', 'timeout', 'max_tokens',
                  'temperature', 'top_p', 'retry', 'api_key_env',
                  'Q', 'reward_ema_factor', 'memory_bonus',
                  'call_interval', 'prune_interval', 'force_non_stream']
        mode_is_tot_api = str(self.mmqs.weight_mode).strip().lower() == 'tot_api'
        defaults = (
            bool(self.mmqs_tot_api_enabled or mode_is_tot_api),
            self.mmqs_tot_api_url,
            self.mmqs_tot_api_model,
            self.mmqs_tot_api_timeout,
            self.mmqs_tot_api_max_tokens,
            self.mmqs_tot_api_temperature,
            self.mmqs_tot_api_top_p,
            self.mmqs_tot_api_retry,
            self.mmqs_tot_api_key_env,
            self.mmqs_tot_q,
            self.mmqs_tot_api_reward_ema_factor,
            self.mmqs_tot_api_memory_bonus,
            self.mmqs_tot_api_call_interval,
            self.mmqs_tot_api_prune_interval,
            self.mmqs_tot_api_force_non_stream
        )
        tot_api_cfg = mmqs_cfg.get('tot_api', {})
        params = [
            defaults[0],                                       
            tot_api_cfg.get('api_url', defaults[1]),
            tot_api_cfg.get('model', defaults[2]),
            tot_api_cfg.get('timeout', defaults[3]),
            tot_api_cfg.get('max_tokens', defaults[4]),
            tot_api_cfg.get('temperature', defaults[5]),
            tot_api_cfg.get('top_p', defaults[6]),
            tot_api_cfg.get('retry', defaults[7]),
            tot_api_cfg.get('api_key_env', defaults[8]),
            tot_api_cfg.get('Q', defaults[9]),
            tot_api_cfg.get('reward_ema_factor', defaults[10]),
            tot_api_cfg.get('memory_bonus', defaults[11]),
            tot_api_cfg.get('call_interval', defaults[12]),
            tot_api_cfg.get('prune_interval', defaults[13]),
            tot_api_cfg.get('force_non_stream', defaults[14])
        ]
        self.mmqs_tot_api = namedtuple('mmqs_tot_api', fields)(*params)

                                                         
        fields = ['enabled', 'warmup_rounds', 'hold_rounds',
                  'acc_drop_threshold', 'loss_rise_threshold', 'ema_factor']
        defaults = (
            self.quality_guard_enabled_effective,
            self.mmqs_quality_guard_warmup_rounds,
            self.mmqs_quality_guard_hold_rounds,
            self.mmqs_quality_guard_acc_drop_threshold,
            self.mmqs_quality_guard_loss_rise_threshold,
            self.mmqs_quality_guard_ema_factor
        )
        params = [
            defaults[0],                     
            quality_guard_cfg.get('warmup_rounds', defaults[1]),
            quality_guard_cfg.get('hold_rounds', defaults[2]),
            quality_guard_cfg.get('acc_drop_threshold', defaults[3]),
            quality_guard_cfg.get('loss_rise_threshold', defaults[4]),
            quality_guard_cfg.get('ema_factor', defaults[5])
        ]
        self.quality_guard = namedtuple('quality_guard', fields)(*params)

                                               
        fields = ['enabled', 'top_m', 'strict_cache_match', 'delay_reduction_ratio']
        defaults = (
            self.prefetch_enabled_effective, self.prefetch_top_m,
            self.prefetch_strict_cache_match, self.prefetch_delay_reduction_ratio
        )
        params = [
            defaults[0],                     
            prefetch_cfg.get('top_m', defaults[1]),
            prefetch_cfg.get('strict_cache_match', defaults[2]),
            prefetch_cfg.get('delay_reduction_ratio', defaults[3])
        ]
        self.prefetch = namedtuple('prefetch', fields)(*params)

                                             
        fields = ['enabled']
        defaults = (self.hybrid_enabled_effective,)
        params = [defaults[0]]                     
        self.hybrid = namedtuple('hybrid', fields)(*params)

        logging.info(
            'effective switches: mmqs=%s quality_guard=%s prefetch=%s hybrid=%s',
            self.mmqs.enabled, self.quality_guard.enabled, self.prefetch.enabled, self.hybrid.enabled
        )
