import random
import math
import numpy as np
import logging
import time
import os
from .tot_agent import TOTAPIAgent, TOTAPIError

def plot(**kwargs):
    """Plot the histograms of given distributions"""
    import matplotlib.pyplot as plt
    num_plots = len(kwargs)
    for i, (key, value) in enumerate(kwargs.items()):
        plt.subplot(1, num_plots, i+1)
        plt.hist(value)
        plt.title(key)
    plt.show()

class Tier(object):
    """Tier objects for client selection"""
    def __init__(self, client_list, probability, credits):
        self.client_list = client_list
        self.p = probability
        self.credits = credits
        self.mean_loss = 10

class ClientSelection(object):
    """Client selection decision making."""
    def __init__(self, clients, select_type, model_name,
                 thpt_ub, rounds, gamma, delay_alpha, semi_period,
                 mmqs_weight_mode='static', mmqs_tot_api_cfg=None,
                 mmqs_W=0):
        self.clients = clients
        self.n_clients = len(self.clients)
        self.select_type = select_type
        self.thpt_ub = thpt_ub
        self.rounds = rounds
        self.gamma = gamma
        self.delay_alpha = delay_alpha
        self.semi_period = semi_period
        self.sel_time = 0.0
        self.force_nonempty_selection = bool(
            int(os.environ.get('MMQS_FORCE_NONEMPTY_SELECTION', '0')) != 0
        )
        self.force_nonempty_allow_thpt_override = bool(
            int(os.environ.get('MMQS_FORCE_NONEMPTY_ALLOW_OVER_THPT', '0')) != 0
        )
        self.mmqs_coverage_floor_enabled = bool(
            int(os.environ.get('MMQS_COVERAGE_FLOOR', '0')) != 0
        )
        self.mmqs_coverage_window = max(
            1, int(self._mmqs_safe_float(os.environ.get('MMQS_COVERAGE_WINDOW', '60'), 60))
        )
        self.mmqs_coverage_min_insert = max(
            1, int(self._mmqs_safe_float(os.environ.get('MMQS_COVERAGE_MIN_INSERT', '1'), 1))
        )
        self.mmqs_coverage_max_share = float(np.clip(
            self._mmqs_safe_float(os.environ.get('MMQS_COVERAGE_MAX_SHARE', '0.35'), 0.35),
            0.0, 1.0
        ))
                                                                             
        self.mmqs_w_data = 0.2975
        self.mmqs_w_modal = 0.15
        self.mmqs_w_perf = 0.2975
        self.mmqs_w_res = 0.085
        self.mmqs_w_cool = 0.085
        self.mmqs_w_fair = 0.085
        self.mmqs_res_lambda = 0.10
        self.mmqs_t_cool = 1.0
        self.mmqs_eps = 1e-12
        self.mmqs_participation_cap = 1000
        self.mmqs_last_valid_weights = None
        self.mmqs_perf_alpha = 0.30
        self.mmqs_perf_default_ema = 1.0
        self.mmqs_perf_last_neutral_ema = self.mmqs_perf_default_ema
        self.mmqs_perf_ema = {}
        self.mmqs_W = int(max(0, self._mmqs_safe_float(mmqs_W, 0)))
        self.mmqs_perf_window = {}
        self.mmqs_last_quality_scores = {}
        self.mmqs_weight_mode = self._mmqs_normalize_weight_mode(mmqs_weight_mode)
        self.mmqs_last_weight_profile = 'static_default'
        self.mmqs_tot_v1_profiles = {
            'tot_warmup': np.array([0.36, 0.02, 0.36, 0.09, 0.08, 0.09], dtype=float),
            'tot_balance': np.array([0.35, 0.02, 0.34, 0.10, 0.10, 0.09], dtype=float),
            'tot_stability': np.array([0.33, 0.02, 0.33, 0.12, 0.11, 0.09], dtype=float),
            'tot_fairness': np.array([0.32, 0.02, 0.33, 0.11, 0.10, 0.12], dtype=float)
        }
                                                                  
                                                                    
        self._mmqs_apply_weight_overrides_from_env()
                                                                            
        self.mmqs_tot_fair_mid_start = 0.35
        self.mmqs_tot_fair_mid_end = 0.65
        self.mmqs_tot_late_stability_start = 0.75
        self.mmqs_tot_tail_static_start = 0.90
        self.mmqs_tot_rho_early = 0.30
        self.mmqs_tot_rho_mid = 0.24
        self.mmqs_tot_rho_late = 0.14
        self.mmqs_tot_rho_tail = 0.00
        self.mmqs_tot_fairness_gap_threshold = 0.55
        self.mmqs_tot_delay_cv_stability_threshold = 0.60
        self.mmqs_tot_fairness_rho_cap = 0.20
                                                                                          
        self.mmqs_tot_late_quality_anchor_enabled = False
        self.mmqs_tot_late_quality_anchor_start_ratio = 0.60
        self.mmqs_tot_late_quality_anchor_max = 1
        self.mmqs_tot_late_quality_anchor_delay_mul = 1.05
        self.mmqs_tot_late_quality_anchor_min_quality = 0.65
                                                                                
        self.mmqs_c23_pool_factor = 2.0
        self.mmqs_c23_min_pool = 8
        self.mmqs_c23_delay_quantile = 0.80
        self.mmqs_c23q_stage_start_ratio = 0.45
        self.mmqs_c23q_stage_end_ratio = 0.70
        self.mmqs_c23q_delay_tol = 0.02
        self.mmqs_c23q_score_tol = 0.01
                                                                   
        self.mmqs_c23q_anchor_ratio = 0.00
        self.mmqs_c23q_anchor_max = 1
        self.mmqs_c23q_anchor_delay_mul = 1.05
        self.mmqs_c23q_anchor_stage_end_ratio = 0.55
                                                                          
        self.mmqs_c23q_fallback_start_ratio = 0.55
        self.mmqs_tot_api_cfg = mmqs_tot_api_cfg or {}
        self.mmqs_tot_api_agent = None
        self.mmqs_tot_api_branch_memory = {}
        self.mmqs_tot_api_last_branch = 'none'
        self.mmqs_tot_api_last_reward = 0.0
        self.mmqs_tot_api_reward_ema_factor = float(np.clip(
            self._mmqs_safe_float(self.mmqs_tot_api_cfg.get('reward_ema_factor', 0.3), 0.3), 0.0, 1.0
        ))
        self.mmqs_tot_api_memory_bonus = float(np.clip(
            self._mmqs_safe_float(self.mmqs_tot_api_cfg.get('memory_bonus', 0.1), 0.1), 0.0, 1.0
        ))
        self.mmqs_tot_q = max(1, int(self._mmqs_safe_float(
            self.mmqs_tot_api_cfg.get('Q', 5), 5
        )))
        self.mmqs_tot_api_weights_memory = {}
        self.mmqs_tot_api_weights_visits = {}
        self.mmqs_tot_api_proxy_bias_ema = 0.0
        self.mmqs_tot_api_proxy_bias_alpha = float(np.clip(
            self._mmqs_safe_float(self.mmqs_tot_api_cfg.get('proxy_bias_alpha', 0.25), 0.25), 0.0, 1.0
        ))
        self.mmqs_tot_api_explore_coef = max(0.0, self._mmqs_safe_float(
            self.mmqs_tot_api_cfg.get('explore_coef', 0.03), 0.03
        ))
        self.mmqs_tot_api_weights_memory_bonus = max(0.0, self._mmqs_safe_float(
            self.mmqs_tot_api_cfg.get('weights_memory_bonus', 0.08), 0.08
        ))
        self.mmqs_last_regional_accuracy = np.nan
        self.mmqs_last_regional_round = -1
        self.mmqs_tot_api_pending_cycle = None
        self.mmqs_tot_api_last_prune = {}
        self.mmqs_tot_api_last_weights = None
        self.mmqs_tot_api_last_profile = 'tot_api_cache_empty'
        self.mmqs_tot_api_last_resolve_round = -1
        self.mmqs_tot_api_branch_plan = {
            'select_round': -1,
            'branches': [],
            'union_client_ids': []
        }
        self.mmqs_tot_api_call_interval = max(1, int(self._mmqs_safe_float(
            self.mmqs_tot_api_cfg.get('call_interval', 3), 3
        )))
        self.mmqs_tot_api_prune_interval = max(1, int(self._mmqs_safe_float(
            self.mmqs_tot_api_cfg.get('prune_interval', 2), 2
        )))
        self.mmqs_tot_api_guard_enabled = bool(self.mmqs_tot_api_cfg.get('guard_enabled', True))
        self.mmqs_tot_api_regional_ema = np.nan
        self.mmqs_tot_api_regional_ema_factor = float(np.clip(
            self._mmqs_safe_float(self.mmqs_tot_api_cfg.get('regional_ema_factor', 0.3), 0.3), 0.0, 1.0
        ))
        self.mmqs_tot_api_drop_threshold = float(np.clip(
            self._mmqs_safe_float(self.mmqs_tot_api_cfg.get('drop_threshold', 0.08), 0.08), 0.0, 1.0
        ))
        self.mmqs_tot_api_fallback_hold_rounds = max(1, int(self._mmqs_safe_float(
            self.mmqs_tot_api_cfg.get('fallback_hold_rounds', 2), 2
        )))
        self.mmqs_tot_api_fallback_until_round = -1
        self._mmqs_init_tot_api_agent()
        self.current_select_step = 0

                                                   
        if self.select_type == 'tier':
            self.tiers = self.tier_profiling()

    def _pick_one_available_fallback(self, cur_thpt):
        available = [c for c in self.clients if getattr(c, 'available', False)]
        if len(available) <= 0:
            return []

        thpt_budget = float(self.thpt_ub - cur_thpt)
        fit = [c for c in available if float(getattr(c, 'throughput', 0.0)) < thpt_budget]
        if len(fit) > 0:
            chosen = min(
                fit,
                key=lambda c: (
                    float(getattr(c, 'delay', 0.0)),
                    float(getattr(c, 'throughput', 0.0)),
                    int(getattr(c, 'client_id', -1)),
                ),
            )
            logging.warning(
                'Force non-empty selection: pick one in-budget client=%s delay=%.6f thpt=%.6f budget=%.6f',
                int(getattr(chosen, 'client_id', -1)),
                float(getattr(chosen, 'delay', 0.0)),
                float(getattr(chosen, 'throughput', 0.0)),
                thpt_budget,
            )
            return [chosen]

        if not self.force_nonempty_allow_thpt_override:
            return []

        chosen = min(
            available,
            key=lambda c: (
                float(getattr(c, 'throughput', 0.0)),
                float(getattr(c, 'delay', 0.0)),
                int(getattr(c, 'client_id', -1)),
            ),
        )
        logging.warning(
            'Force non-empty selection: throughput override client=%s delay=%.6f thpt=%.6f budget=%.6f',
            int(getattr(chosen, 'client_id', -1)),
            float(getattr(chosen, 'delay', 0.0)),
            float(getattr(chosen, 'throughput', 0.0)),
            thpt_budget,
        )
        return [chosen]

    def _ensure_nonempty_selection(self, sample_clients, cur_thpt):
        if len(sample_clients) > 0 or (not self.force_nonempty_selection):
            return sample_clients
        fallback = self._pick_one_available_fallback(cur_thpt)
        if len(fallback) > 0:
            return fallback
        return sample_clients

    @staticmethod
    def _client_id_safe(client):
        try:
            return int(getattr(client, 'client_id', -1))
        except (TypeError, ValueError):
            return -1

    @staticmethod
    def _client_thpt_safe(client):
        try:
            return float(max(0.0, getattr(client, 'throughput', 0.0)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _client_delay_safe(client):
        try:
            return float(max(0.0, getattr(client, 'delay', 0.0)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _client_participation_safe(client):
        try:
            return int(max(0, getattr(client, 'participation_count', 0)))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _client_last_round_safe(client):
        try:
            return int(getattr(client, 'last_participation_round', -1))
        except (TypeError, ValueError):
            return -1

    def _mmqs_apply_coverage_floor(self, sample_clients, ranked_candidates, cur_thpt):
        """Inject under-participated clients with throughput-safe replacement."""
        if (not self.mmqs_coverage_floor_enabled) or self.select_type != 'mmqs':
            return sample_clients
        if len(ranked_candidates) <= 0 or len(sample_clients) <= 0:
            return sample_clients

        cur_step = int(max(0, getattr(self, 'current_select_step', 0)))
        window = int(max(1, self.mmqs_coverage_window))
        min_insert = int(max(1, self.mmqs_coverage_min_insert))
        max_share = float(np.clip(self.mmqs_coverage_max_share, 0.0, 1.0))
        max_replace = int(max(1, np.ceil(len(sample_clients) * max_share)))

        budget = float(max(0.0, float(self.thpt_ub) - float(cur_thpt)))
        selected = list(sample_clients)
        selected_ids = set(self._client_id_safe(c) for c in selected)
        used_thpt = float(sum(self._client_thpt_safe(c) for c in selected))

        overdue = []
        for client in ranked_candidates:
            cid = self._client_id_safe(client)
            if cid in selected_ids:
                continue
            last_r = self._client_last_round_safe(client)
            stale = cur_step - last_r
            if last_r < 0:
                stale = window + 1
            if stale >= window:
                overdue.append(client)

        if len(overdue) <= 0:
            return selected

        overdue = sorted(
            overdue,
            key=lambda c: (
                -int(max(0, cur_step - self._client_last_round_safe(c))),
                self._client_participation_safe(c),
                self._client_delay_safe(c),
                self._client_id_safe(c),
            )
        )

        injected = 0
        replaced = 0
        injected_ids = []

        for cand in overdue:
            if injected >= min_insert:
                break
            cand_id = self._client_id_safe(cand)
            if cand_id in selected_ids:
                continue
            cand_thpt = self._client_thpt_safe(cand)

            if used_thpt + cand_thpt <= budget + self.mmqs_eps:
                selected.append(cand)
                selected_ids.add(cand_id)
                used_thpt += cand_thpt
                injected += 1
                injected_ids.append(cand_id)
                continue

            if replaced >= max_replace:
                continue

            victim_candidates = sorted(
                selected,
                key=lambda c: (
                    self._client_participation_safe(c),
                    -self._client_delay_safe(c),
                    self._client_id_safe(c),
                ),
                reverse=True
            )
            swapped = False
            for victim in victim_candidates:
                victim_id = self._client_id_safe(victim)
                if victim_id < 0:
                    continue
                victim_thpt = self._client_thpt_safe(victim)
                new_used = used_thpt - victim_thpt + cand_thpt
                if new_used <= budget + self.mmqs_eps:
                    idx = selected.index(victim)
                    selected[idx] = cand
                    selected_ids.discard(victim_id)
                    selected_ids.add(cand_id)
                    used_thpt = new_used
                    replaced += 1
                    injected += 1
                    injected_ids.append(cand_id)
                    swapped = True
                    break
            if swapped:
                continue

        if injected > 0:
            logging.info(
                'MMQS coverage floor: step=%s window=%s min_insert=%s injected=%s replaced=%s selected=%s used_thpt=%.6f budget=%.6f ids=%s',
                cur_step, window, min_insert, injected, replaced, len(selected), used_thpt, budget, injected_ids[:8]
            )
        return selected

    def _mmqs_v1_rank(self, candidates, cur_round, cur_thpt=0.0):
        """MMQS v1 ranking with s_data + s_modal + s_perf + s_res + s_cool + s_fair."""
        if len(candidates) == 0:
            return candidates, None

                                                                        
        data_sizes = []
        for c in candidates:
            try:
                data_sizes.append(float(getattr(c, 'num_samples', 0)))
            except (TypeError, ValueError):
                data_sizes.append(0.0)
        data_sizes = np.nan_to_num(np.array(data_sizes, dtype=float),
                                   nan=0.0, posinf=0.0, neginf=0.0)
        max_data = np.max(data_sizes) if len(data_sizes) > 0 else 0.0
        if max_data > 0:
            s_data = data_sizes / max_data
        else:
            s_data = np.zeros((len(candidates),), dtype=float)

                                                                     
        ema_losses = np.full((len(candidates),), np.nan, dtype=float)
        for i, c in enumerate(candidates):
            client_key = self._mmqs_client_key(c)
            raw_loss = getattr(c, 'loss', None)
            try:
                cur_loss = float(raw_loss) if raw_loss is not None else np.nan
            except (TypeError, ValueError):
                cur_loss = np.nan

            proxy_loss = self._mmqs_update_perf_proxy(client_key, cur_loss)
            if np.isfinite(proxy_loss):
                ema_losses[i] = float(proxy_loss)

        valid_ema = ema_losses[np.isfinite(ema_losses)]
        if valid_ema.size > 0:
            neutral_ema = float(np.median(valid_ema))
            self.mmqs_perf_last_neutral_ema = neutral_ema
        elif np.isfinite(self.mmqs_perf_last_neutral_ema):
            neutral_ema = float(self.mmqs_perf_last_neutral_ema)
        else:
            neutral_ema = float(self.mmqs_perf_default_ema)

        missing_mask = ~np.isfinite(ema_losses)
        if np.any(missing_mask):
            ema_losses[missing_mask] = neutral_ema
        ema_losses = np.nan_to_num(ema_losses, nan=neutral_ema, posinf=neutral_ema, neginf=neutral_ema)

        s_perf = np.full((len(candidates),), 0.5, dtype=float)
        perf_min = np.min(ema_losses) if len(ema_losses) > 0 else np.nan
        perf_max = np.max(ema_losses) if len(ema_losses) > 0 else np.nan
        if np.isfinite(perf_min) and np.isfinite(perf_max):
            perf_denom = max(self.mmqs_eps, perf_max - perf_min)
            if perf_denom > self.mmqs_eps:
                s_perf = 1.0 - (ema_losses - perf_min) / perf_denom
                s_perf = np.clip(s_perf, 0.0, 1.0)
                s_perf = np.nan_to_num(s_perf, nan=0.5, posinf=0.5, neginf=0.5)

                                                           
        participations = []
        last_rounds = []
        for c in candidates:
            try:
                p = int(getattr(c, 'participation_count', 0))
            except (TypeError, ValueError):
                p = 0
            p = max(0, min(p, self.mmqs_participation_cap))
            participations.append(float(p))

            try:
                t_last = int(getattr(c, 'last_participation_round', -1))
            except (TypeError, ValueError):
                t_last = -1
            last_rounds.append(t_last)
        participations = np.array(participations, dtype=float)
        last_rounds = np.array(last_rounds, dtype=float)

                                                    
        s_res = np.exp(-self.mmqs_res_lambda * participations)
        s_res = np.maximum(self.mmqs_eps, s_res)
        s_res = np.nan_to_num(s_res, nan=self.mmqs_eps, posinf=1.0, neginf=self.mmqs_eps)
        s_res = np.clip(s_res, self.mmqs_eps, 1.0)

                                                                             
        s_cool = np.ones((len(candidates),), dtype=float)
        valid_last = last_rounds >= 0
        if np.any(valid_last):
            cool_window = max(1.0, float(self.mmqs_t_cool))
            cool_delta = (float(cur_round) - last_rounds[valid_last]) / cool_window
            s_cool[valid_last] = np.clip(cool_delta, 0.0, 1.0)
        s_cool = np.nan_to_num(s_cool, nan=1.0, posinf=1.0, neginf=0.0)

                                                                  
        all_participations = []
        for c in self.clients:
            try:
                cp = int(getattr(c, 'participation_count', 0))
            except (TypeError, ValueError):
                cp = 0
            all_participations.append(max(0, cp))
        max_p = max(1.0, float(np.max(all_participations)) if len(all_participations) > 0 else 1.0)
        s_fair = 1.0 - np.divide(participations, max_p)
        s_fair = np.nan_to_num(s_fair, nan=0.0, posinf=0.0, neginf=0.0)
        s_fair = np.clip(s_fair, 0.0, 1.0)

                                                                                         
        s_modal = np.array([
            self._mmqs_extract_modal_score(c) for c in candidates
        ], dtype=float)
        s_modal = np.nan_to_num(s_modal, nan=1.0, posinf=1.0, neginf=0.0)
        s_modal = np.clip(s_modal, 0.0, 1.0)

                                                                       
        raw_weights, profile_name = self._mmqs_resolve_weight_vector(
            candidates=candidates,
            participations=participations,
            cur_round=cur_round,
            cur_thpt=cur_thpt,
            component_scores={
                's_data': s_data,
                's_modal': s_modal,
                's_perf': s_perf,
                's_res': s_res,
                's_cool': s_cool,
                's_fair': s_fair
            }
        )
        weights = np.maximum(raw_weights, 0.0)
        if np.sum(weights) <= self.mmqs_eps:
            if self.mmqs_last_valid_weights is not None and\
                    self.mmqs_last_valid_weights.shape == weights.shape:
                weights = self.mmqs_last_valid_weights.copy()
            else:
                weights = np.ones_like(weights, dtype=float) / len(weights)
        else:
            weights = weights / np.sum(weights)
        self.mmqs_last_valid_weights = weights.copy()
        self.mmqs_last_weight_profile = profile_name

        scores = (
            weights[0] * s_data +
            weights[1] * s_modal +
            weights[2] * s_perf +
            weights[3] * s_res +
            weights[4] * s_cool +
            weights[5] * s_fair
        )
        scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
        quality_core = np.divide(
            weights[0] * s_data + weights[2] * s_perf,
            max(self.mmqs_eps, weights[0] + weights[2])
        )
        quality_core = np.nan_to_num(quality_core, nan=0.0, posinf=0.0, neginf=0.0)
        quality_core = np.clip(quality_core, 0.0, 1.0)
        self.mmqs_last_quality_scores = {
            self._mmqs_client_key(candidates[i]): float(quality_core[i])
            for i in range(len(candidates))
        }

        if not np.any(np.isfinite(scores)):
            self.mmqs_last_quality_scores = {}
            return None, None

        sorted_idx = sorted(range(len(candidates)),
                            key=lambda i: scores[i],
                            reverse=True)

        ranked_candidates = [candidates[i] for i in sorted_idx]
        ranked_scores = [float(scores[i]) for i in sorted_idx]

        top_n = min(5, len(ranked_candidates))
        top_debug = [
            (ranked_candidates[i].client_id, ranked_scores[i])
            for i in range(top_n)
        ]
        logging.info(
            'MMQS v1 components mean: data=%.4f modal=%.4f perf=%.4f res=%.4f cool=%.4f fair=%.4f mode=%s profile=%s',
            float(np.mean(s_data)),
            float(np.mean(s_modal)),
            float(np.mean(s_perf)),
            float(np.mean(s_res)),
            float(np.mean(s_cool)),
            float(np.mean(s_fair)),
            self.mmqs_weight_mode,
            self.mmqs_last_weight_profile
        )
        logging.info('MMQS v1 top scores: {}'.format(top_debug))
        return ranked_candidates, ranked_scores

    def _mmqs_normalize_weight_mode(self, mode):
        mode_str = str(mode).strip().lower()
        if mode_str not in ('static', 'tot_api'):
            logging.warning('Unsupported mmqs_weight_mode=%s, fallback to static', mode)
            return 'static'
        return mode_str

    def _mmqs_init_tot_api_agent(self):
        enabled = bool(self.mmqs_tot_api_cfg.get('enabled', False)) or (self.mmqs_weight_mode == 'tot_api')
        if not enabled:
            return

        api_url = str(self.mmqs_tot_api_cfg.get('api_url', '')).strip()
        model = str(self.mmqs_tot_api_cfg.get('model', '')).strip()
        if len(api_url) <= 0 or len(model) <= 0:
            logging.warning('MMQS TOT API disabled: api_url/model is empty')
            return

        try:
            self.mmqs_tot_api_agent = TOTAPIAgent(
                api_url=api_url,
                model=model,
                api_key_env=str(self.mmqs_tot_api_cfg.get('api_key_env', '')),
                timeout=self._mmqs_safe_float(self.mmqs_tot_api_cfg.get('timeout', 120.0), 120.0),
                max_tokens=int(self._mmqs_safe_float(self.mmqs_tot_api_cfg.get('max_tokens', 512), 512)),
                temperature=self._mmqs_safe_float(self.mmqs_tot_api_cfg.get('temperature', 0.2), 0.2),
                top_p=self._mmqs_safe_float(self.mmqs_tot_api_cfg.get('top_p', 0.7), 0.7),
                retry=int(self._mmqs_safe_float(self.mmqs_tot_api_cfg.get('retry', 1), 1)),
                force_non_stream=bool(self.mmqs_tot_api_cfg.get('force_non_stream', False))
            )
            logging.info(
                'MMQS TOT API init ok: mode=%s url=%s model=%s key_env=%s',
                self.mmqs_weight_mode, api_url, model,
                str(self.mmqs_tot_api_cfg.get('api_key_env', ''))
            )
        except Exception as exc:                                
            self.mmqs_tot_api_agent = None
            logging.warning('MMQS TOT API init failed, fallback to static: %s', str(exc))

    def update_mmqs_regional_accuracy(self, accuracy, round_id):
        """Inject regional-model validation accuracy for ToT feedback."""
        acc = self._mmqs_safe_float(accuracy, np.nan)
        if not np.isfinite(acc):
            return
        acc = float(np.clip(acc, 0.0, 1.0))
        self.mmqs_last_regional_accuracy = acc
        feedback_round = int(max(0, self.current_select_step))
        try:
            feedback_round = max(feedback_round, int(round_id))
        except (TypeError, ValueError):
            pass
        self.mmqs_last_regional_round = feedback_round

        if self.mmqs_weight_mode != 'tot_api':
            return

        ema_factor = float(np.clip(self.mmqs_tot_api_regional_ema_factor, 0.0, 1.0))
        prev_ema = self._mmqs_safe_float(self.mmqs_tot_api_regional_ema, np.nan)
        if np.isfinite(prev_ema):
            next_ema = (1.0 - ema_factor) * float(prev_ema) + ema_factor * float(acc)
        else:
            next_ema = float(acc)
        self.mmqs_tot_api_regional_ema = float(next_ema)

        if not self.mmqs_tot_api_guard_enabled or (not np.isfinite(prev_ema)):
            return
        acc_drop = float(prev_ema) - float(acc)
        if acc_drop < float(self.mmqs_tot_api_drop_threshold):
            return

        hold_until = feedback_round + int(max(1, self.mmqs_tot_api_fallback_hold_rounds))
        self.mmqs_tot_api_fallback_until_round = max(
            int(self.mmqs_tot_api_fallback_until_round),
            int(hold_until)
        )
        logging.info(
            'MMQS TOT API guard trigger: feedback_round=%s acc=%.6f ema_prev=%.6f drop=%.6f(th=%.6f) hold_until=%s',
            feedback_round, float(acc), float(prev_ema), float(acc_drop),
            float(self.mmqs_tot_api_drop_threshold), int(self.mmqs_tot_api_fallback_until_round)
        )

    def _mmqs_update_perf_proxy(self, client_key, cur_loss):
        """Update reliability proxy by |W|-window mean or EMA fallback."""
        if self.mmqs_W > 0:
            history = self.mmqs_perf_window.get(client_key, [])
            if np.isfinite(cur_loss):
                history = list(history)
                history.append(float(cur_loss))
                if len(history) > self.mmqs_W:
                    history = history[-self.mmqs_W:]
                self.mmqs_perf_window[client_key] = history
            if len(history) > 0:
                return float(np.mean(history))
            return np.nan

        prev_ema = self.mmqs_perf_ema.get(client_key, np.nan)
        if np.isfinite(cur_loss):
            if np.isfinite(prev_ema):
                cur_ema = (1.0 - self.mmqs_perf_alpha) * float(prev_ema) + self.mmqs_perf_alpha * cur_loss
            else:
                cur_ema = float(cur_loss)
            self.mmqs_perf_ema[client_key] = float(cur_ema)
            return float(cur_ema)
        if np.isfinite(prev_ema):
            return float(prev_ema)
        return np.nan

    def _mmqs_extract_modal_score(self, client):
        for field in ('s_modal', 'modal_score', 'mmqs_modal_score'):
            value = getattr(client, field, None)
            if value is None:
                continue
            val = self._mmqs_safe_float(value, 1.0)
            return float(np.clip(val, 0.0, 1.0))

        missing_ratio = getattr(client, 'missing_modal_ratio', None)
        if missing_ratio is not None:
            miss = self._mmqs_safe_float(missing_ratio, 0.0)
            return float(np.clip(1.0 - miss, 0.0, 1.0))

        total_modalities = getattr(client, 'total_modalities', None)
        available_modalities = getattr(client, 'available_modalities', None)
        if total_modalities is not None and available_modalities is not None:
            total = max(1.0, self._mmqs_safe_float(total_modalities, 1.0))
            avail = np.clip(self._mmqs_safe_float(available_modalities, total), 0.0, total)
            return float(np.clip(avail / total, 0.0, 1.0))

        missing_modalities = getattr(client, 'missing_modalities', None)
        if total_modalities is not None and missing_modalities is not None:
            total = max(1.0, self._mmqs_safe_float(total_modalities, 1.0))
            miss = np.clip(self._mmqs_safe_float(missing_modalities, 0.0), 0.0, total)
            return float(np.clip((total - miss) / total, 0.0, 1.0))

        return 1.0

    def _mmqs_resolve_weight_vector(self, candidates, participations, cur_round, cur_thpt=0.0, component_scores=None):
        static_weights = np.array([
            self.mmqs_w_data,
            self.mmqs_w_modal,
            self.mmqs_w_perf,
            self.mmqs_w_res,
            self.mmqs_w_cool,
            self.mmqs_w_fair
        ], dtype=float)
        if self.mmqs_weight_mode == 'static':
            return static_weights, 'static_default'

        metrics = self._mmqs_collect_state_metrics(candidates, participations, cur_round)
        if self.mmqs_weight_mode == 'tot_api':
            weights, profile = self._mmqs_resolve_weight_vector_tot_api(
                static_weights=static_weights,
                metrics=metrics,
                candidates=candidates,
                cur_thpt=cur_thpt,
                component_scores=component_scores
            )
            if weights is not None:
                return weights, profile
            return static_weights, 'tot_api_fallback_static'
        return static_weights, 'static_default'

    def _mmqs_collect_state_metrics(self, candidates, participations, cur_round):
        total_rounds = max(1.0, self._mmqs_safe_float(self.rounds, 1.0))
        stage_ratio = float(np.clip(self._mmqs_safe_float(cur_round, 0.0) / total_rounds, 0.0, 1.0))

        delays = np.array([
            max(0.0, self._mmqs_safe_float(getattr(c, 'delay', 0.0), 0.0))
            for c in candidates
        ], dtype=float)
        delay_mean = float(np.mean(delays)) if delays.size > 0 else 0.0
        delay_std = float(np.std(delays)) if delays.size > 0 else 0.0
        delay_cv = delay_std / max(self.mmqs_eps, delay_mean)

        fairness_gap = 0.0
        if participations.size > 0:
            p_mean = float(np.mean(participations))
            p_max = float(np.max(participations))
            if p_max > self.mmqs_eps:
                fairness_gap = float(np.clip((p_max - p_mean) / p_max, 0.0, 1.0))
        return {
            'stage_ratio': stage_ratio,
            'delay_mean': delay_mean,
            'delay_std': delay_std,
            'delay_cv': delay_cv,
            'fairness_gap': fairness_gap
        }
    def _mmqs_apply_tot_api_regional_feedback(self, metrics, allow_prune_api=True):
        pending = self.mmqs_tot_api_pending_cycle
        if not isinstance(pending, dict):
            return
        if bool(pending.get('feedback_applied', False)):
            return
        regional_scores = pending.get('branch_regional_scores', {})
        if not isinstance(regional_scores, dict) or len(regional_scores) <= 0:
            return

        try:
            select_round = int(pending.get('select_round', -1))
        except (TypeError, ValueError):
            select_round = -1
        feedback_round = int(self._mmqs_safe_float(self.mmqs_last_regional_round, -1))
        if feedback_round < select_round:
            return

        selected_branch = str(pending.get('selected_branch', ''))
        selected_weights_key = str(pending.get('selected_weights_key', '')).strip()
        selected_proxy_reward = float(self._mmqs_safe_float(pending.get('selected_proxy_reward', 0.0), 0.0))
        evaluated = []
        selected_branch_acc = np.nan
        for item in pending.get('evaluated', []):
            if not isinstance(item, dict):
                continue
            name = str(item.get('name', '')).strip()
            weights = self._mmqs_get_item_weights(item)
            if weights.size != 6:
                continue
            b_acc = self._mmqs_safe_float(regional_scores.get(name, np.nan), np.nan)
            if name == selected_branch and np.isfinite(b_acc):
                selected_branch_acc = float(b_acc)
            evaluated.append(
                {
                    'name': name,
                    'weights': weights,
                    'reason': str(item.get('reason', '')).strip(),
                    'proxy_reward': float(item.get('base_reward', item.get('reward', 0.0))),
                    'regional_accuracy': float(b_acc) if np.isfinite(b_acc) else None,
                }
            )
        if len(evaluated) <= 0:
            pending['feedback_applied'] = True
            return
        if not np.isfinite(selected_branch_acc):
            pending['feedback_applied'] = True
            return

        feedback_state = {
            'round': int(max(0, select_round)),
            'feedback_round': int(max(0, feedback_round)),
            'regional_accuracy': float(selected_branch_acc),
            'stage_ratio': float(metrics.get('stage_ratio', 0.0)),
            'delay_cv': float(metrics.get('delay_cv', 0.0)),
            'fairness_gap': float(metrics.get('fairness_gap', 0.0)),
            'proxy_bias_ema': float(self.mmqs_tot_api_proxy_bias_ema),
        }

        prune_source = 'llm'
        prune_meta = {'attempt': 0, 'elapsed_sec': 0.0}
        prune_result = None
        if allow_prune_api:
            try:
                prune_result, prune_meta = self.mmqs_tot_api_agent.prune_and_evolve(
                    state=feedback_state,
                    evaluated_branches=evaluated
                )
            except TOTAPIError as exc:
                prune_source = 'fallback'
                logging.warning('MMQS TOT API prune failed, use fallback evolution: %s', str(exc))
            except Exception as exc:                                
                prune_source = 'fallback'
                logging.warning('MMQS TOT API prune runtime failed, use fallback evolution: %s', str(exc))
        else:
            prune_source = 'throttle'

        if not isinstance(prune_result, dict):
            prune_result = self._mmqs_default_prune_result(evaluated=evaluated, selected_branch=selected_branch)

        winner = str(prune_result.get('winner_branch_id', selected_branch)).strip() or selected_branch
        if not winner:
            winner = str(evaluated[0].get('name', 'branch_0'))
        pruned = str(prune_result.get('pruned_branch_id', '')).strip()
        direction = str(prune_result.get('next_exploration_direction', '')).strip()
        if not direction:
            direction = 'regional_feedback_refine'

        seed_weights = np.array(prune_result.get('next_seed_weights', []), dtype=float)
        if seed_weights.size != 6:
            seed_weights = np.array(
                self._mmqs_default_prune_result(evaluated=evaluated, selected_branch=winner).get('next_seed_weights', []),
                dtype=float
            )
        if seed_weights.size == 6:
            denom = float(np.sum(seed_weights))
            if denom > self.mmqs_eps:
                seed_weights = np.maximum(seed_weights, 0.0) / denom

                                                                                             
        reward_old = float(self.mmqs_tot_api_branch_memory.get(selected_branch, 0.0))
        ema_factor = float(np.clip(self.mmqs_tot_api_reward_ema_factor, 0.0, 1.0))
        reward_ema = (1.0 - ema_factor) * reward_old + ema_factor * float(selected_branch_acc)
        if selected_branch:
            self.mmqs_tot_api_branch_memory[selected_branch] = float(reward_ema)
            self.mmqs_tot_api_last_branch = selected_branch
            self.mmqs_tot_api_last_reward = float(selected_branch_acc)

        if not selected_weights_key:
            selected_weights_key = self._mmqs_weights_key(
                pending.get('selected_weights', [])
            )
        if selected_weights_key:
            weights_old = float(self.mmqs_tot_api_weights_memory.get(selected_weights_key, 0.0))
            weights_new = (1.0 - ema_factor) * weights_old + ema_factor * float(selected_branch_acc)
            self.mmqs_tot_api_weights_memory[selected_weights_key] = float(weights_new)
            prev_visits = int(self._mmqs_safe_float(self.mmqs_tot_api_weights_visits.get(selected_weights_key, 0), 0))
            self.mmqs_tot_api_weights_visits[selected_weights_key] = int(max(0, prev_visits) + 1)

        proxy_error = float(selected_branch_acc) - float(selected_proxy_reward)
        bias_alpha = float(np.clip(self.mmqs_tot_api_proxy_bias_alpha, 0.0, 1.0))
        self.mmqs_tot_api_proxy_bias_ema = float(
            (1.0 - bias_alpha) * float(self.mmqs_tot_api_proxy_bias_ema) + bias_alpha * proxy_error
        )

        self.mmqs_tot_api_last_prune = {
            'source': prune_source,
            'winner_branch_id': winner,
            'pruned_branch_id': pruned,
            'next_exploration_direction': direction,
            'next_seed_weights': [float(v) for v in seed_weights.tolist()] if seed_weights.size == 6 else [],
            'feedback_round': int(max(0, feedback_round)),
            'api_elapsed_sec': float(prune_meta.get('elapsed_sec', 0.0)),
            'api_attempt': int(prune_meta.get('attempt', 0)),
            'regional_accuracy': float(selected_branch_acc),
            'selected_weights_key': str(selected_weights_key),
            'proxy_error': float(proxy_error),
            'proxy_bias_ema': float(self.mmqs_tot_api_proxy_bias_ema),
        }
        pending['feedback_applied'] = True

        logging.info(
            'MMQS TOT API feedback: select_round=%s feedback_round=%s branch=%s regional_acc=%.6f proxy_reward=%.6f proxy_error=%.6f proxy_bias_ema=%.6f prune_source=%s winner=%s pruned=%s api_elapsed=%.4fs attempt=%s',
            select_round, feedback_round, selected_branch, float(selected_branch_acc),
            float(selected_proxy_reward), float(proxy_error), float(self.mmqs_tot_api_proxy_bias_ema),
            prune_source, winner, pruned,
            float(prune_meta.get('elapsed_sec', 0.0)),
            prune_meta.get('attempt', 0)
        )

    def _mmqs_default_prune_result(self, evaluated, selected_branch):
        winner = str(selected_branch or '')
        winner_weights = None
        best_acc = -1.0
        for item in evaluated:
            if not isinstance(item, dict):
                continue
            name = str(item.get('name', '')).strip()
            weights = self._mmqs_get_item_weights(item)
            if weights.size != 6:
                continue
            acc = self._mmqs_safe_float(item.get('regional_accuracy', np.nan), np.nan)
            if np.isfinite(acc) and (acc >= best_acc):
                best_acc = float(acc)
                winner = name
                winner_weights = weights
            elif winner_weights is None and name == winner:
                winner_weights = weights
        if winner_weights is None:
            for item in evaluated:
                weights = self._mmqs_get_item_weights(item)
                if weights.size == 6:
                    winner_weights = weights
                    if not winner:
                        winner = str(item.get('name', 'branch_0'))
                    break
        if winner_weights is None:
            winner_weights = np.array([1.0 / 6.0] * 6, dtype=float)

        pruned = ''
        worst_acc = 1e30
        for item in evaluated:
            name = str(item.get('name', '')).strip()
            if (not name) or (name == winner):
                continue
            acc = self._mmqs_safe_float(item.get('regional_accuracy', np.nan), np.nan)
            if np.isfinite(acc) and acc < worst_acc:
                worst_acc = float(acc)
                pruned = name

        denom = float(np.sum(np.maximum(winner_weights, 0.0)))
        if denom > self.mmqs_eps:
            winner_weights = np.maximum(winner_weights, 0.0) / denom

        return {
            'pruned_branch_id': pruned,
            'winner_branch_id': winner,
            'next_exploration_direction': 'regional_feedback_refine',
            'next_seed_weights': winner_weights,
        }

    def _mmqs_build_tot_api_state(self, metrics, candidates, component_scores):
        arrays = {}
        for key in ('s_data', 's_modal', 's_perf', 's_res', 's_cool', 's_fair'):
            val = component_scores.get(key, []) if isinstance(component_scores, dict) else []
            arr = np.array(val, dtype=float) if len(val) > 0 else np.array([], dtype=float)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arrays[key] = arr

        memory_items = sorted(
            self.mmqs_tot_api_branch_memory.items(),
            key=lambda item: item[1],
            reverse=True
        )
        memory_top = [
            {'name': str(name), 'reward_ema': float(reward)}
            for name, reward in memory_items[:3]
        ]

        weights_items = sorted(
            self.mmqs_tot_api_weights_memory.items(),
            key=lambda item: item[1],
            reverse=True
        )
        weights_memory_top = []
        for key, reward in weights_items[:3]:
            visits = int(self._mmqs_safe_float(self.mmqs_tot_api_weights_visits.get(key, 0), 0))
            row = {'weights_key': str(key), 'reward_ema': float(reward), 'visits': int(max(0, visits))}
            weights_memory_top.append(row)

        regional_acc = self._mmqs_safe_float(self.mmqs_last_regional_accuracy, np.nan)
        if np.isfinite(regional_acc):
            regional_acc = float(np.clip(regional_acc, 0.0, 1.0))
        else:
            regional_acc = None

        last_prune = self.mmqs_tot_api_last_prune if isinstance(self.mmqs_tot_api_last_prune, dict) else {}
        state_last_prune = {}
        if len(last_prune) > 0:
            state_last_prune = {
                'source': str(last_prune.get('source', '')),
                'winner_branch_id': str(last_prune.get('winner_branch_id', '')),
                'pruned_branch_id': str(last_prune.get('pruned_branch_id', '')),
                'next_exploration_direction': str(last_prune.get('next_exploration_direction', '')),
                'next_seed_weights': [
                    float(v) for v in last_prune.get('next_seed_weights', [])
                ] if isinstance(last_prune.get('next_seed_weights', []), list) else [],
                'feedback_round': int(self._mmqs_safe_float(last_prune.get('feedback_round', -1), -1)),
                'proxy_error': float(self._mmqs_safe_float(last_prune.get('proxy_error', 0.0), 0.0)),
                'proxy_bias_ema': float(self._mmqs_safe_float(last_prune.get('proxy_bias_ema', 0.0), 0.0)),
            }

        return {
            'round': int(max(0, self.current_select_step)),
            'round_total': int(max(1, self.rounds)),
            'stage_ratio': float(metrics.get('stage_ratio', 0.0)),
            'regional_accuracy': regional_acc,
            'regional_feedback_round': int(self._mmqs_safe_float(self.mmqs_last_regional_round, -1)),
            'delay_mean': float(metrics.get('delay_mean', 0.0)),
            'delay_std': float(metrics.get('delay_std', 0.0)),
            'delay_cv': float(metrics.get('delay_cv', 0.0)),
            'fairness_gap': float(metrics.get('fairness_gap', 0.0)),
            'Q': int(len(candidates)),
            'component_mean': {
                key: float(np.mean(arrays[key])) if arrays[key].size > 0 else 0.0
                for key in arrays
            },
            'component_std': {
                key: float(np.std(arrays[key])) if arrays[key].size > 0 else 0.0
                for key in arrays
            },
            'memory_top': memory_top,
            'weights_memory_top': weights_memory_top,
            'proxy_bias_ema': float(self.mmqs_tot_api_proxy_bias_ema),
            'last_prune': state_last_prune
        }

    def _mmqs_validate_tot_candidates(
        self,
        api_candidates,
        client_candidates,
        component_scores,
        metrics,
        return_all=False
    ):
        if len(api_candidates) <= 0 or len(client_candidates) <= 0:
            return (None, []) if return_all else None
        if not isinstance(component_scores, dict):
            return (None, []) if return_all else None

        arrays = {}
        for key in ('s_data', 's_modal', 's_perf', 's_res', 's_cool', 's_fair'):
            raw = component_scores.get(key, None)
            if raw is None:
                return (None, []) if return_all else None
            arr = np.nan_to_num(np.array(raw, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            arrays[key] = arr
        n = len(arrays['s_data'])
        if n <= 0:
            return (None, []) if return_all else None

        delays = np.array([
            max(0.0, self._mmqs_safe_float(getattr(c, 'delay', 0.0), 0.0))
            for c in client_candidates
        ], dtype=float)
        if delays.size != n:
            delays = np.zeros((n,), dtype=float)
        max_delay = max(self.mmqs_eps, float(np.max(delays)) if delays.size > 0 else 1.0)
        delays_norm = delays / max_delay

        stage_ratio = float(metrics.get('stage_ratio', 0.0))
        fairness_gap = float(metrics.get('fairness_gap', 0.0))
        delay_cv = float(metrics.get('delay_cv', 0.0))
        if stage_ratio < 0.30:
            w_quality, w_delay, w_fair, w_res = 0.45, 0.30, 0.15, 0.10
        elif stage_ratio < 0.70:
            w_quality, w_delay, w_fair, w_res = 0.55, 0.25, 0.10, 0.10
        else:
            w_quality, w_delay, w_fair, w_res = 0.65, 0.20, 0.10, 0.05

        k_eval = max(1, int(np.ceil(0.25 * n)))
        best = None
        evaluated = []
        for item in api_candidates:
            weights = self._mmqs_get_item_weights(item)
            if weights.size != 6:
                continue
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            weights = np.maximum(weights, 0.0)
            weights_sum = float(np.sum(weights))
            if weights_sum <= self.mmqs_eps:
                continue
            weights = weights / weights_sum

            raw_score = (
                weights[0] * arrays['s_data'] +
                weights[1] * arrays['s_modal'] +
                weights[2] * arrays['s_perf'] +
                weights[3] * arrays['s_res'] +
                weights[4] * arrays['s_cool'] +
                weights[5] * arrays['s_fair']
            )
            sorted_idx = np.argsort(-raw_score)
            top_idx = sorted_idx[:k_eval]
            quality_term = float(np.mean(raw_score[top_idx])) if top_idx.size > 0 else float(np.mean(raw_score))
            delay_term = float(np.mean(delays_norm[top_idx])) if top_idx.size > 0 else float(np.mean(delays_norm))
            fair_term = float(np.mean(arrays['s_fair'][top_idx])) if top_idx.size > 0 else float(np.mean(arrays['s_fair']))
            res_term = float(np.mean(arrays['s_res'][top_idx])) if top_idx.size > 0 else float(np.mean(arrays['s_res']))

            reward = (
                w_quality * quality_term -
                w_delay * delay_term +
                w_fair * fair_term +
                w_res * res_term
            )
            if (fairness_gap >= self.mmqs_tot_fairness_gap_threshold) and (weights[5] < 0.08):
                reward -= 0.03
            if (delay_cv >= self.mmqs_tot_delay_cv_stability_threshold) and ((weights[3] < 0.08) or (weights[4] < 0.08)):
                reward -= 0.03
            if (stage_ratio >= 0.70) and (weights[2] < 0.30):
                reward -= 0.04

            name = str(item.get('name', 'branch'))
            mem_bonus = self.mmqs_tot_api_memory_bonus * float(
                self.mmqs_tot_api_branch_memory.get(name, 0.0)
            )
            weights_key = self._mmqs_weights_key(weights)
            weights_prior = float(self.mmqs_tot_api_weights_memory.get(weights_key, 0.0))
            weights_bonus = self.mmqs_tot_api_weights_memory_bonus * weights_prior
            visits = int(self._mmqs_safe_float(self.mmqs_tot_api_weights_visits.get(weights_key, 0), 0))
            explore_bonus = float(self.mmqs_tot_api_explore_coef / np.sqrt(1.0 + max(0, visits)))
            proxy_corrected = float(reward + self.mmqs_tot_api_proxy_bias_ema)
            reward_total = float(proxy_corrected + mem_bonus + weights_bonus + explore_bonus)
            result = {
                'name': name,
                'weights': weights,
                'reward': reward_total,
                'base_reward': float(reward),
                'proxy_corrected': float(proxy_corrected),
                'memory_bonus': float(mem_bonus),
                'weights_memory_bonus': float(weights_bonus),
                'explore_bonus': float(explore_bonus),
                'weights_key': str(weights_key),
                'weights_prior': float(weights_prior),
                'weights_visits': int(max(0, visits)),
                'reason': str(item.get('reason', '')),
                'parse_source': str(item.get('parse_source', 'unknown'))
            }
            evaluated.append(result)
            if (best is None) or (reward_total > best['reward']):
                best = result
        if return_all:
            return best, evaluated
        return best

    def _mmqs_build_branch_sample_clients(self, candidates, weights, cur_thpt):
        if len(candidates) <= 0:
            return []
        arr = self._mmqs_normalize_weights_vec(weights)
        if arr.size != 6:
            return []
        data_sizes = np.array([
            max(0.0, self._mmqs_safe_float(getattr(c, 'num_samples', 0.0), 0.0))
            for c in candidates
        ], dtype=float)
        max_data = float(np.max(data_sizes)) if data_sizes.size > 0 else 0.0
        s_data = (data_sizes / max_data) if max_data > self.mmqs_eps else np.zeros((len(candidates),), dtype=float)

        ema_losses = np.full((len(candidates),), np.nan, dtype=float)
        for i, c in enumerate(candidates):
            key = self._mmqs_client_key(c)
            raw_loss = self._mmqs_safe_float(getattr(c, 'loss', np.nan), np.nan)
            proxy_loss = self._mmqs_update_perf_proxy(key, raw_loss)
            if np.isfinite(proxy_loss):
                ema_losses[i] = float(proxy_loss)
        valid_ema = ema_losses[np.isfinite(ema_losses)]
        neutral = float(np.median(valid_ema)) if valid_ema.size > 0 else float(self.mmqs_perf_last_neutral_ema)
        if not np.isfinite(neutral):
            neutral = float(self.mmqs_perf_default_ema)
        ema_losses[~np.isfinite(ema_losses)] = neutral
        p_min = float(np.min(ema_losses)) if ema_losses.size > 0 else 0.0
        p_max = float(np.max(ema_losses)) if ema_losses.size > 0 else 0.0
        p_den = max(self.mmqs_eps, p_max - p_min)
        if p_den > self.mmqs_eps:
            s_perf = 1.0 - (ema_losses - p_min) / p_den
        else:
            s_perf = np.full((len(candidates),), 0.5, dtype=float)
        s_perf = np.clip(np.nan_to_num(s_perf, nan=0.5, posinf=0.5, neginf=0.5), 0.0, 1.0)

        participations = np.array([
            float(max(0, min(self.mmqs_participation_cap, int(self._mmqs_safe_float(getattr(c, 'participation_count', 0), 0)))))
            for c in candidates
        ], dtype=float)
        s_res = np.exp(-self.mmqs_res_lambda * participations)
        s_res = np.clip(np.nan_to_num(s_res, nan=self.mmqs_eps, posinf=1.0, neginf=self.mmqs_eps), self.mmqs_eps, 1.0)

        last_rounds = np.array([
            float(int(self._mmqs_safe_float(getattr(c, 'last_participation_round', -1), -1)))
            for c in candidates
        ], dtype=float)
        s_cool = np.ones((len(candidates),), dtype=float)
        valid_last = last_rounds >= 0
        if np.any(valid_last):
            cool_window = max(1.0, float(self.mmqs_t_cool))
            cool_delta = (float(self.current_select_step) - last_rounds[valid_last]) / cool_window
            s_cool[valid_last] = np.clip(cool_delta, 0.0, 1.0)
        s_cool = np.nan_to_num(s_cool, nan=1.0, posinf=1.0, neginf=0.0)

        all_part = []
        for c in self.clients:
            all_part.append(max(0, int(self._mmqs_safe_float(getattr(c, 'participation_count', 0), 0))))
        max_p = max(1.0, float(np.max(all_part)) if len(all_part) > 0 else 1.0)
        s_fair = 1.0 - np.divide(participations, max_p)
        s_fair = np.clip(np.nan_to_num(s_fair, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

        s_modal = np.array([self._mmqs_extract_modal_score(c) for c in candidates], dtype=float)
        s_modal = np.clip(np.nan_to_num(s_modal, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        scores = (
            arr[0] * s_data +
            arr[1] * s_modal +
            arr[2] * s_perf +
            arr[3] * s_res +
            arr[4] * s_cool +
            arr[5] * s_fair
        )
        scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
        sorted_idx = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        ordered = [candidates[i] for i in sorted_idx]

        thpt_budget = float(self.thpt_ub - cur_thpt)
        accum = 0.0
        k = 0
        for c in ordered:
            c_thpt = max(0.0, self._mmqs_safe_float(getattr(c, 'throughput', 0.0), 0.0))
            if accum + c_thpt <= thpt_budget + self.mmqs_eps:
                accum += c_thpt
                k += 1
            else:
                break
        sample = ordered[:k]
        sample = self._ensure_nonempty_selection(sample, cur_thpt)
        sample = self._mmqs_apply_coverage_floor(
            sample_clients=sample,
            ranked_candidates=ordered,
            cur_thpt=cur_thpt
        )
        return sample

    def _mmqs_resolve_weight_vector_tot_api(self, static_weights, metrics, candidates, cur_thpt, component_scores):
        if self.mmqs_tot_api_agent is None:
            logging.warning('MMQS TOT API unavailable, fallback to static')
            return None, 'tot_api_unavailable'

        select_round = int(max(0, self.current_select_step))
        if (
            self.mmqs_tot_api_guard_enabled and
            (select_round <= int(self.mmqs_tot_api_fallback_until_round))
        ):
            logging.info(
                'MMQS TOT API guard fallback: select_round=%s hold_until=%s use=static',
                select_round, int(self.mmqs_tot_api_fallback_until_round)
            )
            return np.array(static_weights, dtype=float), 'tot_api_guard_static'

        try:
                                                                             
                                                                   
            allow_prune_api = (select_round % int(self.mmqs_tot_api_prune_interval) == 0)
            self._mmqs_apply_tot_api_regional_feedback(
                metrics,
                allow_prune_api=allow_prune_api
            )

            if self.mmqs_tot_api_last_weights is not None:
                gap = select_round - int(self.mmqs_tot_api_last_resolve_round)
                if (gap > 0) and (gap < int(self.mmqs_tot_api_call_interval)):
                    logging.info(
                        'MMQS TOT API throttle reuse: select_round=%s last_round=%s call_interval=%s profile=%s',
                        select_round, int(self.mmqs_tot_api_last_resolve_round),
                        int(self.mmqs_tot_api_call_interval), str(self.mmqs_tot_api_last_profile)
                    )
                    return self.mmqs_tot_api_last_weights.copy(), str(self.mmqs_tot_api_last_profile)

            state = self._mmqs_build_tot_api_state(metrics, candidates, component_scores)
            api_candidates, api_meta = self.mmqs_tot_api_agent.propose_weights(
                state=state,
                Q=self.mmqs_tot_q
            )
            best, evaluated = self._mmqs_validate_tot_candidates(
                api_candidates=api_candidates,
                client_candidates=candidates,
                component_scores=component_scores,
                metrics=metrics,
                return_all=True
            )
            if best is None:
                logging.warning('MMQS TOT API got invalid candidates, fallback to static')
                return None, 'tot_api_invalid'

            parse_source = str(best.get('parse_source', 'unknown')).strip().lower() or 'unknown'
            selected_weights, blend_api = self._mmqs_blend_tot_api_weights(
                best.get('weights', []),
                static_weights,
                parse_source=parse_source
            )
            branch_name = str(best['name'])
            reward_proxy = float(best.get('base_reward', best.get('reward', 0.0)))
            reward_proxy_corrected = float(best.get('proxy_corrected', reward_proxy + self.mmqs_tot_api_proxy_bias_ema))
            selected_weights_key = str(self._mmqs_weights_key(selected_weights))
            cycle_pack = {
                'select_round': select_round,
                'selected_branch': branch_name,
                'selected_weights': [float(v) for v in np.array(selected_weights, dtype=float).tolist()],
                'selected_weights_key': selected_weights_key,
                'selected_parse_source': str(parse_source),
                'selected_blend_api': float(blend_api),
                'selected_proxy_reward': float(reward_proxy),
                'selected_proxy_corrected': float(reward_proxy_corrected),
                'evaluated': [
                    {
                        'name': str(item.get('name', '')),
                        'weights': [float(v) for v in np.array(item.get('weights', []), dtype=float).tolist()],
                        'reason': str(item.get('reason', '')),
                        'reward': float(item.get('reward', 0.0)),
                        'base_reward': float(item.get('base_reward', 0.0)),
                        'proxy_corrected': float(item.get('proxy_corrected', 0.0)),
                        'memory_bonus': float(item.get('memory_bonus', 0.0)),
                        'weights_memory_bonus': float(item.get('weights_memory_bonus', 0.0)),
                        'explore_bonus': float(item.get('explore_bonus', 0.0)),
                        'weights_key': str(item.get('weights_key', self._mmqs_weights_key(item.get('weights', [])))),
                        'weights_prior': float(item.get('weights_prior', 0.0)),
                        'weights_visits': int(self._mmqs_safe_float(item.get('weights_visits', 0), 0)),
                        'parse_source': str(item.get('parse_source', 'unknown')),
                    }
                    for item in evaluated
                ],
                'feedback_applied': False,
            }
            self.mmqs_tot_api_pending_cycle = cycle_pack
            self.mmqs_tot_api_last_branch = branch_name
            self.mmqs_tot_api_last_reward = float(reward_proxy)

            last_prune = self.mmqs_tot_api_last_prune if isinstance(self.mmqs_tot_api_last_prune, dict) else {}
            prune_hint = str(last_prune.get('next_exploration_direction', ''))

            branch_plans = []
            union_ids = set()
            for item in evaluated:
                branch_name_i = str(item.get('name', ''))
                weights_i = item.get('weights', [])
                clients_i = self._mmqs_build_branch_sample_clients(
                    candidates=candidates,
                    weights=weights_i,
                    cur_thpt=cur_thpt
                )
                ids_i = []
                for c in clients_i:
                    cid = getattr(c, 'client_id', None)
                    try:
                        cid_int = int(cid)
                    except (TypeError, ValueError):
                        continue
                    ids_i.append(cid_int)
                    union_ids.add(cid_int)
                branch_plans.append({
                    'name': branch_name_i,
                    'weights': [float(v) for v in np.array(weights_i, dtype=float).tolist()],
                    'client_ids': ids_i
                })
            self.mmqs_tot_api_branch_plan = {
                'select_round': int(select_round),
                'branches': branch_plans,
                'union_client_ids': sorted(list(union_ids))
            }

            logging.info(
                'MMQS TOT API resolve: branch=%s parse_source=%s blend_api=%.2f proxy=%.4f proxy_corrected=%.4f total=%.4f base=%.4f mem_bonus=%.4f weights_bonus=%.4f explore_bonus=%.4f api_elapsed=%.4fs attempt=%s stage_ratio=%.4f fairness_gap=%.4f delay_cv=%.4f prune_hint=%s',
                branch_name, parse_source, float(blend_api), reward_proxy, reward_proxy_corrected, float(best.get('reward', 0.0)),
                float(best['base_reward']), float(best['memory_bonus']),
                float(best.get('weights_memory_bonus', 0.0)), float(best.get('explore_bonus', 0.0)),
                float(api_meta.get('elapsed_sec', 0.0)), api_meta.get('attempt', 1),
                float(metrics.get('stage_ratio', 0.0)), float(metrics.get('fairness_gap', 0.0)),
                float(metrics.get('delay_cv', 0.0)), prune_hint
            )
            logging.info(
                'MMQS TOT API weights final: parse_source=%s blend_api=%.2f weights=%s static=%s',
                parse_source, float(blend_api),
                self._mmqs_fmt_weights(selected_weights),
                self._mmqs_fmt_weights(static_weights)
            )
            selected_profile = 'tot_api_{}'.format(branch_name)
            self.mmqs_tot_api_last_weights = selected_weights.copy()
            self.mmqs_tot_api_last_profile = str(selected_profile)
            self.mmqs_tot_api_last_resolve_round = int(select_round)
            return selected_weights, selected_profile
        except TOTAPIError as exc:
            logging.warning('MMQS TOT API call/parse failed, fallback to static: %s', str(exc))
        except Exception as exc:                                
            logging.warning('MMQS TOT API runtime failed, fallback to static: %s', str(exc))
        if self.mmqs_tot_api_last_weights is not None:
            logging.warning(
                'MMQS TOT API fallback uses cached branch profile=%s last_round=%s',
                str(self.mmqs_tot_api_last_profile), int(self.mmqs_tot_api_last_resolve_round)
            )
            return self.mmqs_tot_api_last_weights.copy(), str(self.mmqs_tot_api_last_profile)
        return None, 'tot_api_error'

    def get_mmqs_tot_branch_plan(self):
        if self.mmqs_weight_mode != 'tot_api':
            return None
        plan = self.mmqs_tot_api_branch_plan if isinstance(self.mmqs_tot_api_branch_plan, dict) else None
        if not isinstance(plan, dict):
            return None
        if int(plan.get('select_round', -1)) != int(max(0, self.current_select_step)):
            return None
        if not isinstance(plan.get('branches', []), list) or len(plan.get('branches', [])) <= 0:
            return None
        return plan

    def apply_mmqs_tot_branch_scores(self, branch_scores, feedback_round):
        if self.mmqs_weight_mode != 'tot_api':
            return
        if not isinstance(branch_scores, dict) or len(branch_scores) <= 0:
            return
        pending = self.mmqs_tot_api_pending_cycle
        if not isinstance(pending, dict):
            return
        cleaned = {}
        for name, score in branch_scores.items():
            key = str(name).strip()
            val = self._mmqs_safe_float(score, np.nan)
            if key and np.isfinite(val):
                cleaned[key] = float(np.clip(val, 0.0, 1.0))
        if len(cleaned) <= 0:
            return
        pending['branch_regional_scores'] = cleaned
        try:
            fr = int(feedback_round)
        except (TypeError, ValueError):
            fr = int(max(0, self.current_select_step))
        self.mmqs_last_regional_round = max(int(self.mmqs_last_regional_round), int(fr))
        selected_branch = str(pending.get('selected_branch', '')).strip()
        if selected_branch in cleaned:
            self.mmqs_last_regional_accuracy = float(cleaned[selected_branch])

    def _mmqs_safe_float(self, value, default=0.0):
        try:
            val = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(val):
            return float(default)
        return float(val)

    @staticmethod
    def _mmqs_fmt_weights(weights):
        arr = np.array(weights, dtype=float).reshape((-1,))
        if arr.size <= 0:
            return '[]'
        return '[' + ','.join('{:.4f}'.format(float(v)) for v in arr.tolist()) + ']'

    @staticmethod
    def _mmqs_get_item_weights(item):
        if not isinstance(item, dict):
            return np.array([], dtype=float)
        return np.array(item.get('weights', []), dtype=float)

    def _mmqs_normalize_weights_vec(self, weights, fallback=None):
        arr = np.array(weights, dtype=float).reshape((-1,))
        if arr.size != 6:
            if fallback is None:
                arr = np.array([1.0 / 6.0] * 6, dtype=float)
            else:
                arr = np.array(fallback, dtype=float).reshape((-1,))
                if arr.size != 6:
                    arr = np.array([1.0 / 6.0] * 6, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.maximum(arr, 0.0)
        denom = float(np.sum(arr))
        if denom <= self.mmqs_eps:
            arr = np.array([1.0 / 6.0] * 6, dtype=float)
            denom = float(np.sum(arr))
        return arr / max(self.mmqs_eps, denom)

    @staticmethod
    def _mmqs_tot_api_blend_ratio(parse_source):
        src = str(parse_source or '').strip().lower()
        if src in ('strict_template', 'strict_json'):
            return 0.80
        if ('fallback' in src) or ('partial' in src):
            return 0.50
        return 0.60

    def _mmqs_blend_tot_api_weights(self, api_weights, static_weights, parse_source='unknown'):
        static_weights_norm = self._mmqs_normalize_weights_vec(static_weights)
        api_weights_norm = self._mmqs_normalize_weights_vec(api_weights, fallback=static_weights_norm)
        blend_api = float(np.clip(self._mmqs_tot_api_blend_ratio(parse_source), 0.0, 1.0))
        blended = blend_api * api_weights_norm + (1.0 - blend_api) * static_weights_norm
        blended = self._mmqs_normalize_weights_vec(blended, fallback=static_weights_norm)
        return blended, blend_api

    def _mmqs_apply_tot_profile_preset_from_env(self):
        preset = str(os.environ.get('MMQS_TOT_PROFILE_PRESET', '')).strip().lower()
        if not preset:
            return False
        return False

    def _mmqs_apply_weight_overrides_from_env(self):
        static_map = [
            ('mmqs_w_data', 'MMQS_W_DATA'),
            ('mmqs_w_modal', 'MMQS_W_MODAL'),
            ('mmqs_w_perf', 'MMQS_W_PERF'),
            ('mmqs_w_res', 'MMQS_W_RES'),
            ('mmqs_w_cool', 'MMQS_W_COOL'),
            ('mmqs_w_fair', 'MMQS_W_FAIR'),
        ]
        static_changed = False
        for attr, env_key in static_map:
            raw = os.environ.get(env_key, None)
            if raw is None:
                continue
            setattr(self, attr, max(0.0, self._mmqs_safe_float(raw, getattr(self, attr))))
            static_changed = True
        if static_changed:
            logging.info(
                'MMQS static weights overridden by env: data=%.4f modal=%.4f perf=%.4f res=%.4f cool=%.4f fair=%.4f',
                float(self.mmqs_w_data), float(self.mmqs_w_modal), float(self.mmqs_w_perf),
                float(self.mmqs_w_res), float(self.mmqs_w_cool), float(self.mmqs_w_fair)
            )

        preset_applied = self._mmqs_apply_tot_profile_preset_from_env()

        disable_modal = int(self._mmqs_safe_float(os.environ.get('MMQS_TOT_DISABLE_MODAL', '0'), 0.0)) != 0
        modal_raw = os.environ.get('MMQS_TOT_MODAL', None)
        if disable_modal:
            target_modal = 0.0
            apply_tot_override = True
        elif modal_raw is not None:
            target_modal = float(np.clip(self._mmqs_safe_float(modal_raw, 0.02), 0.0, 1.0))
            apply_tot_override = True
        else:
            target_modal = None
            apply_tot_override = False

        if (not apply_tot_override) and preset_applied:
            return

        if not apply_tot_override:
            return

        remain = max(0.0, 1.0 - float(target_modal))
        for key, vec in list(self.mmqs_tot_v1_profiles.items()):
            arr = np.array(vec, dtype=float).reshape((-1,))
            if arr.size != 6:
                continue
            head = np.maximum(arr[:5], 0.0)
            denom = float(np.sum(head))
            if denom <= self.mmqs_eps:
                arr[:5] = np.array([remain / 5.0] * 5, dtype=float)
            else:
                arr[:5] = (head / denom) * remain
            arr[5] = float(target_modal)
            self.mmqs_tot_v1_profiles[key] = arr

        logging.info(
            'MMQS TOT profiles overridden by env: modal=%.4f disable_modal=%s',
            float(target_modal), str(bool(disable_modal))
        )

    def _mmqs_weights_key(self, weights):
        arr = np.array(weights, dtype=float).reshape((-1,))
        if arr.size != 6:
            return ''
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.maximum(arr, 0.0)
        denom = float(np.sum(arr))
        if denom <= self.mmqs_eps:
            return ''
        arr = arr / denom
                                                                        
        q = [int(np.round(v * 1000.0)) for v in arr.tolist()]
        return ','.join(str(int(x)) for x in q)

    def _mmqs_client_key(self, client):
        client_id = getattr(client, 'client_id', None)
        if client_id is not None:
            return ('cid', client_id)
        return ('obj', id(client))

    def _mmqs_estimate_k(self, ranked_candidates, cur_thpt):
        avail_thpt = max(0.0, float(self.thpt_ub) - float(cur_thpt))
        accum = 0.0
        k_est = 0
        for client in ranked_candidates:
            thpt = self._mmqs_safe_float(getattr(client, 'throughput', 0.0), 0.0)
            thpt = max(0.0, thpt)
            if accum + thpt <= avail_thpt + self.mmqs_eps:
                accum += thpt
                k_est += 1
            else:
                break
        return k_est

    def _mmqs_apply_speed_constraints(self, ranked_candidates, ranked_scores, cur_thpt):
        """C23Q: C23 fast-set ordering + one constrained mid-stage replacement."""
        if len(ranked_candidates) <= 1:
            return ranked_candidates
        if ranked_scores is None or len(ranked_scores) != len(ranked_candidates):
            return ranked_candidates

        total_rounds = max(1, int(self._mmqs_safe_float(getattr(self, 'rounds', 1), 1)))
        stage_ratio = float(self.current_select_step) / float(total_rounds)
        fallback_start = float(np.clip(
            self._mmqs_safe_float(getattr(self, 'mmqs_c23q_fallback_start_ratio', 0.55), 0.55),
            0.0, 1.0
        ))
                                                                    
        use_tot_late_speed_guard = (
            getattr(self, 'mmqs_weight_mode', 'static') in ('tot_api',) and
            stage_ratio >= fallback_start
        )
                                                                                              
        if stage_ratio >= fallback_start and (not use_tot_late_speed_guard):
            logging.info(
                'MMQS C23Q fallback to C2 ranking: stage_ratio=%.4f fallback_start=%.2f',
                stage_ratio, fallback_start
            )
            return ranked_candidates
        if use_tot_late_speed_guard:
            logging.info(
                'MMQS C23Q late speed guard enabled for TOT: stage_ratio=%.4f fallback_start=%.2f',
                stage_ratio, fallback_start
            )

        k_est = self._mmqs_estimate_k(ranked_candidates, cur_thpt)
        if k_est <= 1:
            return ranked_candidates

        quality_map = getattr(self, 'mmqs_last_quality_scores', {})

        def _quality(idx):
            client = ranked_candidates[idx]
            q = quality_map.get(self._mmqs_client_key(client), float(ranked_scores[idx]))
            if not np.isfinite(q):
                q = float(ranked_scores[idx])
            return float(q)

        pool_factor = max(1.0, self._mmqs_safe_float(
            getattr(self, 'mmqs_c23_pool_factor', 2.0), 2.0
        ))
        min_pool = max(1, int(self._mmqs_safe_float(
            getattr(self, 'mmqs_c23_min_pool', 8), 8
        )))
        pool_n = int(np.ceil(pool_factor * k_est))
        pool_n = max(k_est + 1, min_pool, pool_n)
        pool_n = min(len(ranked_candidates), pool_n)
        if pool_n <= k_est:
            return ranked_candidates

        pool_idx = list(range(pool_n))
        delays = np.array([
            max(0.0, self._mmqs_safe_float(getattr(ranked_candidates[i], 'delay', 0.0), 0.0))
            for i in pool_idx
        ], dtype=float)
        if delays.size == 0:
            return ranked_candidates

        delay_q = float(np.clip(
            self._mmqs_safe_float(getattr(self, 'mmqs_c23_delay_quantile', 0.80), 0.80), 0.0, 1.0
        ))
        delay_cap = float(np.quantile(delays, delay_q))
        fast_idx = [i for i in pool_idx if delays[i] <= delay_cap + self.mmqs_eps]
        slow_idx = [i for i in pool_idx if i not in fast_idx]
        if len(fast_idx) == 0:
            fast_idx, slow_idx = pool_idx, []

        fast_sorted = sorted(fast_idx, key=lambda i: float(ranked_scores[i]), reverse=True)

        stage_start = float(np.clip(
            self._mmqs_safe_float(getattr(self, 'mmqs_c23q_stage_start_ratio', 0.45), 0.45),
            0.0, 1.0
        ))
        stage_end = float(np.clip(
            self._mmqs_safe_float(getattr(self, 'mmqs_c23q_stage_end_ratio', 0.70), 0.70),
            0.0, 1.0
        ))
        stage_on = (stage_ratio >= stage_start) and (stage_ratio <= stage_end)
        delay_tol = max(0.0, self._mmqs_safe_float(
            getattr(self, 'mmqs_c23q_delay_tol', 0.02), 0.02
        ))
        score_tol = max(0.0, self._mmqs_safe_float(
            getattr(self, 'mmqs_c23q_score_tol', 0.01), 0.01
        ))
        anchor_ratio = float(np.clip(
            self._mmqs_safe_float(getattr(self, 'mmqs_c23q_anchor_ratio', 0.08), 0.08),
            0.0, 1.0
        ))
        anchor_max = max(0, int(self._mmqs_safe_float(
            getattr(self, 'mmqs_c23q_anchor_max', 1), 1
        )))
        anchor_delay_mul = max(1.0, self._mmqs_safe_float(
            getattr(self, 'mmqs_c23q_anchor_delay_mul', 1.05), 1.05
        ))
        anchor_stage_end = float(np.clip(
            self._mmqs_safe_float(getattr(self, 'mmqs_c23q_anchor_stage_end_ratio', 0.55), 0.55),
            0.0, 1.0
        ))

        swap_applied = 0
        swap_old_id = -1
        swap_new_id = -1
        anchor_applied = 0
        anchor_client_ids = []
        late_anchor_applied = 0
        late_anchor_client_ids = []

        if stage_on and len(fast_sorted) > k_est and k_est >= 1:
            old_idx = fast_sorted[k_est - 1]
            old_delay = float(delays[old_idx])
            old_score = float(ranked_scores[old_idx])

            tail_idx = fast_sorted[k_est:]
            feasible_idx = [
                i for i in tail_idx
                if (float(delays[i]) <= old_delay * (1.0 + delay_tol) + self.mmqs_eps) and
                   (float(ranked_scores[i]) >= old_score - score_tol)
            ]

            if len(feasible_idx) > 0:
                max_part = 1.0
                for idx in fast_sorted:
                    part = float(max(0, getattr(ranked_candidates[idx], 'participation_count', 0)))
                    if part > max_part:
                        max_part = part

                def _swap_rank(i):
                    part = float(max(0, getattr(ranked_candidates[i], 'participation_count', 0)))
                    part_norm = part / max_part
                    return (part_norm, -_quality(i), float(delays[i]), -float(ranked_scores[i]))

                new_idx = sorted(feasible_idx, key=_swap_rank)[0]
                if new_idx != old_idx:
                    selected_prefix = fast_sorted[:k_est]
                    selected_prefix[-1] = new_idx
                    selected_set = set(selected_prefix)
                    fast_sorted = selected_prefix + [i for i in fast_sorted if i not in selected_set]
                    swap_applied = 1
                    swap_old_id = int(getattr(ranked_candidates[old_idx], 'client_id', -1))
                    swap_new_id = int(getattr(ranked_candidates[new_idx], 'client_id', -1))

                                                                                                 
        if k_est >= 4 and anchor_max > 0 and anchor_ratio > 0.0 and stage_on and stage_ratio <= anchor_stage_end:
            anchor_target = min(anchor_max, max(1, int(np.ceil(anchor_ratio * k_est))))
            delay_anchor_cap = delay_cap * anchor_delay_mul + self.mmqs_eps
            quality_sorted = sorted(
                pool_idx,
                key=lambda i: (-_quality(i), delays[i], -float(ranked_scores[i]))
            )
            anchor_idx = []
            anchor_set = set()
            for i in quality_sorted:
                if delays[i] <= delay_anchor_cap:
                    anchor_idx.append(i)
                    anchor_set.add(i)
                if len(anchor_idx) >= anchor_target:
                    break

            if len(anchor_idx) > 0:
                fast_sorted = anchor_idx + [i for i in fast_sorted if i not in anchor_set]
                slow_idx = [i for i in slow_idx if i not in anchor_set]
                anchor_applied = len(anchor_idx)
                anchor_client_ids = [
                    int(getattr(ranked_candidates[i], 'client_id', -1))
                    for i in anchor_idx
                ]

                                                                                          
        late_anchor_enable = bool(
            getattr(self, 'mmqs_tot_late_quality_anchor_enabled', False)
        )
        late_anchor_start = float(np.clip(
            self._mmqs_safe_float(
                getattr(self, 'mmqs_tot_late_quality_anchor_start_ratio', 0.60), 0.60
            ), 0.0, 1.0
        ))
        late_anchor_max = max(0, int(self._mmqs_safe_float(
            getattr(self, 'mmqs_tot_late_quality_anchor_max', 1), 1
        )))
        late_anchor_delay_mul = max(1.0, self._mmqs_safe_float(
            getattr(self, 'mmqs_tot_late_quality_anchor_delay_mul', 1.05), 1.05
        ))
        late_anchor_min_quality = float(np.clip(
            self._mmqs_safe_float(
                getattr(self, 'mmqs_tot_late_quality_anchor_min_quality', 0.65), 0.65
            ), 0.0, 1.0
        ))
        if (
            use_tot_late_speed_guard and late_anchor_enable and
            stage_ratio >= late_anchor_start and k_est >= 4 and late_anchor_max > 0
        ):
            delay_anchor_cap = delay_cap * late_anchor_delay_mul + self.mmqs_eps
            quality_sorted = sorted(
                pool_idx,
                key=lambda i: (-_quality(i), delays[i], -float(ranked_scores[i]))
            )
            late_anchor_idx = []
            late_anchor_set = set()
            for i in quality_sorted:
                if _quality(i) < late_anchor_min_quality:
                    continue
                if delays[i] <= delay_anchor_cap:
                    late_anchor_idx.append(i)
                    late_anchor_set.add(i)
                if len(late_anchor_idx) >= late_anchor_max:
                    break

            if len(late_anchor_idx) > 0:
                fast_sorted = late_anchor_idx + [i for i in fast_sorted if i not in late_anchor_set]
                slow_idx = [i for i in slow_idx if i not in late_anchor_set]
                late_anchor_applied = len(late_anchor_idx)
                late_anchor_client_ids = [
                    int(getattr(ranked_candidates[i], 'client_id', -1))
                    for i in late_anchor_idx
                ]

        slow_sorted = sorted(
            slow_idx,
            key=lambda i: (-float(ranked_scores[i]), delays[i])
        )
        remain_idx = list(range(pool_n, len(ranked_candidates)))
        reordered_idx = fast_sorted + slow_sorted + remain_idx
        reordered = [ranked_candidates[i] for i in reordered_idx]

        logging.info(
            'MMQS C23Q speed constraint: k_est=%d pool_n=%d delay_q=%.2f delay_cap=%.4f fast=%d slow=%d stage_ratio=%.4f stage=[%.2f,%.2f] delay_tol=%.4f score_tol=%.4f swap=%d swap_old=%d swap_new=%d anchor=%d anchor_ratio=%.2f anchor_max=%d anchor_delay_mul=%.2f anchor_stage_end=%.2f anchor_ids=%s late_anchor=%d late_anchor_start=%.2f late_anchor_max=%d late_anchor_delay_mul=%.2f late_anchor_min_q=%.2f late_anchor_ids=%s',
            int(k_est), int(pool_n), delay_q, delay_cap, len(fast_sorted), len(slow_sorted),
            stage_ratio, stage_start, stage_end, delay_tol, score_tol,
            int(swap_applied), int(swap_old_id), int(swap_new_id),
            int(anchor_applied), anchor_ratio, int(anchor_max), anchor_delay_mul,
            anchor_stage_end, anchor_client_ids[:5],
            int(late_anchor_applied), late_anchor_start, int(late_anchor_max),
            late_anchor_delay_mul, late_anchor_min_quality, late_anchor_client_ids[:5]
        )
        return reordered

    def update_grads(self, grads, num_samples, client_id='all'):
                                                                                             
                                                                                            
                                                                                    
                                                                                  
        if client_id == 'all':                                   
            self.grads = grads
            self.num_samples = np.reshape(num_samples, (self.n_clients, 1))
            self.avg_grad = np.sum(
                np.multiply(grads, self.num_samples), axis=0
            ) / np.sum(self.num_samples)        

            self.grads_err_mat = np.zeros((self.n_clients, self.n_clients))
            for i in range(self.n_clients):
                self.grads_err_mat[i, :] = np.sum(
                    np.square(self.grads - self.grads[i]), axis=1
                )

        else:                                     
            self.avg_grad -= self.num_samples[client_id] / np.sum(num_samples) *\
                             self.grads[client_id, :]
            self.grads[client_id, :] = grads
            self.num_samples[client_id] = num_samples
            self.avg_grad += self.num_samples[client_id] / np.sum(num_samples) *\
                             self.grads[client_id, :]

            self.grads_err_mat[client_id, :] = np.sum(
                np.square(self.grads - self.grads[client_id]), axis=1
            )
            self.grads_err_mat[:, client_id] = self.grads_err_mat[client_id, :]

                                        
                                                                   
                                                                              
                                                     

                               

    def tier_profiling(self):
                                       
                                                          

                                                   
                                                            

                                              
        sorted_clients = sorted(self.clients, key=lambda c:c.delay)
        for c in sorted_clients:
            print(c.delay)

                                   
        est_clients_per_round = 5
        m = len(self.clients)/est_clients_per_round

        if m < 1:
            m = 1
        elif m < 5:
            m = math.floor(m)
        elif m <= 10:
            m = 5
        else:
            m = 10

                                      
        credits = math.ceil(self.rounds / m)

                                                    
        p = 1/m

                                     
        clients_per_group = math.floor(len(self.clients)/m)

        tiers = {}
        for i in range(0, m):
            if i != m-1:
                temp = sorted_clients[clients_per_group * i : clients_per_group * (i+1)]
            else:
                temp = sorted_clients[clients_per_group * i : ]
            tiers[i] = Tier(temp, p, credits)

        return tiers

    def tier_change_prob(self):
        selected_tier = self.last_select_tier
        mean = sum(
            [client.loss for client in self.tiers[selected_tier].client_list])
        mean /= len(self.tiers[selected_tier].client_list)
        self.tiers[selected_tier].mean_loss = mean

                                                           

                                      
        sorted_tiers = sorted(self.tiers, key=lambda t: self.tiers[t].mean_loss,
                              reverse=True)

                                      
        credit_cnt = 0
        for tier in sorted_tiers:
            print("Tier Loss" + str(tier) + " : " + str(self.tiers[tier].mean_loss))
            print("Tier Credits" + str(tier) + " : " + str(self.tiers[tier].credits))

            if self.tiers[tier].credits > 0:
                credit_cnt = credit_cnt + 1

                                            
        D = credit_cnt * (credit_cnt - 1) / 2

        i = 0
        for tier in sorted_tiers:

            if self.tiers[tier].credits == 0:
                self.tiers[tier].p = 0
                continue
            elif D > 0:
                temp = (credit_cnt-i)/D
                if temp < 0:
                    temp = 0
                self.tiers[tier].p = temp
            else:
                temp = credit_cnt -i
                if temp < 0:
                    temp = 0
                self.tiers[tier].p = temp
            print("Tier " + str(tier) + " : " + str(self.tiers[tier].p))
            i = i + 1

    def select(self, cur_thpt):
                                                  
        candidates = [c for c in self.clients if c.available]
                                                                              
        candidates = [c for c in candidates if c.throughput < (self.thpt_ub - cur_thpt)]
        flag = np.array([c.available and c.throughput < (self.thpt_ub - cur_thpt)
                         for c in self.clients])

        logging.info('select starts!')
        start = time.time()
        if self.select_type == 'mmqs':
            self.current_select_step += 1

        if self.select_type == 'divfl':
            sample_clients = []
            while len(candidates) > 0 and cur_thpt < self.thpt_ub:
                                                                           
                                                                    
                not_selected = np.array([c.available for c in self.clients])
                selected = ~not_selected
                                     

                                                                                               
                if np.sum(selected) > 1:                                           
                    cur_G = np.min(
                        self.grads_err_mat[not_selected][:, selected], axis=1,
                        keepdims=True)
                elif np.sum(selected) == 1:                                      
                    cur_G = self.grads_err_mat[not_selected, selected]
                else:                                                    
                    cur_G = np.max(self.grads_err_mat, axis=1, keepdims=True)
                                           

                                                                                                       
                                                                                            
                err_rdt = np.maximum(
                    cur_G - self.grads_err_mat[not_selected][:, not_selected],
                    0.0)
                                               

                                                                                                            
                total_err_rdt = np.sum(err_rdt, axis=0)
                select_client = candidates[np.argmax(total_err_rdt)]
                                                     
                sample_clients.append(select_client)

                                                   
                select_client.set_unavailable()
                cur_thpt += select_client.throughput
                candidates = [c for c in self.clients if c.available]

            cur_sel_time = time.time() - start
            self.sel_time += cur_sel_time
            logging.info('select time: {}'.format(cur_sel_time))

            return self._ensure_nonempty_selection(sample_clients, cur_thpt)                          

        else:                                                                           

            if self.select_type == 'random':
                                
                random.shuffle(candidates)

            elif self.select_type == 'high_loss_first':
                                                                         
                candidates = sorted(candidates, key=lambda c: c.loss, reverse=True)

            elif self.select_type == 'short_latency_first':
                                                                         
                candidates = sorted(candidates, key=lambda c: c.delay)

            elif self.select_type == 'short_latency_high_loss_first':
                                                        
                losses = np.array([c.loss for c in candidates])
                mean, var = np.mean(losses), np.std(losses)
                losses = (losses - mean) / var
                delays = np.array([c.delay for c in candidates])
                mean, var = np.mean(delays), np.std(delays)
                delays = (delays - mean) / var

                                                                       
                sorted_idx = sorted(range(len(candidates)),
                                    key=lambda i: losses[i] - self.gamma * delays[i],
                                    reverse=True)
                print([losses[i] for i in sorted_idx])
                print([self.gamma * delays[i] for i in sorted_idx])
                candidates = [candidates[i] for i in sorted_idx]

            elif self.select_type == 'tier':
                                                      
                tiers = [num for num in self.tiers]
                tier_prob = [self.tiers[num].p for num in self.tiers]
                selected_tier = random.choices(tiers, weights=tier_prob)[0]
                print('selected_tier: ', selected_tier)
                credits = self.tiers[selected_tier].credits
                while credits == 0:
                    selected_tier = random.choices(tiers, weights=tier_prob)[0]
                    credits = self.tiers[selected_tier].credits

                self.tiers[selected_tier].credits = credits - 1
                self.last_select_tier = selected_tier

                                                      
                candidates = self.tiers[selected_tier].client_list
                random.shuffle(candidates)

            elif self.select_type == 'oort':
                                              
                delays = np.array([c.delay for c in candidates])
                flag = (delays > self.semi_period)
                delays_inv = flag * (1 / delays) ** self.delay_alpha +\
                             (~flag * np.ones((len(candidates),)))

                losses = np.square(np.array([c.loss for c in candidates]))
                candidates_flag = np.array([c.available for c in self.clients])
                num_samples = self.num_samples[candidates_flag].reshape((-1))
                losses = num_samples * np.sqrt(np.divide(losses, num_samples))

                                                                       
                sorted_idx = sorted(range(len(candidates)),
                                    key=lambda i: losses[i] * delays_inv[i],
                                    reverse=True)
                print([losses[i] for i in sorted_idx])
                print([delays_inv[i] for i in sorted_idx])
                candidates = [candidates[i] for i in sorted_idx]

            elif self.select_type == 'mmqs':
                ranked_candidates, mmqs_scores = self._mmqs_v1_rank(
                    candidates, self.current_select_step, cur_thpt
                )
                if ranked_candidates is None:
                    logging.warning('MMQS v1 invalid scores, fallback to random ordering.')
                    random.shuffle(candidates)
                elif len(ranked_candidates) == 0 and len(candidates) > 0:
                    logging.warning('MMQS v1 empty ranking, fallback to random ordering.')
                    random.shuffle(candidates)
                else:
                    candidates = self._mmqs_apply_speed_constraints(
                        ranked_candidates, mmqs_scores, cur_thpt
                    )

            elif 'coreset' in self.select_type:
                                               
                delays = np.array([c.delay for c in candidates])
                delays_inv = (1 / delays) ** self.delay_alpha

                                                                   
                                                          
                if self.select_type == 'coreset_v1':

                                                
                    self.dissimil_mat = self.grads @ self.grads.T
                    np.fill_diagonal(self.dissimil_mat, 0.0)
                                              

                    eta = self.avg_grad @ self.grads[flag].T        
                    v = - np.sum(self.dissimil_mat[flag][:, flag], axis=1) / (
                                len(candidates) - 1)        
                    div = eta + v

                elif self.select_type == 'coreset_v2':

                                          
                    grads_normed = self.grads / np.linalg.norm(self.grads, axis=1).reshape((-1, 1))
                    avg_grad_normed = self.avg_grad / np.linalg.norm(self.avg_grad)

                                                
                    self.dissimil_mat = grads_normed @ grads_normed.T
                    np.fill_diagonal(self.dissimil_mat, 0.0)
                                              

                    eta = avg_grad_normed @ grads_normed[flag].T        
                    v = - np.sum(self.dissimil_mat[flag][:, flag], axis=1) / (
                                len(candidates) - 1)        

                    loss = np.array([c.loss for c in candidates])
                    div = (eta + v) * loss

                elif self.select_type == 'coreset_v3':

                                                
                    self.dissimil_mat = self.grads @ self.grads.T
                    np.fill_diagonal(self.dissimil_mat, 0.0)
                                              

                    eta = self.avg_grad @ self.grads[flag].T        
                    v = - np.sum(self.dissimil_mat[flag][:, flag], axis=1) / (
                                len(candidates) - 1)        

                    loss = np.array([c.loss for c in candidates])
                    div = (eta + v) * loss

                elif self.select_type == 'coreset_v4':

                                          
                    grads_normed = self.grads / np.linalg.norm(self.grads, axis=1).reshape((-1, 1))
                    avg_grad_normed = self.avg_grad / np.linalg.norm(
                        self.avg_grad)

                                                
                    self.dissimil_mat = grads_normed @ grads_normed.T
                    np.fill_diagonal(self.dissimil_mat, 0.0)
                                              

                    eta = avg_grad_normed @ grads_normed[flag].T        
                    v = - np.sum(self.dissimil_mat[flag][:, flag], axis=1) / (
                                len(candidates) - 1)        

                    div = (eta + v)

                else:
                    raise ValueError(
                        "client select type not implemented: {}".format(
                            self.select_type))

                                                       
                max_div, min_div = np.max(div), np.min(div)
                div = (div - min_div) / (max_div - min_div)

                                                                                                    

                                              
                thpt = np.array([c.throughput for c in candidates])

                                           
                                                                                     
                sorted_idx = sorted(range(len(candidates)),
                                    key=lambda i: div[i] * delays_inv[i] / thpt[i],
                                    reverse=True)
                print([div[i] for i in sorted_idx])
                print([delays_inv[i] for i in sorted_idx])
                candidates = [candidates[i] for i in sorted_idx]

            else:
                raise ValueError(
                    "client select type not implemented: {}".format(self.select_type))

                                                                                 
            thpt_list = [c.throughput for c in candidates]
            accum_thpt = [sum(thpt_list[0:i[0]+1]) for i in enumerate(thpt_list)]
                                                  
                              
            k = len(accum_thpt)
            for index, elem in enumerate(accum_thpt):
                if elem > self.thpt_ub - cur_thpt:
                    k = index
                    break
            sample_clients = candidates[:k]
            sample_clients = self._ensure_nonempty_selection(sample_clients, cur_thpt)
            if self.select_type == 'mmqs':
                sample_clients = self._mmqs_apply_coverage_floor(
                    sample_clients=sample_clients,
                    ranked_candidates=candidates,
                    cur_thpt=cur_thpt
                )
            logging.info('Select: {}'.format(sample_clients))
            if self.select_type == 'mmqs':
                logging.info('MMQS v1 k_selected: {} selected_ids: {}'.format(
                    len(sample_clients),
                    [c.client_id for c in sample_clients]
                ))
            logging.info('Max thpt: {} avail thpt: {} select thpt: {}'.format(accum_thpt[-1] if len(accum_thpt) > 0 else 0,
                                                                       self.thpt_ub - cur_thpt,
                                                                       accum_thpt[k-1] if k > 0 else 0))
            logging.info('Select {} out of {}'.format(k, len(candidates)))

        cur_sel_time = time.time() - start
        self.sel_time += cur_sel_time
        logging.info('select time: {}'.format(cur_sel_time))

                               
        return sample_clients
