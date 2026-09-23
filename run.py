import argparse
import config
import logging
import os
import shutil
import time
import server
import random
import numpy as np
import torch
import glob
from pathlib import Path

               
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./configs/AVE/hybrid_noniid_ave_tuned.json',
                    help='Federated learning configuration file.')
parser.add_argument('-sel', '--selection', type=str, default='random',
                    choices=['random', 'mmqs'],
                    help='Client selection algorithm (paper delivery subset).')
parser.add_argument('-gamma_from', '--cs_gamma_from', type=float, default=0.2,
                    help='Starting weight for delay in client selection.')
parser.add_argument('-gamma_to', '--cs_gamma_to', type=float, default=0.1,
                    help='Finishing weight for delay in client selection')
parser.add_argument('-alpha', '--cs_alpha', type=float, default=1.0,
                    help='Weights for delays')
parser.add_argument('-ass', '--association', type=str, default='random',
                    choices=['random'],
                    help='Client association algorithm (paper delivery subset).')
parser.add_argument('-phi', '--ca_phi', type=float, default=0.1,
                    help='Weight for throughput balancing in client association.')
parser.add_argument('--delay_mode', type=str, default='uniform',
                    choices=['uniform', 'nycmesh'],
                    help='how to generate network delays')
parser.add_argument('--semi_period', type=float, default=70.0,
                    help='Waiting period for semi-async server')
parser.add_argument('--pca_dim', type=int, default=0,
                    help='Dimensions for PCA')
parser.add_argument('--trial', type=int, default=0,
                    help='id for recording multiple runs and setting seeds')
parser.add_argument('--mmqs_enabled', action='store_true',
                    help='Enable MMQS module skeleton. Default is disabled.')
parser.add_argument('--mmqs_loss_aware_topk_enabled', action='store_true',
                    help='Enable loss-aware Top-K client budget controller (paper Eq. (7)-(10)).')
parser.add_argument('--mmqs_beta', type=float, default=5.0,
                    help='Paper symbol beta controlling the sensitivity of loss-aware Dynamic Top-K.')
parser.add_argument('--mmqs_rho_min', type=float, default=0.75,
                    help='Paper symbol rho_min, lower bound of alpha(t_r).')
parser.add_argument('--mmqs_rho_max', type=float, default=1.0,
                    help='Paper symbol rho_max, upper bound of alpha(t_r).')
parser.add_argument('--mmqs_weight_mode', type=str, default='static',
                    choices=['static', 'tot_api'],
                    help='MMQS weight mode placeholder.')
parser.add_argument('--mmqs_W', type=int, default=0,
                    help='Paper symbol W: sliding window size for historical performance.')
parser.add_argument('--mmqs_tot_api_enabled', action='store_true',
                    help='Enable external ToT API reasoning for MMQS.')
parser.add_argument('--mmqs_tot_api_url', type=str,
                    default='',
                    help='ToT API URL (OpenAI-compatible chat completions). Set this manually; no default credentials are provided.')
parser.add_argument('--mmqs_tot_api_model', type=str,
                    default='',
                    help='ToT API model name. Set this manually before running.')
parser.add_argument('--mmqs_tot_api_timeout', type=float, default=120.0,
                    help='ToT API timeout in seconds.')
parser.add_argument('--mmqs_tot_api_max_tokens', type=int, default=512,
                    help='ToT API max_tokens.')
parser.add_argument('--mmqs_tot_api_temperature', type=float, default=0.2,
                    help='ToT API temperature.')
parser.add_argument('--mmqs_tot_api_top_p', type=float, default=0.7,
                    help='ToT API top_p.')
parser.add_argument('--mmqs_tot_api_retry', type=int, default=1,
                    help='ToT API retry count on failures.')
parser.add_argument('--mmqs_tot_api_key_env', type=str, default='',
                    help='Environment variable containing the ToT API key. Set this manually before running.')
parser.add_argument('--mmqs_tot_q', type=int, default=5,
                    help='Paper symbol Q: candidate branch count requested from ToT API.')
parser.add_argument('--mmqs_tot_api_reward_ema_factor', type=float, default=0.3,
                    help='EMA factor for ToT branch reward memory.')
parser.add_argument('--mmqs_tot_api_memory_bonus', type=float, default=0.1,
                    help='Memory bonus weight used in local branch validation.')
parser.add_argument('--mmqs_tot_api_call_interval', type=int, default=3,
                    help='Resolve ToT API every N select rounds; intermediate rounds reuse cached branch.')
parser.add_argument('--mmqs_tot_api_prune_interval', type=int, default=2,
                    help='Allow ToT prune/evolve every N MMQS selection steps.')
parser.add_argument('--mmqs_tot_api_force_non_stream', action='store_true',
                    help='Force non-stream HTTP mode for ToT API.')
parser.add_argument('--mmqs_quality_guard_enabled', action='store_true',
                    help='Enable MMQS quality guard v1. Default is disabled.')
parser.add_argument('--mmqs_quality_guard_warmup_rounds', type=int, default=1,
                    help='Warmup rounds before quality guard can trigger.')
parser.add_argument('--mmqs_quality_guard_hold_rounds', type=int, default=2,
                    help='Hold rounds when quality guard is triggered.')
parser.add_argument('--mmqs_quality_guard_acc_drop_threshold', type=float, default=0.015,
                    help='Accuracy EMA drop threshold for quality guard trigger.')
parser.add_argument('--mmqs_quality_guard_loss_rise_threshold', type=float, default=0.03,
                    help='Loss EMA rise threshold for quality guard trigger.')
parser.add_argument('--mmqs_quality_guard_ema_factor', type=float, default=0.3,
                    help='EMA factor used by quality guard trend detector.')
parser.add_argument('--prefetch_enabled', action='store_true',
                    help='Enable prefetch skeleton. Default is disabled.')
parser.add_argument('--prefetch_top_m', type=int, default=0,
                    help='Top-M candidates for prefetch skeleton.')
parser.add_argument('--prefetch_strict_cache_match', action='store_true',
                    help='Enable strict cache match placeholder.')
parser.add_argument('--prefetch_delay_reduction_ratio', type=float, default=0.0,
                    help='Delay reduction ratio placeholder. No behavior change in Step A.')
parser.add_argument('--hybrid_v1_enabled', action='store_true',
                    help='Enable hybrid-v1 placeholder switch. Default is disabled.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()

             
logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()), datefmt='%H:%M:%S')


def _env_flag_enabled(name, default=False):
    raw = os.environ.get(name, None)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ('1', 'true', 'yes', 'on')


class _NullLogger(object):
    """No-op logger used by terminal-only mode."""

    def log_value(self, *args, **kwargs):
        return None


def _is_windows_file_in_use(err):
    return int(getattr(err, 'winerror', 0) or 0) == 32


def _resolve_path(raw_path):
    if raw_path is None:
        return None
    p = Path(str(raw_path).strip())
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    return p


def _cleanup_terminal_only_outputs(fl_config, tb_folder):
                                                   
    metrics_path = _resolve_path(getattr(fl_config.paths, 'metrics_csv', None))
    if metrics_path is not None and metrics_path.exists():
        try:
            metrics_path.unlink()
            logging.debug('Terminal-only cleanup removed metrics csv: %s', metrics_path)
        except OSError as e:
            logging.warning('Terminal-only cleanup failed to remove metrics csv %s: %s', metrics_path, e)

                                                             
    saved_model_path = _resolve_path(getattr(fl_config.paths, 'saved_model', None))
    if saved_model_path is not None and saved_model_path.exists():
        try:
            shutil.rmtree(str(saved_model_path), ignore_errors=False)
            logging.debug('Terminal-only cleanup removed saved model dir: %s', saved_model_path)
        except OSError as e:
            logging.warning('Terminal-only cleanup failed to remove saved model dir %s: %s', saved_model_path, e)

                                                             
    tb_path = _resolve_path(tb_folder)
    if tb_path is not None and tb_path.exists():
        try:
            shutil.rmtree(str(tb_path), ignore_errors=False)
            logging.debug('Terminal-only cleanup removed tensorboard dir: %s', tb_path)
        except OSError as e:
            if _is_windows_file_in_use(e):
                logging.debug('Terminal-only cleanup skipped busy tensorboard dir: %s', tb_path)
            else:
                logging.warning('Terminal-only cleanup failed to remove tensorboard dir %s: %s', tb_path, e)


def main():
    """Run a federated learning simulation."""
                     
    random.seed(args.trial)
    np.random.seed(args.trial)
    torch.manual_seed(args.trial)

                             
    fl_config = config.Config(args)
    terminal_only = _env_flag_enabled('MMQS_TERMINAL_ONLY', True)

                                                       
    tb_folder = None
    logger = _NullLogger()

                     
    if not os.path.isdir(fl_config.paths.saved_model):
        os.makedirs(fl_config.paths.saved_model)

                                                    
    server_factory = {
        "basic": lambda: server.Server(fl_config),
        "sync": lambda: server.SyncServer(fl_config),
        "async": lambda: server.AsyncServer(fl_config),
        "hybrid": lambda: server.HybridServer(fl_config),
        
    }
    mode = str(fl_config.server.mode).strip().lower()
    if mode not in server_factory:
        raise ValueError(
            "UNSUPPORTED_SERVER_MODE: {} (supported: {})".format(
                fl_config.server.mode, ",".join(sorted(server_factory.keys()))
            )
        )
    fl_server = server_factory[mode]()
    fl_server.boot()

                            
    fl_server.run(logger=logger)

                                       
                                                             
                                                                         
                                                                          

                         
    for f in glob.glob(fl_config.paths.model + '/global*'):
        os.remove(f)

    if terminal_only:
        _cleanup_terminal_only_outputs(fl_config, tb_folder)

    return fl_server


if __name__ == "__main__":
    st = time.time()
    fl_server = main()
    elapsed = time.time() - st
    try:
        mmqs_mode = str(getattr(getattr(fl_server, 'config', None), 'mmqs').weight_mode).strip().lower()
    except Exception:
        mmqs_mode = ''
    if mmqs_mode == 'tot_api':
        gateways = list(getattr(fl_server, 'gateways', []) or [])
        select_time_sum = 0.0
        for gateway in gateways:
            cs_obj = getattr(gateway, 'cs', None)
            if cs_obj is None:
                continue
            try:
                select_time_sum += float(getattr(cs_obj, 'sel_time', 0.0))
            except (TypeError, ValueError):
                continue
        total_execution_time = max(0.0, float(elapsed) - float(select_time_sum))
    else:
        total_execution_time = max(0.0, float(elapsed))
    logging.info('Total execution time=%.6f s', float(total_execution_time))
