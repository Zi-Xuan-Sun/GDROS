import time
import os
import shutil

def process_transformer_cfg(cfg):
    log_dir = ''
    if 'critical_params' in cfg:
        critical_params = [cfg[key] for key in cfg.critical_params]
        for name, param in zip(cfg["critical_params"], critical_params):
            log_dir += "{:s}[{:s}]".format(name, str(param))

    return log_dir

def process_cfg(cfg):
    # Use os.path.join to build path, ensuring cross-platform compatibility
    log_dir_parts = ['logs', cfg.name, cfg.transformer]
    log_dir = os.path.join(*log_dir_parts)
    
    critical_params = [cfg.trainer[key] for key in cfg.critical_params]
    for name, param in zip(cfg["critical_params"], critical_params):
        log_dir += "{:s}[{:s}]".format(name, str(param))

    log_dir += process_transformer_cfg(cfg[cfg.transformer])

    now = time.localtime()
    now_time = '{:02d}_{:02d}_{:02d}_{:02d}'.format(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
    log_dir += cfg.suffix + '(' + now_time + ')'
    cfg.log_dir = log_dir
    try:
        os.makedirs(log_dir)
    except FileExistsError:
        print(f"Directory {log_dir} already exists.")

    try:
        configs_dst = os.path.join(log_dir, 'configs')
        if os.path.exists(configs_dst):
            shutil.rmtree(configs_dst)
        shutil.copytree('configs', configs_dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    except Exception as e:
        print(f"Warning: Failed to copy configs directory: {e}")

    try:
        gdrosnet_dst = os.path.join(log_dir, 'GDROSnet')
        if os.path.exists(gdrosnet_dst):
            shutil.rmtree(gdrosnet_dst)
        shutil.copytree('core/GDROSnet', gdrosnet_dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    except Exception as e:
        print(f"Warning: Failed to copy GDROSnet directory: {e}")
