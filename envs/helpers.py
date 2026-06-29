from envs.dmc_wrapper import DMCGym
import numpy as np
import multiprocess as mp
mp.set_start_method('spawn', force=True) 

# ==========================================
# Multiprocessing Vector Environment
# ==========================================
def worker(remote, parent_remote, env_fn):
    """Worker process for stepping an environment."""
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                res = env.step(data)
                
                # Handle standard gym vs gymnasium tuple lengths
                if len(res) == 4:
                    obs, reward, done, info = res
                    term = done
                    trunc = False
                else:
                    obs, reward, term, trunc, info = res
                    done = term or trunc
                
                info['terminated'] = term
                info['truncated'] = trunc
                
                if done:
                    info['terminal_observation'] = obs
                    reset_res = env.reset()
                    obs = reset_res[0] if isinstance(reset_res, tuple) else reset_res
                    
                remote.send((obs, reward, done, info))
                
            elif cmd == 'reset':
                reset_res = env.reset()
                obs = reset_res[0] if isinstance(reset_res, tuple) else reset_res
                remote.send(obs)
                
            elif cmd == 'get_action_space':
                remote.send(env.action_space)
                
            elif cmd == 'close':
                remote.close()
                break
    except KeyboardInterrupt:
        print("Worker KeyboardInterrupt")
    except (EOFError, ConnectionResetError):
        pass
    except Exception as e:
        import traceback
        print(f"Worker Error ({type(e).__name__}): {e}")
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()

class SubprocVecEnv:
    """Lightweight, safe multiprocessing wrapper for gym environments."""
    def __init__(self, env_fns):
        self.closed = False
        self.n_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.n_envs)])
        self.processes = [
            mp.Process(target=worker, args=(work_remote, remote, env_fn))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.processes:
            p.daemon = True # Ensure processes die if main process exits
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        return np.stack(results)
        
    def get_action_space(self):
        self.remotes[0].send(('get_action_space', None))
        return self.remotes[0].recv()

    def close(self):
        if self.closed: return
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()
        self.closed = True

def make_env(domain, task, seed, height, width, terminate_on_limit=True):
    """Top-level function for pickling safely across OSs."""
    camera_id = 1 if domain == 'cartpole' else 0
    return DMCGym(
        domain=domain,
        task=task,
        task_kwargs={'random': seed},
        height=height,
        width=width,
        camera_id=camera_id,
        terminate_on_limit=terminate_on_limit
    )
